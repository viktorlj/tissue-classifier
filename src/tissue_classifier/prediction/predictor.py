"""v2.0 40-network cross-architecture (2x RealMLP + 2x TabM) ensemble predictor with DifferentiableBlend and policy evaluation."""

from __future__ import annotations

import hashlib
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import torch
from torch import nn

from ..config import ModelSpec, TISSUE_DISPLAY_NAMES


class FoldWeightedCrossEntropy:
    """Unpickling compatibility stub for frozen balanced-arm RealMLP training criterion."""

    def __init__(self, weights: tuple[float, ...] = ()) -> None:
        self.weights = tuple(weights)
        self.calls = 0


def register_unpickle_shims() -> None:
    """Register training-time module stubs so balanced-arm model.joblib loads cleanly standalone."""
    for pkg_name in ("Step08_TrainRealMLP", "lib"):
        mod_name = f"{pkg_name}.weighted_loss"
        if mod_name not in sys.modules:
            pkg = sys.modules.get(pkg_name) or types.ModuleType(pkg_name)
            mod = types.ModuleType(mod_name)
            mod.FoldWeightedCrossEntropy = FoldWeightedCrossEntropy  # type: ignore[attr-defined]
            pkg.weighted_loss = mod  # type: ignore[attr-defined]
            sys.modules[pkg_name] = pkg
            sys.modules[mod_name] = mod


def resolve_compute_device(requested: Optional[str] = None) -> str:
    if requested is not None:
        if requested == "mps" and not torch.backends.mps.is_available():
            return "cpu"
        return requested
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


BOUNDS_THRESHOLD = 1e-7


class TorchQuantileTransform(nn.Module):
    """Exact pure-PyTorch MPS/CPU port of sklearn QuantileTransformer(output_distribution='normal')."""

    def __init__(self, quantiles: np.ndarray, references: np.ndarray, output_distribution: str = "normal") -> None:
        super().__init__()
        q = np.asarray(quantiles, dtype=np.float32)
        r = np.asarray(references, dtype=np.float32)
        self.output_distribution = output_distribution
        self.register_buffer("q_fwd", torch.from_numpy(np.ascontiguousarray(q.T)))
        self.register_buffer("r_fwd", torch.from_numpy(np.ascontiguousarray(r)))
        self.register_buffer("q_bwd", torch.from_numpy(np.ascontiguousarray((-q[::-1, :]).T)))
        self.register_buffer("r_bwd", torch.from_numpy(np.ascontiguousarray(-r[::-1])))

    @classmethod
    def from_tabm_estimator(cls, estimator: Any) -> tuple["TorchQuantileTransform", torch.Tensor, nn.Module, bool]:
        sub = estimator.alg_interface_.sub_split_interfaces[0]

        def _find_normalizer(obj: Any) -> Any:
            if hasattr(obj, "normalizer_"):
                return obj.normalizer_
            for attr in ("tfms", "tfm"):
                child = getattr(obj, attr, None)
                if child is None:
                    continue
                if hasattr(child, "normalizer_"):
                    return child.normalizer_
                if isinstance(child, (list, tuple, nn.ModuleList)):
                    for item in child:
                        found = _find_normalizer(item)
                        if found is not None:
                            return found
                elif child is not obj:
                    found = _find_normalizer(child)
                    if found is not None:
                        return found
            return None

        sk_tfm = _find_normalizer(sub.tfm_)
        if sk_tfm is None:
            raise ValueError("Could not locate fitted TabrQuantileTransformer inside TabM estimator")
        qt = cls(sk_tfm.quantiles_, sk_tfm.references_, output_distribution=sk_tfm.output_distribution)
        return qt, sub.num_col_mask_.detach().cpu().clone(), sub.model_, bool(sub.config.get("average_logits", False))

    @staticmethod
    def _interp_1d_batched(x_t: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
        q_len = xp.shape[1]
        idx = torch.searchsorted(xp, x_t, right=True)
        idx_hi = idx.clamp(1, q_len - 1)
        idx_lo = idx_hi - 1
        x0 = torch.gather(xp, 1, idx_lo)
        x1 = torch.gather(xp, 1, idx_hi)
        y0 = fp[idx_lo]
        y1 = fp[idx_hi]
        t = ((x_t - x0) / (x1 - x0).clamp(min=1e-30)).clamp(0.0, 1.0)
        out = y0 + t * (y1 - y0)
        out = torch.where(x_t <= xp[:, :1], fp[0], out)
        out = torch.where(x_t >= xp[:, -1:], fp[-1], out)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.transpose(0, 1).contiguous()
        fwd = self._interp_1d_batched(x_t, self.q_fwd, self.r_fwd)
        bwd = -self._interp_1d_batched(-x_t, self.q_bwd, self.r_bwd)
        u = 0.5 * (fwd + bwd)
        u = torch.where(x_t >= self.q_fwd[:, -1:], torch.ones_like(u), u)
        u = torch.where(x_t <= self.q_fwd[:, :1], torch.zeros_like(u), u)
        if self.output_distribution == "normal":
            norm_bound = 5.19933758270342
            u_safe = u.clamp(BOUNDS_THRESHOLD, 1.0 - BOUNDS_THRESHOLD)
            out_t = torch.erfinv((2.0 * u_safe - 1.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7)) * 1.4142135623730951
            out_t = torch.where(u <= BOUNDS_THRESHOLD, torch.full_like(out_t, -norm_bound), out_t)
            out_t = torch.where(u >= 1.0 - BOUNDS_THRESHOLD, torch.full_like(out_t, norm_bound), out_t)
        else:
            out_t = u
        return out_t.transpose(0, 1).contiguous()


class _TabMRuntimeWrapper(nn.Module):
    def __init__(self, qt: TorchQuantileTransform, num_col_mask: torch.Tensor, tabm_model: nn.Module, average_logits: bool) -> None:
        super().__init__()
        self.qt = qt
        self.register_buffer("col_indices", torch.nonzero(num_col_mask, as_tuple=False).squeeze(-1).long())
        self.tabm_model = tabm_model
        self.average_logits = average_logits

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        z = torch.index_select(self.qt(inputs), 1, self.col_indices)
        logits_k = self.tabm_model(z, None).float()
        if self.average_logits:
            return torch.softmax(logits_k.mean(dim=1), dim=-1)
        return torch.softmax(logits_k, dim=-1).mean(dim=1)


class _RealMLPRuntimeWrapper(nn.Module):
    def __init__(self, static_model: nn.Module, dynamic_model: nn.Module) -> None:
        super().__init__()
        self.static_model = static_model
        self.dynamic_model = dynamic_model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        data = {
            "x_cont": inputs,
            "x_cat": torch.empty((len(inputs), 0), dtype=torch.long, device=inputs.device),
        }
        transformed = self.static_model(data)
        logits = self.dynamic_model(
            {key: value.unsqueeze(0) for key, value in transformed.items()}
        )["x_cont"]
        if logits.shape[0] != 1:
            raise ValueError("Unexpected native ensemble axis")
        return torch.softmax(logits[0], dim=-1)


class DifferentiableBlend(nn.Module):
    """Frozen v1 (10-network RealMLP) or v2 (40-network 2x RealMLP + 2x TabM) calibrated ensemble."""

    def __init__(
        self,
        model_dir: Path,
        feature_orders: list[list[str]],
        labels: list[str],
        temperature: float,
        alpha_balanced: float = 0.5,
        device: str = "cpu",
        verify_hashes: bool = True,
        vector_scale_a: Optional[list[float]] = None,
        vector_scale_b: Optional[list[float]] = None,
        pooling: Optional[str] = None,
        views: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        super().__init__()
        register_unpickle_shims()
        self.model_dir = Path(model_dir)
        self.feature_orders = feature_orders
        self.widths = [len(features) for features in feature_orders]
        self.labels = list(labels)
        self.temperature = float(temperature)
        self.alpha_balanced = float(alpha_balanced)
        self.compute_device = torch.device(device)

        if vector_scale_a is not None and vector_scale_b is not None:
            self.register_buffer("vec_a", torch.tensor(vector_scale_a, dtype=torch.float32, device=self.compute_device))
            self.register_buffer("vec_b", torch.tensor(vector_scale_b, dtype=torch.float32, device=self.compute_device))
            self.use_vector_scaling = True
        else:
            self.use_vector_scaling = False

        frozen_meta_path = self.model_dir / "models_frozen.json"
        if views is None and frozen_meta_path.exists():
            meta = json.loads(frozen_meta_path.read_text())
            if "views" in meta:
                views = meta["views"]

        self.is_multiview = views is not None
        self.pooling = pooling if pooling is not None else ("logit" if self.is_multiview else "linear")

        if self.is_multiview:
            self.view_specs = list(views or [])
            total_w = sum(float(v.get("weight", 1.0)) for v in self.view_specs)
            self.view_weights = [float(v.get("weight", 1.0)) / total_w for v in self.view_specs]
            self.view_modules = nn.ModuleList()
            for vspec in self.view_specs:
                vname, vfamily = vspec["name"], vspec["family"]
                per_fold_arms = nn.ModuleList()
                for fold, features in enumerate(feature_orders):
                    for arm in ("unweighted", "balanced"):
                        folder = self.model_dir / vname / arm / f"fold_{fold}"
                        bundle = self._load_verified_bundle(folder, features, verify_hashes)
                        per_fold_arms.append(
                            self._wrap_estimator(bundle["model"], vfamily, features).to(self.compute_device)
                        )
                self.view_modules.append(per_fold_arms)
        else:
            self.static = nn.ModuleList()
            self.dynamic = nn.ModuleList()
            for fold, features in enumerate(feature_orders):
                for arm in ("unweighted", "balanced"):
                    folder = self.model_dir / arm / f"fold_{fold}"
                    bundle = self._load_verified_bundle(folder, features, verify_hashes)
                    estimator = bundle["model"]
                    self._check_identity_converter(estimator, features)
                    native = estimator.alg_interface_.model
                    if native.config.get("n_ens", 1) != 1:
                        raise ValueError("Adapter requires the frozen n_ens=1 contract")
                    self.static.append(native.creator.static_model.to(self.compute_device))
                    self.dynamic.append(native.model.to(self.compute_device))

        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def _load_verified_bundle(self, folder: Path, features: list[str], verify_hashes: bool) -> dict[str, Any]:
        model_path = folder / "model.joblib"
        comp_path = folder / "completion.json"
        if verify_hashes and comp_path.exists():
            comp = json.loads(comp_path.read_text())
            expected_hash = comp.get("files", {}).get("model.joblib")
            if expected_hash and _sha256_file(model_path) != expected_hash:
                raise ValueError(f"SHA-256 mismatch for {model_path}")
        bundle = joblib.load(model_path)
        if list(bundle["features"]) != features or list(bundle["labels"]) != self.labels:
            raise ValueError(f"Frozen fold schema/label mismatch in {model_path}")
        return bundle

    @staticmethod
    def _check_identity_converter(estimator: Any, features: list[str]) -> None:
        transformers = estimator.x_converter_.num_tf.transformers_
        selected = [
            column
            for _name, tfm, columns in transformers
            if tfm != "drop"
            for column in columns
        ]
        if selected != features or np.any(estimator.x_converter_.cat_features):
            raise ValueError("Converter must be identity-ordered continuous inputs")

    def _wrap_estimator(self, estimator: Any, family: str, features: list[str]) -> nn.Module:
        self._check_identity_converter(estimator, features)
        if family == "realmlp":
            native = estimator.alg_interface_.model
            if native.config.get("n_ens", 1) != 1:
                raise ValueError("Adapter requires the frozen n_ens=1 contract")
            return _RealMLPRuntimeWrapper(native.creator.static_model, native.model)
        if family == "tabm":
            qt, mask, model, avg_logits = TorchQuantileTransform.from_tabm_estimator(estimator)
            return _TabMRuntimeWrapper(qt, mask, model, avg_logits)
        raise ValueError(f"Unsupported model family: {family}")

    def forward_components(
        self, packed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (calibrated, raw_blend, unweighted_mean, balanced_mean) on CPU."""
        x = packed.to(device=self.compute_device, dtype=torch.float32)
        folds = torch.split(x, self.widths, dim=1)

        if self.is_multiview:
            unw_log_acc = None
            bal_log_acc = None
            for w_v, per_fold_arms in zip(self.view_weights, self.view_modules):
                unw_folds = [per_fold_arms[2 * fold](inputs) for fold, inputs in enumerate(folds)]
                bal_folds = [per_fold_arms[2 * fold + 1](inputs) for fold, inputs in enumerate(folds)]
                p_u = torch.stack(unw_folds).mean(0)
                p_b = torch.stack(bal_folds).mean(0)
                lu = w_v * torch.log(torch.clamp(p_u, min=1e-12, max=1.0))
                lb = w_v * torch.log(torch.clamp(p_b, min=1e-12, max=1.0))
                unw_log_acc = lu if unw_log_acc is None else unw_log_acc + lu
                bal_log_acc = lb if bal_log_acc is None else bal_log_acc + lb
            unweighted = torch.softmax(unw_log_acc, dim=-1)
            balanced = torch.softmax(bal_log_acc, dim=-1)
        else:
            probabilities = []
            for fold, inputs in enumerate(folds):
                for arm in range(2):
                    i = 2 * fold + arm
                    data = {
                        "x_cont": inputs,
                        "x_cat": torch.empty((len(inputs), 0), dtype=torch.long, device=x.device),
                    }
                    transformed = self.static[i](data)
                    logits = self.dynamic[i](
                        {key: value.unsqueeze(0) for key, value in transformed.items()}
                    )["x_cont"]
                    if logits.shape[0] != 1:
                        raise ValueError("Unexpected native ensemble axis")
                    probabilities.append(torch.softmax(logits[0], dim=-1))
            unweighted = torch.stack(probabilities[0::2]).mean(0)
            balanced = torch.stack(probabilities[1::2]).mean(0)

        if self.pooling == "logit":
            raw = torch.softmax(
                (1.0 - self.alpha_balanced) * torch.log(torch.clamp(unweighted, min=1e-12, max=1.0))
                + self.alpha_balanced * torch.log(torch.clamp(balanced, min=1e-12, max=1.0)),
                dim=-1,
            )
        else:
            raw = (1.0 - self.alpha_balanced) * unweighted + self.alpha_balanced * balanced

        log_raw = torch.log(torch.clamp(raw, min=1e-12, max=1.0))
        if self.use_vector_scaling:
            calibrated = torch.softmax(log_raw * self.vec_a[None, :] + self.vec_b[None, :], dim=-1)
        else:
            calibrated = torch.softmax(log_raw / self.temperature, dim=-1)

        return (
            calibrated.to("cpu"),
            raw.to("cpu"),
            unweighted.to("cpu"),
            balanced.to("cpu"),
        )

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        calibrated, _, _, _ = self.forward_components(packed)
        return calibrated.to(packed.device)


@dataclass
class CandidatePrediction:
    rank: int
    tumor_type: str
    display_name: str
    probability: float
    raw_probability: float
    unweighted_probability: float
    balanced_probability: float


@dataclass
class PredictionResult:
    status: str
    model_id: str
    model_display_name: str
    requires_cna: bool
    device: str
    predicted_class: str
    predicted_display_name: str
    confidence: float
    top3_cumulative_probability: float
    margin_top1_vs_top2: float
    normalized_entropy: float
    policy_tier: str
    policy_tier_code: str
    policy_description: str
    policy_flags: dict[str, bool]
    thresholds: dict[str, float]
    endpoint_agreement: bool
    unweighted_top1: str
    balanced_top1: str
    top3: list[CandidatePrediction]
    all_probabilities: dict[str, float]
    raw_probabilities: dict[str, float]
    labels: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "model_id": self.model_id,
            "model_display_name": self.model_display_name,
            "requires_cna": self.requires_cna,
            "device": self.device,
            "predicted_class": self.predicted_class,
            "predicted_display_name": self.predicted_display_name,
            "confidence": self.confidence,
            "top3_cumulative_probability": self.top3_cumulative_probability,
            "margin_top1_vs_top2": self.margin_top1_vs_top2,
            "normalized_entropy": self.normalized_entropy,
            "policy_tier": self.policy_tier,
            "policy_tier_code": self.policy_tier_code,
            "policy_description": self.policy_description,
            "policy_flags": self.policy_flags,
            "thresholds": self.thresholds,
            "endpoint_agreement": self.endpoint_agreement,
            "unweighted_top1": self.unweighted_top1,
            "balanced_top1": self.balanced_top1,
            "top3": [
                {
                    "rank": c.rank,
                    "tumor_type": c.tumor_type,
                    "display_name": c.display_name,
                    "probability": c.probability,
                    "raw_probability": c.raw_probability,
                    "unweighted_probability": c.unweighted_probability,
                    "balanced_probability": c.balanced_probability,
                }
                for c in self.top3
            ],
            "all_probabilities": self.all_probabilities,
            "raw_probabilities": self.raw_probabilities,
        }


class TissuePredictor:
    """High-level wrapper around frozen DifferentiableBlend for single-sample & SHAP batch inference."""

    def __init__(
        self,
        model_dir: Path,
        spec: Optional[ModelSpec] = None,
        device: Optional[str] = None,
        verify_hashes: bool = True,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.spec = spec if spec is not None else ModelSpec.from_dir(self.model_dir)
        self.device = resolve_compute_device(device)
        self.blend = DifferentiableBlend(
            model_dir=self.model_dir,
            feature_orders=self.spec.fold_feature_orders,
            labels=self.spec.labels,
            temperature=self.spec.temperature,
            alpha_balanced=self.spec.alpha_balanced,
            device=self.device,
            verify_hashes=verify_hashes,
            vector_scale_a=self.spec.vector_scale_a,
            vector_scale_b=self.spec.vector_scale_b,
            pooling=self.spec.pooling,
            views=self.spec.views,
        )

    def predict_packed_batch(self, packed_batch: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        """Predict calibrated probabilities (N, 22) for a batch of packed 5-fold feature vectors."""
        outputs = []
        with torch.no_grad():
            for start in range(0, len(packed_batch), batch_size):
                chunk = torch.from_numpy(
                    np.asarray(packed_batch[start : start + batch_size], dtype=np.float32)
                )
                outputs.append(self.blend(chunk).cpu().numpy())
        probs = np.concatenate(outputs, axis=0)
        if not (np.isfinite(probs).all() and np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)):
            raise ValueError("Nonfinite or unnormalized probabilities from DifferentiableBlend")
        return probs

    def predict_single(self, packed_vector: np.ndarray) -> PredictionResult:
        """Compute calibrated probabilities, Top-3 candidates, and frozen policy tier."""
        vec = np.asarray(packed_vector, dtype=np.float32)
        if vec.ndim == 1:
            vec = vec[None, :]
        with torch.no_grad():
            cal_t, raw_t, unw_t, bal_t = self.blend.forward_components(torch.from_numpy(vec))
        cal = cal_t.numpy()[0].astype(np.float64)
        raw = raw_t.numpy()[0].astype(np.float64)
        unw = unw_t.numpy()[0].astype(np.float64)
        bal = bal_t.numpy()[0].astype(np.float64)

        labels = self.spec.labels
        order = np.argsort(-cal, kind="stable")
        top1_idx = int(order[0])
        top2_idx = int(order[1])
        pred_cls = labels[top1_idx]
        pred_disp = self.spec.display_names.get(
            pred_cls, TISSUE_DISPLAY_NAMES.get(pred_cls, pred_cls)
        )

        p_top1 = float(cal[top1_idx])
        p_top3_sum = float(cal[order[:3]].sum())
        margin = float(cal[top1_idx] - cal[top2_idx])
        entropy = float(-(cal * np.log(np.clip(cal, 1e-12, 1.0))).sum() / np.log(len(labels)))

        unw_top1 = labels[int(np.argsort(-unw, kind="stable")[0])]
        bal_top1 = labels[int(np.argsort(-bal, kind="stable")[0])]
        agree = unw_top1 == bal_top1

        thr = self.spec.thresholds
        t_high = float(thr["high_confidence_top1"])
        t_std = float(thr["standard_confidence_top1"])
        t_diff = float(thr["differential_top3_sum"])

        f_high = p_top1 >= t_high
        f_std = p_top1 >= t_std
        f_diff = p_top3_sum >= t_diff

        if f_high:
            tier = "High Confidence"
            tier_code = "HIGH_CONFIDENCE"
            desc = (
                f"Top-1 probability ({p_top1:.1%}) >= {t_high:.0%} frozen threshold "
                "(95% Top-1 accuracy target)."
            )
        elif f_std:
            tier = "Standard Confidence"
            tier_code = "STANDARD_CONFIDENCE"
            desc = (
                f"Top-1 probability ({p_top1:.1%}) >= {t_std:.0%} frozen threshold "
                "(90% Top-1 accuracy target)."
            )
        elif f_diff:
            tier = "Top-3 Differential"
            tier_code = "TOP3_DIFFERENTIAL"
            desc = (
                f"Below single-tissue threshold ({t_std:.0%}), but Top-3 cumulative probability "
                f"({p_top3_sum:.1%}) >= {t_diff:.0%} frozen threshold (95% Top-3 target)."
            )
        else:
            tier = "Low Confidence"
            tier_code = "LOW_CONFIDENCE"
            desc = (
                f"Below frozen single-tissue ({t_std:.0%}) and Top-3 differential ({t_diff:.0%}) "
                "reporting thresholds; interpret cautiously."
            )

        top3_list = []
        for rank_idx, cls_i in enumerate(order[:3], start=1):
            cls_code = labels[int(cls_i)]
            cls_disp = self.spec.display_names.get(
                cls_code, TISSUE_DISPLAY_NAMES.get(cls_code, cls_code)
            )
            top3_list.append(
                CandidatePrediction(
                    rank=rank_idx,
                    tumor_type=cls_code,
                    display_name=cls_disp,
                    probability=float(cal[cls_i]),
                    raw_probability=float(raw[cls_i]),
                    unweighted_probability=float(unw[cls_i]),
                    balanced_probability=float(bal[cls_i]),
                )
            )

        return PredictionResult(
            status="CALLABLE",
            model_id=self.spec.model_id,
            model_display_name=self.spec.display_name,
            requires_cna=self.spec.requires_cna,
            device=self.device,
            predicted_class=pred_cls,
            predicted_display_name=pred_disp,
            confidence=p_top1,
            top3_cumulative_probability=p_top3_sum,
            margin_top1_vs_top2=margin,
            normalized_entropy=entropy,
            policy_tier=tier,
            policy_tier_code=tier_code,
            policy_description=desc,
            policy_flags={
                "meets_high_confidence_top1": f_high,
                "meets_standard_confidence_top1": f_std,
                "meets_top3_differential": f_diff,
            },
            thresholds=thr,
            endpoint_agreement=agree,
            unweighted_top1=unw_top1,
            balanced_top1=bal_top1,
            top3=top3_list,
            all_probabilities={labels[i]: float(cal[i]) for i in range(len(labels))},
            raw_probabilities={labels[i]: float(raw[i]) for i in range(len(labels))},
            labels=labels,
        )
