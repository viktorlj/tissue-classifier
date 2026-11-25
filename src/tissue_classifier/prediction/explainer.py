"""Coherent 5-fold PermutationExplainer across all 22 classes with Top-3 SHAP Explanations."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
import shap

from ..config import ReferenceData
from ..reporting.formatting import (
    classify_feature_modality,
    format_feature_name,
    format_feature_value,
)
from .predictor import PredictionResult, TissuePredictor

DEFAULT_SHAP_SEED = 20261009
ADDITIVITY_TOLERANCE = 2e-5


def display_values(
    packed: np.ndarray, coordinate_map: np.ndarray, n_features: int
) -> np.ndarray:
    """Mean saved-fold input value across folds retaining each named feature."""
    result = np.zeros((len(packed), n_features), dtype="float32")
    for sample in range(len(result)):
        np.add.at(result[sample], coordinate_map, np.asarray(packed[sample], dtype="float32"))
    counts = np.bincount(coordinate_map, minlength=n_features)[None, :]
    return result / np.maximum(counts, 1)


class FoldCoherentMasker(shap.maskers.Masker):
    """Mask each union feature coherently across all 5 folds using the same donor patient."""

    immutable_outputs = True
    supports_delta_masking = False

    def __init__(
        self,
        foreground: np.ndarray,
        backgrounds: np.ndarray,
        coordinate_map: np.ndarray,
        n_features: int,
    ) -> None:
        self.foreground = np.asarray(foreground, dtype="float32")
        self.backgrounds = np.asarray(backgrounds, dtype="float32")
        self.coordinate_map = np.asarray(coordinate_map, dtype="int64")
        self.shape = (len(self.backgrounds), n_features)
        self.display = display_values(self.foreground[None, :], self.coordinate_map, n_features)[0]
        self._invariants = np.ones(self.shape, dtype=bool)
        for row in range(len(self.backgrounds)):
            np.logical_and.at(
                self._invariants[row],
                self.coordinate_map,
                self.foreground == self.backgrounds[row],
            )

    def _check(self, x: np.ndarray) -> None:
        if np.asarray(x).shape != self.display.shape or not np.array_equal(x, self.display):
            raise ValueError("Masker foreground/display pairing changed")

    def apply_bool_mask(self, bool_mask: np.ndarray) -> tuple[np.ndarray]:
        if bool_mask.shape != (self.shape[1],):
            raise ValueError("Named-feature mask shape mismatch")
        return (
            np.where(
                bool_mask[self.coordinate_map][None, :],
                self.foreground[None, :],
                self.backgrounds,
            ),
        )

    def __call__(self, mask: Any, x: np.ndarray) -> tuple[np.ndarray]:
        self._check(x)
        bool_mask = np.asarray(self._standardize_mask(mask, x), dtype=bool)
        return self.apply_bool_mask(bool_mask)

    def invariants(self, x: np.ndarray) -> np.ndarray:
        self._check(x)
        return self._invariants


class _VaryingSubsetMasker(shap.maskers.Masker):
    """Exact Dummy-Axiom invariant-pruned view over FoldCoherentMasker.

    Features identical between foreground and every background donor across all
    5 folds have mathematically identical 0.0 Shapley value; excluding them from
    PermutationExplainer avoids SHAP's 2 * N_total evaluation floor with zero
    approximation error.
    """

    immutable_outputs = True
    supports_delta_masking = False

    def __init__(self, full_masker: FoldCoherentMasker, varying_indices: np.ndarray) -> None:
        self.full_masker = full_masker
        self.varying_indices = np.asarray(varying_indices, dtype=np.int64)
        self.n_total = full_masker.shape[1]
        self.shape = (full_masker.shape[0], len(self.varying_indices))
        self.display = full_masker.display[self.varying_indices]
        self._invariants = full_masker._invariants[:, self.varying_indices]

    def _check(self, x: np.ndarray) -> None:
        if np.asarray(x).shape != self.display.shape or not np.array_equal(x, self.display):
            raise ValueError("Subset masker foreground/display pairing changed")

    def __call__(self, mask: Any, x: np.ndarray) -> tuple[np.ndarray]:
        self._check(x)
        sub_mask = np.asarray(self._standardize_mask(mask, x), dtype=bool)
        full_mask = np.zeros(self.n_total, dtype=bool)
        full_mask[self.varying_indices] = sub_mask
        return self.full_masker.apply_bool_mask(full_mask)

    def invariants(self, x: np.ndarray) -> np.ndarray:
        self._check(x)
        return self._invariants


@dataclass
class FeatureContribution:
    raw_feature: str
    display_name: str
    formatted_value: str
    numeric_value: float
    modality: str
    shap_value: float
    error_std: float


@dataclass
class ClassSHAPExplanation:
    rank: int
    tumor_type: str
    display_name: str
    base_value: float
    predicted_probability: float
    additivity_residual: float
    explanation: shap.Explanation
    top_positive: list[FeatureContribution]
    top_negative: list[FeatureContribution]
    modality_totals: dict[str, float]


@dataclass
class Top3SHAPResult:
    background_donors: int
    permutations: int
    varying_features: int
    total_union_features: int
    elapsed_seconds: float
    max_additivity_residual: float
    all_values: np.ndarray          # shape (n_features, 22)
    all_base_values: np.ndarray     # shape (22,)
    raw_feature_names: list[str]
    display_feature_names: list[str]
    formatted_feature_values: list[str]
    numeric_display_values: np.ndarray
    top3_explanations: list[ClassSHAPExplanation]

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "background_donors": self.background_donors,
            "permutations": self.permutations,
            "varying_features": self.varying_features,
            "total_union_features": self.total_union_features,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "max_additivity_residual": self.max_additivity_residual,
            "top3_explanations": [
                {
                    "rank": c.rank,
                    "tumor_type": c.tumor_type,
                    "display_name": c.display_name,
                    "base_value": c.base_value,
                    "predicted_probability": c.predicted_probability,
                    "additivity_residual": c.additivity_residual,
                    "modality_signed_sum": c.modality_totals,
                    "top_positive_features": [
                        {
                            "feature": f.raw_feature,
                            "label": f.display_name,
                            "value": f.formatted_value,
                            "modality": f.modality,
                            "shap_value": f.shap_value,
                        }
                        for f in c.top_positive[:8]
                    ],
                    "top_negative_features": [
                        {
                            "feature": f.raw_feature,
                            "label": f.display_name,
                            "value": f.formatted_value,
                            "modality": f.modality,
                            "shap_value": f.shap_value,
                        }
                        for f in c.top_negative[:8]
                    ],
                }
                for c in self.top3_explanations
            ],
        }


def build_hgvsp_lookup(usable_maf: pd.DataFrame) -> dict[str, str]:
    """Extract Mutation_ID -> HGVSp_Short mapping from the sample's usable MAF."""
    if usable_maf.empty or "Mutation_ID" not in usable_maf.columns:
        return {}
    for col in ("HGVSp_Short", "HGVSp", "Amino_Acid_Change"):
        if col in usable_maf.columns:
            lookup: dict[str, str] = {}
            for _, row in usable_maf.iterrows():
                mid = str(row["Mutation_ID"]).strip()
                val = str(row[col]).strip()
                if mid and val and val not in {".", "NA", "nan"}:
                    lookup[mid] = val
            return lookup
    return {}


class SHAPExplainer:
    """Coherent 5-fold PermutationExplainer over all 22 classes of DifferentiableBlend."""

    def __init__(
        self,
        predictor: TissuePredictor,
        ref: ReferenceData,
        n_background: int = 32,
        permutations: int = 8,
        seed: int = DEFAULT_SHAP_SEED,
    ) -> None:
        self.predictor = predictor
        self.ref = ref
        self.n_background = n_background
        self.permutations = permutations
        self.seed = seed
        mode: Literal["primary", "without_cna"] = (
            "primary" if predictor.spec.requires_cna else "without_cna"
        )
        (
            self.backgrounds,
            self.feature_names,
            self.coordinate_map,
            self.donor_ids,
        ) = ref.load_shap_background(mode=mode, n_donors=n_background)

    def explain(
        self,
        packed_vector: np.ndarray,
        prediction: PredictionResult,
        usable_maf: Optional[pd.DataFrame] = None,
    ) -> Top3SHAPResult:
        """Compute simultaneous 22-class SHAP values and build Top-3 Explanations."""
        t0 = time.perf_counter()
        fg = np.asarray(packed_vector, dtype=np.float32)
        if fg.ndim == 2:
            fg = fg[0]

        masker = FoldCoherentMasker(
            foreground=fg,
            backgrounds=self.backgrounds,
            coordinate_map=self.coordinate_map,
            n_features=len(self.feature_names),
        )
        varying_indices = np.flatnonzero((~masker.invariants(masker.display)).any(axis=0))
        varying = int(len(varying_indices))
        n_classes = len(self.predictor.spec.labels)
        values = np.zeros((len(self.feature_names), n_classes), dtype=np.float64)
        error_std = np.zeros((len(self.feature_names), n_classes), dtype=np.float64)

        if varying == 0:
            base_values = self.predictor.predict_packed_batch(fg[None, :])[0].astype(np.float64)
        else:
            sub_masker = _VaryingSubsetMasker(masker, varying_indices)
            sub_names = [self.feature_names[int(i)] for i in varying_indices]
            max_evals = max(self.permutations * (2 * varying + 1), 2 * varying + 1)

            explainer = shap.PermutationExplainer(
                self.predictor.predict_packed_batch,
                sub_masker,
                feature_names=sub_names,
                output_names=self.predictor.spec.labels,
                seed=self.seed,
            )
            result = explainer(
                sub_masker.display[None, :],
                max_evals=max_evals,
                batch_size=64,
                error_bounds=True,
                silent=True,
            )
            values[varying_indices] = np.asarray(result.values[0], dtype=np.float64)
            base_values = np.asarray(result.base_values[0], dtype=np.float64)
            if getattr(result, "error_std", None) is not None:
                error_std[varying_indices] = np.asarray(result.error_std[0], dtype=np.float64)

        expected = self.predictor.predict_packed_batch(fg[None, :])[0].astype(np.float64)
        residuals = np.abs(base_values + values.sum(axis=0) - expected)
        max_residual = float(residuals.max())
        if values.shape != (len(self.feature_names), len(self.predictor.spec.labels)):
            raise ValueError(f"Unexpected SHAP values shape: {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite SHAP values encountered")
        if max_residual >= ADDITIVITY_TOLERANCE:
            raise ValueError(
                f"SHAP local additivity check failed: max residual {max_residual:.3e} >= {ADDITIVITY_TOLERANCE}"
            )

        hgvsp_map = build_hgvsp_lookup(usable_maf) if usable_maf is not None else {}
        disp_names = [
            format_feature_name(fname, hgvsp_lookup=hgvsp_map, max_len=36)
            for fname in self.feature_names
        ]
        fmt_vals = [
            format_feature_value(fname, float(masker.display[i]))
            for i, fname in enumerate(self.feature_names)
        ]
        modalities = [classify_feature_modality(fname) for fname in self.feature_names]

        labels = self.predictor.spec.labels
        top3_exps: list[ClassSHAPExplanation] = []
        for cand in prediction.top3:
            cls_idx = labels.index(cand.tumor_type)
            cls_vals = values[:, cls_idx]
            cls_std = error_std[:, cls_idx]
            cls_base = float(base_values[cls_idx])
            cls_pred = float(expected[cls_idx])
            cls_res = float(residuals[cls_idx])

            exp_obj = shap.Explanation(
                values=cls_vals.astype(np.float32),
                base_values=cls_base,
                data=masker.display.copy(),
                display_data=np.array(fmt_vals, dtype=object),
                feature_names=disp_names,
            )

            pos_idx = np.argsort(-cls_vals, kind="stable")
            neg_idx = np.argsort(cls_vals, kind="stable")

            top_pos = [
                FeatureContribution(
                    raw_feature=self.feature_names[int(i)],
                    display_name=disp_names[int(i)],
                    formatted_value=fmt_vals[int(i)],
                    numeric_value=float(masker.display[int(i)]),
                    modality=modalities[int(i)],
                    shap_value=float(cls_vals[int(i)]),
                    error_std=float(cls_std[int(i)]),
                )
                for i in pos_idx[:12]
                if cls_vals[int(i)] > 1e-6
            ]
            top_neg = [
                FeatureContribution(
                    raw_feature=self.feature_names[int(i)],
                    display_name=disp_names[int(i)],
                    formatted_value=fmt_vals[int(i)],
                    numeric_value=float(masker.display[int(i)]),
                    modality=modalities[int(i)],
                    shap_value=float(cls_vals[int(i)]),
                    error_std=float(cls_std[int(i)]),
                )
                for i in neg_idx[:12]
                if cls_vals[int(i)] < -1e-6
            ]

            mod_totals: dict[str, float] = {}
            for m_name, sv_val in zip(modalities, cls_vals):
                mod_totals[m_name] = mod_totals.get(m_name, 0.0) + float(sv_val)

            top3_exps.append(
                ClassSHAPExplanation(
                    rank=cand.rank,
                    tumor_type=cand.tumor_type,
                    display_name=cand.display_name,
                    base_value=cls_base,
                    predicted_probability=cls_pred,
                    additivity_residual=cls_res,
                    explanation=exp_obj,
                    top_positive=top_pos,
                    top_negative=top_neg,
                    modality_totals=mod_totals,
                )
            )

        elapsed = time.perf_counter() - t0
        return Top3SHAPResult(
            background_donors=len(self.backgrounds),
            permutations=self.permutations,
            varying_features=varying,
            total_union_features=len(self.feature_names),
            elapsed_seconds=elapsed,
            max_additivity_residual=max_residual,
            all_values=values.astype(np.float32),
            all_base_values=base_values.astype(np.float32),
            raw_feature_names=self.feature_names,
            display_feature_names=disp_names,
            formatted_feature_values=fmt_vals,
            numeric_display_values=masker.display,
            top3_explanations=top3_exps,
        )
