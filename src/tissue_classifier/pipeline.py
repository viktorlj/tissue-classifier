"""End-to-end single-sample V20 tissue-of-origin pipeline orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Optional

from . import __version__
from .config import ModelSpec, PipelineConfig, ReferenceData
from .prediction.explainer import SHAPExplainer, Top3SHAPResult
from .prediction.predictor import PredictionResult, TissuePredictor
from .preprocessing.compiler import CompilationResult, compile_sample_features
from .reporting.formatting import format_model_mode, interpret_sbs6_spectrum
from .reporting.html_report import generate_html_report
from .reporting.pdf_report import generate_pdf_report

logger = logging.getLogger(__name__)


@dataclass
class PipelineOutput:
    status: str
    sample_id: str
    output_dir: Path
    report_pdf_path: Path
    prediction_json_path: Path
    features_fold0_path: Optional[Path]
    report_html_path: Optional[Path]
    compilation: CompilationResult
    prediction: Optional[PredictionResult]
    shap_result: Optional[Top3SHAPResult]


def run_pipeline(
    config: PipelineConfig,
    predictor_cache: Optional[dict[Path, TissuePredictor]] = None,
    ref: Optional[ReferenceData] = None,
) -> PipelineOutput:
    """Run V20 preprocessing, ensemble inference, Top-3 SHAP, and vector PDF reporting."""
    ref_data = ref if ref is not None else ReferenceData(config.reference_dir)
    requested_spec = ModelSpec.from_dir(config.model_dir)
    fallback_spec = (
        ModelSpec.from_dir(config.fallback_model_dir)
        if config.fallback_model_dir.exists()
        else None
    )

    logger.info("Compiling 5-fold V20 features for sample '%s'...", config.sample_id)
    comp = compile_sample_features(
        config=config,
        ref=ref_data,
        requested_spec=requested_spec,
        fallback_spec=fallback_spec,
    )

    sample_out_dir = Path(config.output_dir) / config.sample_id
    sample_out_dir.mkdir(parents=True, exist_ok=True)

    features_fold0_path: Optional[Path] = None
    if comp.fold_features:
        features_fold0_path = sample_out_dir / "features_fold0.csv"
        comp.fold_features[0].to_csv(features_fold0_path)

    prediction: Optional[PredictionResult] = None
    shap_result: Optional[Top3SHAPResult] = None

    if comp.status == "CALLABLE" and comp.packed_vector is not None and comp.active_model_spec is not None:
        active_dir = (
            config.model_dir
            if comp.active_model_mode == "primary_50pct_blend" or not requested_spec.requires_cna
            else config.fallback_model_dir
        ).resolve()

        if predictor_cache is not None and active_dir in predictor_cache:
            predictor = predictor_cache[active_dir]
        else:
            n_nets = (len(comp.active_model_spec.views) if comp.active_model_spec.views else 1) * comp.active_model_spec.n_folds * len(comp.active_model_spec.arms)
            logger.info("Loading %d-network v2.0 ensemble from %s...", n_nets, active_dir.name)
            predictor = TissuePredictor(
                model_dir=active_dir,
                spec=comp.active_model_spec,
                device=config.device,
            )
            if predictor_cache is not None:
                predictor_cache[active_dir] = predictor

        prediction = predictor.predict_single(comp.packed_vector)
        logger.info(
            "Predicted %s (%.1f%%, %s)",
            prediction.predicted_class,
            prediction.confidence * 100.0,
            prediction.policy_tier,
        )

        if config.compute_shap and config.shap_permutations > 0:
            logger.info(
                "Computing coherent 5-fold Top-3 SHAP (%d donors, %d permutations)...",
                config.shap_background_size,
                config.shap_permutations,
            )
            explainer = SHAPExplainer(
                predictor=predictor,
                ref=ref_data,
                n_background=config.shap_background_size,
                permutations=config.shap_permutations,
            )
            shap_result = explainer.explain(
                packed_vector=comp.packed_vector,
                prediction=prediction,
                usable_maf=comp.maf_result.usable_maf,
            )
            logger.info(
                "SHAP complete in %.2fs (max additivity residual = %.2e)",
                shap_result.elapsed_seconds,
                shap_result.max_additivity_residual,
            )
    else:
        logger.warning(
            "Sample '%s' returned TECHNICAL_NO_CALL (%s): %s",
            config.sample_id,
            comp.no_call_reason,
            comp.diagnostic_message,
        )

    # Write prediction.json
    sbs_interp = interpret_sbs6_spectrum(comp.spectrum_df, comp.maf_result.usable_maf)
    pred_dict = prediction.to_dict() if prediction is not None else None
    if pred_dict is not None:
        pred_dict["model_display_name"] = format_model_mode(comp.active_model_mode)

    pred_payload: dict[str, Any] = {
        "model_name": "TOPO",
        "version": __version__,
        "intended_use": "For Research Use Only — Not for Diagnostic Use",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "sample_id": config.sample_id,
        "status": comp.status,
        "active_model_mode": comp.active_model_mode,
        "model_mode_display": format_model_mode(
            comp.active_model_mode,
            is_no_call=(comp.status == "TECHNICAL_NO_CALL" or prediction is None),
        ),
        "allow_no_cna": config.allow_no_cna,
        "no_call_reason": comp.no_call_reason,
        "diagnostic_message": comp.diagnostic_message,
        "fallback_warning": comp.fallback_warning,
        "warnings": comp.warnings,
        "inputs": {
            "genome": config.genome,
            "age": config.age,
            "sex": config.sex,
            "raw_mutations": comp.maf_result.raw_count,
            "usable_mutations_274g": len(comp.maf_result.usable_maf),
            "sv_supplied": comp.sv_result.supplied,
            "raw_sv": comp.sv_result.raw_count,
            "usable_sv_274g": len(comp.sv_result.usable_sv),
            "seg_supplied": comp.seg_result.supplied,
            "seg_valid": comp.seg_result.valid,
            "cna_observed_bin_fraction": (
                comp.cna_result.observed_bin_fraction if comp.cna_result is not None else None
            ),
            "sbs6_snp_count": int(comp.spectrum_df["Spectrum_SNP_Count"].iloc[0]),
        },
        "sbs6_interpretation": sbs_interp.to_dict(),
        "prediction": pred_dict,
        "shap": shap_result.to_summary_dict() if shap_result is not None else None,
    }
    if config.true_tumor_type is not None:
        pred_payload["true_tumor_type"] = config.true_tumor_type

    prediction_json_path = sample_out_dir / "prediction.json"
    prediction_json_path.write_text(json.dumps(pred_payload, indent=2) + "\n")

    report_pdf_path = sample_out_dir / "report.pdf"
    generate_pdf_report(
        output_path=report_pdf_path,
        config=config,
        comp=comp,
        prediction=prediction,
        shap_res=shap_result,
    )

    report_html_path: Optional[Path] = None
    if config.generate_html:
        report_html_path = sample_out_dir / "report.html"
        generate_html_report(
            output_path=report_html_path,
            config=config,
            comp=comp,
            prediction=prediction,
            shap_res=shap_result,
        )

    return PipelineOutput(
        status=comp.status,
        sample_id=config.sample_id,
        output_dir=sample_out_dir,
        report_pdf_path=report_pdf_path,
        prediction_json_path=prediction_json_path,
        features_fold0_path=features_fold0_path,
        report_html_path=report_html_path,
        compilation=comp,
        prediction=prediction,
        shap_result=shap_result,
    )
