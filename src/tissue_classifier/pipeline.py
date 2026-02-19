"""Main pipeline orchestrator for tissue-of-origin classification."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import PipelineConfig, ReferenceData
from .feature_extractors.clinical import extract_clinical_features
from .feature_extractors.cna import extract_cna_features
from .feature_extractors.deepsig import extract_deepsig_features
from .feature_extractors.mutations import extract_mutation_features
from .feature_extractors.mutfreq import extract_mutfreq_features
from .feature_extractors.sv import extract_sv_features
from .feature_extractors.tert import extract_tert_features
from .prediction.predictor import PredictionResult, TissuePredictor
from .preprocessing.feature_compiler import compile_features
from .preprocessing.liftover import liftover_maf, liftover_seg
from .preprocessing.maf_parser import parse_maf
from .preprocessing.seg_parser import parse_seg

logger = logging.getLogger(__name__)


def run_pipeline(config: PipelineConfig) -> dict:
    """Run the full tissue-of-origin classification pipeline.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration.

    Returns
    -------
    dict
        Dictionary with keys: prediction, features_df, report_path, shap_result.
    """
    ref = ReferenceData(config.reference_dir)

    # Parse inputs
    logger.info("Parsing MAF: %s", config.maf_path)
    maf = parse_maf(config.maf_path)

    # Liftover if hg38
    if config.genome == "hg38":
        logger.info("Lifting over from hg38 to hg19")
        maf = liftover_maf(maf)

    seg = None
    if config.seg_path is not None:
        logger.info("Parsing SEG: %s", config.seg_path)
        seg = parse_seg(config.seg_path)
        if config.genome == "hg38":
            seg = liftover_seg(seg)

    sv_df = None
    if config.sv_path is not None:
        logger.info("Parsing SV: %s", config.sv_path)
        sv_df = pd.read_csv(config.sv_path, sep="\t", comment="#", low_memory=False)

    # Extract features
    feature_series = {
        "mutations": extract_mutation_features(maf, ref),
        "sv": extract_sv_features(sv_df, ref),
        "deepsig": extract_deepsig_features(
            maf, ref, mode=config.deepsig_mode,
            docker_image=config.deepsig_docker_image,
        ),
        "mutfreq": extract_mutfreq_features(maf, ref),
        "clinical": extract_clinical_features(age=config.age, sex=config.sex),
        "tert": extract_tert_features(maf, ref),
    }

    if seg is not None:
        feature_series["cna"] = extract_cna_features(seg, ref)

    # Compile features
    features_df = compile_features(feature_series, ref)

    # Save features
    output_dir = Path(config.output_dir) / config.sample_id
    output_dir.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_dir / "features.csv")

    # Predict
    predictor = TissuePredictor(config.model_dir)
    prediction = predictor.predict(features_df)

    # SHAP explanation
    shap_result = None
    try:
        from .prediction.explainer import SHAPExplainer
        explainer = SHAPExplainer(predictor, ref, nsamples=config.shap_nsamples)
        shap_result = explainer.explain(features_df, prediction.predicted_class)
    except Exception as e:
        logger.warning("SHAP explanation failed: %s", e)

    # Generate report
    from .reporting.html_report import generate_html_report
    report_path = output_dir / "report.html"
    generate_html_report(
        prediction=prediction,
        features_df=features_df,
        ref=ref,
        sample_id=config.sample_id,
        genome=config.genome,
        n_mutations=len(maf),
        n_segments=len(seg) if seg is not None else 0,
        has_seg=seg is not None,
        has_sv=sv_df is not None,
        age=config.age,
        sex=config.sex,
        deepsig_status=config.deepsig_mode,
        shap_result=shap_result,
        shap_nsamples=config.shap_nsamples,
        output_path=report_path,
    )

    logger.info("Pipeline complete. Results in %s", output_dir)
    return {
        "prediction": prediction,
        "features_df": features_df,
        "report_path": report_path,
        "shap_result": shap_result,
    }
