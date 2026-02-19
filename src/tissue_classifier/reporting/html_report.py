"""Jinja2 HTML report generation for tissue classification results."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from jinja2 import Environment, FileSystemLoader

from .. import __version__
from ..config import ReferenceData
from ..prediction.predictor import PredictionResult
from ..prediction.explainer import SHAPResult
from .plots import (
    plot_full_probabilities,
    plot_modality_breakdown,
    plot_shap_waterfall,
    plot_top3_predictions,
)

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "templates"


def _confidence_level(confidence: float) -> str:
    if confidence >= 0.7:
        return "high"
    elif confidence >= 0.4:
        return "moderate"
    return "low"


def _nonzero_features_by_modality(
    features_df: "pd.DataFrame",
    ref: ReferenceData,
) -> dict[str, list[str]]:
    """Group non-zero features by their source modality."""
    import pandas as pd

    manifest = pd.read_csv(ref._dir / "feature_manifest.csv")
    step_map = dict(zip(manifest["feature_name"], manifest["source_step"]))
    step_labels = {
        "Step05_Mutations": "Mutations",
        "Step06_SV": "Structural Variants",
        "Step08_DeepSig": "Mutational Signatures",
        "Step09_MutFreq": "Mutation Frequency",
        "Step10_Clinical": "Clinical",
        "Step11_TERT": "TERT Promoter",
        "Step12_CNA": "Copy Number",
    }
    result: dict[str, list[str]] = {v: [] for v in step_labels.values()}
    row = features_df.iloc[0]
    for feat in features_df.columns:
        if row[feat] != 0:
            step = step_map.get(feat, "")
            label = step_labels.get(step, "Other")
            result[label].append(feat)
    # Remove empty modalities
    return {k: v for k, v in result.items() if v}


def generate_html_report(
    prediction: PredictionResult,
    features_df: "pd.DataFrame",
    ref: ReferenceData,
    sample_id: str = "SAMPLE",
    genome: str = "hg19",
    n_mutations: int = 0,
    n_segments: int = 0,
    has_seg: bool = False,
    has_sv: bool = False,
    age: Optional[float] = None,
    sex: Optional[str] = None,
    deepsig_status: str = "Skipped",
    shap_result: Optional[SHAPResult] = None,
    shap_nsamples: int = 500,
    output_path: Optional[Path] = None,
) -> str:
    """Generate an HTML report from prediction results.

    Parameters
    ----------
    prediction : PredictionResult
        Model prediction result.
    features_df : pd.DataFrame
        Compiled feature DataFrame.
    ref : ReferenceData
        Reference data instance.
    sample_id : str
        Sample identifier.
    output_path : Optional[Path]
        If provided, write the HTML to this path.

    Returns
    -------
    str
        HTML string.
    """
    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)))
    template = env.get_template("report.html")

    nonzero_mods = _nonzero_features_by_modality(features_df, ref)
    n_nonzero = sum(len(v) for v in nonzero_mods.values())

    # Generate plots
    plot_top3 = plot_top3_predictions(prediction.top3)
    plot_probabilities = plot_full_probabilities(prediction.all_probabilities)
    plot_modality = plot_modality_breakdown(nonzero_mods)
    plot_shap = None
    if shap_result is not None:
        plot_shap = plot_shap_waterfall(
            shap_result.top_features, prediction.predicted_class,
        )

    html = template.render(
        sample_id=sample_id,
        predicted_class=prediction.predicted_class,
        confidence=prediction.confidence,
        confidence_level=_confidence_level(prediction.confidence),
        top3=prediction.top3,
        plot_top3=plot_top3,
        plot_probabilities=plot_probabilities,
        plot_modality=plot_modality,
        plot_shap=plot_shap,
        nonzero_features_by_modality=nonzero_mods,
        n_nonzero_features=n_nonzero,
        n_total_features=len(features_df.columns),
        genome=genome,
        n_mutations=n_mutations,
        n_segments=n_segments,
        has_seg=has_seg,
        has_sv=has_sv,
        age=age,
        sex=sex,
        deepsig_status=deepsig_status,
        shap_nsamples=shap_nsamples,
        version=__version__,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        n_classes=len(prediction.class_labels),
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(html)
        logger.info("Report written to %s", output_path)

    return html
