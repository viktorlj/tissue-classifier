"""Optional standalone HTML report generator (--html) with Top-3 V20 SHAP waterfalls."""

from __future__ import annotations

import base64
from datetime import datetime
import io
from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .. import __version__
from ..config import PipelineConfig, TISSUE_DISPLAY_NAMES
from ..prediction.explainer import ClassSHAPExplanation, Top3SHAPResult
from ..prediction.predictor import PredictionResult
from ..preprocessing.compiler import CompilationResult
from .formatting import OKABE_ITO, apply_report_rcparams
from .pdf_report import draw_shap_waterfall_ax

TEMPLATE_DIR = Path(__file__).parent / "templates"


def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _plot_probabilities_b64(prediction: PredictionResult) -> str:
    apply_report_rcparams()
    fig, ax = plt.subplots(figsize=(9.5, 3.3))
    items = sorted(prediction.all_probabilities.items(), key=lambda kv: -kv[1])
    codes = [k for k, _ in items]
    probs = [v for _, v in items]
    names = [TISSUE_DISPLAY_NAMES.get(k, k) for k in codes]
    colors = [
        OKABE_ITO["blue"] if i == 0 else ("#56B4E9" if i < 3 else "#CBD5E1")
        for i in range(len(codes))
    ]
    x = np.arange(len(codes))
    ax.bar(x, probs, color=colors, width=0.68)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=36, ha="right", rotation_mode="anchor", fontsize=8.0)
    ax.set_ylim(0, max(max(probs) * 1.15, 0.25))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylabel("Calibrated Probability")
    ax.set_title("22-Class RealMLP + TabM Calibrated Probability Distribution", loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    return _fig_to_b64(fig)


def _plot_waterfall_b64(cls_exp: ClassSHAPExplanation, max_display: int = 14) -> str:
    apply_report_rcparams()
    fig = plt.figure(figsize=(8.8, 0.36 * max_display + 1.4))
    ax = fig.add_axes([0.38, 0.14, 0.58, 0.76])
    draw_shap_waterfall_ax(
        ax, cls_exp, max_display=max_display, title_prefix=f"Rank #{cls_exp.rank}: "
    )
    return _fig_to_b64(fig)


def generate_html_report(
    output_path: Path,
    config: PipelineConfig,
    comp: CompilationResult,
    prediction: Optional[PredictionResult],
    shap_res: Optional[Top3SHAPResult],
) -> Path:
    """Render an optional standalone HTML report mirroring the V20 PDF report."""
    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)), autoescape=True)
    template = env.get_template("report.html")

    prob_b64 = _plot_probabilities_b64(prediction) if prediction is not None else None
    waterfall_cards = []
    if shap_res is not None:
        for exp in shap_res.top3_explanations:
            mdisp = 15 if exp.rank == 1 else 12
            waterfall_cards.append(
                {
                    "rank": exp.rank,
                    "tumor_type": exp.tumor_type,
                    "display_name": exp.display_name,
                    "probability": exp.predicted_probability,
                    "base_value": exp.base_value,
                    "plot_b64": _plot_waterfall_b64(exp, max_display=mdisp),
                    "top_positive": exp.top_positive[:6],
                    "top_negative": exp.top_negative[:4],
                }
            )

    html = template.render(
        sample_id=config.sample_id,
        genome=config.genome,
        age=config.age,
        sex=config.sex,
        true_tumor_type=config.true_tumor_type,
        comp=comp,
        prediction=prediction,
        shap_res=shap_res,
        prob_b64=prob_b64,
        waterfall_cards=waterfall_cards,
        n_usable_mut=len(comp.maf_result.usable_maf),
        n_raw_mut=comp.maf_result.raw_count,
        n_usable_sv=len(comp.sv_result.usable_sv),
        sv_supplied=comp.sv_result.supplied,
        cna_segments=comp.cna_result.n_autosomal_x_segments if comp.cna_result else 0,
        cna_coverage=(
            f"{comp.cna_result.observed_bin_fraction:.1%}" if comp.cna_result else "N/A"
        ),
        version=__version__,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    return output_path
