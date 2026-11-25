"""Multi-page vector PDF clinical & molecular report generator for TOPO."""

from __future__ import annotations

from datetime import datetime
import textwrap
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Patch
import numpy as np

from .. import __version__
from ..config import PipelineConfig, TISSUE_DISPLAY_NAMES
from ..prediction.explainer import ClassSHAPExplanation, Top3SHAPResult
from ..prediction.predictor import PredictionResult
from ..preprocessing.compiler import CompilationResult
from .formatting import (
    OKABE_ITO,
    SBS6Interpretation,
    apply_report_rcparams,
    format_model_mode,
    format_model_tag,
    format_tissue_name,
    interpret_sbs6_spectrum,
)
from .logo import draw_topo_svg

PAGE_WIDTH = 8.5
PAGE_HEIGHT = 11.0

TIER_COLORS = {
    "HIGH_CONFIDENCE": "#009E73",
    "STANDARD_CONFIDENCE": "#0072B2",
    "TOP3_DIFFERENTIAL": "#E69F00",
    "LOW_CONFIDENCE": "#D55E00",
}


def _new_page(
    sample_id: str,
    model_tag: str,
    timestamp: str,
    page_num: int,
    total_pages: int,
) -> plt.Figure:
    fig = plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT), facecolor="white")
    # Running header & footer canvas
    bg = fig.add_axes([0, 0, 1, 1], frameon=False)
    bg.set_xlim(0, 1)
    bg.set_ylim(0, 1)
    bg.set_xticks([])
    bg.set_yticks([])

    if page_num == 1:
        # Full TOPO logotype lockup on Page 1 header
        draw_topo_svg(fig, [0.058, 0.928, 0.235, 0.058], variant="logo", include_subtitle=True)
        bg.text(
            0.94,
            0.968,
            "Clinical & Molecular Report",
            fontsize=10.5,
            fontweight="bold",
            color="#0F172A",
            va="center",
            ha="right",
        )
        bg.text(
            0.94,
            0.944,
            f"Sample: {sample_id[:34]}   |   Page {page_num} of {total_pages}",
            fontsize=7.8,
            color="#475569",
            va="center",
            ha="right",
        )
        bg.plot([0.06, 0.94], [0.920, 0.920], color="#CBD5E1", lw=0.95)
    else:
        # Compact TOPO icon + running title on continuation pages
        draw_topo_svg(fig, [0.058, 0.952, 0.036, 0.029], variant="icon", include_subtitle=False)
        bg.text(
            0.097,
            0.965,
            "TOPO — Clinical & Molecular Report",
            fontsize=9.2,
            fontweight="bold",
            color="#0F172A",
            va="center",
            ha="left",
        )
        bg.text(
            0.94,
            0.965,
            f"Sample: {sample_id[:34]}   |   Page {page_num} of {total_pages}",
            fontsize=7.8,
            color="#475569",
            va="center",
            ha="right",
        )
        bg.plot([0.06, 0.94], [0.947, 0.947], color="#CBD5E1", lw=0.9)

    bg.plot([0.06, 0.94], [0.038, 0.038], color="#CBD5E1", lw=0.8)
    bg.text(
        0.06,
        0.023,
        f"TOPO v{__version__}  •  {model_tag}  •  Generated {timestamp}",
        fontsize=7.3,
        color="#64748B",
        va="center",
        ha="left",
    )
    bg.text(
        0.94,
        0.023,
        "For Research Use Only — Not for Diagnostic Use",
        fontsize=7.3,
        color="#64748B",
        va="center",
        ha="right",
    )
    return fig


def _draw_card(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    facecolor: str = "#F8FAFC",
    edgecolor: str = "#CBD5E1",
    lw: float = 0.9,
    rounding: float = 0.010,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.0,rounding_size={rounding}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(patch)


def _format_delta(val: float) -> str:
    if abs(val) >= 0.01:
        return f"{val:+.3f}"
    if abs(val) >= 0.0005:
        return f"{val:+.4f}"
    return f"{val:+.1e}"


def draw_shap_waterfall_ax(
    ax: plt.Axes,
    cls_exp: ClassSHAPExplanation,
    max_display: int = 15,
    title_prefix: str = "",
) -> None:
    """Render a crisp, non-overlapping vector SHAP waterfall onto a provided Axes."""
    exp = cls_exp.explanation
    vals = np.asarray(exp.values, dtype=np.float64)
    base = float(exp.base_values)
    pred = float(cls_exp.predicted_probability)
    disp_data = exp.display_data if exp.display_data is not None else exp.data
    fnames = list(exp.feature_names)

    n_total = len(vals)
    k = min(max_display, n_total)
    order = np.argsort(-np.abs(vals), kind="stable")

    if n_total > k:
        n_ind = k - 1
        top_indices = order[:n_ind]
        rem_sum = float(vals[order[n_ind:]].sum())
        # Rows ordered from bottom (y=0) to top (y=k-1)
        deltas = [rem_sum] + [float(vals[i]) for i in reversed(top_indices)]
        ylabels = [f"Remaining {n_total - n_ind:,} features"] + [
            f"{disp_data[i]} = {fnames[i]}" for i in reversed(top_indices)
        ]
    else:
        top_indices = order[:k]
        deltas = [float(vals[i]) for i in reversed(top_indices)]
        ylabels = [f"{disp_data[i]} = {fnames[i]}" for i in reversed(top_indices)]

    n_rows = len(deltas)
    starts = np.zeros(n_rows, dtype=np.float64)
    ends = np.zeros(n_rows, dtype=np.float64)
    cursor = base
    for r in range(n_rows):
        starts[r] = cursor
        cursor += deltas[r]
        ends[r] = cursor

    all_x = np.concatenate([[base, pred], starts, ends])
    min_x = float(all_x.min())
    max_x = float(all_x.max())
    span = max(max_x - min_x, 0.035)
    pad = 0.24 * span
    ax.set_xlim(min_x - pad, max_x + pad)
    ax.set_ylim(-0.75, n_rows - 0.15)

    # Prior & Final vertical reference lines
    ax.axvline(base, color="#94A3B8", linestyle=":", linewidth=0.9, zorder=1)
    ax.axvline(pred, color="#334155", linestyle="--", linewidth=0.9, zorder=1)

    bar_h = 0.62
    text_offset = 0.015 * span

    for r in range(n_rows):
        d = deltas[r]
        left = min(starts[r], ends[r])
        width = max(abs(d), 0.0015 * span)
        color = OKABE_ITO["blue"] if d >= 0 else OKABE_ITO["vermilion"]
        ax.barh(
            r,
            width,
            left=left,
            height=bar_h,
            color=color,
            edgecolor="none",
            alpha=0.92,
            zorder=3,
        )
        # Step connector to next row
        if r < n_rows - 1:
            ax.plot(
                [ends[r], ends[r]],
                [r + bar_h / 2.0, r + 1 - bar_h / 2.0],
                color="#94A3B8",
                linestyle="--",
                linewidth=0.65,
                zorder=2,
            )
        lbl = _format_delta(d)
        if d >= 0:
            ax.text(
                ends[r] + text_offset,
                r,
                lbl,
                va="center",
                ha="left",
                fontsize=7.6,
                fontweight="bold",
                color=color,
                zorder=4,
            )
        else:
            ax.text(
                ends[r] - text_offset,
                r,
                lbl,
                va="center",
                ha="right",
                fontsize=7.6,
                fontweight="bold",
                color=color,
                zorder=4,
            )

    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(ylabels, fontsize=7.8, color="#1E293B")
    ax.grid(axis="x", linestyle=":", alpha=0.35, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#94A3B8")

    ax.set_xlabel(
        f"Calibrated Probability Scale   (Background Prior E[f(X)] = {base:.3f}  →  "
        f"Calibrated P = {pred:.3f})",
        fontsize=8.0,
        color="#334155",
        labelpad=4,
    )
    heading = (
        f"{title_prefix}{cls_exp.display_name} ({cls_exp.tumor_type})  —  "
        f"Calibrated P = {pred:.2%}"
    )
    ax.set_title(heading, loc="left", fontsize=8.8, fontweight="bold", color="#0F172A", pad=14)
    ax.text(
        0.0,
        1.015,
        "Blue (+): supporting evidence   |   Vermilion (–): opposing evidence",
        transform=ax.transAxes,
        fontsize=7.2,
        color="#475569",
        va="bottom",
        ha="left",
    )


def _render_page1_summary(
    pdf: PdfPages,
    config: PipelineConfig,
    comp: CompilationResult,
    prediction: Optional[PredictionResult],
    sbs_interp: SBS6Interpretation,
    timestamp: str,
    model_tag: str,
    total_pages: int,
) -> None:
    fig = _new_page(config.sample_id, model_tag, timestamp, 1, total_pages)
    canvas = fig.axes[0]

    # 1. Sample Metadata Header Card
    _draw_card(canvas, 0.06, 0.836, 0.88, 0.072, facecolor="#F8FAFC", edgecolor="#CBD5E1")
    genome_str = "hg19 (GRCh37)" if config.genome == "hg19" else "hg38 → hg19 (pyliftover)"
    age_str = f"{config.age:.0f} yr" if config.age is not None else "Imputed"
    sex_str = config.sex if config.sex is not None else "Imputed"
    mode_str = format_model_mode(
        comp.active_model_mode,
        is_no_call=(comp.status == "TECHNICAL_NO_CALL" or prediction is None),
    )

    meta_items = [
        (0.075, 0.188, 0.887, "Sample ID", config.sample_id[:26]),
        (0.075, 0.188, 0.857, "Genome Build", genome_str),
        (0.405, 0.505, 0.887, "Age / Sex", f"{age_str}  /  {sex_str}"),
        (0.405, 0.505, 0.857, "Model Mode", mode_str),
        (0.735, 0.835, 0.887, "Report Date", timestamp[:10]),
        (
            0.735,
            0.835,
            0.857,
            "True Origin" if config.true_tumor_type else "Model",
            format_tissue_name(config.true_tumor_type)
            if config.true_tumor_type
            else f"TOPO v{__version__}",
        ),
    ]
    for x_lbl, x_val, y_pos, k_lbl, v_lbl in meta_items:
        canvas.text(x_lbl, y_pos, f"{k_lbl}:", fontsize=7.4, color="#64748B", fontweight="bold", va="center")
        canvas.text(x_val, y_pos, v_lbl, fontsize=7.5, color="#0F172A", va="center")

    # Handle TECHNICAL_NO_CALL Page 1
    if comp.status == "TECHNICAL_NO_CALL" or prediction is None:
        _draw_card(
            canvas,
            0.06,
            0.595,
            0.88,
            0.225,
            facecolor="#FEF2F2",
            edgecolor="#DC2626",
            lw=1.4,
        )
        canvas.text(
            0.085,
            0.788,
            "TECHNICAL NO-CALL — PREDICTION WITHHELD BY TOPO QUALITY GATE",
            fontsize=11.0,
            fontweight="bold",
            color="#991B1B",
            va="center",
        )
        canvas.text(
            0.085,
            0.755,
            f"Reason Code:  {comp.no_call_reason}",
            fontsize=9.5,
            fontweight="bold",
            color="#7F1D1D",
            va="center",
        )
        wrapped_msg = "\n".join(textwrap.wrap(comp.diagnostic_message or "", width=95))
        canvas.text(
            0.085,
            0.715,
            wrapped_msg,
            fontsize=8.3,
            color="#1E293B",
            va="top",
            linespacing=1.4,
        )
        remediation = (
            "Recommended Action: Supply a matched non-overlapping hg19/hg38 copy-number segmentation (.seg) file "
            "covering autosomal chromosomes, or pass --allow-no-cna to explicitly invoke the TOPO CNA-free "
            "fallback model."
            if comp.no_call_reason == "MISSING_OR_INCOMPATIBLE_CNA"
            else "Recommended Action: Verify somatic variant filtering, genome build (--genome hg19 vs hg38), and "
            "coverage across the shared 274-gene panel target regions."
        )
        canvas.text(
            0.085,
            0.635,
            "\n".join(textwrap.wrap(remediation, width=98)),
            fontsize=8.0,
            color="#475569",
            style="italic",
            va="top",
            linespacing=1.35,
        )

        # Input Modality QC Table on lower half of No-Call Page 1
        _draw_qc_card(canvas, comp, config, sbs_interp, x=0.06, y=0.25, w=0.88, h=0.31)
        pdf.savefig(fig)
        plt.close(fig)
        return

    # CALLABLE Page 1
    cursor_top = 0.820
    if comp.fallback_warning:
        _draw_card(
            canvas,
            0.06,
            0.750,
            0.88,
            0.068,
            facecolor="#FFFBEB",
            edgecolor="#D97706",
            lw=1.3,
        )
        canvas.text(
            0.08,
            0.800,
            "WARNING — TOPO NO-CNA FALLBACK MODEL ACTIVE (--allow-no-cna)",
            fontsize=8.8,
            fontweight="bold",
            color="#92400E",
            va="center",
        )
        warn_lines = "\n".join(textwrap.wrap(comp.fallback_warning, width=106))
        canvas.text(
            0.08,
            0.780,
            warn_lines,
            fontsize=7.5,
            color="#78350F",
            va="top",
            linespacing=1.28,
        )
        cursor_top = 0.736

    # Prominent Prediction & Policy Tier Callout Card
    card_h = 0.118
    card_y = cursor_top - card_h
    tier_color = TIER_COLORS.get(prediction.policy_tier_code, "#334155")
    _draw_card(canvas, 0.06, card_y, 0.88, card_h, facecolor="#F8FAFC", edgecolor=tier_color, lw=1.4)

    canvas.text(
        0.08,
        card_y + card_h - 0.022,
        "PREDICTED TISSUE OF ORIGIN"
        if prediction.policy_tier_code != "TOP3_DIFFERENTIAL"
        else "LEAD CANDIDATE (REPORT AS TOP-3 DIFFERENTIAL)",
        fontsize=7.8,
        fontweight="bold",
        color="#475569",
        va="center",
    )
    canvas.text(
        0.08,
        card_y + card_h - 0.055,
        f"{prediction.predicted_display_name}  ({prediction.predicted_class})",
        fontsize=14.5,
        fontweight="bold",
        color="#0F172A",
        va="center",
    )
    desc_wrapped = "\n".join(textwrap.wrap(prediction.policy_description, width=68))
    canvas.text(
        0.08,
        card_y + 0.036,
        desc_wrapped,
        fontsize=7.6,
        color="#334155",
        va="top",
        linespacing=1.25,
    )

    # Right side of Prediction Callout: Probability + Policy Pill Badge
    canvas.text(
        0.92,
        card_y + card_h - 0.032,
        f"{prediction.confidence:.1%}",
        fontsize=18.0,
        fontweight="bold",
        color=tier_color,
        va="center",
        ha="right",
    )
    # Policy Pill
    pill_w, pill_h = 0.21, 0.028
    _draw_card(
        canvas,
        0.92 - pill_w,
        card_y + card_h - 0.076,
        pill_w,
        pill_h,
        facecolor=tier_color,
        edgecolor=tier_color,
        lw=0.5,
        rounding=0.012,
    )
    canvas.text(
        0.92 - pill_w / 2.0,
        card_y + card_h - 0.076 + pill_h / 2.0,
        prediction.policy_tier.upper(),
        fontsize=7.6,
        fontweight="bold",
        color="white",
        va="center",
        ha="center",
    )
    canvas.text(
        0.92,
        card_y + 0.016,
        f"Top-3 Sum: {prediction.top3_cumulative_probability:.1%}   |   "
        f"Margin: {prediction.margin_top1_vs_top2:+.1%}   |   "
        f"Arm Agree: {'Yes' if prediction.endpoint_agreement else 'No'}",
        fontsize=7.3,
        color="#475569",
        va="center",
        ha="right",
    )

    # Middle Section: Left = Top-3 Horizontal Bar Chart, Right = Modality QC Summary Card
    mid_top = card_y - 0.020
    mid_h = 0.225
    mid_y = mid_top - mid_h

    ax_top3 = fig.add_axes([0.20, mid_y + 0.032, 0.27, mid_h - 0.060])
    ranks_rev = list(reversed(prediction.top3))
    y_pos = np.arange(len(ranks_rev))
    probs_rev = [c.probability for c in ranks_rev]
    bar_colors = ["#94A3B8", "#56B4E9", OKABE_ITO["blue"]][-len(ranks_rev) :]
    ax_top3.barh(y_pos, probs_rev, height=0.56, color=bar_colors, edgecolor="none")
    ax_top3.set_yticks(y_pos)
    ax_top3.set_yticklabels(
        [f"#{c.rank} {c.display_name}" for c in ranks_rev],
        fontsize=8.0,
        fontweight="bold",
        color="#1E293B",
    )
    ax_top3.set_xlim(0, 1.24)
    ax_top3.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
    ax_top3.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=7.4)
    t_std = prediction.thresholds["standard_confidence_top1"]
    t_high = prediction.thresholds["high_confidence_top1"]
    ax_top3.axvline(t_std, color="#64748B", linestyle="--", lw=0.8, alpha=0.7)
    ax_top3.axvline(t_high, color="#009E73", linestyle=":", lw=0.9, alpha=0.8)
    for idx_b, p_val in enumerate(probs_rev):
        ax_top3.text(
            p_val + 0.025,
            idx_b,
            f"{p_val:.1%}",
            va="center",
            ha="left",
            fontsize=8.0,
            fontweight="bold",
            color="#0F172A",
        )
    ax_top3.spines["top"].set_visible(False)
    ax_top3.spines["right"].set_visible(False)
    ax_top3.set_title("Top-3 Candidate Differential", loc="left", fontsize=9.2, pad=6)
    ax_top3.set_xlabel("Calibrated Probability", fontsize=7.8)

    _draw_qc_card(canvas, comp, config, sbs_interp, x=0.51, y=mid_y, w=0.43, h=mid_h)

    # Bottom Section: Full 22-Class Calibrated Probability Bar Chart
    bot_y = 0.124
    bot_h = max(mid_y - bot_y - 0.052, 0.150)
    ax_all = fig.add_axes([0.085, bot_y, 0.855, bot_h])
    sorted_items = sorted(prediction.all_probabilities.items(), key=lambda kv: -kv[1])
    cls_codes = [k for k, _ in sorted_items]
    cls_probs = [v for _, v in sorted_items]
    cls_short = [TISSUE_DISPLAY_NAMES.get(k, k) for k in cls_codes]
    colors_22 = [
        OKABE_ITO["blue"] if i == 0 else ("#56B4E9" if i < 3 else "#CBD5E1")
        for i in range(len(cls_codes))
    ]
    x22 = np.arange(len(cls_codes))
    ax_all.bar(x22, cls_probs, width=0.68, color=colors_22, edgecolor="#64748B", linewidth=0.35)
    ax_all.set_ylim(0, max(max(cls_probs) * 1.18, 0.25))
    ax_all.set_xticks(x22)
    ax_all.set_xticklabels(cls_short, rotation=38, ha="right", rotation_mode="anchor", fontsize=6.6)
    ax_all.set_ylabel("Calibrated P", fontsize=7.8)
    ax_all.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax_all.grid(axis="y", linestyle=":", alpha=0.35)
    ax_all.spines["top"].set_visible(False)
    ax_all.spines["right"].set_visible(False)
    ax_all.set_title(
        "All 22 Tumor Classes — TOPO Calibrated Probability Distribution",
        loc="left",
        fontsize=9.2,
        pad=5,
    )
    for i in range(min(3, len(cls_probs))):
        if cls_probs[i] >= 0.015:
            ax_all.text(
                x22[i],
                cls_probs[i] + max(max(cls_probs) * 0.025, 0.008),
                f"{cls_probs[i]:.1%}",
                ha="center",
                va="bottom",
                fontsize=6.8,
                fontweight="bold",
                color="#0F172A",
            )

    pdf.savefig(fig)
    plt.close(fig)


def _draw_qc_card(
    canvas: plt.Axes,
    comp: CompilationResult,
    config: PipelineConfig,
    sbs_interp: SBS6Interpretation,
    x: float,
    y: float,
    w: float,
    h: float,
) -> None:
    _draw_card(canvas, x, y, w, h, facecolor="#F8FAFC", edgecolor="#CBD5E1")
    canvas.text(
        x + 0.02,
        y + h - 0.024,
        "Input Modality & Quality Control Summary",
        fontsize=8.8,
        fontweight="bold",
        color="#0F172A",
        va="center",
    )

    n_mut = len(comp.maf_result.usable_maf)
    raw_mut = comp.maf_result.raw_count
    sv_str = (
        f"{len(comp.sv_result.usable_sv)} usable ({comp.sv_result.raw_count} raw)"
        if comp.sv_result.supplied
        else "WARN: Not supplied (0 SVs assumed)"
    )
    if comp.cna_result is not None:
        cna_str = (
            f"{comp.cna_result.n_autosomal_x_segments} segs "
            f"({comp.cna_result.observed_bin_fraction:.1%} bins)"
        )
    elif comp.seg_result.supplied:
        cna_str = f"Invalid ({(comp.seg_result.error_reason or '')[:24]})"
    else:
        cna_str = "Not supplied"

    if sbs_interp.evaluable:
        spec_str = f"{sbs_interp.n_snvs} SNVs — {sbs_interp.short_tag}"
    else:
        spec_str = f"<4 SNVs ({sbs_interp.n_snvs}; not evaluable)"

    clin_str = (
        f"Age={'Yes' if config.age is not None else 'Imputed'}, "
        f"Sex={'Yes' if config.sex is not None else 'Imputed'}"
    )

    rows = [
        ("Mutations (274g)", f"{n_mut} usable ({raw_mut} raw)"),
        ("Structural Vars", sv_str),
        ("Copy Number", cna_str),
        ("SBS-6 Spectrum", spec_str),
        ("Clinical", clin_str),
    ]
    step = (h - 0.056) / len(rows)
    for idx, (rk, rv) in enumerate(rows):
        ry = y + h - 0.058 - idx * step
        canvas.text(x + 0.02, ry, rk + ":", fontsize=7.4, fontweight="bold", color="#475569", va="center")
        val_color = "#B45309" if "WARN" in rv or "Invalid" in rv or "Not supplied" in rv else "#0F172A"
        canvas.text(x + 0.155, ry, rv[:38], fontsize=7.4, color=val_color, va="center")


def _render_page2_rank1_shap(
    pdf: PdfPages,
    config: PipelineConfig,
    shap_res: Top3SHAPResult,
    timestamp: str,
    model_tag: str,
    total_pages: int,
) -> None:
    fig = _new_page(config.sample_id, model_tag, timestamp, 2, total_pages)
    canvas = fig.axes[0]
    exp1 = shap_res.top3_explanations[0]

    canvas.text(
        0.06,
        0.922,
        f"SHAP Feature Attribution — Rank #1 Predicted Origin: {exp1.display_name}",
        fontsize=10.2,
        fontweight="bold",
        color="#0F172A",
        va="center",
    )
    canvas.text(
        0.06,
        0.903,
        f"TOPO PermutationSHAP attribution ({shap_res.background_donors} development background donors, "
        f"{shap_res.permutations} antithetic cycles).",
        fontsize=7.6,
        color="#475569",
        va="center",
    )

    ax_wf = fig.add_axes([0.365, 0.435, 0.565, 0.430])
    draw_shap_waterfall_ax(ax_wf, exp1, max_display=15, title_prefix="Rank #1: ")

    # Key Supporting (+) & Counter-Evidence (-) Features Card
    _draw_card(canvas, 0.06, 0.055, 0.88, 0.320, facecolor="#F8FAFC", edgecolor="#CBD5E1")
    canvas.text(
        0.08,
        0.352,
        f"Key Supporting (+) & Counter-Evidence (–) Features for {exp1.display_name}",
        fontsize=8.8,
        fontweight="bold",
        color="#0F172A",
        va="center",
    )
    # Compact Net Modality ΔP summary line sized to fit cleanly inside card boundaries
    short_mod_names = {
        "Copy Number (CNA)": "CNA",
        "Somatic Mutations": "Mutations",
        "Structural Variants": "SVs",
        "SBS-6 Spectrum": "SBS-6",
        "Clinical (Age/Sex)": "Clinical",
    }
    mod_parts = []
    for mod_label in [
        "Copy Number (CNA)",
        "Somatic Mutations",
        "Structural Variants",
        "SBS-6 Spectrum",
        "Clinical (Age/Sex)",
    ]:
        m_val = exp1.modality_totals.get(mod_label, 0.0)
        mod_parts.append(f"{short_mod_names[mod_label]}: {m_val:+.3f}")
    canvas.text(
        0.08,
        0.330,
        "Net Modality ΔP:  " + "  |  ".join(mod_parts),
        fontsize=7.2,
        color="#334155",
        va="center",
    )

    # Build up to 10 table rows (up to 6 top supporting + 4 top counter-evidence)
    combined = exp1.top_positive[:6] + exp1.top_negative[:4]
    headers = ["Direction", "Feature", "Sample Value", "Modality", "Δ Calibrated P"]
    col_x = [0.08, 0.205, 0.50, 0.645, 0.920]
    col_align = ["left", "left", "left", "left", "right"]
    canvas.plot([0.075, 0.925], [0.312, 0.312], color="#CBD5E1", lw=0.7)
    for cx, h_txt, h_ha in zip(col_x, headers, col_align):
        canvas.text(cx, 0.298, h_txt, fontsize=7.5, fontweight="bold", color="#475569", va="center", ha=h_ha)
    canvas.plot([0.075, 0.925], [0.285, 0.285], color="#CBD5E1", lw=0.7)

    if not combined:
        canvas.text(0.08, 0.255, "No non-zero feature shifts above threshold.", fontsize=7.8, color="#64748B")
    else:
        row_step = 0.0215
        for idx, fc in enumerate(combined[:10]):
            ry = 0.268 - idx * row_step
            is_pos = fc.shap_value >= 0
            dir_txt = "Supporting (+)" if is_pos else "Opposing (–)"
            dir_col = OKABE_ITO["blue"] if is_pos else OKABE_ITO["vermilion"]
            canvas.text(col_x[0], ry, dir_txt, fontsize=7.3, fontweight="bold", color=dir_col, va="center")
            canvas.text(col_x[1], ry, fc.display_name[:34], fontsize=7.3, color="#0F172A", va="center")
            canvas.text(col_x[2], ry, fc.formatted_value[:16], fontsize=7.3, color="#334155", va="center")
            canvas.text(col_x[3], ry, fc.modality.split(" (")[0], fontsize=7.2, color="#475569", va="center")
            canvas.text(
                col_x[4],
                ry,
                _format_delta(fc.shap_value),
                fontsize=7.4,
                fontweight="bold",
                color=dir_col,
                va="center",
                ha="right",
            )

    pdf.savefig(fig)
    plt.close(fig)


def _render_page3_rank2_rank3_shap(
    pdf: PdfPages,
    config: PipelineConfig,
    shap_res: Top3SHAPResult,
    timestamp: str,
    model_tag: str,
    total_pages: int,
) -> None:
    fig = _new_page(config.sample_id, model_tag, timestamp, 3, total_pages)
    canvas = fig.axes[0]

    canvas.text(
        0.06,
        0.922,
        "Differential Comparison — SHAP Waterfalls for Rank #2 & Rank #3 Candidates",
        fontsize=10.2,
        fontweight="bold",
        color="#0F172A",
        va="center",
    )
    canvas.text(
        0.06,
        0.903,
        "Direct side-by-side comparison of molecular evidence supporting or opposing the secondary differential diagnoses.",
        fontsize=7.6,
        color="#475569",
        va="center",
    )

    if len(shap_res.top3_explanations) >= 2:
        ax_r2 = fig.add_axes([0.365, 0.545, 0.565, 0.320])
        draw_shap_waterfall_ax(
            ax_r2, shap_res.top3_explanations[1], max_display=12, title_prefix="Rank #2: "
        )

    canvas.plot([0.06, 0.94], [0.478, 0.478], color="#E2E8F0", lw=0.9)

    if len(shap_res.top3_explanations) >= 3:
        ax_r3 = fig.add_axes([0.365, 0.095, 0.565, 0.320])
        draw_shap_waterfall_ax(
            ax_r3, shap_res.top3_explanations[2], max_display=12, title_prefix="Rank #3: "
        )

    pdf.savefig(fig)
    plt.close(fig)


def _render_page4_molecular_profile(
    pdf: PdfPages,
    config: PipelineConfig,
    comp: CompilationResult,
    sbs_interp: SBS6Interpretation,
    timestamp: str,
    model_tag: str,
    page_num: int,
    total_pages: int,
) -> None:
    fig = _new_page(config.sample_id, model_tag, timestamp, page_num, total_pages)
    canvas = fig.axes[0]

    canvas.text(
        0.06,
        0.922,
        "Sample Molecular Profile",
        fontsize=10.4,
        fontweight="bold",
        color="#0F172A",
        va="center",
    )

    # 1. Top-Left Card: Detected Somatic Mutations in 274-Gene Panel
    _draw_card(canvas, 0.06, 0.625, 0.55, 0.273, facecolor="#F8FAFC", edgecolor="#CBD5E1")
    maf = comp.maf_result.usable_maf
    canvas.text(
        0.078,
        0.876,
        f"Usable Somatic Mutations in 274-Gene Panel (n = {len(maf)})",
        fontsize=8.6,
        fontweight="bold",
        color="#0F172A",
        va="center",
    )
    m_headers = ["Gene", "Locus / Allele", "Classification"]
    m_x = [0.078, 0.190, 0.425]
    for cx, ht in zip(m_x, m_headers):
        canvas.text(cx, 0.852, ht, fontsize=7.2, fontweight="bold", color="#475569", va="center")
    canvas.plot([0.075, 0.595], [0.840, 0.840], color="#CBD5E1", lw=0.6)

    if maf.empty:
        canvas.text(0.078, 0.812, "0 usable somatic mutations in target domain.", fontsize=7.6, color="#64748B")
    else:
        max_m = 9
        for idx, (_, r) in enumerate(maf.head(max_m).iterrows()):
            ry = 0.822 - idx * 0.020
            gene = str(r["Hugo_Symbol"])[:11]
            hgvsp = ""
            for hcol in ("HGVSp_Short", "HGVSp"):
                if hcol in r and str(r[hcol]).strip() not in {"", ".", "NA", "nan"}:
                    hgvsp = str(r[hcol]).strip()
                    break
            locus = (
                hgvsp[:24]
                if hgvsp
                else f"chr{r['Chromosome']}:{r['Start_Position']} {str(r['Reference_Allele'])[:3]}>{str(r['Tumor_Seq_Allele2'])[:3]}"
            )
            vcls = str(r["Variant_Classification"]).replace("_Mutation", "").replace("_", " ")[:19]
            canvas.text(m_x[0], ry, gene, fontsize=7.1, fontweight="bold", color="#0F172A", va="center")
            canvas.text(m_x[1], ry, locus[:25], fontsize=7.0, color="#334155", va="center")
            canvas.text(m_x[2], ry, vcls, fontsize=7.0, color="#475569", va="center")
        if len(maf) > max_m:
            canvas.text(
                0.078,
                0.638,
                f"+ {len(maf) - max_m} additional usable panel variants not shown.",
                fontsize=7.0,
                style="italic",
                color="#64748B",
                va="center",
            )

    # 2. Top-Right Card: Detected Structural Variants
    _draw_card(canvas, 0.63, 0.625, 0.31, 0.273, facecolor="#F8FAFC", edgecolor="#CBD5E1")
    sv_df = comp.sv_result.usable_sv
    canvas.text(
        0.648,
        0.876,
        f"Structural Variants (n = {len(sv_df)})",
        fontsize=8.6,
        fontweight="bold",
        color="#0F172A",
        va="center",
    )
    canvas.plot([0.645, 0.925], [0.858, 0.858], color="#CBD5E1", lw=0.6)
    if not comp.sv_result.supplied:
        canvas.text(
            0.648,
            0.822,
            "SV file not supplied (--sv omitted);\ntreated as 0 observed SVs.",
            fontsize=7.4,
            color="#B45309",
            va="top",
            linespacing=1.35,
        )
    elif sv_df.empty:
        canvas.text(
            0.648,
            0.822,
            "0 usable somatic SVs detected\nin 274-gene panel.",
            fontsize=7.4,
            color="#64748B",
            va="top",
            linespacing=1.35,
        )
    else:
        max_sv = 9
        for idx, (_, sr) in enumerate(sv_df.head(max_sv).iterrows()):
            ry = 0.838 - idx * 0.021
            lbl = str(sr["SV_Label"])
            sv_cls = str(sr.get("Class", "SV")).strip() or "SV"
            canvas.text(
                0.648,
                ry,
                f"• {lbl[:20]} ({sv_cls[:10]})",
                fontsize=7.2,
                fontweight="bold",
                color="#009E73",
                va="center",
            )
        if len(sv_df) > max_sv:
            canvas.text(
                0.648,
                0.638,
                f"+ {len(sv_df) - max_sv} additional SVs not shown.",
                fontsize=7.0,
                style="italic",
                color="#64748B",
                va="center",
            )

    # 3. Middle Card: SBS-6 Mutational Spectrum & Etiology Interpretation
    _draw_card(canvas, 0.06, 0.372, 0.88, 0.230, facecolor="#F8FAFC", edgecolor="#CBD5E1")
    canvas.text(
        0.078,
        0.582,
        "SBS-6 Mutational Spectrum & Putative Etiology Interpretation",
        fontsize=8.8,
        fontweight="bold",
        color="#0F172A",
        va="center",
    )

    ax_sbs = fig.add_axes([0.115, 0.408, 0.355, 0.138])
    sbs_labels = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]
    sbs_counts = [float(sbs_interp.counts[c]) for c in sbs_labels]
    sbs_colors = ["#56B4E9", "#334155", "#D55E00", "#94A3B8", "#009E73", "#CC79A7"]
    bars = ax_sbs.bar(np.arange(6), sbs_counts, width=0.60, color=sbs_colors)
    ax_sbs.set_xticks(np.arange(6))
    ax_sbs.set_xticklabels(sbs_labels, fontsize=7.4, fontweight="bold", color="#1E293B")
    ymax_sbs = max(max(sbs_counts) * 1.32, 1.0)
    ax_sbs.set_ylim(0, ymax_sbs)
    ax_sbs.set_ylabel("SNV Count", fontsize=7.4)
    ax_sbs.tick_params(axis="y", labelsize=7.0)
    ax_sbs.set_facecolor("#F8FAFC")
    ax_sbs.grid(axis="y", linestyle=":", alpha=0.35)
    ax_sbs.spines["top"].set_visible(False)
    ax_sbs.spines["right"].set_visible(False)
    for bar, cnt in zip(bars, sbs_counts):
        if cnt > 0:
            ax_sbs.text(
                bar.get_x() + bar.get_width() / 2.0,
                cnt + ymax_sbs * 0.03,
                f"{int(round(cnt))}",
                ha="center",
                va="bottom",
                fontsize=7.2,
                fontweight="bold",
                color="#0F172A",
            )

    # Right half of SBS-6 Card: Interpretation Badge + Details
    interp_x = 0.505
    badge_w = 0.415
    badge_h = 0.028
    _draw_card(
        canvas,
        interp_x,
        0.528,
        badge_w,
        badge_h,
        facecolor=sbs_interp.badge_color,
        edgecolor=sbs_interp.badge_color,
        lw=0.4,
        rounding=0.008,
    )
    canvas.text(
        interp_x + 0.014,
        0.528 + badge_h / 2.0,
        sbs_interp.label.upper(),
        fontsize=7.5,
        fontweight="bold",
        color="white",
        va="center",
        ha="left",
    )

    titv_str = f"{sbs_interp.ti_tv_ratio:.2f}" if sbs_interp.ti_tv_ratio is not None else "N/A"
    if sbs_interp.evaluable and sbs_interp.dominant_class is not None and sbs_interp.dominant_fraction is not None:
        dom_txt = f"{sbs_interp.dominant_class} ({sbs_interp.dominant_fraction:.0%})"
    else:
        dom_txt = "N/A"
    metrics_line = f"SNVs: {sbs_interp.n_snvs}  |  Dominant: {dom_txt}  |  Ti/Tv: {titv_str}"
    if sbs_interp.has_cc_tt_dnp:
        metrics_line += "  |  CC>TT DNP"
    canvas.text(
        interp_x,
        0.507,
        metrics_line,
        fontsize=7.0,
        fontweight="bold",
        color="#334155",
        va="center",
    )

    detail_wrapped = "\n".join(textwrap.wrap(sbs_interp.detail, width=56))
    canvas.text(
        interp_x,
        0.486,
        detail_wrapped,
        fontsize=7.5,
        color="#1E293B",
        va="top",
        linespacing=1.35,
    )
    caveat_wrapped = "\n".join(
        textwrap.wrap(
            f"{sbs_interp.confidence_note}. Heuristic 6-channel panel guide; "
            "not formal COSMIC SBS-96 signature deconvolution.",
            width=62,
        )
    )
    canvas.text(
        interp_x,
        0.388,
        caveat_wrapped,
        fontsize=6.6,
        style="italic",
        color="#64748B",
        va="bottom",
        linespacing=1.25,
    )

    # 4. Bottom Card: Chromosome-Level CNA Profile (chr1..chr22, chrX)
    _draw_card(canvas, 0.06, 0.062, 0.88, 0.288, facecolor="#F8FAFC", edgecolor="#CBD5E1")
    if comp.cna_result is not None and not comp.cna_result.chrom_summary.empty:
        canvas.text(
            0.078,
            0.330,
            f"Chromosome Copy Number Profile — Mean Within-Sample Signed Rank "
            f"(Observed Bins: {comp.cna_result.observed_bin_fraction:.1%})",
            fontsize=8.6,
            fontweight="bold",
            color="#0F172A",
            va="center",
        )
        ax_cna = fig.add_axes([0.115, 0.112, 0.800, 0.195])
        cdf = comp.cna_result.chrom_summary
        c_names = [c.removeprefix("chr") for c in cdf["chrom"].tolist()]
        c_vals = cdf["mean_signed_rank"].fillna(0.0).to_numpy(dtype=float)
        c_cols = [OKABE_ITO["vermilion"] if v >= 0 else OKABE_ITO["blue"] for v in c_vals]
        xc = np.arange(len(c_names))
        ax_cna.bar(xc, c_vals, width=0.66, color=c_cols, alpha=0.88)
        ax_cna.axhline(0.0, color="#334155", lw=0.7)
        ax_cna.set_ylim(-1.05, 1.05)
        ax_cna.set_xticks(xc)
        ax_cna.set_xticklabels(c_names, fontsize=7.3)
        ax_cna.set_xlabel("Chromosome (hg19 10 Mb Coverage-Aware Bins)", fontsize=7.5)
        ax_cna.set_ylabel("Mean Rank", fontsize=7.5)
        ax_cna.set_facecolor("#F8FAFC")
        ax_cna.grid(axis="y", linestyle=":", alpha=0.35)
        ax_cna.spines["top"].set_visible(False)
        ax_cna.spines["right"].set_visible(False)
        ax_cna.legend(
            handles=[
                Patch(facecolor=OKABE_ITO["vermilion"], alpha=0.88, label="Relative Gain (+)"),
                Patch(facecolor=OKABE_ITO["blue"], alpha=0.88, label="Relative Loss (–)"),
            ],
            loc="upper right",
            frameon=False,
            ncol=2,
            fontsize=7.2,
        )
    else:
        canvas.text(
            0.50,
            0.220,
            "Copy Number Profile Unavailable",
            fontsize=9.5,
            fontweight="bold",
            color="#475569",
            ha="center",
            va="center",
        )
        canvas.text(
            0.50,
            0.185,
            comp.seg_result.error_reason or "No valid CNA segments provided for chromosome summary.",
            fontsize=8.0,
            color="#64748B",
            ha="center",
            va="center",
        )

    pdf.savefig(fig)
    plt.close(fig)


def generate_pdf_report(
    output_path: Path,
    config: PipelineConfig,
    comp: CompilationResult,
    prediction: Optional[PredictionResult],
    shap_res: Optional[Top3SHAPResult],
) -> Path:
    """Render the complete multi-page clinical vector PDF report."""
    apply_report_rcparams()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    sbs_interp = interpret_sbs6_spectrum(comp.spectrum_df, comp.maf_result.usable_maf)

    if comp.status == "TECHNICAL_NO_CALL" or prediction is None:
        model_tag = format_model_tag(comp.active_model_mode, is_no_call=True)
        total_pages = 2
        with PdfPages(output_path) as pdf:
            _render_page1_summary(
                pdf, config, comp, None, sbs_interp, timestamp, model_tag, total_pages
            )
            _render_page4_molecular_profile(
                pdf, config, comp, sbs_interp, timestamp, model_tag, 2, total_pages
            )
        return output_path

    model_tag = format_model_tag(comp.active_model_mode, is_no_call=False)
    has_shap = shap_res is not None and len(shap_res.top3_explanations) > 0
    total_pages = 4 if has_shap else 2

    with PdfPages(output_path) as pdf:
        _render_page1_summary(
            pdf, config, comp, prediction, sbs_interp, timestamp, model_tag, total_pages
        )
        if has_shap and shap_res is not None:
            _render_page2_rank1_shap(
                pdf, config, shap_res, timestamp, model_tag, total_pages
            )
            _render_page3_rank2_rank3_shap(
                pdf, config, shap_res, timestamp, model_tag, total_pages
            )
            _render_page4_molecular_profile(
                pdf, config, comp, sbs_interp, timestamp, model_tag, 4, total_pages
            )
        else:
            _render_page4_molecular_profile(
                pdf, config, comp, sbs_interp, timestamp, model_tag, 2, total_pages
            )

    return output_path
