"""Matplotlib plots for tissue classification reports (base64 PNG output)."""
from __future__ import annotations

import base64
import io
import logging
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)


def _fig_to_base64(fig: plt.Figure) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def plot_top3_predictions(
    top3: list[tuple[str, float]],
) -> str:
    """Bar chart of top-3 predicted tissue types.

    Parameters
    ----------
    top3 : list[tuple[str, float]]
        List of (class_name, probability) tuples.

    Returns
    -------
    str
        Base64-encoded PNG image.
    """
    fig, ax = plt.subplots(figsize=(6, 3))
    classes = [c for c, _ in top3]
    probs = [p * 100 for _, p in top3]
    colors = ["#1565C0", "#42A5F5", "#90CAF9"]
    ax.barh(classes[::-1], probs[::-1], color=colors[::-1])
    ax.set_xlabel("Probability (%)")
    ax.set_xlim(0, 100)
    for i, (c, p) in enumerate(zip(classes[::-1], probs[::-1])):
        ax.text(p + 1, i, f"{p:.1f}%", va="center", fontsize=10)
    ax.set_title("Top-3 Predictions")
    fig.tight_layout()
    return _fig_to_base64(fig)


def plot_full_probabilities(
    all_probs: dict[str, float],
) -> str:
    """Bar chart of all class probabilities.

    Parameters
    ----------
    all_probs : dict[str, float]
        Dict mapping class name to probability.

    Returns
    -------
    str
        Base64-encoded PNG image.
    """
    sorted_items = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
    classes = [c for c, _ in sorted_items]
    probs = [p * 100 for _, p in sorted_items]

    fig, ax = plt.subplots(figsize=(8, max(6, len(classes) * 0.3)))
    colors = ["#1565C0" if p > 5 else "#90CAF9" for p in probs]
    ax.barh(classes[::-1], probs[::-1], color=colors[::-1])
    ax.set_xlabel("Probability (%)")
    ax.set_title("Full Probability Distribution")
    fig.tight_layout()
    return _fig_to_base64(fig)


def plot_shap_waterfall(
    top_features: list[tuple[str, float, float]],
    predicted_class: str,
) -> str:
    """SHAP waterfall-style horizontal bar chart.

    Parameters
    ----------
    top_features : list[tuple[str, float, float]]
        List of (feature_name, shap_value, feature_value) tuples.
    predicted_class : str
        The predicted class being explained.

    Returns
    -------
    str
        Base64-encoded PNG image.
    """
    fig, ax = plt.subplots(figsize=(8, max(4, len(top_features) * 0.35)))
    names = [f"{n} = {v:.2g}" for n, sv, v in top_features]
    values = [sv for _, sv, _ in top_features]
    colors = ["#4CAF50" if v > 0 else "#F44336" for v in values]

    ax.barh(names[::-1], values[::-1], color=colors[::-1])
    ax.set_xlabel(f"SHAP value (impact on {predicted_class})")
    ax.set_title(f"Top Features Driving Prediction: {predicted_class}")
    ax.axvline(x=0, color="black", linewidth=0.5)
    fig.tight_layout()
    return _fig_to_base64(fig)


def plot_modality_breakdown(
    nonzero_by_modality: dict[str, list[str]],
) -> str:
    """Pie chart of non-zero features by modality.

    Parameters
    ----------
    nonzero_by_modality : dict[str, list[str]]
        Dict mapping modality name to list of non-zero feature names.

    Returns
    -------
    str
        Base64-encoded PNG image.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    labels = list(nonzero_by_modality.keys())
    sizes = [len(v) for v in nonzero_by_modality.values()]

    if sum(sizes) == 0:
        ax.text(0.5, 0.5, "No non-zero features", ha="center", va="center")
    else:
        ax.pie(sizes, labels=labels, autopct="%1.0f%%", startangle=90)

    ax.set_title("Non-Zero Features by Modality")
    fig.tight_layout()
    return _fig_to_base64(fig)
