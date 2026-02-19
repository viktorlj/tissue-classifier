"""KernelSHAP explainer for AutoGluon ensemble."""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ReferenceData

logger = logging.getLogger(__name__)


@dataclass
class SHAPResult:
    """Result from SHAP explanation."""
    shap_values: np.ndarray
    expected_values: np.ndarray
    feature_names: list[str]
    class_labels: list[str]
    predicted_class: str
    top_features: list[tuple[str, float, float]] = field(default_factory=list)


class SHAPExplainer:
    """KernelSHAP explainer for tissue classification model."""

    def __init__(
        self,
        predictor: "TissuePredictor",
        ref: ReferenceData,
        nsamples: int = 500,
    ) -> None:
        import shap

        self.predictor = predictor
        self.nsamples = nsamples
        self.class_labels = predictor.class_labels
        self.feature_names = ref.training_feature_order

        bg_path = ref._dir / "shap_background.pkl"
        if not bg_path.exists():
            raise FileNotFoundError(f"SHAP background not found: {bg_path}")
        with open(bg_path, "rb") as f:
            self.background = pickle.load(f)

        self.explainer = shap.KernelExplainer(self._predict_proba, self.background)
        self.expected_values = np.array(self.explainer.expected_value)

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        df = pd.DataFrame(X, columns=self.feature_names[:X.shape[1]])
        return self.predictor.predictor.predict_proba(df).values

    def explain(
        self,
        features_df: pd.DataFrame,
        predicted_class: str,
        n_top: int = 15,
    ) -> SHAPResult:
        """Compute SHAP values for a prediction.

        Parameters
        ----------
        features_df : pd.DataFrame
            Single-row feature DataFrame.
        predicted_class : str
            The predicted class to explain.
        n_top : int
            Number of top features to return.

        Returns
        -------
        SHAPResult
            SHAP explanation result.
        """
        logger.info("Computing KernelSHAP (nsamples=%d)...", self.nsamples)
        feature_names = list(features_df.columns)
        sv = self.explainer.shap_values(
            features_df.values, nsamples=self.nsamples, silent=True,
        )
        if isinstance(sv, list):
            sv = np.stack(sv, axis=-1)
        sv = sv.squeeze(axis=0)

        cls_idx = self.class_labels.index(predicted_class)
        cls_shap = sv[:, cls_idx]
        top_idx = np.argsort(np.abs(cls_shap))[-n_top:][::-1]
        fv = features_df.iloc[0].values
        top_features = [
            (feature_names[i], float(cls_shap[i]), float(fv[i]))
            for i in top_idx
        ]

        logger.info(
            "SHAP done. Top 3: %s",
            ", ".join(f"{n} ({v:+.4f})" for n, v, _ in top_features[:3]),
        )
        return SHAPResult(
            shap_values=sv,
            expected_values=self.expected_values,
            feature_names=feature_names,
            class_labels=self.class_labels,
            predicted_class=predicted_class,
            top_features=top_features,
        )
