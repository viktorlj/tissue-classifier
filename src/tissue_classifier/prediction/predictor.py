"""AutoGluon model prediction wrapper."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Result from tissue-of-origin prediction."""
    predicted_class: str
    confidence: float
    top3: list[tuple[str, float]]
    all_probabilities: dict[str, float]
    class_labels: list[str] = field(default_factory=list)
    proba_array: np.ndarray = field(default_factory=lambda: np.array([]))


class TissuePredictor:
    """Wrapper around AutoGluon TabularPredictor for tissue classification."""

    def __init__(self, model_dir: Path, model_name: str | None = None) -> None:
        from autogluon.tabular import TabularPredictor

        self.model_dir = model_dir
        logger.info("Loading AutoGluon model from %s", model_dir)
        self.predictor = TabularPredictor.load(str(model_dir))
        self.model_name = model_name or self.predictor.model_best
        self.class_labels = sorted(self.predictor.class_labels)
        logger.info(
            "Model loaded: %d classes, model=%s",
            len(self.class_labels), self.model_name,
        )

    def predict(self, features_df: pd.DataFrame) -> PredictionResult:
        """Run prediction on a single-row feature DataFrame.

        Parameters
        ----------
        features_df : pd.DataFrame
            Single-row DataFrame with features in training order.

        Returns
        -------
        PredictionResult
            Prediction result with class, confidence, and probabilities.
        """
        pred = self.predictor.predict(features_df, model=self.model_name)
        proba_df = self.predictor.predict_proba(features_df, model=self.model_name)
        predicted_class = pred.iloc[0]
        proba_row = proba_df.iloc[0]
        all_probs = proba_row.to_dict()
        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        top3 = [(cls, float(p)) for cls, p in sorted_probs[:3]]
        confidence = top3[0][1]
        proba_array = np.array([all_probs.get(c, 0.0) for c in self.class_labels])
        logger.info("Prediction: %s (%.1f%%)", predicted_class, confidence * 100)
        return PredictionResult(
            predicted_class=str(predicted_class),
            confidence=float(confidence),
            top3=top3,
            all_probabilities=all_probs,
            class_labels=self.class_labels,
            proba_array=proba_array,
        )
