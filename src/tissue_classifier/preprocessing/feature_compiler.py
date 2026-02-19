"""Merge all feature modalities and align to training feature order."""
from __future__ import annotations

import logging

import pandas as pd

from ..config import ReferenceData

logger = logging.getLogger(__name__)


def compile_features(
    feature_series: dict[str, pd.Series],
    ref: ReferenceData,
) -> pd.DataFrame:
    """Compile all feature Series into a single-row DataFrame aligned to training order.

    Parameters
    ----------
    feature_series : dict[str, pd.Series]
        Dict mapping modality name to feature Series.
    ref : ReferenceData
        Reference data instance.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with exactly 4333 features in training order.
    """
    training_order = ref.training_feature_order
    deepsig_names = set(ref.deepsig_features)

    # Concatenate all feature series
    all_features = pd.concat(feature_series.values())

    # Reindex to training order, filling missing with 0
    aligned = all_features.reindex(training_order, fill_value=0.0)

    # Imputation rules matching training pipeline:
    # - DeepSig SBS* features: NaN -> -1 (sentinel for "not computed")
    # - AGE_AT_SEQ_REPORT: NaN -> 58.9 (median from training)
    # - SEX: NaN -> 0 (mode from training = Female)
    # - Everything else: NaN -> 0
    for feat in training_order:
        if pd.isna(aligned[feat]):
            if feat in deepsig_names:
                aligned[feat] = -1.0
            elif feat == "AGE_AT_SEQ_REPORT":
                aligned[feat] = 58.9
            elif feat == "SEX":
                aligned[feat] = 0.0
            else:
                aligned[feat] = 0.0

    assert len(aligned) == len(training_order), (
        f"Expected {len(training_order)} features, got {len(aligned)}"
    )

    result = aligned.to_frame().T
    result.index = ["SAMPLE"]
    logger.info(
        "Compiled %d features (%d non-zero)",
        len(training_order),
        (result.iloc[0] != 0).sum(),
    )
    return result
