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
        Single-row DataFrame with exactly 4320 features in training order.
    """
    training_order = ref.training_feature_order

    # Concatenate all feature series
    all_features = pd.concat(feature_series.values())

    # Reindex to training order, filling missing with 0
    aligned = all_features.reindex(training_order, fill_value=0.0)

    # Identify spectrum fraction features and Ti/Tv ratio
    spectrum_fraction_names = {f for f in training_order if 'Fraction' in f and f.startswith(('C_', 'T_'))}
    spectrum_fraction_names.add('Ti_Tv_Ratio')

    # Track if spectrum data is available
    has_spectrum = any(not pd.isna(aligned.get(f, float('nan'))) for f in spectrum_fraction_names)

    # Imputation rules matching training pipeline:
    # - Spectrum fraction features: NaN -> 0.0
    # - AGE_AT_SEQ_REPORT: NaN -> 63.0 (training median)
    # - SEX: NaN -> 0.0
    # - Everything else: NaN -> 0.0
    for feat in training_order:
        if pd.isna(aligned[feat]):
            if feat in spectrum_fraction_names:
                aligned[feat] = 0.0  # NaN fractions -> 0
            elif feat == "AGE_AT_SEQ_REPORT":
                aligned[feat] = 63.0  # training median
            elif feat == "SEX":
                aligned[feat] = 0.0
            else:
                aligned[feat] = 0.0

    # Add the has_spectrum_data flag (added during training preprocessing)
    if 'has_spectrum_data' in training_order:
        aligned['has_spectrum_data'] = 1.0 if has_spectrum else 0.0

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
