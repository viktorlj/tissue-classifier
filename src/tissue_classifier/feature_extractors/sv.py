"""Structural variant feature extraction (~186 features)."""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from ..config import ReferenceData

logger = logging.getLogger(__name__)


def extract_sv_features(
    sv_df: Optional[pd.DataFrame],
    ref: ReferenceData,
) -> pd.Series:
    """Extract SV features from a structural variant DataFrame.

    Parameters
    ----------
    sv_df : Optional[pd.DataFrame]
        SV DataFrame with Site1_Hugo_Symbol and Site2_Hugo_Symbol columns,
        or None if no SV data.
    ref : ReferenceData
        Reference data instance.

    Returns
    -------
    pd.Series
        Series of 186 SV features.
    """
    sv_feature_names = ref.recurrent_sv_labels
    result = pd.Series(0, index=sv_feature_names, dtype=float)

    if sv_df is None or (hasattr(sv_df, "empty") and sv_df.empty):
        if "has_SV_data" in result.index:
            result["has_SV_data"] = 0
        logger.info("SV features: no SV data, all zeros")
        return result

    sv_labels = []
    for _, row in sv_df.iterrows():
        g1 = str(row.get("Site1_Hugo_Symbol", "")).strip()
        g2 = str(row.get("Site2_Hugo_Symbol", "")).strip()
        if g1 and g2 and g1 != "nan" and g2 != "nan" and g1 != g2:
            label = "-".join(sorted([g1, g2]))
        elif g1 and g1 != "nan":
            label = f"{g1}_SV"
        elif g2 and g2 != "nan":
            label = f"{g2}_SV"
        else:
            continue
        sv_labels.append(label)

    for label in sv_labels:
        if label in result.index:
            result[label] = 1

    if "SV_Total_Count" in result.index:
        result["SV_Total_Count"] = len(sv_labels)
    if "has_SV_data" in result.index:
        result["has_SV_data"] = 1

    logger.info("SV features: %d non-zero out of %d", (result != 0).sum(), len(result))
    return result
