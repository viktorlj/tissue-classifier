"""Mutation frequency feature extraction (10 features)."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import ReferenceData

logger = logging.getLogger(__name__)


def extract_mutfreq_features(
    maf: pd.DataFrame,
    ref: ReferenceData,
) -> pd.Series:
    """Extract mutation frequency features from a MAF DataFrame.

    Parameters
    ----------
    maf : pd.DataFrame
        Parsed MAF DataFrame.
    ref : ReferenceData
        Reference data instance.

    Returns
    -------
    pd.Series
        Series of 10 mutation frequency features.
    """
    feature_names = ref.mutfreq_features
    result = pd.Series(0.0, index=feature_names, dtype=float)

    if maf.empty:
        return result

    vt_counts = maf["Variant_Type"].value_counts()
    for vt in ["SNP", "INS", "DEL", "DNP", "TNP"]:
        result[f"{vt}_Count"] = int(vt_counts.get(vt, 0))

    total = result[["SNP_Count", "INS_Count", "DEL_Count", "DNP_Count", "TNP_Count"]].sum()
    result["Total_Mutation_Count"] = int(total)
    denom = max(total, 1)
    result["Indel_Fraction"] = (result["INS_Count"] + result["DEL_Count"]) / denom
    result["DNP_TNP_Fraction"] = (result["DNP_Count"] + result["TNP_Count"]) / denom
    result["SNP_Fraction"] = result["SNP_Count"] / denom
    result["Log_TMB"] = np.log1p(total)

    logger.info("MutFreq features: Total_Mutation_Count=%d", int(total))
    return result
