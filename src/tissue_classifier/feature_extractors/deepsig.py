"""DeepSig mutational signature feature extraction (13 features)."""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from ..config import ReferenceData

logger = logging.getLogger(__name__)


def extract_deepsig_features(
    maf: pd.DataFrame,
    ref: ReferenceData,
    mode: str = "skip",
    mutation_count: Optional[int] = None,
) -> pd.Series:
    """Extract DeepSig mutational signature features.

    Parameters
    ----------
    maf : pd.DataFrame
        Parsed MAF DataFrame.
    ref : ReferenceData
        Reference data instance.
    mode : str
        One of 'docker', 'subprocess', 'skip'. Default 'skip'.
    mutation_count : Optional[int]
        Override mutation count for minimum threshold check.

    Returns
    -------
    pd.Series
        Series of 13 DeepSig features. Returns NaN values for skip/fallback
        modes (will be imputed to -1 by the feature compiler).
    """
    feature_names = ref.deepsig_features

    if mode == "skip":
        logger.info("DeepSig: skipped, returning NaN for %d features", len(feature_names))
        return pd.Series(float("nan"), index=feature_names, dtype=float)

    n_muts = mutation_count if mutation_count is not None else len(maf)
    if n_muts < 4:
        logger.warning("DeepSig: only %d mutations (need >=4), returning NaN", n_muts)
        return pd.Series(float("nan"), index=feature_names, dtype=float)

    if mode in ("docker", "subprocess"):
        logger.warning("DeepSig %s mode not implemented, falling back to NaN", mode)
        return pd.Series(float("nan"), index=feature_names, dtype=float)

    return pd.Series(float("nan"), index=feature_names, dtype=float)
