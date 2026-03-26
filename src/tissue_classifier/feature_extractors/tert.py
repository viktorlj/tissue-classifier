"""TERT promoter mutation feature extraction (~18 features)."""
from __future__ import annotations

import logging

import pandas as pd

from ..config import ReferenceData

logger = logging.getLogger(__name__)


def extract_tert_features(
    maf: pd.DataFrame,
    ref: ReferenceData,
) -> pd.Series:
    """Extract TERT promoter mutation features.

    Parameters
    ----------
    maf : pd.DataFrame
        Parsed MAF DataFrame.
    ref : ReferenceData
        Reference data instance.

    Returns
    -------
    pd.Series
        Series of 18 TERT features (binary).
    """
    tert_feature_names = ref.recurrent_tert_mutations
    result = pd.Series(0, index=tert_feature_names, dtype=float)

    tert_maf = maf[maf["Hugo_Symbol"] == "TERT"]
    if tert_maf.empty:
        logger.info("TERT features: no TERT mutations found")
        return result

    tert_ids = (
        "TERT_" + tert_maf["Chromosome"].astype(str) + "_"
        + tert_maf["Start_Position"].astype(str) + "_"
        + tert_maf["Reference_Allele"].astype(str) + "_"
        + tert_maf["Tumor_Seq_Allele2"].astype(str)
    )

    has_non_recurrent = False
    for tert_id in tert_ids:
        if tert_id in result.index:
            result[tert_id] = 1
        else:
            has_non_recurrent = True

    if has_non_recurrent and "TERT_other" in result.index:
        result["TERT_other"] = 1

    logger.info("TERT features: %d non-zero out of %d", (result != 0).sum(), len(result))
    return result
