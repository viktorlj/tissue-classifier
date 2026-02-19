"""Clinical feature extraction (2 features)."""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

CLINICAL_FEATURES = ["AGE_AT_SEQ_REPORT", "SEX"]


def extract_clinical_features(
    age: Optional[float] = None,
    sex: Optional[str] = None,
) -> pd.Series:
    """Extract clinical features.

    Parameters
    ----------
    age : Optional[float]
        Age at sequencing report.
    sex : Optional[str]
        Sex string: Female/F -> 0.0, Male/M -> 1.0.

    Returns
    -------
    pd.Series
        Series of 2 clinical features. NaN for missing values
        (imputed by compiler: AGE->58.9, SEX->0).
    """
    result = pd.Series(dtype=float, index=CLINICAL_FEATURES)
    result["AGE_AT_SEQ_REPORT"] = float(age) if age is not None else float("nan")

    if sex is not None:
        s = sex.strip().upper()
        if s in ("MALE", "M"):
            result["SEX"] = 1.0
        elif s in ("FEMALE", "F"):
            result["SEX"] = 0.0
        else:
            result["SEX"] = float("nan")
    else:
        result["SEX"] = float("nan")

    logger.info("Clinical features: age=%s, sex=%s", age, sex)
    return result
