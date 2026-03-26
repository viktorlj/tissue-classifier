"""SBS-6 mutation spectrum feature extraction (14 features)."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import ReferenceData

logger = logging.getLogger(__name__)

# Minimum number of SNPs required for meaningful spectrum proportions
MIN_SNPS = 4

# Complement map for converting purine-ref SNPs to pyrimidine convention
_COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}

# The six SBS classes in pyrimidine-reference convention
SBS6_CLASSES = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]

# Transitions (Ti) and transversions (Tv)
_TRANSITIONS = {"C>T", "T>C"}
_TRANSVERSIONS = {"C>A", "C>G", "T>A", "T>G"}


def classify_sbs6(ref: str, alt: str) -> str | None:
    """Classify a single SNP into one of 6 SBS classes using pyrimidine convention.

    Parameters
    ----------
    ref : str
        Reference allele (single base).
    alt : str
        Alternate allele (single base).

    Returns
    -------
    str or None
        One of 'C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G', or None if invalid.
    """
    ref = ref.upper().strip()
    alt = alt.upper().strip()

    if ref not in _COMPLEMENT or alt not in _COMPLEMENT:
        return None
    if ref == alt:
        return None

    # Convert to pyrimidine reference convention
    if ref in ("A", "G"):
        ref = _COMPLEMENT[ref]
        alt = _COMPLEMENT[alt]

    sbs_class = f"{ref}>{alt}"
    if sbs_class in SBS6_CLASSES:
        return sbs_class
    return None


def extract_spectrum_features(maf: pd.DataFrame, ref: ReferenceData) -> pd.Series:
    """Extract SBS-6 mutation spectrum features from a MAF DataFrame.

    Parameters
    ----------
    maf : pd.DataFrame
        Parsed MAF DataFrame.
    ref : ReferenceData
        Reference data instance.

    Returns
    -------
    pd.Series
        Series of 14 spectrum features: 6 counts, 6 fractions, Ti/Tv ratio,
        and total SNP count.
    """
    feature_names = ref.spectrum_features

    # Filter to SNPs only
    if "Variant_Type" in maf.columns:
        snps = maf[maf["Variant_Type"] == "SNP"].copy()
    else:
        snps = pd.DataFrame()

    # Classify each SNP
    if len(snps) > 0 and "Reference_Allele" in snps.columns and "Tumor_Seq_Allele2" in snps.columns:
        snps = snps.assign(
            sbs6_class=snps.apply(
                lambda row: classify_sbs6(row["Reference_Allele"], row["Tumor_Seq_Allele2"]),
                axis=1,
            )
        )
        snps = snps.dropna(subset=["sbs6_class"])
    else:
        snps = pd.DataFrame(columns=["sbs6_class"])

    n_snps = len(snps)

    # Count per SBS-6 class
    counts = {}
    for cls in SBS6_CLASSES:
        col_name = cls.replace(">", "_") + "_Count"  # e.g. C_A_Count
        counts[col_name] = (snps["sbs6_class"] == cls).sum() if n_snps > 0 else 0

    # Build result
    result = {}

    # Counts (always populated, even below threshold)
    for col_name, count in counts.items():
        result[col_name] = float(count)

    # Fractions and Ti/Tv: NaN if below minimum SNP threshold
    if n_snps >= MIN_SNPS:
        total = max(n_snps, 1)  # clip to avoid div by 0
        for cls in SBS6_CLASSES:
            count_name = cls.replace(">", "_") + "_Count"
            frac_name = cls.replace(">", "_") + "_Fraction"
            result[frac_name] = counts[count_name] / total

        # Ti/Tv ratio
        ti_count = sum(counts[cls.replace(">", "_") + "_Count"] for cls in _TRANSITIONS)
        tv_count = sum(counts[cls.replace(">", "_") + "_Count"] for cls in _TRANSVERSIONS)
        result["Ti_Tv_Ratio"] = ti_count / max(tv_count, 1)
    else:
        for cls in SBS6_CLASSES:
            frac_name = cls.replace(">", "_") + "_Fraction"
            result[frac_name] = float("nan")
        result["Ti_Tv_Ratio"] = float("nan")

    result["Spectrum_SNP_Count"] = float(n_snps)

    return pd.Series(result).reindex(feature_names).astype(float)
