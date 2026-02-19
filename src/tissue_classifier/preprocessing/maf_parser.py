"""Parse and validate MAF files."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "Hugo_Symbol", "Chromosome", "Start_Position", "End_Position",
    "Reference_Allele", "Tumor_Seq_Allele2", "Variant_Classification", "Variant_Type",
]


def parse_maf(path: Path) -> pd.DataFrame:
    """Parse a MAF file. Strips 'chr' prefix, fixes SNV->SNP."""
    df = pd.read_csv(path, sep="\t", comment="#", low_memory=False)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"MAF missing required columns: {missing}")
    df["Chromosome"] = df["Chromosome"].astype(str).str.replace("^chr", "", regex=True)
    df["Variant_Type"] = df["Variant_Type"].replace({"SNV": "SNP"})
    df["Start_Position"] = pd.to_numeric(df["Start_Position"], errors="coerce").astype("Int64")
    df["End_Position"] = pd.to_numeric(df["End_Position"], errors="coerce").astype("Int64")
    logger.info("Parsed MAF: %d variants from %s", len(df), path.name)
    return df
