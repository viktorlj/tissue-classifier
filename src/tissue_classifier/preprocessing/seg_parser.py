"""Parse and validate SEG files."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["ID", "chrom", "loc.start", "loc.end", "seg.mean"]


def parse_seg(path: Path) -> pd.DataFrame:
    """Parse a SEG file into a cleaned DataFrame.

    Parameters
    ----------
    path : Path
        Path to the SEG file (tab-separated).

    Returns
    -------
    pd.DataFrame
        Cleaned SEG with chr prefix added if missing.
    """
    df = pd.read_csv(path, sep="\t", comment="#", low_memory=False)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"SEG missing required columns: {missing}")
    df["chrom"] = df["chrom"].astype(str)
    mask = ~df["chrom"].str.startswith("chr")
    df.loc[mask, "chrom"] = "chr" + df.loc[mask, "chrom"]
    df["loc.start"] = pd.to_numeric(df["loc.start"], errors="coerce").astype("Int64")
    df["loc.end"] = pd.to_numeric(df["loc.end"], errors="coerce").astype("Int64")
    df["seg.mean"] = pd.to_numeric(df["seg.mean"], errors="coerce")
    logger.info("Parsed SEG: %d segments from %s", len(df), path.name)
    return df
