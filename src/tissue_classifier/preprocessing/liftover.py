"""Coordinate liftover hg38 -> hg19 via pyliftover."""
from __future__ import annotations

import logging

import pandas as pd
from pyliftover import LiftOver

logger = logging.getLogger(__name__)


def liftover_maf(maf: pd.DataFrame) -> pd.DataFrame:
    """Liftover MAF coordinates from hg38 to hg19.

    Parameters
    ----------
    maf : pd.DataFrame
        MAF DataFrame with Chromosome (no chr prefix), Start_Position, End_Position.

    Returns
    -------
    pd.DataFrame
        MAF with converted coordinates; unmappable variants dropped.
    """
    lo = LiftOver("hg38", "hg19")
    n_before = len(maf)
    new_starts, new_ends, keep = [], [], []
    for _, row in maf.iterrows():
        chrom = f"chr{row['Chromosome']}"
        sr = lo.convert_coordinate(chrom, int(row["Start_Position"]))
        er = lo.convert_coordinate(chrom, int(row["End_Position"]))
        if sr and er:
            new_starts.append(sr[0][1])
            new_ends.append(er[0][1])
            keep.append(True)
        else:
            new_starts.append(None)
            new_ends.append(None)
            keep.append(False)
    maf = maf.copy()
    maf["Start_Position"] = new_starts
    maf["End_Position"] = new_ends
    maf = maf[keep].reset_index(drop=True)
    n_dropped = n_before - len(maf)
    if n_dropped > 0:
        logger.warning("Liftover: dropped %d unmappable variants", n_dropped)
    return maf


def liftover_seg(seg: pd.DataFrame) -> pd.DataFrame:
    """Liftover SEG coordinates from hg38 to hg19.

    Parameters
    ----------
    seg : pd.DataFrame
        SEG DataFrame with chrom (chr prefix), loc.start, loc.end.

    Returns
    -------
    pd.DataFrame
        SEG with converted coordinates; unmappable segments dropped.
    """
    lo = LiftOver("hg38", "hg19")
    n_before = len(seg)
    new_starts, new_ends, keep = [], [], []
    for _, row in seg.iterrows():
        sr = lo.convert_coordinate(row["chrom"], int(row["loc.start"]))
        er = lo.convert_coordinate(row["chrom"], int(row["loc.end"]))
        if sr and er:
            new_starts.append(sr[0][1])
            new_ends.append(er[0][1])
            keep.append(True)
        else:
            new_starts.append(None)
            new_ends.append(None)
            keep.append(False)
    seg = seg.copy()
    seg["loc.start"] = new_starts
    seg["loc.end"] = new_ends
    seg = seg[keep].reset_index(drop=True)
    n_dropped = n_before - len(seg)
    if n_dropped > 0:
        logger.warning("SEG liftover: dropped %d unmappable segments", n_dropped)
    return seg
