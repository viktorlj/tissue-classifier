"""Coordinate liftover (hg38 -> hg19) for MAF, SEG, and SV inputs via pyliftover."""

from __future__ import annotations

import logging
import pandas as pd
from pyliftover import LiftOver

logger = logging.getLogger(__name__)


def _get_liftover() -> LiftOver:
    return LiftOver("hg38", "hg19")


def liftover_maf(maf: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Liftover 1-based inclusive MAF coordinates from hg38 to hg19.

    Returns (lifted_maf, n_dropped).
    """
    if maf.empty:
        return maf.copy(), 0
    lo = _get_liftover()
    new_chroms: list[str] = []
    new_starts: list[str] = []
    new_ends: list[str] = []
    keep: list[bool] = []

    for _, row in maf.iterrows():
        raw_chr = str(row["Chromosome"]).strip().removeprefix("chr")
        chrom = f"chr{raw_chr}"
        try:
            s0 = int(float(row["Start_Position"])) - 1
            e0 = int(float(row["End_Position"])) - 1
        except (TypeError, ValueError):
            new_chroms.append(raw_chr)
            new_starts.append("")
            new_ends.append("")
            keep.append(False)
            continue

        sr = lo.convert_coordinate(chrom, s0)
        er = lo.convert_coordinate(chrom, e0)
        if sr and er and sr[0][0] == er[0][0]:
            target_chr = str(sr[0][0]).removeprefix("chr")
            p1 = int(sr[0][1]) + 1
            p2 = int(er[0][1]) + 1
            s_new, e_new = (p1, p2) if p1 <= p2 else (p2, p1)
            new_chroms.append(target_chr)
            new_starts.append(str(s_new))
            new_ends.append(str(e_new))
            keep.append(True)
        else:
            new_chroms.append(raw_chr)
            new_starts.append("")
            new_ends.append("")
            keep.append(False)

    out = maf.copy()
    out["Chromosome"] = new_chroms
    out["Start_Position"] = new_starts
    out["End_Position"] = new_ends
    out = out.loc[keep].reset_index(drop=True)
    if "NCBI_Build" in out.columns:
        out["NCBI_Build"] = "GRCh37"
    dropped = len(maf) - len(out)
    if dropped > 0:
        logger.warning("MAF hg38->hg19 liftover dropped %d unmappable variants.", dropped)
    return out, dropped


def liftover_seg(seg: pd.DataFrame, chrom_sizes: dict[str, int]) -> tuple[pd.DataFrame, int]:
    """Liftover SEG intervals from hg38 to hg19 cleanly without introducing overlaps."""
    if seg.empty:
        return seg.copy(), 0
    lo = _get_liftover()
    rows = []
    dropped = 0

    for _, row in seg.iterrows():
        raw_chr = str(row["chrom"]).strip().removeprefix("chr").replace("23", "X").replace("24", "Y")
        chrom = f"chr{raw_chr}"
        try:
            s_val = int(float(row["loc.start"]))
            e_val = int(float(row["loc.end"]))
        except (TypeError, ValueError):
            dropped += 1
            continue

        sr = lo.convert_coordinate(chrom, max(s_val, 0))
        er = lo.convert_coordinate(chrom, max(e_val - 1, 0))
        if not (sr and er and sr[0][0] == chrom and er[0][0] == chrom):
            dropped += 1
            continue

        s_new = int(sr[0][1])
        e_new = int(er[0][1]) + 1
        if e_new <= s_new or chrom not in chrom_sizes:
            dropped += 1
            continue

        max_len = chrom_sizes[chrom]
        s_new = max(0, min(s_new, max_len - 1))
        e_new = max(s_new + 1, min(e_new, max_len))
        updated = row.copy()
        updated["chrom"] = raw_chr
        updated["loc.start"] = s_new
        updated["loc.end"] = e_new
        rows.append(updated)

    if not rows:
        return seg.iloc[0:0].copy(), dropped

    out = pd.DataFrame(rows).sort_values(["ID", "chrom", "loc.start", "loc.end"]).reset_index(drop=True)
    # Trim any minor endpoint overlaps created by cross-assembly block shifts
    cleaned = []
    for (_sid, _chr), grp in out.groupby(["ID", "chrom"], sort=False):
        cursor = -1
        for _, r in grp.iterrows():
            s = int(r["loc.start"])
            e = int(r["loc.end"])
            if s < cursor:
                s = cursor
            if e <= s:
                dropped += 1
                continue
            rc = r.copy()
            rc["loc.start"] = s
            rc["loc.end"] = e
            cursor = e
            cleaned.append(rc)

    if not cleaned:
        return seg.iloc[0:0].copy(), dropped
    return pd.DataFrame(cleaned).reset_index(drop=True), dropped
