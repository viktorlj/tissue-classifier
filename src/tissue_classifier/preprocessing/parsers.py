"""Exact V20 MAF, SV, and SEG parsers and shared-panel domain filters."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .liftover import liftover_maf, liftover_seg

logger = logging.getLogger(__name__)

REQUIRED_MAF_COLUMNS = (
    "Hugo_Symbol",
    "Chromosome",
    "Start_Position",
    "End_Position",
    "Reference_Allele",
    "Tumor_Seq_Allele2",
    "Variant_Classification",
    "Variant_Type",
)

REQUIRED_SEG_COLUMNS = ("ID", "chrom", "loc.start", "loc.end", "seg.mean")
VALID_VARIANT_TYPES = {"SNP", "SNV", "INS", "DEL", "DNP", "TNP"}
VALID_SEG_CHROMS = {str(i) for i in range(1, 23)} | {"X", "Y"}


def contained(
    chrom: pd.Series | np.ndarray,
    start: pd.Series | np.ndarray,
    end: pd.Series | np.ndarray,
    lookup: dict[str, list[list[int]]],
) -> np.ndarray:
    """Exact V20 half-open BED containment check (1-based inclusive MAF positions)."""
    result = np.zeros(len(start), dtype=bool)
    chrom_arr = np.asarray(chrom).astype(str)
    starts = np.asarray(start, dtype=float) - 1.0
    ends = np.asarray(end, dtype=float)
    for c, intervals in lookup.items():
        rows = np.flatnonzero(chrom_arr == str(c))
        if not len(rows) or not len(intervals):
            continue
        x = np.asarray(intervals, dtype=float)
        ix = np.searchsorted(x[:, 0], starts[rows], side="right") - 1
        safe = np.maximum(ix, 0)
        result[rows] = (
            (ix >= 0)
            & np.isfinite(starts[rows])
            & np.isfinite(ends[rows])
            & (ends[rows] > starts[rows])
            & (ends[rows] <= x[safe, 1])
        )
    return result


def _clean_int_str(series: pd.Series) -> pd.Series:
    nums = pd.to_numeric(series, errors="coerce")
    out = pd.Series("", index=series.index, dtype=str)
    valid = nums.notna() & np.isfinite(nums)
    out.loc[valid] = nums.loc[valid].astype(np.int64).astype(str)
    return out


@dataclass
class MafParseResult:
    raw_count: int
    usable_maf: pd.DataFrame
    tert_promoter_maf: pd.DataFrame = field(default_factory=pd.DataFrame)
    liftover_dropped: int = 0
    warnings: list[str] = field(default_factory=list)


def parse_and_filter_maf(
    maf_path: Path,
    sample_id: str,
    useful_genes: set[str],
    bed_lookup: dict[str, list[list[int]]],
    genome: str = "hg19",
) -> MafParseResult:
    """Parse a MAF file and apply exact V20 shared 274-gene + BED target filters."""
    df = pd.read_csv(maf_path, sep="\t", comment="#", dtype=str, keep_default_na=False)
    missing = [c for c in REQUIRED_MAF_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"MAF missing required columns: {missing}")

    warnings_list: list[str] = []

    if "Tumor_Sample_Barcode" in df.columns and len(df) > 0:
        barcodes = df["Tumor_Sample_Barcode"].astype(str).str.strip()
        if sample_id in set(barcodes):
            df = df.loc[barcodes == sample_id].copy()
        elif barcodes.nunique() > 1:
            raise ValueError(
                f"Multi-sample MAF does not contain sample_id '{sample_id}'. "
                f"Available barcodes (first 5): { sorted(set(barcodes))[:5] }"
            )
        else:
            df = df.copy()
            df["Tumor_Sample_Barcode"] = sample_id
    else:
        df = df.copy()
        df["Tumor_Sample_Barcode"] = sample_id

    raw_count = len(df)
    if df.empty:
        df["Mutation_ID"] = pd.Series(dtype=str)
        return MafParseResult(raw_count=0, usable_maf=df, warnings=warnings_list)

    df["Chromosome"] = df["Chromosome"].astype(str).str.strip().str.removeprefix("chr")
    df["Hugo_Symbol"] = df["Hugo_Symbol"].astype(str).str.strip()
    df["Reference_Allele"] = df["Reference_Allele"].astype(str).str.strip()
    df["Tumor_Seq_Allele2"] = df["Tumor_Seq_Allele2"].astype(str).str.strip()
    df["Variant_Classification"] = df["Variant_Classification"].astype(str).str.strip()
    df["Variant_Type"] = df["Variant_Type"].astype(str).str.strip()

    liftover_dropped = 0
    if genome == "hg38":
        df, liftover_dropped = liftover_maf(df)
        if liftover_dropped > 0:
            warnings_list.append(f"Liftover hg38->hg19 dropped {liftover_dropped} unmappable MAF row(s).")
    elif "NCBI_Build" in df.columns:
        builds = set(df["NCBI_Build"].astype(str).str.strip().unique()) - {"", "."}
        if builds and any("38" in b for b in builds):
            warnings_list.append(
                f"MAF NCBI_Build indicates {sorted(builds)} while --genome hg19 was selected."
            )

    if df.empty:
        df["Mutation_ID"] = pd.Series(dtype=str)
        return MafParseResult(
            raw_count=raw_count,
            usable_maf=df,
            liftover_dropped=liftover_dropped,
            warnings=warnings_list,
        )

    keep = df["Hugo_Symbol"].isin(useful_genes)

    if "Mutation_Status" in df.columns:
        ms = df["Mutation_Status"].astype(str).str.strip()
        keep &= ms.eq("") | ms.str.upper().eq("SOMATIC")

    if "FILTER" in df.columns:
        flt = df["FILTER"].astype(str).str.strip()
        keep &= flt.eq("") | flt.isin(["PASS", "."])

    if genome == "hg19" and "NCBI_Build" in df.columns:
        nb = df["NCBI_Build"].astype(str).str.strip()
        keep &= nb.eq("") | nb.isin(["GRCh37", "hg19", "37", "."])

    d = df.loc[keep].copy()
    if d.empty:
        d["Mutation_ID"] = pd.Series(dtype=str)
        return MafParseResult(
            raw_count=raw_count,
            usable_maf=d,
            tert_promoter_maf=pd.DataFrame(),
            liftover_dropped=liftover_dropped,
            warnings=warnings_list,
        )

    # Capture somatic PASS TERT 5'Flank promoter mutations BEFORE common BED clipping
    tert_prom = d[d["Hugo_Symbol"].eq("TERT") & d["Variant_Classification"].eq("5'Flank")].copy()
    if not tert_prom.empty:
        tert_prom["Start_Position"] = _clean_int_str(tert_prom["Start_Position"])
        tert_prom["Tumor_Sample_Barcode"] = sample_id

    start_num = pd.to_numeric(d["Start_Position"], errors="coerce")
    end_num = pd.to_numeric(d["End_Position"], errors="coerce")
    in_bed = contained(d["Chromosome"], start_num, end_num, bed_lookup)
    valid_alleles = (
        d["Reference_Allele"].ne("")
        & d["Tumor_Seq_Allele2"].ne("")
        & d["Reference_Allele"].ne("NA")
        & d["Tumor_Seq_Allele2"].ne("NA")
        & d["Reference_Allele"].ne(d["Tumor_Seq_Allele2"])
        & d["Variant_Type"].isin(VALID_VARIANT_TYPES)
    )
    d = d.loc[in_bed & valid_alleles].copy()
    if d.empty:
        d["Mutation_ID"] = pd.Series(dtype=str)
        return MafParseResult(
            raw_count=raw_count,
            usable_maf=d,
            tert_promoter_maf=tert_prom,
            liftover_dropped=liftover_dropped,
            warnings=warnings_list,
        )

    d["Start_Position"] = _clean_int_str(d["Start_Position"])
    d["End_Position"] = _clean_int_str(d["End_Position"])
    d["Tumor_Sample_Barcode"] = sample_id

    dedup_cols = [
        "Tumor_Sample_Barcode",
        "Chromosome",
        "Start_Position",
        "End_Position",
        "Reference_Allele",
        "Tumor_Seq_Allele2",
        "Hugo_Symbol",
    ]
    d = d.drop_duplicates(subset=dedup_cols).reset_index(drop=True)

    # Exact V20 Step 04 annotation: Mutation_ID & SNV -> SNP normalization
    d["Mutation_ID"] = (
        d["Hugo_Symbol"]
        + "_"
        + d["Chromosome"]
        + "_"
        + d["Start_Position"]
        + "_"
        + d["Reference_Allele"]
        + "_"
        + d["Tumor_Seq_Allele2"]
    )
    d["Variant_Type"] = d["Variant_Type"].replace({"SNV": "SNP"})
    return MafParseResult(
        raw_count=raw_count,
        usable_maf=d,
        tert_promoter_maf=tert_prom,
        liftover_dropped=liftover_dropped,
        warnings=warnings_list,
    )


def label_sv(row: pd.Series) -> Optional[str]:
    """Exact V20 Step 06 SV feature naming rule."""
    genes = sorted(
        {str(row.get(f"Site{i}_Hugo_Symbol", "")).strip() for i in (1, 2)}
        - {"", "nan", "None"}
    )
    if not genes:
        return None
    return genes[0] + "_SV" if len(genes) == 1 else "-".join(genes)


@dataclass
class SvParseResult:
    supplied: bool
    raw_count: int
    usable_sv: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


def parse_and_filter_sv(
    sv_path: Optional[Path],
    sample_id: str,
    useful_genes: set[str],
) -> SvParseResult:
    """Parse optional SV file and apply exact V20 usable_gene_sv domain filter."""
    empty_df = pd.DataFrame(
        columns=[
            "Sample_Id",
            "Site1_Hugo_Symbol",
            "Site2_Hugo_Symbol",
            "Original_Site1_Hugo_Symbol",
            "Original_Site2_Hugo_Symbol",
            "SV_Label",
        ]
    )
    if sv_path is None:
        return SvParseResult(
            supplied=False,
            raw_count=0,
            usable_sv=empty_df,
            warnings=["SV file not supplied; structural variant counts treated as 0."],
        )

    df = pd.read_csv(sv_path, sep="\t", comment="#", dtype=str, keep_default_na=False)
    if df.empty:
        return SvParseResult(supplied=True, raw_count=0, usable_sv=empty_df)

    if "Sample_Id" not in df.columns and "Tumor_Sample_Barcode" in df.columns:
        df["Sample_Id"] = df["Tumor_Sample_Barcode"]
    elif "Sample_Id" not in df.columns:
        df["Sample_Id"] = sample_id

    for req_col in ("Site1_Hugo_Symbol", "Site2_Hugo_Symbol"):
        if req_col not in df.columns:
            raise ValueError(f"SV file missing required column: '{req_col}'")

    sids = df["Sample_Id"].astype(str).str.strip()
    if sample_id in set(sids):
        df = df.loc[sids == sample_id].copy()
    elif sids.nunique() > 1:
        raise ValueError(f"Multi-sample SV file does not contain sample_id '{sample_id}'.")
    else:
        df = df.copy()
        df["Sample_Id"] = sample_id

    raw_count = len(df)
    if df.empty:
        return SvParseResult(supplied=True, raw_count=0, usable_sv=empty_df)

    df["Sample_Id"] = sample_id
    df["Site1_Hugo_Symbol"] = df["Site1_Hugo_Symbol"].astype(str).str.strip()
    df["Site2_Hugo_Symbol"] = df["Site2_Hugo_Symbol"].astype(str).str.strip()

    keep = pd.Series(True, index=df.index)
    if "SV_Status" in df.columns:
        st = df["SV_Status"].astype(str).str.strip()
        keep &= st.eq("") | st.str.upper().eq("SOMATIC")
    if "DNA_Support" in df.columns:
        dna = df["DNA_Support"].astype(str).str.strip()
        keep &= ~dna.str.upper().eq("NO")

    keep &= df["Site1_Hugo_Symbol"].isin(useful_genes) | df["Site2_Hugo_Symbol"].isin(useful_genes)
    sv = df.loc[keep].copy()
    if sv.empty:
        return SvParseResult(supplied=True, raw_count=raw_count, usable_sv=empty_df)

    for i in (1, 2):
        col = f"Site{i}_Hugo_Symbol"
        if f"Original_{col}" not in sv.columns:
            sv[f"Original_{col}"] = sv[col]
        sv[col] = sv[col].where(sv[col].isin(useful_genes), "")

    dedup_candidates = [
        "Sample_Id",
        "Original_Site1_Hugo_Symbol",
        "Original_Site2_Hugo_Symbol",
        "Site1_Chromosome",
        "Site2_Chromosome",
        "Site1_Position",
        "Site2_Position",
        "Class",
    ]
    present_dedup = [c for c in dedup_candidates if c in sv.columns]
    if present_dedup:
        sv = sv.drop_duplicates(subset=present_dedup)

    sv["SV_Label"] = sv.apply(label_sv, axis=1)
    sv = sv.dropna(subset=["SV_Label"]).reset_index(drop=True)
    return SvParseResult(supplied=True, raw_count=raw_count, usable_sv=sv)


@dataclass
class SegParseResult:
    supplied: bool
    valid: bool
    raw_count: int
    segments: pd.DataFrame
    error_reason: Optional[str] = None
    liftover_dropped: int = 0
    warnings: list[str] = field(default_factory=list)


def parse_and_validate_seg(
    seg_path: Optional[Path],
    sample_id: str,
    chrom_sizes: dict[str, int],
    genome: str = "hg19",
) -> SegParseResult:
    """Parse and strictly validate a SEG file per V20 non-overlapping hg19 contract."""
    empty_seg = pd.DataFrame(columns=["ID", "chrom", "loc.start", "loc.end", "seg.mean"])
    if seg_path is None:
        return SegParseResult(
            supplied=False,
            valid=False,
            raw_count=0,
            segments=empty_seg,
            error_reason="SEG file not provided",
        )

    df = pd.read_csv(seg_path, sep="\t", comment="#", dtype=str, keep_default_na=False)
    missing = [c for c in REQUIRED_SEG_COLUMNS if c not in df.columns]
    if missing:
        return SegParseResult(
            supplied=True,
            valid=False,
            raw_count=len(df),
            segments=empty_seg,
            error_reason=f"SEG missing required columns: {missing}",
        )

    if df.empty:
        return SegParseResult(
            supplied=True,
            valid=False,
            raw_count=0,
            segments=empty_seg,
            error_reason="SEG file contains 0 segment rows",
        )

    ids = df["ID"].astype(str).str.strip()
    if sample_id in set(ids):
        df = df.loc[ids == sample_id].copy()
    elif ids.nunique() > 1:
        return SegParseResult(
            supplied=True,
            valid=False,
            raw_count=len(df),
            segments=empty_seg,
            error_reason=f"Multi-sample SEG file does not contain sample_id '{sample_id}'",
        )
    else:
        df = df.copy()

    df["ID"] = sample_id
    raw_count = len(df)
    if raw_count == 0:
        return SegParseResult(
            supplied=True,
            valid=False,
            raw_count=0,
            segments=empty_seg,
            error_reason="No segment rows for sample",
        )

    df["chrom"] = (
        df["chrom"]
        .astype(str)
        .str.strip()
        .str.removeprefix("chr")
        .replace({"23": "X", "24": "Y"})
    )

    warnings_list: list[str] = []
    liftover_dropped = 0
    if genome == "hg38":
        df, liftover_dropped = liftover_seg(df, chrom_sizes)
        if liftover_dropped > 0:
            warnings_list.append(
                f"SEG hg38->hg19 liftover dropped/trimmed {liftover_dropped} segment(s)."
            )
        if df.empty:
            return SegParseResult(
                supplied=True,
                valid=False,
                raw_count=raw_count,
                segments=empty_seg,
                error_reason="No valid segments remained after hg38->hg19 liftover",
                liftover_dropped=liftover_dropped,
                warnings=warnings_list,
            )

    num_cols = ["loc.start", "loc.end", "seg.mean"]
    if "num.mark" in df.columns:
        num_cols.append("num.mark")
    nums = df[num_cols].apply(pd.to_numeric, errors="coerce")

    good = (
        np.isfinite(nums).all(axis=1)
        & nums["loc.start"].ge(0)
        & nums["loc.end"].gt(nums["loc.start"])
        & df["chrom"].isin(VALID_SEG_CHROMS)
    )
    if "num.mark" in nums.columns:
        good &= nums["num.mark"].ge(0)

    if not bool(good.all()):
        n_bad = int((~good).sum())
        return SegParseResult(
            supplied=True,
            valid=False,
            raw_count=raw_count,
            segments=empty_seg,
            error_reason=f"Invalid segment coordinates, chromosome names, or NaN values in {n_bad} row(s)",
            liftover_dropped=liftover_dropped,
            warnings=warnings_list,
        )

    df["loc.start"] = nums["loc.start"].astype(np.int64)
    df["loc.end"] = nums["loc.end"].astype(np.int64)
    df["seg.mean"] = nums["seg.mean"].astype(np.float64)
    if "num.mark" in nums.columns:
        df["num.mark"] = nums["num.mark"].astype(np.int64)

    # Verify bounds against hg19 chromosome sizes
    for chrom, grp in df.groupby("chrom", sort=False):
        max_size = chrom_sizes.get(f"chr{chrom}")
        if max_size is not None and (grp["loc.end"] > max_size).any():
            return SegParseResult(
                supplied=True,
                valid=False,
                raw_count=raw_count,
                segments=empty_seg,
                error_reason=f"Segment coordinate exceeds hg19 chr{chrom} length ({max_size:,} bp)",
                liftover_dropped=liftover_dropped,
                warnings=warnings_list,
            )

    df = df.sort_values(["ID", "chrom", "loc.start", "loc.end"]).reset_index(drop=True)
    previous_end = df.groupby(["ID", "chrom"])["loc.end"].cummax().groupby([df["ID"], df["chrom"]]).shift()
    if df["loc.start"].lt(previous_end).any():
        n_ov = int(df["loc.start"].lt(previous_end).sum())
        return SegParseResult(
            supplied=True,
            valid=False,
            raw_count=raw_count,
            segments=empty_seg,
            error_reason=f"Overlapping or duplicate CNA segments detected ({n_ov} overlap(s))",
            liftover_dropped=liftover_dropped,
            warnings=warnings_list,
        )

    if not df["chrom"].ne("Y").any():
        return SegParseResult(
            supplied=True,
            valid=False,
            raw_count=raw_count,
            segments=empty_seg,
            error_reason="SEG file has no autosomal or chrX segments",
            liftover_dropped=liftover_dropped,
            warnings=warnings_list,
        )

    return SegParseResult(
        supplied=True,
        valid=True,
        raw_count=raw_count,
        segments=df,
        liftover_dropped=liftover_dropped,
        warnings=warnings_list,
    )
