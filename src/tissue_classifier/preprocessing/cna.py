"""Exact v2 coverage-aware 10 Mb genomic bin CNA extraction with scale-invariant
signed-rank + winsorized-RMS (Z_win) normalization and 24 spatial shape summaries.

Unmeasured genomic intervals remain NaN (never zero-filled before fold preprocessing).
Assay-specific coverage fractions and raw seg.mean threshold cutoffs are excluded
from model features; observed_bin_fraction is returned solely for QC & coverage gating.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from ..config import CNA_BIN_SIZE_BP

AUTOSOMAL_AND_X_CHROMS = [str(i) for i in range(1, 23)] + ["X"]

HG19_CENTROMERES = {
    "1": 125_000_000, "2": 93_300_000, "3": 91_000_000, "4": 50_400_000,
    "5": 48_400_000, "6": 61_000_000, "7": 59_900_000, "8": 45_600_000,
    "9": 49_000_000, "10": 40_200_000, "11": 53_700_000, "12": 35_800_000,
    "13": 17_900_000, "14": 17_600_000, "15": 19_000_000, "16": 36_600_000,
    "17": 24_000_000, "18": 17_200_000, "19": 26_500_000, "20": 27_500_000,
    "21": 13_200_000, "22": 14_700_000, "X": 60_600_000,
}
ACROCENTRIC_CHROMS = {"13", "14", "15", "21", "22"}


@dataclass
class CnaExtractionResult:
    features: pd.DataFrame
    chrom_summary: pd.DataFrame
    observed_bin_fraction: float
    n_autosomal_x_segments: int


def extract_cna_features(
    segments: pd.DataFrame,
    sample_id: str,
    chrom_sizes: dict[str, int],
    bin_size: int = CNA_BIN_SIZE_BP,
) -> CnaExtractionResult:
    """Compute exact v2 scale-invariant CNA features for one sample without zero-filling unmeasured bins."""
    ids = [sample_id]
    seg = segments[segments["chrom"].ne("Y")].copy()
    if seg.empty:
        raise ValueError("Sample has no autosomal/chrX CNA segments")

    ordered = seg.sort_values(["ID", "chrom", "loc.start", "loc.end"])
    previous = (
        ordered.groupby(["ID", "chrom"])["loc.end"]
        .cummax()
        .groupby([ordered["ID"], ordered["chrom"]])
        .shift()
    )
    if ordered["loc.start"].lt(previous).any():
        raise ValueError("Overlapping or duplicate CNA segments")

    names: list[str] = []
    widths: list[int] = []
    chrom_slices: dict[str, tuple[int, int]] = {}
    for chrom in AUTOSOMAL_AND_X_CHROMS:
        start_index = len(names)
        length = chrom_sizes["chr" + chrom]
        for start in range(0, length, bin_size):
            end = min(start + bin_size, length)
            names.append(f"CNA_bin__chr{chrom}_{start}_{end}")
            widths.append(end - start)
        chrom_slices[chrom] = (start_index, len(names))

    sums = np.zeros((1, len(names)), dtype=np.float64)
    covered = np.zeros_like(sums)

    for chrom, (offset, _stop) in chrom_slices.items():
        d = seg[seg["chrom"].eq(chrom)]
        if d.empty:
            continue
        starts = d["loc.start"].to_numpy(dtype=np.int64)
        ends = d["loc.end"].to_numpy(dtype=np.int64)
        if np.any(ends > chrom_sizes["chr" + chrom]):
            raise ValueError(f"Segment beyond hg19 chr{chrom} size")
        values = d["seg.mean"].to_numpy(dtype=np.float64)
        first = starts // bin_size
        last = (ends - 1) // bin_size
        for k in range(int((last - first).max()) + 1):
            bins = first + k
            valid = bins <= last
            s = np.zeros(int(valid.sum()), dtype=np.int64)
            b = bins[valid]
            v = values[valid]
            overlap = np.minimum(ends[valid], (b + 1) * bin_size) - np.maximum(
                starts[valid], b * bin_size
            )
            np.add.at(sums, (s, b + offset), v * overlap)
            np.add.at(covered, (s, b + offset), overlap)

    widths_arr = np.asarray(widths, dtype=np.float64)
    if np.any(covered > widths_arr + 1e-6):
        raise ValueError("Overlapping segments double-count measured bases")

    raw = np.divide(sums, covered, out=np.full_like(sums, np.nan), where=covered > 0)
    coverage = covered / widths_arr
    if not np.isfinite(raw).any(axis=1).all():
        raise ValueError("Sample has no measured autosomal/X CNA bins")

    nz = np.isfinite(raw) & (raw != 0)
    ranks = rankdata(
        np.where(nz, np.abs(raw), np.nan), axis=1, method="average", nan_policy="omit"
    )
    ranks = np.sign(raw) * ranks / np.maximum(nz.sum(axis=1, keepdims=True), 1)
    ranks[np.isfinite(raw) & (raw == 0)] = 0.0

    # Exact linear-scale-invariant winsorized RMS Z_win bins
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        p02 = np.nanpercentile(raw, 2, axis=1, keepdims=True)
        p98 = np.nanpercentile(raw, 98, axis=1, keepdims=True)
        rw = np.clip(raw, p02, p98)
        med = np.nanmedian(rw, axis=1, keepdims=True)
        rc = rw - med
        rms = np.sqrt(np.nanmean(rc ** 2, axis=1, keepdims=True))
        z_mat = np.clip((raw - med) / np.maximum(rms, 1e-6), -5.0, 5.0)
        z_mat[~np.isfinite(raw)] = np.nan

    data: dict[str, np.ndarray] = {
        name: ranks[:, i].astype("float32") for i, name in enumerate(names)
    }
    chrom_rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for chrom, (start, stop) in chrom_slices.items():
            mean_rank = np.nanmean(ranks[:, start:stop], axis=1)
            obs_frac = (coverage[:, start:stop] > 0).mean(axis=1)
            mean_raw = np.nanmean(raw[:, start:stop], axis=1)
            data[f"CNA_chr{chrom}_mean_rank"] = mean_rank.astype("float32")
            chrom_rows.append(
                {
                    "chrom": f"chr{chrom}",
                    "mean_seg_log2": float(mean_raw[0]) if np.isfinite(mean_raw[0]) else np.nan,
                    "mean_signed_rank": float(mean_rank[0]) if np.isfinite(mean_rank[0]) else np.nan,
                    "observed_bin_fraction": float(obs_frac[0]),
                    "n_bins": int(stop - start),
                    "measured_bins": int((coverage[0, start:stop] > 0).sum()),
                }
            )
        data["CNA_mean_rank"] = np.nanmean(ranks, axis=1).astype("float32")
        data["CNA_std_rank"] = np.nanstd(ranks, axis=1).astype("float32")
        data["CNA_median_rank"] = np.nanmedian(ranks, axis=1).astype("float32")
        data["CNA_mad_rank"] = np.nanmedian(
            np.abs(ranks - np.nanmedian(ranks, axis=1, keepdims=True)), axis=1
        ).astype("float32")

    for i, name in enumerate(names):
        data[name.replace("CNA_bin__", "CNA_zbin__")] = z_mat[:, i].astype("float32")

    # 24 exact scale-invariant spatial / focality / amplitude shape summaries
    arm_to_cols: dict[str, list[int]] = {}
    col_to_arm: list[str | None] = []
    for idx, col in enumerate(names):
        rest = col.removeprefix("CNA_bin__chr")
        chrom, start_s, end_s = rest.split("_")
        mid = 0.5 * (int(start_s) + int(end_s))
        arm = "p" if mid < HG19_CENTROMERES[chrom] else "q"
        if chrom in ACROCENTRIC_CHROMS and arm == "p":
            col_to_arm.append(None)
            continue
        arm_name = f"chr{chrom}{arm}"
        arm_to_cols.setdefault(arm_name, []).append(idx)
        col_to_arm.append(arm_name)

    arm_names = sorted(
        arm_to_cols.keys(),
        key=lambda s: (int(s[3:-1]) if s[3:-1].isdigit() else 23, 0 if s[-1] == "p" else 1),
    )
    arm_idx_map = {name: i for i, name in enumerate(arm_names)}
    arm_rank_mat = np.full((1, len(arm_names)), np.nan, dtype=np.float64)
    arm_z_mat = np.full((1, len(arm_names)), np.nan, dtype=np.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i, arm_name in enumerate(arm_names):
            cols = arm_to_cols[arm_name]
            arm_rank_mat[:, i] = np.nanmean(ranks[:, cols], axis=1)
            arm_z_mat[:, i] = np.nanmean(z_mat[:, cols], axis=1)

        resid_r = np.full_like(ranks, np.nan)
        resid_z = np.full_like(z_mat, np.nan)
        for idx, arm_name in enumerate(col_to_arm):
            if arm_name is not None:
                ai = arm_idx_map[arm_name]
                resid_r[:, idx] = ranks[:, idx] - arm_rank_mat[:, ai]
                resid_z[:, idx] = z_mat[:, idx] - arm_z_mat[:, ai]

        inv: dict[str, np.ndarray] = {}
        inv["CNA_inv_arm_rank_std"] = np.nanstd(arm_rank_mat, axis=1)
        inv["CNA_inv_arm_rank_iqr"] = np.nanpercentile(arm_rank_mat, 75, axis=1) - np.nanpercentile(arm_rank_mat, 25, axis=1)
        inv["CNA_inv_arm_pos_frac"] = np.nanmean(arm_rank_mat > 0.25, axis=1)
        inv["CNA_inv_arm_neg_frac"] = np.nanmean(arm_rank_mat < -0.25, axis=1)

        flips = np.zeros(1, dtype=np.float64)
        for chrom in AUTOSOMAL_AND_X_CHROMS:
            if chrom in ACROCENTRIC_CHROMS:
                continue
            p_m = arm_rank_mat[:, arm_idx_map[f"chr{chrom}p"]]
            q_m = arm_rank_mat[:, arm_idx_map[f"chr{chrom}q"]]
            flips += np.where(np.isfinite(p_m) & np.isfinite(q_m) & ((p_m * q_m) < -0.05), 1.0, 0.0)
        inv["CNA_inv_chrom_sign_flip_count"] = flips

        inv["CNA_inv_focal_resid_std"] = np.nanstd(resid_r, axis=1)
        inv["CNA_inv_focal_pos_frac"] = np.nanmean(resid_r > 0.35, axis=1)
        inv["CNA_inv_focal_neg_frac"] = np.nanmean(resid_r < -0.35, axis=1)
        inv["CNA_inv_focal_max_pos"] = np.nanmax(resid_r, axis=1)
        inv["CNA_inv_focal_min_neg"] = np.nanmin(resid_r, axis=1)

        z0, z1 = z_mat[:, :-1], z_mat[:, 1:]
        vp = np.isfinite(z0) & np.isfinite(z1)
        inv["CNA_inv_z_lag1"] = np.nansum(np.where(vp, z0 * z1, 0.0), axis=1) / np.maximum(vp.sum(axis=1), 1)
        mad_w = np.nanmedian(np.abs(rc), axis=1) * 1.4826
        inv["CNA_inv_mad_rms_ratio"] = mad_w / np.maximum(rms[:, 0], 1e-6)
        var_z = np.maximum(np.nanvar(z_mat, axis=1), 1e-6)
        inv["CNA_inv_arm_r2"] = np.clip(1.0 - np.nanvar(resid_z, axis=1) / var_z, 0.0, 1.0)
        inv["CNA_inv_focal_z_std"] = np.nanstd(resid_z, axis=1)
        inv["CNA_inv_focal_z_max"] = np.nanmax(resid_z, axis=1)
        inv["CNA_inv_focal_z_min"] = np.nanmin(resid_z, axis=1)
        inv["CNA_inv_z_gt2_frac"] = np.nanmean(z_mat > 2.0, axis=1)
        inv["CNA_inv_z_gt35_frac"] = np.nanmean(z_mat > 3.5, axis=1)
        inv["CNA_inv_z_lt_m2_frac"] = np.nanmean(z_mat < -2.0, axis=1)
        inv["CNA_inv_z_lt_m35_frac"] = np.nanmean(z_mat < -3.5, axis=1)
        inv["CNA_inv_z_q95"] = np.nanpercentile(z_mat, 95, axis=1)
        inv["CNA_inv_z_q05"] = np.nanpercentile(z_mat, 5, axis=1)
        inv["CNA_inv_z_kurt_log"] = np.log1p(np.clip(np.nanmean(z_mat ** 4, axis=1), 0.0, 1000.0))
        inv["CNA_inv_rank_iqr"] = np.nanpercentile(ranks, 75, axis=1) - np.nanpercentile(ranks, 25, axis=1)

    for k, v in inv.items():
        data[k] = np.where(np.isfinite(v), v, 0.0).astype("float32")

    f = pd.DataFrame(data, index=pd.Index(ids, name="SAMPLE_ID")).astype("float32")
    if np.isinf(f.to_numpy()).any():
        raise ValueError("Infinite CNA feature encountered")

    obs_total = float((coverage > 0).mean())
    return CnaExtractionResult(
        features=f,
        chrom_summary=pd.DataFrame(chrom_rows),
        observed_bin_fraction=obs_total,
        n_autosomal_x_segments=len(seg),
    )
