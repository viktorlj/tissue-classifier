"""CNA segment feature extraction (~416 features)."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import ReferenceData

logger = logging.getLogger(__name__)

# HG19 chromosome sizes
CHR_SIZES = {
    "chr1": 249250621, "chr2": 242951149, "chr3": 198022430, "chr4": 191154276,
    "chr5": 180915260, "chr6": 171115067, "chr7": 159138663, "chr8": 146364022,
    "chr9": 141213431, "chr10": 135534747, "chr11": 135006516, "chr12": 133851895,
    "chr13": 115169878, "chr14": 107349540, "chr15": 102531392, "chr16": 90354753,
    "chr17": 81195210, "chr18": 78077248, "chr19": 59128983, "chr20": 63025520,
    "chr21": 48129895, "chr22": 51304566, "chrX": 155270560,
}
BIN_SIZE = 10_000_000
CHROMOSOMES = [f"chr{i}" for i in range(1, 23)] + ["chrX"]


def _create_genomic_bins() -> pd.DataFrame:
    bins = []
    for chrom in CHROMOSOMES:
        for start in range(0, CHR_SIZES[chrom], BIN_SIZE):
            end = min(start + BIN_SIZE, CHR_SIZES[chrom])
            bins.append({
                "chrom": chrom, "start": start, "end": end,
                "bin_name": f"{chrom}:{start}-{end}",
            })
    return pd.DataFrame(bins)


def _segment_to_bins(seg: pd.DataFrame, bins_df: pd.DataFrame) -> dict[str, float]:
    features = {b["bin_name"]: 0.0 for _, b in bins_df.iterrows()}
    for _, s in seg.iterrows():
        ob = bins_df[
            (bins_df["chrom"] == s["chrom"])
            & (bins_df["start"] < s["loc.end"])
            & (bins_df["end"] > s["loc.start"])
        ]
        for _, b in ob.iterrows():
            overlap = min(s["loc.end"], b["end"]) - max(s["loc.start"], b["start"])
            frac = overlap / (b["end"] - b["start"])
            features[b["bin_name"]] += s["seg.mean"] * frac
    return features


def _advanced_features(seg: pd.DataFrame) -> dict[str, float]:
    f: dict[str, float] = {}
    f["mean_cn"] = seg["seg.mean"].mean()
    f["std_cn"] = seg["seg.mean"].std()
    f["median_cn"] = seg["seg.mean"].median()
    f["mad_cn"] = (seg["seg.mean"] - seg["seg.mean"].median()).abs().median()
    f["num_segments"] = len(seg)
    vals = seg["seg.mean"]
    for thresh, name in [(0.3, "gain"), (0.6, "high_gain"), (1.0, "amp")]:
        f[f"num_{name}"] = int((vals > thresh).sum())
    for thresh, name in [(-0.3, "loss"), (-0.6, "deep_loss"), (-1.0, "hom_del")]:
        f[f"num_{name}"] = int((vals < thresh).sum())
    for chrom in CHROMOSOMES:
        cs = seg[seg["chrom"] == chrom]
        if len(cs) > 0:
            f[f"{chrom}_mean"] = cs["seg.mean"].mean()
            f[f"{chrom}_std"] = cs["seg.mean"].std()
            f[f"{chrom}_num_segments"] = len(cs)
        else:
            f[f"{chrom}_mean"] = 0.0
            f[f"{chrom}_std"] = 0.0
            f[f"{chrom}_num_segments"] = 0
    sizes = seg["loc.end"] - seg["loc.start"]
    f["mean_segment_size"] = sizes.mean()
    f["median_segment_size"] = sizes.median()
    f["total_genome_size"] = sizes.sum()
    return f


def _focal_events(seg: pd.DataFrame) -> dict[str, float]:
    sizes = seg["loc.end"] - seg["loc.start"]
    vals = seg["seg.mean"]
    focal = sizes < BIN_SIZE
    fa = focal & (vals > 0.6)
    hfa = focal & (vals > 1.0)
    fd = focal & (vals < -0.6)
    dfd = focal & (vals < -1.0)
    return {
        "num_focal_amps": int(fa.sum()),
        "num_high_focal_amps": int(hfa.sum()),
        "num_focal_dels": int(fd.sum()),
        "num_deep_focal_dels": int(dfd.sum()),
        "num_broad_amps": int((~focal & (vals > 0.3)).sum()),
        "num_broad_dels": int((~focal & (vals < -0.3)).sum()),
        "total_focal_amp_size": float(sizes[fa].sum()),
        "total_focal_del_size": float(sizes[fd].sum()),
        "mean_focal_amp_size": float(sizes[fa].mean()) if fa.any() else 0.0,
        "mean_focal_del_size": float(sizes[fd].mean()) if fd.any() else 0.0,
    }


def _instability_metrics(seg: pd.DataFrame) -> dict[str, float]:
    ss = seg.sort_values(["chrom", "loc.start"])
    bp = 0
    pc, pe = None, None
    for _, s in ss.iterrows():
        if pc == s["chrom"] and pe is not None and s["loc.start"] > pe + 1:
            bp += 1
        elif pc is not None and pc != s["chrom"]:
            bp += 1
        pc, pe = s["chrom"], s["loc.end"]
    cn = ss["seg.mean"].diff().dropna()
    osc = int(((cn > 0) != (cn.shift(1) > 0)).sum()) if len(cn) > 1 else 0
    sizes = ss["loc.end"] - ss["loc.start"]
    wv = float(np.average(ss["seg.mean"].values ** 2, weights=sizes)) if len(ss) > 0 else 0.0
    gfa = (abs(ss["seg.mean"]) > 0.3).sum() / max(len(ss), 1)
    return {
        "num_breakpoints": bp,
        "num_oscillations": osc,
        "weighted_cn_variance": wv,
        "genome_fraction_altered": float(gfa),
    }


def _chromothripsis(seg: pd.DataFrame, min_osc: int = 10) -> dict[str, float]:
    scores = []
    for _, cs in seg.groupby("chrom"):
        if len(cs) < 3:
            continue
        cs = cs.sort_values("loc.start")
        states = cs["seg.mean"].apply(
            lambda x: 1 if x > 0.3 else (-1 if x < -0.3 else 0)
        )
        osc = (states.diff() != 0).sum() - 1
        scores.append(1 if osc >= min_osc else 0)
    return {
        "chromothripsis_score": sum(scores),
        "num_affected_chroms": sum(scores),
        "has_chromothripsis_signature": int(sum(scores) > 0),
    }


def _rank_normalize_bins(bin_features: dict[str, float]) -> dict[str, float]:
    """Rank-based CNA normalization.

    For samples with ≥10 non-zero bins, replace non-zero bin values with
    signed ranks normalized to [-1, 1]: sign(value) * rank(|value|) / n_nonzero.
    This eliminates platform-dependent CNA scale differences while preserving
    the relative ordering of gains and losses.
    """
    bin_keys = [k for k in bin_features if ":" in k]
    bin_vals = np.array([bin_features[k] for k in bin_keys])

    nonzero = bin_vals != 0
    if nonzero.sum() < 10:
        return bin_features

    normalized = bin_features.copy()
    nz_vals = bin_vals[nonzero]
    signs = np.sign(nz_vals)
    abs_vals = np.abs(nz_vals)
    ranks = np.argsort(np.argsort(abs_vals)).astype(float) + 1
    ranks /= len(ranks)
    ranked = signs * ranks

    nz_idx = 0
    for i, k in enumerate(bin_keys):
        if nonzero[i]:
            normalized[k] = float(ranked[nz_idx])
            nz_idx += 1

    # Recalculate chromosome means from rank-normalized bins
    for chrom in CHROMOSOMES:
        chrom_bins = [k for k in bin_keys if k.startswith(f"{chrom}:")]
        if chrom_bins:
            chrom_vals = [normalized[k] for k in chrom_bins]
            nonzero_chrom = [v for v in chrom_vals if v != 0]
            normalized[f"{chrom}_mean"] = np.mean(nonzero_chrom) if nonzero_chrom else 0.0

    # Recalculate summary stats from rank-normalized bins
    norm_vals = np.array([normalized[k] for k in bin_keys])
    nonzero_norm = norm_vals[norm_vals != 0]
    if len(nonzero_norm) > 0:
        normalized["mean_cn"] = float(np.mean(nonzero_norm))
        normalized["std_cn"] = float(np.std(nonzero_norm))
        normalized["median_cn"] = float(np.median(nonzero_norm))
        normalized["mad_cn"] = float(np.median(np.abs(nonzero_norm - np.median(nonzero_norm))))

    return normalized


def extract_cna_features(
    seg: pd.DataFrame,
    ref: ReferenceData,
) -> pd.Series:
    """Extract CNA features from a SEG DataFrame.

    Parameters
    ----------
    seg : pd.DataFrame
        Parsed SEG DataFrame.
    ref : ReferenceData
        Reference data instance.

    Returns
    -------
    pd.Series
        CNA features aligned to cna_features.json, with per-sample z-score
        normalization of bin values to match v19 training pipeline.
    """
    cna_feature_names = ref.cna_features

    if seg.empty:
        return pd.Series(0.0, index=cna_feature_names, dtype=float)

    bins_df = _create_genomic_bins()
    features: dict[str, float] = {}
    features.update(_segment_to_bins(seg, bins_df))
    features.update(_advanced_features(seg))
    features.update(_focal_events(seg))
    features.update(_instability_metrics(seg))
    features.update(_chromothripsis(seg))

    # Per-sample rank-based normalization of CNA bins (v18 pipeline)
    features = _rank_normalize_bins(features)

    result = pd.Series(0.0, index=cna_feature_names, dtype=float)
    for k, v in features.items():
        if k in result.index:
            result[k] = v
    result = result.fillna(0.0)

    logger.info("CNA features: %d non-zero out of %d", (result != 0).sum(), len(result))
    return result
