"""Feature label/value formatting and publication styling for clinical PDF & HTML reports."""

from __future__ import annotations

from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..config import TISSUE_DISPLAY_NAMES

# Colorblind-safe Okabe-Ito palette
OKABE_ITO = {
    "blue": "#0072B2",
    "vermilion": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "slate": "#334155",
}

MODALITY_COLORS: dict[str, str] = {
    "Copy Number (CNA)": "#0072B2",
    "Somatic Mutations": "#D55E00",
    "Structural Variants": "#009E73",
    "SBS-6 Spectrum": "#CC79A7",
    "Clinical (Age/Sex)": "#E69F00",
}

KNOWN_HOTSPOT_ALIASES: dict[str, str] = {
    "TERT_5_1295228_G_A": "TERT C228T Promoter",
    "TERT_5_1295250_G_A": "TERT C250T Promoter",
    "BRAF_7_140453136_A_T": "BRAF V600E",
    "KRAS_12_25398284_C_A": "KRAS G12V",
    "KRAS_12_25398284_C_T": "KRAS G12D",
    "KRAS_12_25398285_C_A": "KRAS G12C",
    "KRAS_12_25398285_C_T": "KRAS G12S",
    "KRAS_12_25398285_C_G": "KRAS G12R",
    "EGFR_7_55259515_T_G": "EGFR L858R",
    "IDH1_2_209113112_C_T": "IDH1 R132H",
    "PIK3CA_3_178936091_G_A": "PIK3CA E545K",
    "PIK3CA_3_178952085_A_G": "PIK3CA H1047R",
}


def apply_report_rcparams() -> None:
    """Configure Matplotlib for crisp TrueType vector PDF rendering."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 8.8,
            "axes.titlesize": 10.0,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.8,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.28,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def format_tissue_name(code: str, include_code: bool = False) -> str:
    disp = TISSUE_DISPLAY_NAMES.get(code, code.replace("_", " ").title())
    if include_code and disp.upper() != code.upper():
        return f"{disp} ({code})"
    return disp


def classify_feature_modality(feature_name: str) -> str:
    """Map any compiled feature name to one of the 5 V20 biological modalities."""
    base = feature_name[:-9] if feature_name.endswith("__missing") else feature_name
    if base in ("AGE_AT_SEQ_REPORT", "SEX", "AGE_AT_SEQ_REPORT_missing", "SEX_missing"):
        return "Clinical (Age/Sex)"
    if base.startswith("CNA_"):
        return "Copy Number (CNA)"
    if base in ("Ti_Tv_Ratio", "Spectrum_SNP_Count", "has_spectrum_data") or (
        base.startswith(("C_", "T_")) and base.endswith(("_Count", "_Fraction"))
    ):
        return "SBS-6 Spectrum"
    if (
        base in ("SV_Total_Count", "has_observed_SV")
        or base.endswith("_SV")
        or ("-" in base and "_" not in base)
    ):
        return "Structural Variants"
    return "Somatic Mutations"


def format_feature_name(
    feature_name: str,
    hgvsp_lookup: Optional[dict[str, str]] = None,
    max_len: int = 38,
) -> str:
    """Convert internal compiled feature name to clean clinical display text."""
    is_missing_ind = feature_name.endswith("__missing")
    base = feature_name[:-9] if is_missing_ind else feature_name

    exact_names = {
        "AGE_AT_SEQ_REPORT": "Age at Sequencing",
        "AGE_AT_SEQ_REPORT_missing": "Age Missingness",
        "SEX": "Biological Sex",
        "SEX_missing": "Sex Missingness",
        "Total_Mutation_Count": "Panel Mutation Burden",
        "Log_TMB": "Log(1 + Panel Mutations)",
        "SNP_Count": "SNV Count",
        "INS_Count": "Insertion Count",
        "DEL_Count": "Deletion Count",
        "DNP_Count": "DNP Count",
        "TNP_Count": "TNP Count",
        "SNP_Fraction": "SNV Fraction",
        "Indel_Fraction": "Indel Fraction",
        "DNP_TNP_Fraction": "DNP/TNP Fraction",
        "TERT_other": "TERT Non-Hotspot Mut",
        "TERT_promoter_C228T": "TERT Promoter C228T",
        "TERT_promoter_C250T": "TERT Promoter C250T",
        "TERT_promoter_other": "TERT Promoter (Other 5'Flank)",
        "TERT_promoter_any": "TERT Promoter Mutation",
        "SV_Total_Count": "Panel SV Count",
        "has_observed_SV": "Any Somatic SV",
        "Ti_Tv_Ratio": "Ti/Tv Ratio",
        "Spectrum_SNP_Count": "SBS-6 SNV Count",
        "has_spectrum_data": "SBS-6 Evaluable (>=4 SNVs)",
        "CNA_mean_rank": "CNA Genome Mean Rank",
        "CNA_std_rank": "CNA Genome SD Rank",
        "CNA_median_rank": "CNA Genome Median Rank",
        "CNA_mad_rank": "CNA Genome MAD Rank",
        "CNA_inv_arm_rank_std": "CNA Arm Rank SD",
        "CNA_inv_arm_rank_iqr": "CNA Arm Rank IQR",
        "CNA_inv_arm_pos_frac": "CNA Gained Arms Fraction",
        "CNA_inv_arm_neg_frac": "CNA Lost Arms Fraction",
        "CNA_inv_chrom_sign_flip_count": "CNA Chrom p/q Sign Flips",
        "CNA_inv_focal_resid_std": "CNA Focal Residual SD",
        "CNA_inv_focal_pos_frac": "CNA Focal Gain Fraction",
        "CNA_inv_focal_neg_frac": "CNA Focal Loss Fraction",
        "CNA_inv_focal_max_pos": "CNA Peak Focal Gain",
        "CNA_inv_focal_min_neg": "CNA Peak Focal Loss",
        "CNA_inv_z_lag1": "CNA Spatial Autocorrelation",
        "CNA_inv_mad_rms_ratio": "CNA Tail-to-Bulk Ratio",
        "CNA_inv_arm_r2": "CNA Arm-Level Variance R²",
        "CNA_inv_focal_z_std": "CNA Focal Z-Score SD",
        "CNA_inv_focal_z_max": "CNA Peak Focal Z-Amp",
        "CNA_inv_focal_z_min": "CNA Peak Focal Z-Del",
        "CNA_inv_z_gt2_frac": "CNA High-Gain Fraction (Z>2)",
        "CNA_inv_z_gt35_frac": "CNA Extreme Amp Fraction (Z>3.5)",
        "CNA_inv_z_lt_m2_frac": "CNA Deep-Loss Fraction (Z<-2)",
        "CNA_inv_z_lt_m35_frac": "CNA Extreme Del Fraction (Z<-3.5)",
        "CNA_inv_z_q95": "CNA 95th Percentile Z",
        "CNA_inv_z_q05": "CNA 5th Percentile Z",
        "CNA_inv_z_kurt_log": "CNA Tail Kurtosis (Log)",
        "CNA_inv_rank_iqr": "CNA Genome Rank IQR",
        "CNA_observed_bin_fraction": "CNA Genome Bin Coverage",
        "CNA_num_segments": "CNA Total Segments",
        "CNA_num_focal_amp": "CNA Focal Amplifications",
        "CNA_num_focal_del": "CNA Focal Deletions",
        "CNA_num_broad_gain": "CNA Broad Gains",
        "CNA_num_broad_loss": "CNA Broad Losses",
    }

    if base in exact_names:
        label = exact_names[base]
    elif base in KNOWN_HOTSPOT_ALIASES:
        label = KNOWN_HOTSPOT_ALIASES[base]
    elif hgvsp_lookup and base in hgvsp_lookup and hgvsp_lookup[base]:
        gene = base.split("_", 1)[0]
        label = f"{gene} {hgvsp_lookup[base]}"
    elif base.startswith("CNA_bin__chr"):
        parts = base.replace("CNA_bin__", "").split("_")
        chrom = parts[0]
        s_mb = int(parts[1]) // 1_000_000
        e_mb = round(int(parts[2]) / 1_000_000)
        label = f"CNA {chrom}:{s_mb}-{e_mb} Mb"
    elif base.startswith("CNA_zbin__chr"):
        parts = base.replace("CNA_zbin__", "").split("_")
        chrom = parts[0]
        s_mb = int(parts[1]) // 1_000_000
        e_mb = round(int(parts[2]) / 1_000_000)
        label = f"CNA Z {chrom}:{s_mb}-{e_mb} Mb"
    elif base.startswith("CNA_coverage__chr"):
        parts = base.replace("CNA_coverage__", "").split("_")
        chrom = parts[0]
        s_mb = int(parts[1]) // 1_000_000
        e_mb = round(int(parts[2]) / 1_000_000)
        label = f"CNA Cov {chrom}:{s_mb}-{e_mb} Mb"
    elif base.startswith("CNA_chr") and base.endswith("_mean_rank"):
        chrom = base.replace("CNA_", "").replace("_mean_rank", "")
        label = f"CNA {chrom} Mean Rank"
    elif base.startswith("CNA_chr") and base.endswith("_observed_bin_fraction"):
        chrom = base.replace("CNA_", "").replace("_observed_bin_fraction", "")
        label = f"CNA {chrom} Coverage"
    elif base.startswith("CNA_num_") and base.endswith("_segments"):
        kind = base.replace("CNA_num_", "").replace("_segments", "").replace("_", " ")
        label = f"CNA {kind.title()} Segs"
    elif base.startswith("CNA_observed_length_fraction_"):
        kind = base.replace("CNA_observed_length_fraction_", "").replace("_", " ")
        label = f"CNA {kind.title()} Span %"
    elif base.startswith("CNA_observed_length_"):
        kind = base.replace("CNA_observed_length_", "").replace("_", " ")
        label = f"CNA {kind.title()} Span"
    elif base.startswith(("C_", "T_")) and base.endswith("_Fraction"):
        sub = base.replace("_Fraction", "").replace("_", ">")
        label = f"SBS {sub} Fraction"
    elif base.startswith(("C_", "T_")) and base.endswith("_Count"):
        sub = base.replace("_Count", "").replace("_", ">")
        label = f"SBS {sub} Count"
    elif base.endswith("_mut_count"):
        gene = base[:-10]
        label = f"{gene} Mut Count"
    elif base.endswith(("_missense", "_truncating", "_inframe", "_splice", "_other")):
        gene, vtype = base.rsplit("_", 1)
        vmap = {
            "missense": "Missense",
            "truncating": "Truncating",
            "inframe": "In-Frame",
            "splice": "Splice",
            "other": "Other Mut",
        }
        label = f"{gene} {vmap[vtype]}"
    elif base.endswith("_SV"):
        gene = base[:-3]
        label = f"SV: {gene}"
    elif "-" in base and "_" not in base:
        label = f"SV: {base}"
    else:
        # Enriched genomic hotspot: GENE_CHROM_POS_REF_ALT
        parts = base.split("_")
        if len(parts) == 5 and parts[2].isdigit():
            gene, chrom, pos, ref, alt = parts
            if len(ref) <= 3 and len(alt) <= 3:
                label = f"{gene} chr{chrom}:{pos} {ref}>{alt}"
            else:
                label = f"{gene} chr{chrom}:{pos} indel"
        else:
            label = base.replace("_", " ")

    if is_missing_ind:
        label += " [unmeasured]"
    if len(label) > max_len:
        label = label[: max_len - 1] + "…"
    return label


def format_feature_value(feature_name: str, value: float) -> str:
    """Format a sample's transformed float32 feature value cleanly for SHAP display."""
    if feature_name.endswith("__missing"):
        return "Unmeasured" if value > 0.5 else "Measured"
    if feature_name in ("AGE_AT_SEQ_REPORT_missing", "SEX_missing"):
        return "Missing" if value > 0.5 else "Provided"
    if feature_name == "AGE_AT_SEQ_REPORT":
        return f"{value:.0f} yr"
    if feature_name == "SEX":
        return "Male" if value >= 0.5 else "Female"
    if feature_name == "has_spectrum_data":
        return "Yes" if value > 0.5 else "No"
    if feature_name == "has_observed_SV":
        return "Present" if value > 0.5 else "None"

    if feature_name.startswith(("CNA_bin__", "CNA_zbin__")) or (
        feature_name.startswith("CNA_chr") and feature_name.endswith("_mean_rank")
    ) or feature_name in (
        "CNA_mean_rank",
        "CNA_median_rank",
        "CNA_inv_focal_max_pos",
        "CNA_inv_focal_min_neg",
        "CNA_inv_focal_z_max",
        "CNA_inv_focal_z_min",
        "CNA_inv_z_q95",
        "CNA_inv_z_q05",
        "CNA_inv_z_lag1",
    ):
        return f"{value:+.2f}"
    if feature_name == "CNA_inv_chrom_sign_flip_count":
        return f"{int(round(value))}"
    if feature_name.startswith("CNA_inv_") and feature_name.endswith("_frac"):
        return f"{value * 100:.1f}%"
    if feature_name.startswith("CNA_inv_") or feature_name in ("CNA_std_rank", "CNA_mad_rank", "Log_TMB", "Ti_Tv_Ratio"):
        return f"{value:.2f}"
    if (
        feature_name.startswith("CNA_coverage__")
        or feature_name.endswith("_observed_bin_fraction")
        or feature_name == "CNA_observed_bin_fraction"
        or feature_name.startswith("CNA_observed_length_fraction_")
        or feature_name.endswith("_Fraction")
    ):
        return f"{value * 100:.1f}%"
    if feature_name.startswith("CNA_observed_length_"):
        return f"{value / 1e6:.1f} Mb"
    if (
        feature_name.startswith("CNA_num_")
        or feature_name.endswith("_Count")
        or feature_name.endswith("_mut_count")
    ):
        return f"{int(round(value))}"

    # Binary mutation / hotspot / SV indicator
    return "Present" if value > 0.5 else "Absent"
