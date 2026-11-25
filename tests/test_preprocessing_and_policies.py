"""Unit and policy tests for V20 preprocessing, missing-CNA handling, liftover, and standalone hygiene."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tissue_classifier.config import PipelineConfig, ReferenceData
from tissue_classifier.pipeline import run_pipeline
from tissue_classifier.prediction.predictor import TissuePredictor
from tissue_classifier.preprocessing.cna import extract_cna_features
from tissue_classifier.preprocessing.parsers import parse_and_filter_maf, parse_and_validate_seg


def test_standalone_no_symlinks_or_legacy_references(project_root: Path) -> None:
    """Ensure zero symlinks outside .venv and zero legacy repository references."""
    forbidden = "_TissueOfOrigin_" + "Genie18_Claude"
    for p in project_root.rglob("*"):
        rel_parts = p.relative_to(project_root).parts
        if not rel_parts or rel_parts[0] in {".venv", ".pytest_cache", "__pycache__"}:
            continue
        assert not p.is_symlink(), f"Forbidden symlink inside tissue-classifier: {p}"
        if p.is_file() and p.suffix in {".py", ".toml", ".json", ".yaml", ".yml", ".md", ".html"}:
            text = p.read_text(errors="ignore")
            assert forbidden not in text, f"Forbidden legacy reference found in {p}"


def test_zero_genie_sample_level_data_in_package(project_root: Path) -> None:
    """AACR GENIE compliance: ensure zero individual GENIE sample IDs exist anywhere in tissue-classifier."""
    import re

    pat = re.compile(r"GENIE-(MSK|UCSF|VICC|DFCI|JHU|YALE|NKI|GRCC|MDA|VHIO|UHN|PROV|WAKE|DUKE|CHOP|COLU|CRUK|PHL|SCI|STJ|UCAM|VUMC)")
    scan_exts = {".py", ".toml", ".json", ".yaml", ".yml", ".md", ".html", ".csv", ".tsv", ".maf", ".seg", ".sv", ".txt"}
    for p in project_root.rglob("*"):
        rel_parts = p.relative_to(project_root).parts
        if not rel_parts or rel_parts[0] in {".venv", ".pytest_cache", "__pycache__"}:
            continue
        assert pat.search(p.name) is None, f"GENIE sample filename found in redistributable package: {p}"
        if p.is_file() and p.suffix.lower() in scan_exts:
            text = p.read_text(errors="ignore")
            assert pat.search(text) is None, f"GENIE sample-level ID found inside {p}"

    # Verify bundled shap_background.npz uses strictly synthetic aggregate-derived IDs
    bg_path = project_root / "reference_data" / "shap_background.npz"
    with np.load(bg_path, allow_pickle=False) as bg:
        sids = [str(x) for x in bg["sample_ids"].tolist()]
        assert len(sids) == 64
        for sid in sids:
            assert sid.startswith("SYNTH_BG_"), f"Non-synthetic background ID found: {sid}"
            assert pat.search(sid) is None


def test_cna_never_zero_fills_unmeasured_bins(ref_data: ReferenceData) -> None:
    """Unmeasured 10 Mb genomic intervals must stay NaN, never zero-filled."""
    seg = pd.DataFrame(
        {
            "ID": ["S1", "S1"],
            "chrom": ["1", "8"],
            "loc.start": [1_000_000, 120_000_000],
            "loc.end": [25_000_000, 135_000_000],
            "seg.mean": [-0.45, 0.85],
        }
    )
    res = extract_cna_features(seg, "S1", ref_data.chrom_sizes)
    df = res.features
    # chr2 had zero segments -> all chr2 bins must be NaN, NOT 0.0!
    chr2_bins = [c for c in df.columns if c.startswith("CNA_bin__chr2_")]
    assert len(chr2_bins) == 25  # ceil(243,199,373 / 10,000,000) = 25
    assert df[chr2_bins].isna().all(axis=None)
    chr2_zbins = [c for c in df.columns if c.startswith("CNA_zbin__chr2_")]
    assert len(chr2_zbins) == 25 and df[chr2_zbins].isna().all(axis=None)
    assert np.isnan(float(df["CNA_chr2_mean_rank"].iloc[0]))
    assert "CNA_chr2_observed_bin_fraction" not in df.columns
    assert float(res.chrom_summary.set_index("chrom").loc["chr2", "observed_bin_fraction"]) == 0.0
    # Measured bins on chr8 are finite and positive
    assert float(df["CNA_bin__chr8_120000000_130000000"].iloc[0]) > 0.0


def test_overlapping_segments_rejected(tmp_path: Path, ref_data: ReferenceData) -> None:
    seg_path = tmp_path / "overlap.seg"
    pd.DataFrame(
        {
            "ID": ["S1", "S1"],
            "chrom": ["chr1", "chr1"],
            "loc.start": [10_000, 5_000_000],
            "loc.end": [10_000_000, 15_000_000],
            "seg.mean": [0.2, -0.3],
        }
    ).to_csv(seg_path, sep="\t", index=False)
    parsed = parse_and_validate_seg(seg_path, "S1", ref_data.chrom_sizes)
    assert not parsed.valid
    assert "Overlapping" in (parsed.error_reason or "")


def test_clinical_tumor_only_maf_blank_status_and_dot_filter(
    tmp_path: Path, demo_dir: Path, ref_data: ReferenceData
) -> None:
    """Tumor-only clinical MAFs with blank Mutation_Status and '.' FILTER are accepted."""
    manifest = pd.read_csv(demo_dir / "demo_manifest.csv")
    sample_maf = pd.read_csv(demo_dir / manifest.iloc[0]["maf_file"], sep="\t", dtype=str)
    clinical_maf = sample_maf.copy()
    clinical_maf["Mutation_Status"] = ""
    clinical_maf["FILTER"] = "."
    maf_out = tmp_path / "tumor_only.maf"
    clinical_maf.to_csv(maf_out, sep="\t", index=False)

    res = parse_and_filter_maf(
        maf_out,
        sample_id="TUMOR_ONLY_CASE",
        useful_genes=ref_data.useful_genes,
        bed_lookup=ref_data.bed_lookup,
    )
    assert len(res.usable_maf) == len(sample_maf)


def test_no_usable_mutation_or_sv_returns_technical_no_call(
    tmp_path: Path, ref_data: ReferenceData, predictor_cache: dict[Path, TissuePredictor]
) -> None:
    """Zero usable panel mutations and zero SVs -> TECHNICAL_NO_CALL + valid PDF."""
    empty_panel_maf = tmp_path / "off_target.maf"
    pd.DataFrame(
        {
            "Hugo_Symbol": ["NON_PANEL_GENE_XYZ"],
            "Chromosome": ["1"],
            "Start_Position": ["100000"],
            "End_Position": ["100000"],
            "Reference_Allele": ["A"],
            "Tumor_Seq_Allele2": ["G"],
            "Variant_Classification": ["Missense_Mutation"],
            "Variant_Type": ["SNP"],
        }
    ).to_csv(empty_panel_maf, sep="\t", index=False)

    cfg = PipelineConfig(
        maf_path=empty_panel_maf,
        output_dir=tmp_path / "out",
        sample_id="NO_SIGNAL_01",
        compute_shap=False,
    )
    out = run_pipeline(cfg, predictor_cache=predictor_cache, ref=ref_data)
    assert out.status == "TECHNICAL_NO_CALL"
    assert out.compilation.no_call_reason == "NO_USABLE_MUTATION_OR_SV"
    assert out.prediction is None
    assert out.report_pdf_path.exists() and out.report_pdf_path.stat().st_size > 4_000
    payload = json.loads(out.prediction_json_path.read_text())
    assert payload["status"] == "TECHNICAL_NO_CALL"
    assert payload["no_call_reason"] == "NO_USABLE_MUTATION_OR_SV"


def test_missing_seg_defaults_to_technical_no_call(
    tmp_path: Path,
    demo_dir: Path,
    ref_data: ReferenceData,
    predictor_cache: dict[Path, TissuePredictor],
) -> None:
    """Missing --seg without --allow-no-cna must return TECHNICAL_NO_CALL and write report.pdf."""
    manifest = pd.read_csv(demo_dir / "demo_manifest.csv")
    row = manifest.iloc[0]
    cfg = PipelineConfig(
        maf_path=demo_dir / str(row["maf_file"]),
        seg_path=None,
        age=float(row["age"]),
        sex=str(row["sex"]),
        allow_no_cna=False,
        output_dir=tmp_path / "out_nocall",
        sample_id="MISSING_SEG_DEFAULT",
        compute_shap=False,
    )
    out = run_pipeline(cfg, predictor_cache=predictor_cache, ref=ref_data)
    assert out.status == "TECHNICAL_NO_CALL"
    assert out.compilation.no_call_reason == "MISSING_OR_INCOMPATIBLE_CNA"
    assert out.prediction is None
    assert out.report_pdf_path.exists() and out.report_pdf_path.stat().st_size > 4_000


def test_allow_no_cna_routes_to_fallback_without_cna(
    tmp_path: Path,
    demo_dir: Path,
    ref_data: ReferenceData,
    predictor_cache: dict[Path, TissuePredictor],
) -> None:
    """Passing --allow-no-cna when --seg is omitted routes to fallback_without_cna and writes report.pdf."""
    manifest = pd.read_csv(demo_dir / "demo_manifest.csv")
    row = manifest.iloc[0]
    sv_file = str(row.get("sv_file", "") or "").strip()
    sv_path = (demo_dir / sv_file) if (sv_file and sv_file != "nan") else None

    cfg = PipelineConfig(
        maf_path=demo_dir / str(row["maf_file"]),
        seg_path=None,
        sv_path=sv_path,
        age=float(row["age"]),
        sex=str(row["sex"]),
        allow_no_cna=True,
        output_dir=tmp_path / "out_fallback",
        sample_id="FALLBACK_NO_CNA_01",
        compute_shap=True,
        shap_permutations=2,
        generate_html=True,
    )
    out = run_pipeline(cfg, predictor_cache=predictor_cache, ref=ref_data)
    assert out.status == "CALLABLE"
    assert out.compilation.active_model_mode == "fallback_without_cna"
    assert out.compilation.fallback_warning is not None
    assert out.prediction is not None
    assert out.prediction.model_id == "fallback_without_cna"
    assert np.isclose(sum(out.prediction.all_probabilities.values()), 1.0, atol=1e-5)
    assert out.shap_result is not None
    assert out.shap_result.max_additivity_residual < 2e-5
    assert out.report_pdf_path.exists() and out.report_pdf_path.stat().st_size > 10_000
    assert out.report_html_path is not None and out.report_html_path.exists()


def test_hg38_liftover_braf_v600e(tmp_path: Path, ref_data: ReferenceData) -> None:
    """Verify hg38 BRAF V600E (chr7:140753336 A>T) lifts over to exact hg19 chr7:140453136."""
    maf_hg38 = tmp_path / "braf_hg38.maf"
    pd.DataFrame(
        {
            "Hugo_Symbol": ["BRAF"],
            "Chromosome": ["chr7"],
            "Start_Position": ["140753336"],
            "End_Position": ["140753336"],
            "Reference_Allele": ["A"],
            "Tumor_Seq_Allele2": ["T"],
            "Variant_Classification": ["Missense_Mutation"],
            "Variant_Type": ["SNP"],
        }
    ).to_csv(maf_hg38, sep="\t", index=False)

    res = parse_and_filter_maf(
        maf_hg38,
        sample_id="HG38_CASE",
        useful_genes=ref_data.useful_genes,
        bed_lookup=ref_data.bed_lookup,
        genome="hg38",
    )
    assert len(res.usable_maf) == 1
    assert res.usable_maf.iloc[0]["Start_Position"] == "140453136"
    assert res.usable_maf.iloc[0]["Mutation_ID"] == "BRAF_7_140453136_A_T"
