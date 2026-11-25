"""Unit tests for TOPO branding, SBS-6 interpretation, logo rendering, and report outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tissue_classifier.config import PipelineConfig, ReferenceData
from tissue_classifier.pipeline import run_pipeline
from tissue_classifier.prediction.predictor import TissuePredictor
from tissue_classifier.preprocessing.modalities import extract_spectrum_features
from tissue_classifier.reporting.formatting import (
    format_model_mode,
    format_model_tag,
    interpret_sbs6_spectrum,
)
from tissue_classifier.reporting.logo import TOPO_ICON_SVG, TOPO_LOGO_SVG, load_logo_svg_b64


def _make_maf(snv_pairs: list[tuple[str, str]], dnps: list[tuple[str, str]] | None = None) -> pd.DataFrame:
    rows = []
    for ref, alt in snv_pairs:
        rows.append(
            {
                "Tumor_Sample_Barcode": "S1",
                "Variant_Type": "SNP",
                "Reference_Allele": ref,
                "Tumor_Seq_Allele2": alt,
            }
        )
    for ref, alt in dnps or []:
        rows.append(
            {
                "Tumor_Sample_Barcode": "S1",
                "Variant_Type": "DNP",
                "Reference_Allele": ref,
                "Tumor_Seq_Allele2": alt,
            }
        )
    return pd.DataFrame(rows)


def test_topo_logo_assets_and_model_labels() -> None:
    assert TOPO_LOGO_SVG.exists() and not TOPO_LOGO_SVG.is_symlink()
    assert TOPO_ICON_SVG.exists() and not TOPO_ICON_SVG.is_symlink()
    b64 = load_logo_svg_b64("logo")
    assert len(b64) > 500
    assert format_model_mode("primary_50pct_blend") == "TOPO Primary (with CNA)"
    assert format_model_mode("fallback_without_cna") == "TOPO Fallback (No-CNA)"
    assert format_model_mode("primary_50pct_blend", is_no_call=True) == "TOPO Quality Gate (No-Call)"
    assert format_model_tag("primary_50pct_blend") == "TOPO Primary"


def test_sbs6_interpretation_tobacco_uv_deamination_and_low_snv() -> None:
    # Tobacco smoke-like (4/6 C>A transversions)
    maf_lung = _make_maf([("C", "A"), ("G", "T"), ("C", "A"), ("G", "T"), ("T", "G"), ("C", "G")])
    spec_lung = extract_spectrum_features(maf_lung, "S1")
    interp_lung = interpret_sbs6_spectrum(spec_lung, maf_lung)
    assert interp_lung.evaluable
    assert interp_lung.pattern_code == "TOBACCO_LIKE"
    assert "Tobacco" in interp_lung.label

    # UV exposure-like (10/11 C>T + GG>AA DNP)
    maf_mel = _make_maf([("C", "T")] * 5 + [("G", "A")] * 5 + [("A", "T")], dnps=[("GG", "AA")])
    spec_mel = extract_spectrum_features(maf_mel, "S1")
    interp_mel = interpret_sbs6_spectrum(spec_mel, maf_mel)
    assert interp_mel.evaluable
    assert interp_mel.has_cc_tt_dnp
    assert interp_mel.pattern_code == "UV_LIKE"
    assert "UV" in interp_mel.label

    # Low-N C>T (3/4 C>T, no CC>TT DNP) -> C>T deamination, NOT UV
    maf_crc = _make_maf([("C", "T"), ("C", "T"), ("G", "A"), ("G", "T")])
    spec_crc = extract_spectrum_features(maf_crc, "S1")
    interp_crc = interpret_sbs6_spectrum(spec_crc, maf_crc)
    assert interp_crc.evaluable
    assert interp_crc.pattern_code == "CT_DEAMINATION"

    # Insufficient SNVs (<4)
    maf_low = _make_maf([("C", "T"), ("C", "A")])
    spec_low = extract_spectrum_features(maf_low, "S1")
    interp_low = interpret_sbs6_spectrum(spec_low, maf_low)
    assert not interp_low.evaluable
    assert interp_low.pattern_code == "NOT_EVALUABLE"


def test_topo_report_json_html_pdf_contract(
    tmp_path: Path,
    demo_dir: Path,
    ref_data: ReferenceData,
    predictor_cache: dict[Path, TissuePredictor],
) -> None:
    """Verify TOPO branding, logo embedding, SBS-6 interpretation, and removal of fold compatibility."""
    cfg = PipelineConfig(
        maf_path=demo_dir / "SYNTH_LUNG_01.maf",
        seg_path=demo_dir / "SYNTH_LUNG_01.seg",
        age=68.0,
        sex="Female",
        genome="hg19",
        output_dir=tmp_path / "topo_report_check",
        sample_id="SYNTH_LUNG_01",
        compute_shap=False,
        generate_html=True,
    )
    out = run_pipeline(cfg, predictor_cache=predictor_cache, ref=ref_data)
    assert out.status == "CALLABLE"
    assert out.report_pdf_path.exists() and out.report_pdf_path.stat().st_size > 10_000
    assert out.report_html_path is not None and out.report_html_path.exists()

    payload = json.loads(out.prediction_json_path.read_text())
    assert payload["model_name"] == "TOPO"
    assert payload["intended_use"] == "For Research Use Only — Not for Diagnostic Use"
    assert payload["model_mode_display"] == "TOPO Primary (with CNA)"
    assert payload["prediction"]["model_display_name"] == "TOPO Primary (with CNA)"
    assert "fold_compatibility" not in payload
    assert payload["sbs6_interpretation"]["pattern_code"] == "TOBACCO_LIKE"

    html = out.report_html_path.read_text()
    assert "data:image/svg+xml;base64," in html
    assert "TOPO Primary (with CNA)" in html
    assert "Tobacco Smoke-Like" in html
    assert "For Research Use Only" in html
    assert "Decision-Support" not in html
    assert "5-Fold Technical Compatibility" not in html
    assert "RealMLP + TabM" not in html

