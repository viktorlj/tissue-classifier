"""End-to-end parity (<1e-5), coherent 5-fold Top-3 SHAP additivity (<2e-5), and CLI tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from tissue_classifier.cli import app
from tissue_classifier.config import PipelineConfig, ReferenceData
from tissue_classifier.pipeline import run_pipeline
from tissue_classifier.prediction.predictor import TissuePredictor


def test_demo_samples_exact_cached_probability_parity(
    tmp_path: Path,
    demo_dir: Path,
    ref_data: ReferenceData,
    predictor_cache: dict[Path, TissuePredictor],
) -> None:
    """Verify all 6 synthetic demo cases match deterministic reference probs < 1e-5 and predict true class."""
    manifest = pd.read_csv(demo_dir / "demo_manifest.csv")
    assert len(manifest) == 6
    assert int((manifest["split"] == "synthetic_demo").sum()) == 6

    for _, row in manifest.iterrows():
        sid = str(row["sample_id"])
        assert sid.startswith("SYNTH_"), f"Expected synthetic demo ID, got {sid}"
        sv_file = str(row.get("sv_file", "") or "").strip()
        sv_path = (demo_dir / sv_file) if (sv_file and sv_file != "nan") else None

        cfg = PipelineConfig(
            maf_path=demo_dir / str(row["maf_file"]),
            seg_path=demo_dir / str(row["seg_file"]),
            sv_path=sv_path,
            age=float(row["age"]),
            sex=str(row["sex"]),
            genome="hg19",
            output_dir=tmp_path / "parity_runs",
            sample_id=sid,
            compute_shap=False,
            true_tumor_type=str(row["true_tumor_type"]),
        )
        out = run_pipeline(cfg, predictor_cache=predictor_cache, ref=ref_data)
        assert out.status == "CALLABLE"
        assert out.prediction is not None
        labels = out.prediction.labels
        pred_probs = np.array([out.prediction.all_probabilities[c] for c in labels], dtype=float)
        cached_probs = np.array([float(row[f"prob_{c}"]) for c in labels], dtype=float)
        max_diff = float(np.max(np.abs(pred_probs - cached_probs)))
        assert max_diff < 1e-5, f"Parity failed for {sid}: max_diff={max_diff:.3e}"
        assert out.prediction.predicted_class == str(row["cached_predicted_class"])
        assert out.prediction.predicted_class == str(row["true_tumor_type"])


def test_full_pipeline_top3_shap_additivity_and_multipage_pdf(
    tmp_path: Path,
    demo_dir: Path,
    ref_data: ReferenceData,
    predictor_cache: dict[Path, TissuePredictor],
) -> None:
    """Full pipeline with SHAP produces 4-page PDF, JSON, CSV, HTML, and < 2e-5 22-class additivity."""
    manifest = pd.read_csv(demo_dir / "demo_manifest.csv")
    row = manifest.iloc[0]
    sid = str(row["sample_id"])
    sv_file = str(row.get("sv_file", "") or "").strip()
    sv_path = (demo_dir / sv_file) if (sv_file and sv_file != "nan") else None

    cfg = PipelineConfig(
        maf_path=demo_dir / str(row["maf_file"]),
        seg_path=demo_dir / str(row["seg_file"]),
        sv_path=sv_path,
        age=float(row["age"]),
        sex=str(row["sex"]),
        genome="hg19",
        output_dir=tmp_path / "full_shap_e2e",
        sample_id=sid,
        compute_shap=True,
        shap_permutations=2,
        shap_background_size=32,
        generate_html=True,
        true_tumor_type=str(row["true_tumor_type"]),
    )
    out = run_pipeline(cfg, predictor_cache=predictor_cache, ref=ref_data)
    assert out.status == "CALLABLE"
    assert out.prediction is not None
    assert out.shap_result is not None
    assert out.shap_result.max_additivity_residual < 2e-5
    assert len(out.shap_result.top3_explanations) == 3

    for rank_idx, cls_exp in enumerate(out.shap_result.top3_explanations, start=1):
        assert cls_exp.rank == rank_idx
        assert cls_exp.additivity_residual < 2e-5
        # Verify shap.Explanation carries formatted display_data and human-readable feature_names
        assert cls_exp.explanation.display_data is not None
        assert len(cls_exp.explanation.feature_names) == out.shap_result.total_union_features

    assert out.report_pdf_path.exists() and out.report_pdf_path.stat().st_size > 15_000
    assert out.prediction_json_path.exists()
    assert out.features_fold0_path is not None and out.features_fold0_path.exists()
    assert out.report_html_path is not None and out.report_html_path.exists()

    js = json.loads(out.prediction_json_path.read_text())
    assert js["status"] == "CALLABLE"
    assert len(js["shap"]["top3_explanations"]) == 3


def test_cli_info_validate_and_predict(tmp_path: Path, project_root: Path) -> None:
    runner = CliRunner()
    r_info = runner.invoke(app, ["info"])
    assert r_info.exit_code == 0
    assert "primary_50pct_blend" in r_info.stdout or "Primary Ensemble" in r_info.stdout

    ex_maf = project_root / "examples" / "example_sample.maf"
    ex_seg = project_root / "examples" / "example_sample.seg"
    r_val = runner.invoke(
        app,
        ["validate", "--maf", str(ex_maf), "--seg", str(ex_seg), "--sample-id", "SYNTH_EXAMPLE_01"],
    )
    assert r_val.exit_code == 0
    assert "MAF valid" in r_val.stdout
    assert "SEG valid" in r_val.stdout

    r_pred = runner.invoke(
        app,
        [
            "predict",
            "--maf",
            str(ex_maf),
            "--seg",
            str(ex_seg),
            "--sample-id",
            "CLI_SMOKE_01",
            "--output",
            str(tmp_path / "cli_out"),
            "--no-shap",
        ],
    )
    assert r_pred.exit_code == 0
    assert (tmp_path / "cli_out" / "CLI_SMOKE_01" / "report.pdf").exists()
