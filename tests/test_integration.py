"""Integration tests: run full CLI predict and check output files."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tissue_classifier.cli import app
from tissue_classifier.config import ReferenceData

FIXTURES = Path(__file__).parent / "fixtures"
REAL_SAMPLES = FIXTURES / "real_samples"
METADATA_PATH = REAL_SAMPLES / "sample_metadata.json"
REFERENCE_DIR = Path(__file__).parent.parent / "reference_data"

runner = CliRunner()


def _first_sample_id() -> str | None:
    if not METADATA_PATH.exists():
        return None
    with open(METADATA_PATH) as f:
        meta = json.load(f)
    return next(iter(meta))


@pytest.mark.slow
@pytest.mark.skipif(not METADATA_PATH.exists(), reason="Real sample fixtures not available")
class TestCLIIntegration:
    """Run the full CLI predict command and validate outputs."""

    def test_predict_produces_outputs(self, tmp_path):
        sample_id = _first_sample_id()
        assert sample_id is not None

        maf_path = REAL_SAMPLES / f"{sample_id}.maf"
        seg_path = REAL_SAMPLES / f"{sample_id}.seg"

        args = [
            "predict",
            "--maf", str(maf_path),
            "--seg", str(seg_path),
            "--output", str(tmp_path),
            "--sample-id", sample_id,
            "--deepsig", "skip",
            "--shap-nsamples", "10",
        ]
        result = runner.invoke(app, args)
        assert result.exit_code == 0, f"CLI failed: {result.output}"

        output_dir = tmp_path / sample_id

        # Check features.csv exists
        features_csv = output_dir / "features.csv"
        assert features_csv.exists(), "features.csv not produced"

        # Check report.html exists
        report_html = output_dir / "report.html"
        assert report_html.exists(), "report.html not produced"

        # Validate features.csv has correct number of features
        import pandas as pd
        features_df = pd.read_csv(features_csv, index_col=0)
        ref = ReferenceData(REFERENCE_DIR)
        expected_n = len(ref.training_feature_order)
        assert features_df.shape[1] == expected_n, (
            f"Expected {expected_n} features, got {features_df.shape[1]}"
        )

    def test_predict_valid_class(self, tmp_path):
        """Check that the prediction is a valid tumor type."""
        sample_id = _first_sample_id()
        assert sample_id is not None

        maf_path = REAL_SAMPLES / f"{sample_id}.maf"
        seg_path = REAL_SAMPLES / f"{sample_id}.seg"

        from tissue_classifier.config import PipelineConfig
        from tissue_classifier.pipeline import run_pipeline

        cfg = PipelineConfig(
            maf_path=maf_path,
            seg_path=seg_path,
            deepsig_mode="skip",
            output_dir=tmp_path,
            sample_id=sample_id,
            shap_nsamples=10,
        )
        result = run_pipeline(cfg)
        prediction = result["prediction"]

        ref = ReferenceData(REFERENCE_DIR)
        valid_classes = set(ref.class_labels.keys())
        assert prediction.predicted_class in valid_classes, (
            f"Predicted '{prediction.predicted_class}' not in valid classes"
        )
