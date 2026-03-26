"""End-to-end tests with real GENIE test-split samples.

These tests compare the inference pipeline output against ground-truth
feature vectors from the training pipeline, and verify predictions.

Marked as slow — run with: pytest -m slow
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tissue_classifier.config import PipelineConfig, ReferenceData
from tissue_classifier.pipeline import run_pipeline

FIXTURES = Path(__file__).parent / "fixtures" / "real_samples"
METADATA_PATH = FIXTURES / "sample_metadata.json"
TEST_PARQUET = (
    Path(__file__).parent.parent.parent
    / "ClaudeAnalysis_V2" / "_Data" / "Compiled" / "test.parquet"
)
REFERENCE_DIR = Path(__file__).parent.parent / "reference_data"


def _load_sample_metadata() -> dict[str, str]:
    with open(METADATA_PATH) as f:
        return json.load(f)


def _sample_ids():
    if not METADATA_PATH.exists():
        return []
    return list(_load_sample_metadata().keys())


@pytest.fixture(scope="module")
def ground_truth_df() -> pd.DataFrame:
    """Load ground-truth feature matrix from training pipeline."""
    return pd.read_parquet(TEST_PARQUET)


@pytest.fixture(scope="module")
def ref() -> ReferenceData:
    return ReferenceData(REFERENCE_DIR)


@pytest.mark.slow
@pytest.mark.skipif(not METADATA_PATH.exists(), reason="Real sample fixtures not available")
@pytest.mark.skipif(not TEST_PARQUET.exists(), reason="Test parquet not available")
@pytest.mark.parametrize("sample_id", _sample_ids())
class TestFeatureComparison:
    """Compare pipeline-produced features against training pipeline ground truth."""

    def test_feature_vector_matches(self, sample_id, ground_truth_df, ref, tmp_path):
        maf_path = FIXTURES / f"{sample_id}.maf"
        seg_path = FIXTURES / f"{sample_id}.seg"

        cfg = PipelineConfig(
            maf_path=maf_path,
            seg_path=seg_path if seg_path.exists() else None,
            output_dir=tmp_path,
            sample_id=sample_id,
            shap_nsamples=10,  # minimal for speed
        )
        result = run_pipeline(cfg)
        features_df = result["features_df"]

        # Load ground truth for this sample
        assert sample_id in ground_truth_df.index, f"{sample_id} not in test parquet"
        gt_row = ground_truth_df.loc[sample_id]

        # Get feature order (exclude TUMOR_TYPE label)
        feature_order = ref.training_feature_order
        # SV features differ because we don't provide SV file in test;
        # clinical features (AGE, SEX) differ because we don't provide them
        sv_features = {f for f in feature_order if "SV" in f or f.startswith("has_SV")}
        clinical_skip = {"AGE_AT_SEQ_REPORT", "SEX"}
        skip_features = sv_features | clinical_skip

        mismatches = []
        for feat in feature_order:
            if feat in skip_features:
                continue
            if feat not in features_df.columns:
                continue

            pipeline_val = features_df[feat].iloc[0]
            gt_val = gt_row[feat] if feat in gt_row.index else np.nan

            if pd.isna(pipeline_val) and pd.isna(gt_val):
                continue
            if pd.isna(pipeline_val) or pd.isna(gt_val):
                mismatches.append((feat, pipeline_val, gt_val))
                continue
            if abs(pipeline_val - gt_val) > 0.01 * max(abs(gt_val), 1.0):
                mismatches.append((feat, pipeline_val, gt_val))

        if mismatches:
            msg = f"\n{len(mismatches)} feature mismatches for {sample_id}:\n"
            for feat, pv, gv in mismatches[:20]:
                msg += f"  {feat}: pipeline={pv}, ground_truth={gv}\n"
            if len(mismatches) > 5:
                pytest.fail(msg)
            else:
                import warnings
                warnings.warn(msg)


@pytest.mark.slow
@pytest.mark.skipif(not METADATA_PATH.exists(), reason="Real sample fixtures not available")
@pytest.mark.parametrize("sample_id", _sample_ids())
class TestPredictionAccuracy:
    """Check that predicted tumor type matches known label (or is in top-3)."""

    def test_prediction_in_top3(self, sample_id, tmp_path):
        metadata = _load_sample_metadata()
        expected_type = metadata[sample_id]

        maf_path = FIXTURES / f"{sample_id}.maf"
        seg_path = FIXTURES / f"{sample_id}.seg"

        cfg = PipelineConfig(
            maf_path=maf_path,
            seg_path=seg_path if seg_path.exists() else None,
            output_dir=tmp_path,
            sample_id=sample_id,
            shap_nsamples=10,
        )
        result = run_pipeline(cfg)
        prediction = result["prediction"]

        # Check if expected type is in top-3
        top3 = [cls for cls, _ in prediction.top3]
        assert expected_type in top3, (
            f"{sample_id}: expected {expected_type} in top-3, "
            f"got {top3} (predicted: {prediction.predicted_class})"
        )
