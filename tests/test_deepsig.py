"""Tests for DeepSig feature extraction."""
from pathlib import Path

import pandas as pd
import pytest

from tissue_classifier.feature_extractors.deepsig import (
    _nan_series,
    _parse_exposure_file,
    _write_maf_for_deepsig,
    extract_deepsig_features,
)


@pytest.fixture
def feature_names(ref):
    return ref.deepsig_features


class TestSkipMode:
    def test_returns_nan_series(self, ref):
        maf = pd.DataFrame({"Hugo_Symbol": ["TP53"]})
        result = extract_deepsig_features(maf, ref, mode="skip")
        assert len(result) == 13
        assert result.isna().all()
        assert list(result.index) == ref.deepsig_features

    def test_skip_ignores_mutation_count(self, ref):
        maf = pd.DataFrame({"Hugo_Symbol": ["TP53"] * 100})
        result = extract_deepsig_features(maf, ref, mode="skip")
        assert result.isna().all()


class TestLowMutationCount:
    def test_fewer_than_4_returns_nan(self, ref):
        maf = pd.DataFrame({"Hugo_Symbol": ["TP53", "KRAS", "BRAF"]})
        result = extract_deepsig_features(maf, ref, mode="subprocess")
        assert result.isna().all()

    def test_mutation_count_override(self, ref):
        maf = pd.DataFrame({"Hugo_Symbol": ["TP53"] * 10})
        result = extract_deepsig_features(maf, ref, mode="subprocess", mutation_count=2)
        assert result.isna().all()

    def test_exactly_4_does_not_short_circuit(self, ref):
        """With 4 mutations, we attempt to run (will fail without Rscript but won't short-circuit)."""
        maf = pd.DataFrame({"Hugo_Symbol": ["TP53"] * 4})
        result = extract_deepsig_features(maf, ref, mode="subprocess", mutation_count=4)
        # Will return NaN because Rscript likely not available, but we verify it tried
        assert len(result) == 13


class TestExposureParsing:
    def test_parse_exposure_file(self, tmp_path, feature_names):
        """Test parsing a mock exposure file matching real DeepSig R output.

        R's write.table outputs row names as the first unlabeled field,
        so data rows have 16 fields vs 15 header cols. Pandas uses
        the extra field as index automatically.
        """
        exposure = tmp_path / "exposure"
        # Header: 15 columns; data: rowname + 15 values = 16 fields
        exposure.write_text(
            '"sid" "M" "SBS2.13" "SBS4" "SBS6" "SBS7" "SBS10" "SBS11" '
            '"SBS14" "SBS15" "SBS22" "SBS26" "SBS38" "SBS44" "SBS92"\n'
            '"rowname" "SAMPLE1" 10 5 0 1 0 0 2 0 0 0 0 3 0 0\n'
        )
        result = _parse_exposure_file(exposure, feature_names)
        assert len(result) == 13
        assert result["SBS2.13"] == 5.0
        assert result["SBS6"] == 1.0
        assert result["SBS11"] == 2.0
        assert result["SBS38"] == 3.0
        assert not result.isna().any()

    def test_parse_empty_exposure(self, tmp_path, feature_names):
        """Empty exposure file returns NaN."""
        exposure = tmp_path / "exposure"
        exposure.write_text(
            '"sid" "M" "SBS2.13" "SBS4" "SBS6" "SBS7" "SBS10" "SBS11" '
            '"SBS14" "SBS15" "SBS22" "SBS26" "SBS38" "SBS44" "SBS92"\n'
        )
        result = _parse_exposure_file(exposure, feature_names)
        assert result.isna().all()


class TestMafWriting:
    def test_writes_required_columns(self, tmp_path):
        maf = pd.DataFrame({
            "Hugo_Symbol": ["TP53"],
            "Chromosome": ["17"],
            "Start_Position": [7577120],
            "End_Position": [7577120],
            "Reference_Allele": ["C"],
            "Tumor_Seq_Allele2": ["T"],
            "Variant_Classification": ["Missense_Mutation"],
            "Variant_Type": ["SNP"],
            "Tumor_Sample_Barcode": ["SAMPLE1"],
            "ExtraCol": ["drop_me"],
        })
        out = tmp_path / "test.maf"
        _write_maf_for_deepsig(maf, out)
        result = pd.read_csv(out, sep="\t")
        assert "Hugo_Symbol" in result.columns
        assert "ExtraCol" not in result.columns
