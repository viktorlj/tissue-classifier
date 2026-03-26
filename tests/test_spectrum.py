"""Tests for SBS-6 mutation spectrum feature extraction."""

import numpy as np
import pandas as pd
import pytest

from tissue_classifier.feature_extractors.spectrum import (
    classify_sbs6,
    extract_spectrum_features,
)


class TestClassifySBS6:
    """Test the classify_sbs6 function for all 6 classes."""

    def test_c_to_a(self):
        assert classify_sbs6("C", "A") == "C>A"

    def test_c_to_g(self):
        assert classify_sbs6("C", "G") == "C>G"

    def test_c_to_t(self):
        assert classify_sbs6("C", "T") == "C>T"

    def test_t_to_a(self):
        assert classify_sbs6("T", "A") == "T>A"

    def test_t_to_c(self):
        assert classify_sbs6("T", "C") == "T>C"

    def test_t_to_g(self):
        assert classify_sbs6("T", "G") == "T>G"

    def test_purine_ref_g_to_a_becomes_c_to_t(self):
        """G>A on purine strand should be classified as C>T (pyrimidine convention)."""
        assert classify_sbs6("G", "A") == "C>T"

    def test_purine_ref_g_to_t_becomes_c_to_a(self):
        assert classify_sbs6("G", "T") == "C>A"

    def test_purine_ref_g_to_c_becomes_c_to_g(self):
        assert classify_sbs6("G", "C") == "C>G"

    def test_purine_ref_a_to_t_becomes_t_to_a(self):
        assert classify_sbs6("A", "T") == "T>A"

    def test_purine_ref_a_to_g_becomes_t_to_c(self):
        assert classify_sbs6("A", "G") == "T>C"

    def test_purine_ref_a_to_c_becomes_t_to_g(self):
        assert classify_sbs6("A", "C") == "T>G"

    def test_same_allele_returns_none(self):
        assert classify_sbs6("C", "C") is None

    def test_invalid_allele_returns_none(self):
        assert classify_sbs6("N", "A") is None
        assert classify_sbs6("C", "X") is None


class TestNaNThreshold:
    """Test that <4 SNPs gives NaN fractions but valid counts."""

    def test_below_threshold_nan_fractions(self, ref):
        """With fewer than 4 SNPs, fractions and Ti/Tv should be NaN."""
        maf = pd.DataFrame({
            "Hugo_Symbol": ["TP53", "KRAS", "BRAF"],
            "Variant_Type": ["SNP", "SNP", "SNP"],
            "Reference_Allele": ["C", "C", "T"],
            "Tumor_Seq_Allele2": ["A", "T", "G"],
            "Variant_Classification": ["Missense_Mutation"] * 3,
        })
        result = extract_spectrum_features(maf, ref)

        # Counts should be valid (not NaN)
        assert result["C_A_Count"] == 1.0
        assert result["C_T_Count"] == 1.0
        assert result["T_G_Count"] == 1.0
        assert result["Spectrum_SNP_Count"] == 3.0

        # Fractions and Ti/Tv should be NaN
        assert np.isnan(result["C_A_Fraction"])
        assert np.isnan(result["C_T_Fraction"])
        assert np.isnan(result["T_G_Fraction"])
        assert np.isnan(result["Ti_Tv_Ratio"])

    def test_zero_snps(self, ref):
        """No SNPs at all should produce zero counts and NaN fractions."""
        maf = pd.DataFrame({
            "Hugo_Symbol": ["TP53"],
            "Variant_Type": ["DEL"],
            "Reference_Allele": ["CG"],
            "Tumor_Seq_Allele2": ["-"],
            "Variant_Classification": ["Frame_Shift_Del"],
        })
        result = extract_spectrum_features(maf, ref)

        assert result["Spectrum_SNP_Count"] == 0.0
        assert result["C_A_Count"] == 0.0
        assert np.isnan(result["C_A_Fraction"])
        assert np.isnan(result["Ti_Tv_Ratio"])


class TestNormalCase:
    """Test that SNPs produce correct proportions."""

    def test_correct_proportions(self, ref):
        """4 SNPs should produce correct fractions."""
        maf = pd.DataFrame({
            "Hugo_Symbol": ["TP53", "KRAS", "BRAF", "EGFR"],
            "Variant_Type": ["SNP", "SNP", "SNP", "SNP"],
            "Reference_Allele": ["C", "C", "T", "T"],
            "Tumor_Seq_Allele2": ["A", "A", "C", "G"],
            "Variant_Classification": ["Missense_Mutation"] * 4,
        })
        result = extract_spectrum_features(maf, ref)

        assert len(result) == 14
        assert result["Spectrum_SNP_Count"] == 4.0
        assert result["C_A_Count"] == 2.0
        assert result["T_C_Count"] == 1.0
        assert result["T_G_Count"] == 1.0
        assert result["C_A_Fraction"] == pytest.approx(0.5)
        assert result["T_C_Fraction"] == pytest.approx(0.25)
        assert result["T_G_Fraction"] == pytest.approx(0.25)
        assert result["C_G_Fraction"] == pytest.approx(0.0)
        assert result["C_T_Fraction"] == pytest.approx(0.0)
        assert result["T_A_Fraction"] == pytest.approx(0.0)

    def test_ti_tv_ratio(self, ref):
        """Verify Ti/Tv ratio: Ti=C>T+T>C, Tv=C>A+C>G+T>A+T>G."""
        # 2 transitions (C>T, T>C), 2 transversions (C>A, T>G)
        maf = pd.DataFrame({
            "Hugo_Symbol": ["G1", "G2", "G3", "G4"],
            "Variant_Type": ["SNP", "SNP", "SNP", "SNP"],
            "Reference_Allele": ["C", "T", "C", "T"],
            "Tumor_Seq_Allele2": ["T", "C", "A", "G"],
            "Variant_Classification": ["Missense_Mutation"] * 4,
        })
        result = extract_spectrum_features(maf, ref)

        # Ti = 2 (C>T + T>C), Tv = 2 (C>A + T>G)
        assert result["Ti_Tv_Ratio"] == pytest.approx(1.0)

    def test_feature_order_matches_reference(self, ref):
        """Result index must match ref.spectrum_features exactly."""
        maf = pd.DataFrame({
            "Hugo_Symbol": ["TP53"] * 5,
            "Variant_Type": ["SNP"] * 5,
            "Reference_Allele": ["C", "C", "C", "T", "T"],
            "Tumor_Seq_Allele2": ["A", "G", "T", "A", "C"],
            "Variant_Classification": ["Missense_Mutation"] * 5,
        })
        result = extract_spectrum_features(maf, ref)
        assert list(result.index) == ref.spectrum_features
