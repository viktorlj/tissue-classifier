"""Tests for feature compiler."""

import json

import pandas as pd

from tissue_classifier.feature_extractors.clinical import extract_clinical_features
from tissue_classifier.feature_extractors.cna import extract_cna_features
from tissue_classifier.feature_extractors.deepsig import extract_deepsig_features
from tissue_classifier.feature_extractors.mutations import extract_mutation_features
from tissue_classifier.feature_extractors.mutfreq import extract_mutfreq_features
from tissue_classifier.feature_extractors.sv import extract_sv_features
from tissue_classifier.feature_extractors.tert import extract_tert_features
from tissue_classifier.preprocessing.feature_compiler import compile_features
from tissue_classifier.preprocessing.maf_parser import parse_maf
from tissue_classifier.preprocessing.seg_parser import parse_seg


def test_compile_all_features(sample_maf_path, sample_seg_path, ref):
    """Compile all features and verify output shape is exactly 4333."""
    maf = parse_maf(sample_maf_path)
    seg = parse_seg(sample_seg_path)

    feature_series = {
        "mutations": extract_mutation_features(maf, ref),
        "sv": extract_sv_features(None, ref),
        "deepsig": extract_deepsig_features(maf, ref, mode="skip"),
        "mutfreq": extract_mutfreq_features(maf, ref),
        "clinical": extract_clinical_features(age=65, sex="Male"),
        "tert": extract_tert_features(maf, ref),
        "cna": extract_cna_features(seg, ref),
    }

    result = compile_features(feature_series, ref)

    assert result.shape == (1, 4333), f"Expected (1, 4333), got {result.shape}"
    training_features = ref.training_feature_order
    assert list(result.columns) == training_features
    # No NaN values
    assert result.isna().sum().sum() == 0
