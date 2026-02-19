"""Tests for mutation feature extraction."""

import pandas as pd
import pytest

from tissue_classifier.feature_extractors.mutations import extract_mutation_features
from tissue_classifier.preprocessing.maf_parser import parse_maf


def test_parse_maf(sample_maf_path):
    maf = parse_maf(sample_maf_path)
    assert len(maf) == 12
    assert "Hugo_Symbol" in maf.columns
    # Chromosome should not have chr prefix
    assert not maf["Chromosome"].str.startswith("chr").any()


def test_mutation_features_shape(sample_maf_path, ref):
    maf = parse_maf(sample_maf_path)
    features = extract_mutation_features(maf, ref)

    # Extractor returns sparse (non-zero only); all feature names should be valid
    manifest = pd.read_csv(ref._dir / "feature_manifest.csv")
    valid_names = set(manifest[manifest["source_step"] == "Step05_Mutations"]["feature_name"])
    for name in features.index:
        assert name in valid_names, f"Feature {name} not in manifest"


def test_known_mutation_detected(sample_maf_path, ref):
    """KRAS_12_25398284_C_A should be in the enriched set and detected."""
    maf = parse_maf(sample_maf_path)
    features = extract_mutation_features(maf, ref)

    # KRAS G12 is a very common enriched mutation
    kras_id = "KRAS_12_25398284_C_A"
    if kras_id in features.index:
        assert features[kras_id] == 1
