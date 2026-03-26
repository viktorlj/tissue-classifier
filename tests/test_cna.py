"""Tests for CNA feature extraction."""

from tissue_classifier.feature_extractors.cna import extract_cna_features
from tissue_classifier.preprocessing.seg_parser import parse_seg


def test_parse_seg(sample_seg_path):
    seg = parse_seg(sample_seg_path)
    assert len(seg) == 15
    assert seg["chrom"].str.startswith("chr").all()


def test_cna_features_shape(sample_seg_path, ref):
    seg = parse_seg(sample_seg_path)
    features = extract_cna_features(seg, ref)
    assert len(features) == len(ref.cna_features)


def test_cna_features_nonzero(sample_seg_path, ref):
    seg = parse_seg(sample_seg_path)
    features = extract_cna_features(seg, ref)
    assert (features != 0).sum() > 0
