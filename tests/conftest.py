"""Shared test fixtures."""

from pathlib import Path

import pytest

from tissue_classifier.config import ReferenceData

FIXTURES_DIR = Path(__file__).parent / "fixtures"
REFERENCE_DIR = Path(__file__).parent.parent / "reference_data"


@pytest.fixture
def sample_maf_path():
    return FIXTURES_DIR / "sample.maf"


@pytest.fixture
def sample_seg_path():
    return FIXTURES_DIR / "sample.seg"


@pytest.fixture
def reference_dir():
    return REFERENCE_DIR


@pytest.fixture
def ref():
    return ReferenceData(REFERENCE_DIR)
