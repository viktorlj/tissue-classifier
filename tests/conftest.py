"""Shared pytest fixtures for V20 tissue-classifier."""

from __future__ import annotations

from pathlib import Path
import pytest

from tissue_classifier.config import PROJECT_ROOT, ReferenceData
from tissue_classifier.prediction.predictor import TissuePredictor


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def demo_dir(project_root: Path) -> Path:
    return project_root / "demo_data"


@pytest.fixture(scope="session")
def ref_data() -> ReferenceData:
    return ReferenceData()


@pytest.fixture(scope="session")
def predictor_cache() -> dict[Path, TissuePredictor]:
    return {}
