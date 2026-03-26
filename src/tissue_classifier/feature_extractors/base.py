"""Base protocol for feature extractors."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class FeatureExtractor(Protocol):
    def extract(self, **kwargs) -> pd.Series: ...
