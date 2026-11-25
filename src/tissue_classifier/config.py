"""Configuration and reference data management for V20 tissue-classifier."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, Field, field_validator

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent.parent
REFERENCE_DIR = PROJECT_ROOT / "reference_data"
MODELS_ROOT = PROJECT_ROOT / "models"
PRIMARY_MODEL_DIR = MODELS_ROOT / "primary_50pct_blend"
FALLBACK_MODEL_DIR = MODELS_ROOT / "fallback_without_cna"

CNA_BIN_SIZE_BP = 10_000_000
N_FOLDS = 5

TISSUE_DISPLAY_NAMES: dict[str, str] = {
    "BLADDER": "Bladder / Urothelial",
    "BREAST": "Breast",
    "CNS": "CNS / Glioma",
    "COLORECTAL": "Colorectal",
    "ESOPHAGOGASTRIC": "Esophagogastric",
    "GERM_CELL": "Germ Cell",
    "GIST": "GIST",
    "GYN": "Gynecological",
    "HEAD_NECK": "Head & Neck",
    "HEPATO": "Hepatocellular",
    "LUNG": "Lung",
    "MBL": "Mature B-Cell Neoplasm",
    "MESOTHELIOMA": "Mesothelioma",
    "NEUROBLASTOMA": "Neuroblastoma",
    "PANCREATOBILIARY": "Pancreatobiliary",
    "PROSTATE": "Prostate",
    "RENAL": "Renal",
    "RETINOBLASTOMA": "Retinoblastoma",
    "SKIN_AND_MELANOMA": "Skin / Melanoma",
    "SOFT_TISSUE": "Soft Tissue Sarcoma",
    "THYROID": "Thyroid",
    "UVEAL_MEL": "Uveal Melanoma",
}


class PipelineConfig(BaseModel):
    """Configuration for a single-sample V20 tissue-of-origin classification run."""

    maf_path: Path
    seg_path: Optional[Path] = None
    sv_path: Optional[Path] = None
    age: Optional[float] = None
    sex: Optional[str] = None
    genome: Literal["hg19", "hg38"] = "hg19"
    allow_no_cna: bool = False
    model_dir: Path = PRIMARY_MODEL_DIR
    fallback_model_dir: Path = FALLBACK_MODEL_DIR
    reference_dir: Path = REFERENCE_DIR
    output_dir: Path = Path("./results")
    sample_id: str = "SAMPLE"
    compute_shap: bool = True
    shap_permutations: int = Field(default=4, ge=0)
    shap_background_size: int = Field(default=32, ge=4, le=64)
    generate_html: bool = False
    true_tumor_type: Optional[str] = None
    device: Optional[str] = None

    @field_validator("sex")
    @classmethod
    def _normalize_sex(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        cleaned = str(v).strip()
        if cleaned == "" or cleaned.lower() in {"unknown", "na", "nan", "none"}:
            return None
        low = cleaned.lower()
        if low in {"male", "m", "1", "1.0"}:
            return "Male"
        if low in {"female", "f", "0", "0.0"}:
            return "Female"
        raise ValueError(f"Unsupported sex value '{v}'; expected Male or Female.")

    @classmethod
    def from_yaml(cls, path: Path) -> "PipelineConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        if data.get("shap_permutations", 8) == 0:
            data.setdefault("compute_shap", False)
        return cls(**data)


class ModelSpec(BaseModel):
    """Frozen ensemble specification loaded from model_spec.json (v1 RealMLP or v2 2xRealMLP+2xTabM)."""

    version: str = "2.0.0"
    model_id: str
    display_name: str
    requires_cna: bool
    alpha_balanced: float
    temperature: float
    vector_scale_a: Optional[list[float]] = None
    vector_scale_b: Optional[list[float]] = None
    pooling: str = "linear"
    views: Optional[list[dict[str, Any]]] = None
    min_cna_observed_bin_fraction: float = 0.85
    n_folds: int = 5
    arms: list[str] = ["unweighted", "balanced"]
    labels: list[str]
    display_names: dict[str, str]
    thresholds: dict[str, float]
    policies: list[dict[str, Any]]
    fold_feature_orders: list[list[str]]
    expected_internal_accuracy: Optional[float] = None
    expected_external_vicc_accuracy: Optional[float] = None

    @classmethod
    def from_dir(cls, model_dir: Path) -> "ModelSpec":
        spec_path = Path(model_dir) / "model_spec.json"
        if not spec_path.exists():
            raise FileNotFoundError(f"Model specification not found: {spec_path}")
        return cls(**json.loads(spec_path.read_text()))


class ReferenceData:
    """Lazily loads and caches V20 reference data and 5-fold preprocessing states."""

    def __init__(self, ref_dir: Path = REFERENCE_DIR) -> None:
        self.ref_dir = Path(ref_dir)
        self._cache: dict[str, Any] = {}

    @property
    def useful_genes(self) -> set[str]:
        if "useful_genes" not in self._cache:
            df = pd.read_csv(self.ref_dir / "UsefulGenes.csv")
            self._cache["useful_genes"] = set(df["USEFUL_GENES"].astype(str).str.strip())
        return self._cache["useful_genes"]

    @property
    def bed_lookup(self) -> dict[str, list[list[int]]]:
        if "bed_lookup" not in self._cache:
            bed_df = pd.read_csv(self.ref_dir / "common_targets.bed.tsv", sep="\t")
            lookup: dict[str, list[list[int]]] = {}
            for chrom, group in bed_df.groupby("chrom", sort=False):
                intervals = sorted(group[["start", "end"]].to_numpy(dtype=int).tolist())
                merged: list[list[int]] = []
                for s, e in intervals:
                    if merged and s <= merged[-1][1]:
                        merged[-1][1] = max(merged[-1][1], e)
                    else:
                        merged.append([int(s), int(e)])
                lookup[str(chrom).removeprefix("chr")] = merged
            self._cache["bed_lookup"] = lookup
        return self._cache["bed_lookup"]

    @property
    def chrom_sizes(self) -> dict[str, int]:
        if "chrom_sizes" not in self._cache:
            df = pd.read_csv(self.ref_dir / "hg19_primary_chrom_sizes.tsv", sep="\t")
            self._cache["chrom_sizes"] = {
                str(r["chrom"]): int(r["size"]) for _, r in df.iterrows()
            }
        return self._cache["chrom_sizes"]

    @property
    def oncotree_mapping(self) -> pd.DataFrame:
        if "oncotree_mapping" not in self._cache:
            self._cache["oncotree_mapping"] = pd.read_csv(self.ref_dir / "oncotree_mapping.csv")
        return self._cache["oncotree_mapping"]

    def fold_event_state(self, fold: int) -> dict[str, Any]:
        key = f"event_state_{fold}"
        if key not in self._cache:
            p = self.ref_dir / "folds" / f"fold_{fold}" / "event_feature_state.json"
            self._cache[key] = json.loads(p.read_text())
        return self._cache[key]

    def fold_preprocessing_state(self, fold: int) -> dict[str, Any]:
        key = f"prep_state_{fold}"
        if key not in self._cache:
            p = self.ref_dir / "folds" / f"fold_{fold}" / "preprocessing_state.json"
            self._cache[key] = json.loads(p.read_text())
        return self._cache[key]

    def load_shap_background(
        self, mode: Literal["primary", "without_cna"], n_donors: int = 32
    ) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
        """Return (backgrounds_packed, common_features, coordinate_map, donor_ids)."""
        bg_path = self.ref_dir / "shap_background.npz"
        if not bg_path.exists():
            raise FileNotFoundError(f"SHAP background archive not found: {bg_path}")
        with np.load(bg_path, allow_pickle=False) as archive:
            donor_ids = archive["sample_ids"][:n_donors].astype(str).tolist()
            prefix = "primary" if mode == "primary" else "without_cna"
            packed = np.asarray(archive[f"{prefix}_packed"][:n_donors], dtype=np.float32)
            features = archive[f"{prefix}_common_features"].astype(str).tolist()
            coord_map = np.asarray(archive[f"{prefix}_coordinate_map"], dtype=np.int64)
        return packed, features, coord_map, donor_ids
