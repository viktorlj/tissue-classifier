"""Configuration management for tissue-classifier."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

PACKAGE_DIR = Path(__file__).parent
PROJECT_ROOT = PACKAGE_DIR.parent.parent
REFERENCE_DIR = PROJECT_ROOT / "reference_data"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "AutoGluon"


class PipelineConfig(BaseModel):
    """Configuration for the tissue classification pipeline."""
    maf_path: Path
    seg_path: Optional[Path] = None
    sv_path: Optional[Path] = None
    age: Optional[float] = None
    sex: Optional[str] = None
    genome: str = Field(default="hg19", pattern="^(hg19|hg38)$")
    model_dir: Path = DEFAULT_MODEL_DIR
    reference_dir: Path = REFERENCE_DIR
    deepsig_mode: str = Field(default="skip", pattern="^(docker|subprocess|skip)$")
    deepsig_docker_image: str = "tissue-classifier-deepsig:latest"
    output_dir: Path = Path("./results")
    sample_id: str = "SAMPLE"
    shap_nsamples: int = 500

    @classmethod
    def from_yaml(cls, path: Path) -> PipelineConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)


PROTEIN_AFFECTING = [
    "Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins",
    "In_Frame_Del", "In_Frame_Ins", "Splice_Site", "Translation_Start_Site",
]

VARIANT_TYPE_MAP = {
    "Missense_Mutation": "missense",
    "Nonsense_Mutation": "truncating",
    "Frame_Shift_Del": "truncating",
    "Frame_Shift_Ins": "truncating",
    "In_Frame_Del": "inframe",
    "In_Frame_Ins": "inframe",
    "Splice_Site": "splice",
    "Splice_Region": "splice",
}


class ReferenceData:
    """Lazily loads and caches reference data files."""

    def __init__(self, ref_dir: Path = REFERENCE_DIR) -> None:
        self._dir = ref_dir
        self._cache: dict = {}

    def _load_json(self, name: str) -> list | dict:
        if name not in self._cache:
            with open(self._dir / name) as f:
                self._cache[name] = json.load(f)
        return self._cache[name]

    @property
    def training_feature_order(self) -> list[str]:
        return self._load_json("training_feature_order.json")

    @property
    def class_labels(self) -> dict[str, int]:
        return self._load_json("class_labels.json")

    @property
    def class_names(self) -> list[str]:
        labels = self.class_labels
        return [k for k, _ in sorted(labels.items(), key=lambda x: x[1])]

    @property
    def recurrent_sv_labels(self) -> list[str]:
        return self._load_json("recurrent_sv_labels.json")

    @property
    def recurrent_tert_mutations(self) -> list[str]:
        return self._load_json("recurrent_tert_mutations.json")

    @property
    def deepsig_features(self) -> list[str]:
        return self._load_json("deepsig_features.json")

    @property
    def mutfreq_features(self) -> list[str]:
        return self._load_json("mutfreq_features.json")

    @property
    def clinical_features(self) -> list[str]:
        return self._load_json("clinical_features.json")

    @property
    def cna_features(self) -> list[str]:
        return self._load_json("cna_features.json")

    @property
    def top_50_genes(self) -> list[str]:
        return self._load_json("top_50_genes.json")

    @property
    def recurrent_mutations(self) -> list[str]:
        return self._load_json("recurrent_mutations.json")

    @property
    def gene_level_mutations(self) -> list[str]:
        return self._load_json("gene_level_mutations.json")

    def load_useful_genes(self) -> set[str]:
        """Load the set of useful gene symbols from UsefulGenes.csv."""
        import pandas as pd
        df = pd.read_csv(self._dir / "UsefulGenes.csv")
        return set(df["USEFUL_GENES"].tolist())

    def load_tissue_specific_mutations(self) -> set[str]:
        """Load the set of tissue-specific enriched mutation IDs."""
        import pandas as pd
        df = pd.read_csv(self._dir / "tissue_specific_mutations.csv")
        return set(df["Mutation_ID"].tolist())
