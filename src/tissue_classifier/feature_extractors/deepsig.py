"""DeepSig mutational signature feature extraction (13 features)."""
from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd

from ..config import PACKAGE_DIR, PROJECT_ROOT, ReferenceData

logger = logging.getLogger(__name__)

R_SCRIPTS_DIR = PROJECT_ROOT / "r_scripts"

# Required MAF columns for maf2cat3.R
MAF_COLUMNS = [
    "Hugo_Symbol", "Chromosome", "Start_Position", "End_Position",
    "Reference_Allele", "Tumor_Seq_Allele2", "Variant_Classification",
    "Variant_Type", "Tumor_Sample_Barcode",
]


def _nan_series(feature_names: list[str]) -> pd.Series:
    """Return a NaN Series with the given feature names."""
    return pd.Series(float("nan"), index=feature_names, dtype=float)


def _write_maf_for_deepsig(maf: pd.DataFrame, path: Path) -> None:
    """Write a minimal MAF file with the 9 required columns."""
    cols = [c for c in MAF_COLUMNS if c in maf.columns]
    maf[cols].to_csv(path, sep="\t", index=False)


def _parse_exposure_file(exposure_path: Path, feature_names: list[str]) -> pd.Series:
    """Parse the space-delimited exposure file from DeepSig.

    R's write.table produces: quoted column names, row names as first field.
    Header has N cols but data rows have N+1 fields (rowname prepended).
    pandas auto-uses the extra field as index. Column names may be quoted.
    """
    df = pd.read_csv(exposure_path, sep=r"\s+", engine="python")
    # Strip quotes from column names
    df.columns = [c.strip('"') for c in df.columns]
    # Drop sid and M columns
    sig_cols = [c for c in df.columns if c not in ("sid", "M")]
    if len(df) == 0:
        return _nan_series(feature_names)
    row = df[sig_cols].iloc[0]
    # Reindex to match expected feature order
    return row.reindex(feature_names).astype(float)


def _run_subprocess(
    maf: pd.DataFrame,
    feature_names: list[str],
    ref_sig_path: Optional[Path] = None,
) -> pd.Series:
    """Run DeepSig via local Rscript subprocess."""
    rscript = shutil.which("Rscript")
    if rscript is None:
        logger.warning("DeepSig subprocess: Rscript not found on PATH")
        return _nan_series(feature_names)

    maf2cat3_script = R_SCRIPTS_DIR / "maf2cat3.R"
    deepsig_script = R_SCRIPTS_DIR / "deepsig.R"

    if not maf2cat3_script.exists() or not deepsig_script.exists():
        logger.warning("DeepSig subprocess: R scripts not found in %s", R_SCRIPTS_DIR)
        return _nan_series(feature_names)

    with tempfile.TemporaryDirectory(prefix="deepsig_") as tmpdir:
        tmp = Path(tmpdir)
        maf_path = tmp / "input.maf"
        catalog_path = tmp / "catalog.txt"
        output_dir = tmp / "output"
        output_dir.mkdir()

        _write_maf_for_deepsig(maf, maf_path)

        # Step 1: maf2cat3
        try:
            subprocess.run(
                [rscript, str(maf2cat3_script), str(maf_path), str(catalog_path)],
                check=True, capture_output=True, text=True, timeout=300,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.warning("DeepSig maf2cat3 failed: %s", e)
            return _nan_series(feature_names)

        if not catalog_path.exists():
            logger.warning("DeepSig: maf2cat3 produced no output")
            return _nan_series(feature_names)

        # Step 2: deepsig
        try:
            subprocess.run(
                [rscript, str(deepsig_script),
                 "-i", str(catalog_path),
                 "-o", str(output_dir),
                 "-c", "pancancer", "-q"],
                check=True, capture_output=True, text=True, timeout=300,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.warning("DeepSig signature extraction failed: %s", e)
            return _nan_series(feature_names)

        exposure_path = output_dir / "exposure"
        if not exposure_path.exists():
            logger.warning("DeepSig: no exposure file produced")
            return _nan_series(feature_names)

        return _parse_exposure_file(exposure_path, feature_names)


def _run_docker(
    maf: pd.DataFrame,
    feature_names: list[str],
    docker_image: str = "tissue-classifier-deepsig:latest",
) -> pd.Series:
    """Run DeepSig via Docker container."""
    docker = shutil.which("docker")
    if docker is None:
        logger.warning("DeepSig docker: docker not found on PATH")
        return _nan_series(feature_names)

    with tempfile.TemporaryDirectory(prefix="deepsig_") as tmpdir:
        tmp = Path(tmpdir)
        maf_path = tmp / "input.maf"
        catalog_path = tmp / "catalog.txt"
        output_dir = tmp / "output"
        output_dir.mkdir()

        _write_maf_for_deepsig(maf, maf_path)

        docker_base = [
            docker, "run", "--rm",
            "-v", f"{tmp}:/data",
            docker_image,
        ]

        # Step 1: maf2cat3
        try:
            subprocess.run(
                docker_base + [
                    "Rscript", "/opt/deepsig/maf2cat3.R",
                    "/data/input.maf", "/data/catalog.txt",
                ],
                check=True, capture_output=True, text=True, timeout=600,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.warning("DeepSig docker maf2cat3 failed: %s", e)
            return _nan_series(feature_names)

        if not catalog_path.exists():
            logger.warning("DeepSig docker: maf2cat3 produced no output")
            return _nan_series(feature_names)

        # Step 2: deepsig
        try:
            subprocess.run(
                docker_base + [
                    "Rscript", "/opt/deepsig/deepsig.R",
                    "-i", "/data/catalog.txt",
                    "-o", "/data/output",
                    "-c", "pancancer", "-q",
                ],
                check=True, capture_output=True, text=True, timeout=600,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.warning("DeepSig docker signature extraction failed: %s", e)
            return _nan_series(feature_names)

        exposure_path = output_dir / "exposure"
        if not exposure_path.exists():
            logger.warning("DeepSig docker: no exposure file produced")
            return _nan_series(feature_names)

        return _parse_exposure_file(exposure_path, feature_names)


def extract_deepsig_features(
    maf: pd.DataFrame,
    ref: ReferenceData,
    mode: str = "skip",
    mutation_count: Optional[int] = None,
    docker_image: str = "tissue-classifier-deepsig:latest",
) -> pd.Series:
    """Extract DeepSig mutational signature features.

    Parameters
    ----------
    maf : pd.DataFrame
        Parsed MAF DataFrame.
    ref : ReferenceData
        Reference data instance.
    mode : str
        One of 'docker', 'subprocess', 'skip'. Default 'skip'.
    mutation_count : Optional[int]
        Override mutation count for minimum threshold check.
    docker_image : str
        Docker image name for docker mode.

    Returns
    -------
    pd.Series
        Series of 13 DeepSig features. Returns NaN values for skip/fallback
        modes (will be imputed to -1 by the feature compiler).
    """
    feature_names = ref.deepsig_features

    if mode == "skip":
        logger.info("DeepSig: skipped, returning NaN for %d features", len(feature_names))
        return _nan_series(feature_names)

    n_muts = mutation_count if mutation_count is not None else len(maf)
    if n_muts < 4:
        logger.warning("DeepSig: only %d mutations (need >=4), returning NaN", n_muts)
        return _nan_series(feature_names)

    if mode == "subprocess":
        return _run_subprocess(maf, feature_names)
    elif mode == "docker":
        return _run_docker(maf, feature_names, docker_image=docker_image)

    logger.warning("DeepSig: unknown mode '%s', returning NaN", mode)
    return _nan_series(feature_names)
