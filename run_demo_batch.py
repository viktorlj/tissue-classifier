#!/usr/bin/env python3
"""Run all bundled demo_data samples through V20 tissue-classifier and verify cached parity."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
import pandas as pd

from tissue_classifier.config import PROJECT_ROOT, PipelineConfig, ReferenceData
from tissue_classifier.pipeline import run_pipeline
from tissue_classifier.prediction.predictor import TissuePredictor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo-dir",
        type=Path,
        default=PROJECT_ROOT / "demo_data",
        help="Directory containing demo_manifest.csv and genomic files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "demo_results",
        help="Output directory for generated reports",
    )
    parser.add_argument(
        "--shap-permutations",
        type=int,
        default=4,
        help="Antithetic PermutationSHAP cycles per sample (default 4; 2 for rapid ~78s SHAP; 0 to skip)",
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Also render standalone report.html alongside report.pdf",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    manifest_path = args.demo_dir / "demo_manifest.csv"
    manifest = pd.read_csv(manifest_path)

    ref = ReferenceData()
    predictor_cache: dict[Path, TissuePredictor] = {}

    print(f"Running V20 tissue-classifier on {len(manifest)} demo samples -> {args.output_dir}")
    print("-" * 104)
    print(
        f"{'Sample ID':<30s} {'Split':<14s} {'True':<18s} {'Predicted':<18s} "
        f"{'Conf':>7s} {'MaxDiff':>10s} {'Time':>6s}"
    )
    print("-" * 104)

    max_global_diff = 0.0
    n_correct = 0

    for _, row in manifest.iterrows():
        sid = str(row["sample_id"])
        maf_path = args.demo_dir / str(row["maf_file"])
        seg_path = args.demo_dir / str(row["seg_file"])
        sv_file = str(row.get("sv_file", "") or "").strip()
        sv_path = (args.demo_dir / sv_file) if (sv_file and sv_file != "nan") else None

        age = float(row["age"]) if pd.notna(row["age"]) else None
        sex = str(row["sex"]) if pd.notna(row["sex"]) and str(row["sex"]).strip() else None
        true_cls = str(row["true_tumor_type"])

        t0 = time.perf_counter()
        cfg = PipelineConfig(
            maf_path=maf_path,
            seg_path=seg_path,
            sv_path=sv_path,
            age=age,
            sex=sex,
            genome="hg19",
            output_dir=args.output_dir,
            sample_id=sid,
            compute_shap=args.shap_permutations > 0,
            shap_permutations=args.shap_permutations,
            generate_html=args.html,
            true_tumor_type=true_cls,
        )
        out = run_pipeline(cfg, predictor_cache=predictor_cache, ref=ref)
        elapsed = time.perf_counter() - t0

        assert out.prediction is not None, f"Unexpected no-call on demo sample {sid}"
        labels = out.prediction.labels
        pred_vec = np.array([out.prediction.all_probabilities[lbl] for lbl in labels], dtype=float)
        cached_vec = np.array([float(row[f"prob_{lbl}"]) for lbl in labels], dtype=float)
        max_diff = float(np.max(np.abs(pred_vec - cached_vec)))
        max_global_diff = max(max_global_diff, max_diff)

        if out.prediction.predicted_class == true_cls:
            n_correct += 1

        assert max_diff < 1e-5, f"Parity exceeded 1e-5 for {sid}: {max_diff:.3e}"
        assert out.report_pdf_path.exists() and out.report_pdf_path.stat().st_size > 5_000

        print(
            f"{sid:<30s} {str(row['split']):<14s} {true_cls:<18s} "
            f"{out.prediction.predicted_class:<18s} {out.prediction.confidence:>6.1%} "
            f"{max_diff:>10.2e} {elapsed:>5.1f}s"
        )

    print("-" * 104)
    print(
        f"Demo batch complete: {n_correct}/{len(manifest)} Top-1 matches | "
        f"max 22-class parity error = {max_global_diff:.3e} (< 1e-5)"
    )


if __name__ == "__main__":
    main()
