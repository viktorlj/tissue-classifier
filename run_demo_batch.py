#!/usr/bin/env python3
"""Run the 10 demo samples through the tissue-classifier pipeline and compare with old predictions."""

import sys
import time
from pathlib import Path

# Ensure we import from the local source
sys.path.insert(0, str(Path("src").resolve()))

from tissue_classifier.config import PipelineConfig, ReferenceData
from tissue_classifier.pipeline import run_pipeline

DEMO_DIR = Path("demo_data")
OUTPUT_DIR = Path("results_v19")

# Ground truth from GENIE clinical data
GROUND_TRUTH = {
    "GENIE-MSK-P-0016546-T01-IM6": {"cancer": "BLADDER", "oncotree": "BLCA", "age": 72, "sex": "Male"},
    "GENIE-MSK-P-0031532-T01-IM6": {"cancer": "BREAST", "oncotree": "BRCA", "age": 65, "sex": "Female"},
    "GENIE-MSK-P-0042600-T01-IM6": {"cancer": "CNS", "oncotree": "GBM", "age": 73, "sex": "Female"},
    "GENIE-MSK-P-0071028-T01-IM7": {"cancer": "COLORECTAL", "oncotree": "READ", "age": 27, "sex": "Male"},
    "GENIE-MSK-P-0079362-T01-IM7": {"cancer": "GYN", "oncotree": "UDDC", "age": 54, "sex": "Female"},
    "GENIE-MSK-P-0090767-T01-IM7": {"cancer": "ESOPHAGOGASTRIC", "oncotree": "STAD", "age": 82, "sex": "Male"},
    "GENIE-UCSF-15252-7549T": {"cancer": "HEAD_NECK", "oncotree": "LXSC", "age": 71, "sex": "Male"},
    "GENIE-UCSF-15399-6820T": {"cancer": "GERM_CELL", "oncotree": "GMN", "age": 38, "sex": "Male"},
    "GENIE-UCSF-16316-7333T": {"cancer": "GIST", "oncotree": "GIST", "age": 66, "sex": "Male"},
    "GENIE-UCSF-782801-63491T": {"cancer": "HEPATO", "oncotree": "HCC", "age": 74, "sex": "Male"},
}

# Old v2 model predictions for comparison
OLD_PREDICTIONS = {
    "GENIE-MSK-P-0016546-T01-IM6": "BLADDER",
    "GENIE-MSK-P-0031532-T01-IM6": "ESOPHAGOGASTRIC",
    "GENIE-MSK-P-0042600-T01-IM6": "CNS",
    "GENIE-MSK-P-0071028-T01-IM7": "COLORECTAL",
    "GENIE-MSK-P-0079362-T01-IM7": "GYN",
    "GENIE-MSK-P-0090767-T01-IM7": "ESOPHAGOGASTRIC",
    "GENIE-UCSF-15252-7549T": "LUNG",
    "GENIE-UCSF-15399-6820T": "SOFT_TISSUE",
    "GENIE-UCSF-16316-7333T": "GIST",
    "GENIE-UCSF-782801-63491T": "HEPATO",
}

import logging
logging.basicConfig(level=logging.WARNING)

results = {}
for sid, gt in GROUND_TRUTH.items():
    maf_path = DEMO_DIR / f"{sid}.maf"
    seg_path = DEMO_DIR / f"{sid}.seg"

    if not maf_path.exists():
        print(f"  SKIP {sid}: no MAF file")
        continue

    t0 = time.time()
    cfg = PipelineConfig(
        maf_path=maf_path,
        seg_path=seg_path if seg_path.exists() else None,
        age=gt["age"],
        sex=gt["sex"],
        genome="hg19",
        output_dir=OUTPUT_DIR,
        sample_id=sid,
        shap_nsamples=0,  # Skip SHAP for speed
    )

    try:
        result = run_pipeline(cfg)
        pred = result["prediction"]
        elapsed = time.time() - t0
        results[sid] = {
            "v19_pred": pred.predicted_class,
            "v19_conf": pred.confidence,
            "v19_top3": pred.top3,
            "true": gt["cancer"],
            "old_pred": OLD_PREDICTIONS.get(sid, "?"),
            "elapsed": elapsed,
        }
        correct = "OK" if pred.predicted_class == gt["cancer"] else "MISS"
        old_match = "=" if pred.predicted_class == OLD_PREDICTIONS.get(sid) else "DIFF"
        print(f"  {sid}: {pred.predicted_class} ({pred.confidence:.1%}) [{correct}] "
              f"[old={OLD_PREDICTIONS.get(sid, '?')} {old_match}] ({elapsed:.1f}s)")
    except Exception as e:
        print(f"  {sid}: ERROR — {e}")
        import traceback
        traceback.print_exc()
        results[sid] = {"error": str(e)}

# Summary
print("\n" + "=" * 85)
print(f"{'Sample':<35s} {'True':>15s} {'V19 Pred':>15s} {'V2 Pred':>15s} {'Match':>5s}")
print("-" * 85)

v19_correct = 0
v2_correct = 0
same_pred = 0

for sid, r in results.items():
    if "error" in r:
        print(f"  {sid:<35s} ERROR")
        continue
    true = r["true"]
    v19 = r["v19_pred"]
    v2 = r["old_pred"]
    v19_ok = "OK" if v19 == true else ""
    v2_ok = "OK" if v2 == true else ""
    same = "=" if v19 == v2 else "diff"

    if v19 == true: v19_correct += 1
    if v2 == true: v2_correct += 1
    if v19 == v2: same_pred += 1

    print(f"  {sid:<35s} {true:>15s} {v19:>15s} {v2:>15s} {same:>5s}")

n = len([r for r in results.values() if "error" not in r])
print("-" * 85)
print(f"  V19 accuracy: {v19_correct}/{n} ({v19_correct/n:.0%})")
print(f"  V2 accuracy:  {v2_correct}/{n} ({v2_correct/n:.0%})")
print(f"  Same prediction: {same_pred}/{n} ({same_pred/n:.0%})")
