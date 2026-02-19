# Project: Tissue-of-Origin Single-Sample Inference Pipeline

## Objective
Standalone inference pipeline that takes a patient's MAF/SEG files, extracts 4,333 features, runs the AutoGluon WeightedEnsemble_L3 classifier, and generates an HTML report with predictions and SHAP explanations.

## Current Status
Full pipeline working end-to-end including SHAP explanations. CLI predict command generates HTML report with predictions, confidence, and KernelSHAP feature importance. 7/7 unit tests passing. AutoGluon uses 2,302 of 4,333 features internally.
**Last updated:** 2026-02-19

## Active Tasks
- [x] Phase 0: Project scaffold + reference data extraction
- [x] Phase 1: Input parsing (MAF, SEG) + hg38->hg19 liftover
- [x] Phase 2: Feature extractors (7 modules: mutations, SV, DeepSig, mutfreq, clinical, TERT, CNA)
- [x] Phase 3: Feature compiler (merge + align to training order)
- [x] Phase 4: Prediction (AutoGluon) + KernelSHAP explainer
- [x] Phase 5: HTML report generation (self-contained, base64 plots)
- [x] Phase 6: CLI (Typer: predict, validate, info commands)
- [ ] Phase 7: Docker for DeepSig
- [ ] Phase 8: End-to-end testing with real GENIE samples
- [ ] README documentation

## Key Findings
- 4,333 features: 3,688 mutations + 186 SV + 13 DeepSig + 10 MutFreq + 2 Clinical + 18 TERT + 416 CNA
- SEX encoding: Female=0, Male=1 (from Step02)
- DeepSig: skip mode (fallback to zeros) when R/Docker not available
- AutoGluon internally uses 2,302 features (drops some during training); pipeline passes all 4,333
- SHAP background shape must be (50, 4333) to match training feature order, not model's internal 2,302
- SHAP top features on synthetic data: SEX, BRAF_mut_count, TP53_mut_count — biologically sensible

## Data Inventory
| Dataset | Source | Location | Notes |
|---------|--------|----------|-------|
| Feature manifest | Step13 | reference_data/feature_manifest.csv | 4,333 features |
| Class labels | Step14 | reference_data/class_labels.json | 23 classes |
| SHAP background | Generated | reference_data/shap_background.pkl | kmeans k=50 |
| AutoGluon model | Symlink | models/AutoGluon/ -> ClaudeAnalysis_V2 | WeightedEnsemble_L3 |
