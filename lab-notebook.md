# Lab Notebook: Tissue-of-Origin Inference Pipeline

## 2026-02-19: Initial pipeline implementation

**Session context:** Claude Opus 4.6, TissueClassifier project setup

### Goal
Create a standalone single-sample inference pipeline for the tissue-of-origin classifier.

### Method
- Scaffolded project with pyproject.toml, src layout, reference_data/
- Extracted reference data from ClaudeAnalysis_V2: feature manifest (4,333 features), class labels (23 classes), tissue-specific mutations, gene lists
- Implemented 7 feature extractors matching Step05-Step12 logic
- Generated SHAP background (kmeans k=50) from test set
- Created Typer CLI with predict/validate/info commands
- HTML report template with Jinja2

### Results
- 7/7 unit tests passing
- Feature compilation produces exactly 4,333 features in correct training order
- CLI info/validate commands working
- SEX encoding verified: Female=0, Male=1

### Decisions & Next Steps
- DeepSig uses skip mode by default (returns zeros); Docker/subprocess modes implemented but untested
- Need end-to-end test with real GENIE sample to verify prediction accuracy
- Need to write README and user documentation

## 2026-02-19: SHAP explainer fix and full end-to-end verification

**Session context:** Claude Opus 4.6, continuation session — fixing SHAP and verifying pipeline

### Goal
Fix SHAP explainer crash and verify the complete pipeline works end-to-end (MAF+SEG → features → prediction → SHAP → HTML report).

### Method
- Diagnosed SHAP error: `_predict_proba` used `feature_metadata.get_features()` (2,302 model-internal features) to name DataFrame columns, but SHAP passes arrays with 4,333 columns (matching training data)
- Fixed by storing `ref.training_feature_order` (4,333 names) in the explainer and using those as column names
- AutoGluon's `predict_proba` handles the extra columns gracefully (ignores features it doesn't use)
- Cleaned `__pycache__`, reinstalled package, ran full CLI test

### Results
- SHAP now working: KernelSHAP with nsamples=500 completes successfully
- Top SHAP features for synthetic sample (predicted SKIN_AND_MELANOMA at 23.5%):
  - SEX (+0.041), BRAF_mut_count (+0.037), TP53_mut_count (+0.023)
- HTML report: 275 KB (includes SHAP waterfall, probability distribution, top-3 bar chart)
- 7/7 unit tests passing
- AutoGluon model uses 2,302 of 4,333 features internally (dropped during training)

### Interpretation
The pipeline is fully functional end-to-end. SHAP feature attributions are biologically sensible (BRAF and TP53 mutations driving melanoma prediction). The 275 KB report is self-contained HTML with base64-embedded plots.

### Decisions & Next Steps
- Pipeline is feature-complete for Phases 0-6
- Still needed: Phase 7 (Docker for DeepSig), Phase 8 (real GENIE sample testing), README
- Should test with a real GENIE sample from the test set to verify feature vector matches training pipeline output
