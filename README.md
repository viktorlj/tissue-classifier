<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo/topo_logo_dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/logo/topo_logo.svg">
    <img alt="TOPO — Tissue-of-Origin for Pathology & Oncology" src="assets/logo/topo_logo.svg" width="520">
  </picture>
</p>

# TOPO (v2.0) — Tissue-of-Origin for Pathology & Oncology

Standalone clinical and research command-line package (`tissue-classifier`) for predicting a tumor's **tissue of origin (22 classes)** from targeted panel sequencing (`MAF` + `SEG` + optional `SV`, `age`, `sex`) using **TOPO**, a calibrated AACR GENIE v20 4-view cross-architecture ensemble (`2× RealMLP + 2× TabM`, 40 networks).

---

## Key Features
- **Calibrated Cross-Architecture Ensemble**: 40 frozen PyTabKit networks (`2× RealMLP + 2× TabM` × 5 folds × unweighted & balanced arms) evaluated in pure PyTorch (MPS/CPU) with LogitPool aggregation and per-class vector scaling.
- **Confidence-Tiered Clinical Policies**: Frozen thresholds for **High Confidence** (`>= 73%` Top-1: `96.2%` internal / `91.6%` external VICC accuracy), **Standard** (`>= 46%` Top-1: `90.6%` / `81.6%`), and **Top-3 Differential** (`96.0%` / `89.9%` at `100%` coverage).
- **Scale-Invariant CNA & Fallback Mode**: 10 Mb signed-rank and winsorized-RMS CNA profiles (`hg19`, with `--genome hg38` liftover) plus `TERT` promoter hotspots (`C228T`/`C250T`). Samples missing `--seg` or `< 85%` observed-bin coverage emit `TECHNICAL_NO_CALL` by default, or route to a 40-network CNA-free fallback ensemble via `--allow-no-cna`.
- **Exact All-Class SHAP & 4-Page Vector PDF Reports**: Coherent `PermutationSHAP` over synthetic aggregate-derived backgrounds (zero individual GENIE records redistributed) generating `report.pdf` (executive summary, Rank #1–#3 SHAP waterfalls, molecular profile, and 5-fold audit) plus optional `--html`.

---

## Installation

```bash
uv sync --python 3.13
```

### Download Pretrained Model Weights (`v2.0.0`)

Because the 80 frozen ensemble checkpoints (`~2.1 GB` uncompressed) exceed standard Git limits, they are distributed via the [`v2.0.0` GitHub Release](https://github.com/viktorlj/tissue-classifier/releases/tag/v2.0.0):

```bash
# Download and extract both primary (with CNA) and fallback (CNA-free) model bundles
gh release download v2.0.0 --repo viktorlj/tissue-classifier \
    -p "primary_50pct_blend.tar.gz" \
    -p "fallback_without_cna.tar.gz" \
    -p "SHA256SUMS.txt"

shasum -a 256 -c SHA256SUMS.txt
tar -xzf primary_50pct_blend.tar.gz
tar -xzf fallback_without_cna.tar.gz
```

*(Alternatively, download the `.tar.gz` archives directly from the Releases page and extract them in the repository root so `models/primary_50pct_blend/` and `models/fallback_without_cna/` exist.)*

---

## CLI Usage

```bash
# 1. Inspect bundled models, thresholds, and reference data
.venv/bin/tissue-classifier info

# 2. Validate input genomic files
.venv/bin/tissue-classifier validate \
    --maf examples/example_sample.maf \
    --seg examples/example_sample.seg \
    --sv examples/example_sample.sv

# 3. Run full prediction + Top-3 SHAP waterfall multi-page PDF report
.venv/bin/tissue-classifier predict \
    --maf examples/example_sample.maf \
    --seg examples/example_sample.seg \
    --sv examples/example_sample.sv \
    --age 61 --sex Female \
    --sample-id CASE_001 \
    --output ./results

# 4. Run also generating optional HTML report (--html) or from hg38 (--genome hg38)
.venv/bin/tissue-classifier predict \
    --maf sample_hg38.maf \
    --seg sample_hg38.seg \
    --genome hg38 \
    --html \
    --sample-id CASE_HG38 \
    --output ./results

# 5. CNA-free fallback mode when copy-number segmentation is unavailable
.venv/bin/tissue-classifier predict \
    --maf examples/example_sample.maf \
    --allow-no-cna \
    --sample-id CASE_NOCNA \
    --output ./results
```

---

## Testing & Demo Batch Verification

```bash
# Run unit, numerical cache-parity (< 1e-5), no-call, fallback, and SHAP additivity tests:
.venv/bin/pytest tests/ -v

# Generate full multi-page PDF reports on the 6 bundled synthetic demo cases (zero sample-level GENIE data):
.venv/bin/python run_demo_batch.py --shap-permutations 2 --html
```
