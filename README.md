# tissue-classifier (v2.0) — Cross-Architecture (`2× RealMLP + 2× TabM`) Tissue-of-Origin Inference & Top-3 SHAP PDF Reports

Standalone clinical and research command-line package for predicting a tumor's **tissue of origin (22 classes)** from targeted panel sequencing (`MAF` + `SEG` + optional `SV` + clinical `age`/`sex`) using the calibrated **v2.0 AACR GENIE v20 4-view cross-architecture ensemble (`2× RealMLP + 2× TabM`, 40 networks)**.

---

## Key Features
- **Calibrated v2.0 Cross-Architecture Ensemble (`models/primary_50pct_blend/`)**:
  - Evaluates all **40 frozen PyTabKit networks** (`realmlp_v1`, `realmlp_v2`, `tabm_v1`, `tabm_v2` × 5 patient-grouped folds × unweighted & class-balanced arms) in a unified pure-PyTorch MPS/CPU `DifferentiableBlend` (including exact `TorchQuantileTransform` parity for TabM), applies equal **LogitPool** across views + 50/50 **LogitPool** across arms, and calibrates via **L2-regularized per-class Vector Scaling**.
  - Applies frozen source-selected confidence policies:
    - **High Confidence (`>= 73%` Top-1)**: 95% target accuracy (`96.21%` internal test accuracy at `77.82%` coverage; `91.59%` external VICC accuracy at `62.04%` eligible coverage).
    - **Standard Confidence (`>= 46%` Top-1)**: 90% target accuracy (`90.58%` internal test accuracy at `92.64%` coverage; `81.61%` external VICC accuracy at `85.89%` coverage).
    - **Top-3 Differential (`100%` coverage)**: 95% Top-3 target accuracy (`95.96%` internal test Top-3 accuracy; `89.88%` external VICC Top-3 accuracy).
- **Scale-Invariant CNA Processing, `TERT` Promoter Features & `--allow-no-cna` Fallback**:
  - Computes 10 Mb signed-rank (`CNA_bin__*`) and scale-invariant winsorized-RMS (`CNA_zbin__*` + 24 `CNA_inv_*` summaries) profiles (`hg19`, corrected `chr2 = 243,199,373` bp) plus `TERT` promoter `5'Flank` hotspot indicators (`C228T`, `C250T`, `other`, `any`); unmeasured genomic bins are **never** zero-filled and assay-specific coverage/missingness shortcuts are excluded.
  - **Default**: Samples missing `--seg` or below the `>= 85%` genome-wide observed-bin coverage gate receive a clear `TECHNICAL_NO_CALL` diagnostic PDF report.
  - **`--allow-no-cna`**: Routes inference and SHAP through the retrained **40-network CNA-free fallback ensemble** (`models/fallback_without_cna/`, policies `>= 83%` High-Conf / `>= 68%` Standard / `>= 84%` Top-3) with a prominent clinical warning banner in `report.pdf`.
- **Coherent All-Class `PermutationSHAP` (Exact Invariant-Pruned) & Multi-Page Vector `report.pdf`**:
  - Uses `FoldCoherentMasker` with exact Dummy-Axiom invariant pruning across 32 aggregate-derived synthetic background profiles (`SYNTH_BG_00..63`, configurable via `--shap-background 16|32|64` and `--shap-permutations 4` default, or `--shap-permutations 2` for rapid ~78s MPS runs; **zero individual GENIE patient records redistributed**) to compute exact additive SHAP values across all 22 calibrated class probabilities simultaneously (`max |base + sum(shap) - P(class)| < 2e-5`).
  - Produces a 4-page vector PDF report (`<output>/<sample_id>/report.pdf`):
    - **Page 1**: Executive Clinical Summary, Policy Tier Badge, Top-3 Candidate Bar Chart, and Full 22-Class Probability Profile.
    - **Page 2**: Vector SHAP Waterfall Plot & Supporting/Counter-Evidence Table for **Rank #1** Most Likely Tumor Type.
    - **Page 3**: Vector SHAP Waterfall Plots for **Rank #2** and **Rank #3** Most Likely Tumor Types.
    - **Page 4**: Sample Molecular Profile (Detected Somatic Mutations, SVs, Chromosome CNA Profile, SBS-6 Spectrum) & 5-Fold Technical Audit.

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
