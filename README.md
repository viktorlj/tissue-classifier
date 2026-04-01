# Tissue-of-Origin Classifier

Single-sample tissue-of-origin inference pipeline using an AutoGluon ensemble trained on AACR GENIE v18 data.

## Quick Start

```bash
# Install
uv venv && uv sync

# Basic prediction (mutations + copy number)
tissue-classifier predict --maf sample.maf --seg sample.seg --age 65 --sex Male

# With structural variants
tissue-classifier predict --maf sample.maf --seg sample.seg --sv sample.sv --age 65 --sex Male

# With all options
tissue-classifier predict \
    --maf sample.maf \
    --seg sample.seg \
    --sv sample.sv \
    --age 65 --sex Male \
    --genome hg19 \
    --output ./results \
    --sample-id PATIENT_001

# Validate input files
tissue-classifier validate --maf sample.maf --seg sample.seg

# Show model info
tissue-classifier info
```

## Prerequisites

- Python >= 3.11
- uv (for environment management)

## Input Files

### MAF (required)
Tab-delimited mutation file with columns: Hugo_Symbol, Chromosome, Start_Position, End_Position, Reference_Allele, Tumor_Seq_Allele2, Variant_Classification, Variant_Type, Tumor_Sample_Barcode.

### SEG (optional)
Tab-delimited copy number segmentation file with columns: ID, chrom, loc.start, loc.end, seg.mean.

### SV (optional)
Tab-delimited structural variant file with columns: Sample_Id, Site1_Hugo_Symbol, Site2_Hugo_Symbol.

## Model

- CatBoost + XGBoost ensemble via AutoGluon (22 tumor types, ~4,320 features)
- Rank-based CNA normalization for cross-platform compatibility
- Rare mutation grouping for robust feature representation

| Split | Balanced Accuracy | Top-3 Accuracy | @70% Confidence |
|-------|-------------------|----------------|-----------------|
| Test (MSK + UCSF-IDTV5) | 83.2% | 94.4% | 94.7% (76% coverage) |
| Holdout (UCSF-NIMV4) | 68.5% | 93.3% | 93.5% (67% coverage) |

Feature modalities: somatic mutations, structural variants, SBS-6 mutation spectrum, mutation frequency, copy number alterations, TERT promoter, clinical (age, sex).

Training pipeline: [too-panelseq](https://github.com/viktorlj/too-panelseq)

## Output

Self-contained HTML report with:
- Top-3 predictions with confidence scores
- Full probability distribution across 22 tumor types
- SHAP feature explanations (when background data available)
- Non-zero features by modality
- Input data summary
