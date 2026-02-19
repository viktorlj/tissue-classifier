# Tissue-of-Origin Classifier

Single-sample tissue-of-origin inference pipeline using an AutoGluon ensemble trained on AACR GENIE v18 data.

## Quick Start

```bash
# Install
uv venv && uv sync

# Basic prediction
tissue-classifier predict --maf sample.maf --seg sample.seg --age 65 --sex Male

# With all options
tissue-classifier predict \
    --maf sample.maf \
    --seg sample.seg \
    --age 65 --sex Male \
    --genome hg19 \
    --output ./results \
    --sample-id PATIENT_001

# Validate input files
tissue-classifier validate --maf sample.maf --seg sample.seg

# Show model info
tissue-classifier info
```

## Input Files

### MAF (required)
Tab-delimited mutation file with columns: Hugo_Symbol, Chromosome, Start_Position, End_Position, Reference_Allele, Tumor_Seq_Allele2, Variant_Classification, Variant_Type, Tumor_Sample_Barcode.

### SEG (optional)
Tab-delimited copy number segmentation file with columns: ID, chrom, loc.start, loc.end, seg.mean.

### SV (optional)
Tab-delimited structural variant file with columns: Sample_Id, Site1_Hugo_Symbol, Site2_Hugo_Symbol.

## Model

- AutoGluon WeightedEnsemble_L3 (23 tumor types, 4,333 features)
- Training performance: 81.1% balanced accuracy, 95.5% top-3 accuracy
- 7 feature modalities: mutations, structural variants, mutational signatures, mutation frequency, clinical, TERT promoter, copy number alterations

## Output

Self-contained HTML report with:
- Top-3 predictions with confidence scores
- Full probability distribution across 23 tumor types
- SHAP feature explanations (when background data available)
- Non-zero features by modality
- Input data summary
