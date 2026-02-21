# Example Input Data

Synthetic example files demonstrating the expected input formats for
`tissue-classifier predict`. These are **not real patient data** — all
variants and copy-number segments are fabricated for illustration.

## Files

### `example_sample.maf` — Mutation Annotation Format

Tab-separated file with somatic mutations. Required columns:

| Column               | Description                          | Example           |
|----------------------|--------------------------------------|-------------------|
| Hugo_Symbol          | HGNC gene symbol                     | TP53              |
| Chromosome           | Chromosome (with or without `chr`)   | 17                |
| Start_Position       | Genomic start (1-based, hg19)        | 7577120           |
| End_Position         | Genomic end                          | 7577120           |
| Reference_Allele     | Reference base(s)                    | C                 |
| Tumor_Seq_Allele2    | Alternate base(s)                    | T                 |
| Variant_Classification | Mutation effect type               | Missense_Mutation |
| Variant_Type         | SNP, DNP, INS, DEL                   | SNP               |

Optional but recommended:

| Column                 | Description              | Example             |
|------------------------|--------------------------|---------------------|
| Tumor_Sample_Barcode   | Sample identifier        | EXAMPLE-SAMPLE-001  |
| HGVSp_Short            | Protein change (HGVS)   | p.R248W             |

**Variant_Classification values used by the classifier:**
- `Missense_Mutation`, `Nonsense_Mutation`
- `Frame_Shift_Del`, `Frame_Shift_Ins`
- `In_Frame_Del`, `In_Frame_Ins`
- `Splice_Site`, `Splice_Region`
- `Translation_Start_Site`

### `example_sample.seg` — Segmented Copy Number

Tab-separated file with copy-number segments. Required columns:

| Column     | Description                          | Example            |
|------------|--------------------------------------|--------------------|
| ID         | Sample identifier                    | EXAMPLE-SAMPLE-001 |
| chrom      | Chromosome (with or without `chr`)   | 17                 |
| loc.start  | Segment start (1-based, hg19)        | 11430              |
| loc.end    | Segment end                          | 7577000            |
| seg.mean   | Log2 copy-number ratio               | -0.3210            |

Optional:

| Column     | Description           | Example |
|------------|-----------------------|---------|
| num.mark   | Number of probes      | 2341    |

**seg.mean interpretation:**
- ~0.0: diploid (no change)
- &gt;0.3: gain
- &lt;-0.3: loss
- &gt;0.7: high-level amplification
- &lt;-1.0: deep (homozygous) deletion

## Usage

```bash
# Minimal (MAF only)
tissue-classifier predict --maf examples/example_sample.maf \
    --sample-id EXAMPLE-SAMPLE-001

# With copy-number data
tissue-classifier predict --maf examples/example_sample.maf \
    --seg examples/example_sample.seg \
    --sample-id EXAMPLE-SAMPLE-001

# Full options
tissue-classifier predict --maf examples/example_sample.maf \
    --seg examples/example_sample.seg \
    --sample-id EXAMPLE-SAMPLE-001 \
    --age 65 --sex Female \
    --genome hg19 \
    --output ./results
```

## Genome Build

Input coordinates should be **hg19** (GRCh37). If your data is hg38, use `--genome hg38` and the pipeline will liftover automatically.
