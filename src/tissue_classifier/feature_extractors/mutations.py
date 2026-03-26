"""Mutation feature extraction (~3,688 features).

Block A: Enriched binary mutations (recurrent_mutations.json) - 2409 features
Block B: Gene-level variant types (gene_level_mutations.json) - 1229 features
Block C: Top-50 gene mutation counts (top_50_genes.json) - 50 features
"""
from __future__ import annotations

import logging

import pandas as pd

from ..config import PROTEIN_AFFECTING, VARIANT_TYPE_MAP, ReferenceData

logger = logging.getLogger(__name__)

# Include Splice_Region for mutation extraction (matches training pipeline)
# but NOT in config.PROTEIN_AFFECTING
_PROTEIN_AFFECTING_EXTENDED = PROTEIN_AFFECTING + ["Splice_Region"]


def extract_mutation_features(
    maf: pd.DataFrame,
    ref: ReferenceData,
) -> pd.Series:
    """Extract mutation features from a MAF DataFrame.

    Returns a sparse Series containing only non-zero features.
    The feature compiler will reindex to full 4333 features.

    Parameters
    ----------
    maf : pd.DataFrame
        Parsed MAF DataFrame.
    ref : ReferenceData
        Reference data instance.

    Returns
    -------
    pd.Series
        Sparse Series of non-zero mutation features.
    """
    useful_genes = ref.load_useful_genes()
    enriched_set = ref.load_tissue_specific_mutations()
    recurrent_mutation_names = set(ref.recurrent_mutations)
    gene_level_names = set(ref.gene_level_mutations)
    top_50 = ref.top_50_genes

    # Filter to useful genes and protein-affecting variants
    mask = (
        maf["Hugo_Symbol"].isin(useful_genes)
        & maf["Variant_Classification"].isin(_PROTEIN_AFFECTING_EXTENDED)
    )
    maf_filtered = maf[mask].copy()

    # Build mutation IDs
    maf_filtered["Mutation_ID"] = (
        maf_filtered["Hugo_Symbol"].astype(str) + "_"
        + maf_filtered["Chromosome"].astype(str) + "_"
        + maf_filtered["Start_Position"].astype(str) + "_"
        + maf_filtered["Reference_Allele"].astype(str) + "_"
        + maf_filtered["Tumor_Seq_Allele2"].astype(str)
    )

    features: dict[str, float] = {}

    # Block A: Enriched binary mutations
    sample_mut_ids = set(maf_filtered["Mutation_ID"].unique())
    for mut_id in sample_mut_ids:
        if mut_id in recurrent_mutation_names:
            features[mut_id] = 1.0

    # Block B: Gene-level variant type features
    for _, row in maf_filtered.iterrows():
        mut_type = VARIANT_TYPE_MAP.get(row["Variant_Classification"], "other")
        if mut_type == "other":
            continue
        fname = f"{row['Hugo_Symbol']}_{mut_type}"
        if fname in gene_level_names:
            features[fname] = 1.0

    # Block C: Top-50 gene mutation counts
    gene_counts = maf_filtered["Hugo_Symbol"].value_counts()
    for gene in top_50:
        fname = f"{gene}_mut_count"
        count = int(gene_counts.get(gene, 0))
        if count > 0:
            features[fname] = float(count)

    result = pd.Series(features, dtype=float)
    logger.info(
        "Mutation features: %d non-zero", len(result),
    )
    return result
