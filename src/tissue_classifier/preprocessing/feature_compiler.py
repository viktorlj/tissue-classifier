"""Merge all feature modalities and align to training feature order."""
from __future__ import annotations

import logging

import pandas as pd

from ..config import ReferenceData

logger = logging.getLogger(__name__)


def _apply_rare_mutation_grouping(
    all_features: pd.Series,
    rare_groups: dict[str, list[str]],
) -> pd.Series:
    """Group rare enriched mutations by gene (OR aggregation).

    Matches the v19 training pipeline: rare alleles (<0.5% prevalence) are
    collapsed into gene-level groups using OR logic (max across alleles).
    """
    grouped = {}
    cols_to_drop = set()

    for gene, alleles in rare_groups.items():
        present = [a for a in alleles if a in all_features.index]
        if present:
            grouped[f"{gene}_rare_enriched"] = max(all_features[a] for a in present)
            cols_to_drop.update(present)

    # Drop individual rare alleles, add grouped features
    result = all_features.drop(labels=[c for c in cols_to_drop if c in all_features.index],
                                errors="ignore")
    for name, val in grouped.items():
        result[name] = val

    return result


def compile_features(
    feature_series: dict[str, pd.Series],
    ref: ReferenceData,
) -> pd.DataFrame:
    """Compile all feature Series into a single-row DataFrame aligned to training order.

    Parameters
    ----------
    feature_series : dict[str, pd.Series]
        Dict mapping modality name to feature Series.
    ref : ReferenceData
        Reference data instance.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with features in training order.
    """
    training_order = ref.training_feature_order

    # Concatenate all feature series, dropping duplicates (keep first)
    all_features = pd.concat(feature_series.values())
    all_features = all_features[~all_features.index.duplicated(keep='first')]

    # Apply rare mutation grouping (v19: 2,398 rare alleles → 246 gene groups)
    rare_groups = ref.rare_mutation_groups
    if rare_groups:
        all_features = _apply_rare_mutation_grouping(all_features, rare_groups)

    # Reindex to training order, filling missing with 0
    aligned = all_features.reindex(training_order, fill_value=0.0)

    # Identify spectrum fraction features and Ti/Tv ratio
    spectrum_fraction_names = {f for f in training_order if 'Fraction' in f and f.startswith(('C_', 'T_'))}
    spectrum_fraction_names.add('Ti_Tv_Ratio')

    # Track if spectrum data is available
    has_spectrum = any(not pd.isna(aligned.get(f, float('nan'))) for f in spectrum_fraction_names)

    # Load imputation values from training data
    try:
        imp = ref.imputation_values
        age_median = imp.get("age_median", 62.0)
    except Exception:
        age_median = 62.0

    # Imputation rules matching v19 training pipeline
    for feat in training_order:
        if pd.isna(aligned[feat]):
            if feat in spectrum_fraction_names:
                aligned[feat] = 0.0
            elif feat == "AGE_AT_SEQ_REPORT":
                aligned[feat] = age_median
            elif feat == "SEX":
                aligned[feat] = 0.0
            else:
                aligned[feat] = 0.0

    # Add the has_spectrum_data flag (added during training preprocessing)
    if 'has_spectrum_data' in training_order:
        aligned['has_spectrum_data'] = 1.0 if has_spectrum else 0.0

    assert len(aligned) == len(training_order), (
        f"Expected {len(training_order)} features, got {len(aligned)}"
    )

    result = aligned.to_frame().T
    result.index = ["SAMPLE"]
    logger.info(
        "Compiled %d features (%d non-zero)",
        len(training_order),
        (result.iloc[0] != 0).sum(),
    )
    return result
