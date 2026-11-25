"""5-fold V20 feature compiler, missing-feature guard, and fallback_without_cna projection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd

from ..config import ModelSpec, N_FOLDS, PipelineConfig, ReferenceData
from .cna import CnaExtractionResult, extract_cna_features
from .modalities import (
    extract_clinical_features,
    extract_fold_mutation_features,
    extract_fold_sv_features,
    extract_spectrum_features,
    extract_tert_promoter_features,
)
from .parsers import (
    MafParseResult,
    SegParseResult,
    SvParseResult,
    parse_and_filter_maf,
    parse_and_filter_sv,
    parse_and_validate_seg,
)


@dataclass
class FoldCompatibilityRecord:
    fold: int
    compatible_primary: bool
    compatible_active: bool
    n_compiled_features: int
    unsupported_missing_features: list[str] = field(default_factory=list)


@dataclass
class CompilationResult:
    status: Literal["CALLABLE", "TECHNICAL_NO_CALL"]
    active_model_mode: Literal["primary_50pct_blend", "fallback_without_cna", "none"]
    active_model_spec: Optional[ModelSpec]
    no_call_reason: Optional[str]
    diagnostic_message: Optional[str]
    fallback_warning: Optional[str]
    maf_result: MafParseResult
    sv_result: SvParseResult
    seg_result: SegParseResult
    cna_result: Optional[CnaExtractionResult]
    spectrum_df: pd.DataFrame
    clinical_df: pd.DataFrame
    fold_features: list[pd.DataFrame]
    packed_vector: Optional[np.ndarray]
    fold_audit: list[FoldCompatibilityRecord]
    warnings: list[str] = field(default_factory=list)


def _expand_raw(raw: pd.DataFrame, state: dict[str, Any]) -> pd.DataFrame:
    if list(raw.columns) != state["raw_columns"]:
        raise ValueError("Raw feature schema/order drift against frozen preprocessing_state.json")
    result = raw.copy()
    emit_clin_missing = state.get("version") == "realmlp_fold_v1"
    for col, value in state["clinical_imputation"].items():
        if emit_clin_missing:
            result[col + "_missing"] = result[col].isna().astype("float32")
        result[col] = result[col].fillna(value)
    fractions = state["spectrum_fraction_columns"]
    result["has_spectrum_data"] = raw[fractions].notna().any(axis=1).astype("float32")
    result[fractions] = result[fractions].fillna(0.0)
    return result


def _transform_single_fold(
    raw: pd.DataFrame,
    state: dict[str, Any],
    drop_cna_from_guard: bool = False,
    projected_columns: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Apply frozen fold preprocessing state once and evaluate genome-wide CNA coverage & missingness."""
    if np.isinf(raw.to_numpy(dtype=np.float64)).any():
        raise ValueError("Infinite inference feature encountered")
    expanded = _expand_raw(raw, state)

    guarded = [
        c
        for c in state["missing_guard_columns"]
        if c not in state["fitting_missing_supported"]
        and (not drop_cna_from_guard or not c.startswith("CNA_"))
    ]
    if guarded:
        unexpected_mask = raw[guarded].isna().iloc[0]
        missing_cols = [col for col, is_na in unexpected_mask.items() if bool(is_na)]
    else:
        missing_cols = []

    # v2 genome-wide CNA coverage gate (>= min_cna_observed_bin_fraction measured 10 Mb bins)
    cna_bins = state.get("cna_bin_columns", [])
    if not drop_cna_from_guard and cna_bins:
        min_cov = float(state.get("min_cna_observed_bin_fraction", 0.85))
        obs_frac = float(raw[cna_bins].notna().mean(axis=1).iloc[0])
        if obs_frac < min_cov:
            missing_cols.append(f"INSUFFICIENT_CNA_COVERAGE({obs_frac:.3f}<{min_cov:.2f})")

    values = expanded[state["numeric_feature_order"]].copy()
    for col in state["missing_indicators"]:
        values[col + "__missing"] = values[col].isna().astype("float32")
    values = (
        values.fillna(state["numeric_imputation"])
        .reindex(columns=state["feature_order"])
        .astype("float32")
    )
    if projected_columns is not None:
        values = values.reindex(columns=projected_columns).astype("float32")
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("Nonfinite compiled features after fold transformation")
    return values, missing_cols


def _placeholder_cna_frame(sample_id: str, prep_state_0: dict[str, Any]) -> pd.DataFrame:
    cna_cols = [c for c in prep_state_0["raw_columns"] if c.startswith("CNA_")]
    return pd.DataFrame(
        np.nan,
        index=pd.Index([sample_id], name="SAMPLE_ID"),
        columns=cna_cols,
        dtype="float32",
    )


def compile_sample_features(
    config: PipelineConfig,
    ref: ReferenceData,
    requested_spec: ModelSpec,
    fallback_spec: Optional[ModelSpec] = None,
) -> CompilationResult:
    """Parse inputs, enforce V20 eligibility & CNA policies, and compile 5-fold inputs."""
    sid = config.sample_id
    all_warnings: list[str] = []

    maf_res = parse_and_filter_maf(
        maf_path=config.maf_path,
        sample_id=sid,
        useful_genes=ref.useful_genes,
        bed_lookup=ref.bed_lookup,
        genome=config.genome,
    )
    all_warnings.extend(maf_res.warnings)

    sv_res = parse_and_filter_sv(
        sv_path=config.sv_path,
        sample_id=sid,
        useful_genes=ref.useful_genes,
    )
    all_warnings.extend(sv_res.warnings)

    seg_res = parse_and_validate_seg(
        seg_path=config.seg_path,
        sample_id=sid,
        chrom_sizes=ref.chrom_sizes,
        genome=config.genome,
    )
    all_warnings.extend(seg_res.warnings)

    spectrum_df = extract_spectrum_features(maf_res.usable_maf, sid)
    clinical_df = extract_clinical_features(sid, config.age, config.sex)

    cna_res: Optional[CnaExtractionResult] = None
    cna_error: Optional[str] = seg_res.error_reason
    if seg_res.valid:
        try:
            cna_res = extract_cna_features(seg_res.segments, sid, ref.chrom_sizes)
        except Exception as exc:
            cna_res = None
            cna_error = str(exc)

    # Gate 1: Molecular signal eligibility (must have >=1 usable mutation or SV in 274-gene domain)
    if len(maf_res.usable_maf) == 0 and len(sv_res.usable_sv) == 0:
        msg = (
            f"Zero usable somatic mutations (from {maf_res.raw_count} raw MAF rows) and "
            f"zero usable structural variants (from {sv_res.raw_count} raw SV rows) "
            "intersected the shared 274-gene target domain."
        )
        return CompilationResult(
            status="TECHNICAL_NO_CALL",
            active_model_mode="none",
            active_model_spec=None,
            no_call_reason="NO_USABLE_MUTATION_OR_SV",
            diagnostic_message=msg,
            fallback_warning=None,
            maf_result=maf_res,
            sv_result=sv_res,
            seg_result=seg_res,
            cna_result=cna_res,
            spectrum_df=spectrum_df,
            clinical_df=clinical_df,
            fold_features=[],
            packed_vector=None,
            fold_audit=[],
            warnings=all_warnings,
        )

    prep_0 = ref.fold_preprocessing_state(0)
    cna_df = cna_res.features if cna_res is not None else _placeholder_cna_frame(sid, prep_0)
    include_tert = "TERT_promoter_any" in prep_0["raw_columns"]
    tert_df = extract_tert_promoter_features(maf_res.tert_promoter_maf, sid) if include_tert else None

    # Build per-fold raw DataFrames
    raw_by_fold: list[pd.DataFrame] = []
    for fold in range(N_FOLDS):
        ev_state = ref.fold_event_state(fold)
        mut_df = extract_fold_mutation_features(maf_res.usable_maf, sid, ev_state["mutation"])
        sv_df = extract_fold_sv_features(sv_res.usable_sv, sid, ev_state["sv"])
        parts = [mut_df, tert_df, sv_df, spectrum_df, cna_df, clinical_df] if include_tert else [mut_df, sv_df, spectrum_df, cna_df, clinical_df]
        raw = pd.concat(parts, axis=1)
        raw_by_fold.append(raw)

    # Case A: User explicitly requested fallback_without_cna model_dir
    if not requested_spec.requires_cna:
        fold_dfs: list[pd.DataFrame] = []
        audit_rows: list[FoldCompatibilityRecord] = []
        for fold in range(N_FOLDS):
            pstate = ref.fold_preprocessing_state(fold)
            _, prim_missing = _transform_single_fold(raw_by_fold[fold], pstate, drop_cna_from_guard=False)
            vals, nocna_missing = _transform_single_fold(
                raw_by_fold[fold],
                pstate,
                drop_cna_from_guard=True,
                projected_columns=requested_spec.fold_feature_orders[fold],
            )
            fold_dfs.append(vals)
            audit_rows.append(
                FoldCompatibilityRecord(
                    fold=fold,
                    compatible_primary=(cna_res is not None and len(prim_missing) == 0),
                    compatible_active=(len(nocna_missing) == 0),
                    n_compiled_features=vals.shape[1],
                    unsupported_missing_features=nocna_missing,
                )
            )
        if any(not r.compatible_active for r in audit_rows):
            return CompilationResult(
                status="TECHNICAL_NO_CALL",
                active_model_mode="none",
                active_model_spec=None,
                no_call_reason="UNSUPPORTED_MISSING_FEATURE",
                diagnostic_message="Unsupported missing non-CNA feature across saved folds.",
                fallback_warning=None,
                maf_result=maf_res,
                sv_result=sv_res,
                seg_result=seg_res,
                cna_result=cna_res,
                spectrum_df=spectrum_df,
                clinical_df=clinical_df,
                fold_features=fold_dfs,
                packed_vector=None,
                fold_audit=audit_rows,
                warnings=all_warnings,
            )
        packed = np.concatenate([df.to_numpy(dtype=np.float32)[0] for df in fold_dfs], axis=0)
        fb_warn = (
            "CNA-free fallback ensemble active (expected accuracy: 71.3% internal test / 63.0% external VICC, "
            "compared with 86.5% primary ensemble with CNA)."
        )
        return CompilationResult(
            status="CALLABLE",
            active_model_mode="fallback_without_cna",
            active_model_spec=requested_spec,
            no_call_reason=None,
            diagnostic_message=None,
            fallback_warning=fb_warn,
            maf_result=maf_res,
            sv_result=sv_res,
            seg_result=seg_res,
            cna_result=cna_res,
            spectrum_df=spectrum_df,
            clinical_df=clinical_df,
            fold_features=fold_dfs,
            packed_vector=packed,
            fold_audit=audit_rows,
            warnings=all_warnings,
        )

    # Case B: Primary model requested (requires valid & 5-fold compatible CNA unless --allow-no-cna)
    primary_fold_dfs: list[pd.DataFrame] = []
    primary_audit: list[FoldCompatibilityRecord] = []
    primary_all_ok = cna_res is not None

    for fold in range(N_FOLDS):
        pstate = ref.fold_preprocessing_state(fold)
        vals, prim_missing = _transform_single_fold(
            raw_by_fold[fold],
            pstate,
            drop_cna_from_guard=False,
            projected_columns=requested_spec.fold_feature_orders[fold],
        )
        if cna_res is None and not prim_missing:
            prim_missing = ["CNA_SEG_MISSING_OR_INVALID"]
        fold_ok = (cna_res is not None) and (len(prim_missing) == 0)
        if not fold_ok:
            primary_all_ok = False
        primary_fold_dfs.append(vals)
        primary_audit.append(
            FoldCompatibilityRecord(
                fold=fold,
                compatible_primary=fold_ok,
                compatible_active=fold_ok,
                n_compiled_features=vals.shape[1],
                unsupported_missing_features=prim_missing,
            )
        )

    if primary_all_ok:
        packed = np.concatenate([df.to_numpy(dtype=np.float32)[0] for df in primary_fold_dfs], axis=0)
        return CompilationResult(
            status="CALLABLE",
            active_model_mode="primary_50pct_blend",
            active_model_spec=requested_spec,
            no_call_reason=None,
            diagnostic_message=None,
            fallback_warning=None,
            maf_result=maf_res,
            sv_result=sv_res,
            seg_result=seg_res,
            cna_result=cna_res,
            spectrum_df=spectrum_df,
            clinical_df=clinical_df,
            fold_features=primary_fold_dfs,
            packed_vector=packed,
            fold_audit=primary_audit,
            warnings=all_warnings,
        )

    # Primary CNA failed (missing, invalid, or unsupported unmeasured 10 Mb CNA bins)
    if not config.allow_no_cna:
        if cna_res is None:
            detail = f"CNA segmentation unavailable or invalid ({cna_error})."
        else:
            missing_union = sorted(
                {f for rec in primary_audit for f in rec.unsupported_missing_features}
            )
            sample_preview = ", ".join(missing_union[:5])
            if len(missing_union) > 5:
                sample_preview += f", ... (+{len(missing_union) - 5} more)"
            detail = (
                f"Provided CNA profile leaves {len(missing_union)} required genomic feature(s) "
                f"unmeasured without training missingness support ({sample_preview})."
            )
        detail += (
            " Primary V20 model never imputes placeholder zeros for unmeasured CNA intervals. "
            "Pass --allow-no-cna to run the dedicated retrained fallback_without_cna ensemble."
        )
        return CompilationResult(
            status="TECHNICAL_NO_CALL",
            active_model_mode="none",
            active_model_spec=None,
            no_call_reason="MISSING_OR_INCOMPATIBLE_CNA",
            diagnostic_message=detail,
            fallback_warning=None,
            maf_result=maf_res,
            sv_result=sv_res,
            seg_result=seg_res,
            cna_result=cna_res,
            spectrum_df=spectrum_df,
            clinical_df=clinical_df,
            fold_features=[],
            packed_vector=None,
            fold_audit=primary_audit,
            warnings=all_warnings,
        )

    # --allow-no-cna is True -> project to fallback_without_cna!
    if fallback_spec is None:
        fallback_spec = ModelSpec.from_dir(config.fallback_model_dir)

    fb_fold_dfs: list[pd.DataFrame] = []
    fb_audit: list[FoldCompatibilityRecord] = []
    for fold in range(N_FOLDS):
        pstate = ref.fold_preprocessing_state(fold)
        vals, nocna_missing = _transform_single_fold(
            raw_by_fold[fold],
            pstate,
            drop_cna_from_guard=True,
            projected_columns=fallback_spec.fold_feature_orders[fold],
        )
        fb_fold_dfs.append(vals)
        fb_audit.append(
            FoldCompatibilityRecord(
                fold=fold,
                compatible_primary=primary_audit[fold].compatible_primary,
                compatible_active=(len(nocna_missing) == 0),
                n_compiled_features=vals.shape[1],
                unsupported_missing_features=nocna_missing,
            )
        )

    if any(not r.compatible_active for r in fb_audit):
        return CompilationResult(
            status="TECHNICAL_NO_CALL",
            active_model_mode="none",
            active_model_spec=None,
            no_call_reason="MISSING_OR_INCOMPATIBLE_CNA",
            diagnostic_message="Unsupported missing non-CNA feature encountered even after CNA exclusion.",
            fallback_warning=None,
            maf_result=maf_res,
            sv_result=sv_res,
            seg_result=seg_res,
            cna_result=cna_res,
            spectrum_df=spectrum_df,
            clinical_df=clinical_df,
            fold_features=[],
            packed_vector=None,
            fold_audit=fb_audit,
            warnings=all_warnings,
        )

    packed_fb = np.concatenate([df.to_numpy(dtype=np.float32)[0] for df in fb_fold_dfs], axis=0)
    trigger = cna_error if cna_res is None else "incompatible CNA bin coverage for primary model"
    fb_warn = (
        f"NO-CNA FALLBACK MODE (--allow-no-cna): Triggered due to {trigger}. "
        "Predictions use the retrained V20 fallback_without_cna ensemble "
        "(71.3% internal test / 63.0% external VICC accuracy vs 86.5% primary ensemble with CNA)."
    )
    all_warnings.append(fb_warn)
    return CompilationResult(
        status="CALLABLE",
        active_model_mode="fallback_without_cna",
        active_model_spec=fallback_spec,
        no_call_reason=None,
        diagnostic_message=None,
        fallback_warning=fb_warn,
        maf_result=maf_res,
        sv_result=sv_res,
        seg_result=seg_res,
        cna_result=cna_res,
        spectrum_df=spectrum_df,
        clinical_df=clinical_df,
        fold_features=fb_fold_dfs,
        packed_vector=packed_fb,
        fold_audit=fb_audit,
        warnings=all_warnings,
    )
