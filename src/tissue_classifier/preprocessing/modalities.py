"""Exact V20 fold-specific mutation, SV, SBS-6 spectrum, and clinical feature extractors."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

PROTEIN_CLASSES = (
    "Missense_Mutation",
    "Nonsense_Mutation",
    "Frame_Shift_Del",
    "Frame_Shift_Ins",
    "In_Frame_Del",
    "In_Frame_Ins",
    "Splice_Site",
    "Splice_Region",
    "Translation_Start_Site",
)

SBS_CLASSES = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]


def variant_group(vc: str) -> str:
    if vc == "Missense_Mutation":
        return "missense"
    if vc in ("Frame_Shift_Del", "Frame_Shift_Ins", "Nonsense_Mutation"):
        return "truncating"
    if vc in ("In_Frame_Del", "In_Frame_Ins"):
        return "inframe"
    if vc in ("Splice_Site", "Splice_Region"):
        return "splice"
    return "other"


def _indicators(
    data: pd.DataFrame, value_column: str, columns: list[str], ids: list[str]
) -> pd.DataFrame:
    if data.empty or not columns:
        return pd.DataFrame(0.0, index=ids, columns=columns, dtype="float32")
    frame = pd.crosstab(data["Tumor_Sample_Barcode"], data[value_column])
    return frame.reindex(index=ids, columns=columns, fill_value=0).gt(0).astype("float32")


def extract_fold_mutation_features(
    maf: pd.DataFrame, sample_id: str, mutation_state: dict
) -> pd.DataFrame:
    """Exact V20 Step 05 fold-fitted mutation feature transformation."""
    ids = [sample_id]
    if maf.empty:
        protein = maf.copy()
        protein["Gene_Type"] = pd.Series(dtype=str)
    else:
        protein = maf[maf["Variant_Classification"].isin(PROTEIN_CLASSES)].copy()
        protein["Gene_Type"] = (
            protein["Hugo_Symbol"] + "_" + protein["Variant_Classification"].map(variant_group)
        )

    recurrent = protein[protein["Mutation_ID"].isin(mutation_state["enriched_ids"])]
    nonrecurrent = protein[~protein["Mutation_ID"].isin(mutation_state["enriched_ids"])]

    a = _indicators(recurrent, "Mutation_ID", mutation_state["enriched_ids"], ids)
    b = _indicators(nonrecurrent, "Gene_Type", mutation_state["nonrecurrent_gene_types"], ids)

    if protein.empty:
        c = pd.DataFrame(
            0.0,
            index=ids,
            columns=[g + "_mut_count" for g in mutation_state["top_genes"]],
            dtype="float32",
        )
    else:
        counts = protein.groupby(["Tumor_Sample_Barcode", "Hugo_Symbol"]).size().unstack(fill_value=0)
        c = counts.reindex(index=ids, columns=mutation_state["top_genes"], fill_value=0).astype("float32")
        c.columns = [gene + "_mut_count" for gene in mutation_state["top_genes"]]

    tert = protein[protein["Hugo_Symbol"].eq("TERT")]
    d = _indicators(tert, "Mutation_ID", mutation_state["tert_ids"], ids)
    d["TERT_other"] = pd.Series(
        pd.Index(ids).isin(
            tert.loc[~tert["Mutation_ID"].isin(mutation_state["tert_ids"]), "Tumor_Sample_Barcode"]
        ),
        index=ids,
        dtype="float32",
    )
    d = d.drop(columns=d.columns.intersection(a.columns.union(b.columns).union(c.columns)))

    variants = ["SNP", "INS", "DEL", "DNP", "TNP"]
    if maf.empty:
        e = pd.DataFrame(0.0, index=ids, columns=[v + "_Count" for v in variants], dtype="float32")
    else:
        freq = maf.groupby(["Tumor_Sample_Barcode", "Variant_Type"]).size().unstack(fill_value=0)
        e = freq.reindex(index=ids, columns=variants, fill_value=0).astype("float32")
        e.columns = [v + "_Count" for v in variants]

    e["Total_Mutation_Count"] = e.sum(axis=1)
    denominator = e["Total_Mutation_Count"].clip(lower=1)
    e["Indel_Fraction"] = (e["INS_Count"] + e["DEL_Count"]) / denominator
    e["DNP_TNP_Fraction"] = (e["DNP_Count"] + e["TNP_Count"]) / denominator
    e["SNP_Fraction"] = e["SNP_Count"] / denominator
    e["Log_TMB"] = np.log1p(e["Total_Mutation_Count"])

    result = pd.concat([a, b, c, d, e], axis=1).fillna(0).astype("float32")
    result.index = pd.Index(ids, name="SAMPLE_ID")
    if result.columns.has_duplicates:
        raise ValueError("Mutation feature name collision")
    return result


def extract_tert_promoter_features(tert_maf: pd.DataFrame, sample_id: str) -> pd.DataFrame:
    """Extract somatic PASS hg19 TERT 5'Flank promoter features (C228T, C250T, other, any)."""
    cols = ["TERT_promoter_C228T", "TERT_promoter_C250T", "TERT_promoter_other", "TERT_promoter_any"]
    out = pd.DataFrame(0.0, index=pd.Index([sample_id], name="SAMPLE_ID"), columns=cols, dtype=np.float32)
    if tert_maf is None or tert_maf.empty:
        return out
    sub = tert_maf[
        tert_maf["Hugo_Symbol"].astype(str).str.strip().eq("TERT")
        & tert_maf["Variant_Classification"].astype(str).str.strip().eq("5'Flank")
    ]
    if sub.empty:
        return out
    pos = pd.to_numeric(sub["Start_Position"], errors="coerce")
    ref = sub["Reference_Allele"].astype(str).str.strip().str.upper()
    alt = sub["Tumor_Seq_Allele2"].astype(str).str.strip().str.upper()
    c228 = (pos == 1295228) & (ref == "G") & (alt == "A")
    c250 = (pos == 1295250) & (ref == "G") & (alt == "A")
    other = ~(c228 | c250)
    if bool(c228.any()):
        out.loc[sample_id, "TERT_promoter_C228T"] = 1.0
    if bool(c250.any()):
        out.loc[sample_id, "TERT_promoter_C250T"] = 1.0
    if bool(other.any()):
        out.loc[sample_id, "TERT_promoter_other"] = 1.0
    out.loc[sample_id, "TERT_promoter_any"] = 1.0
    return out


def extract_fold_sv_features(
    sv: pd.DataFrame, sample_id: str, sv_state: dict
) -> pd.DataFrame:
    """Exact V20 Step 06 fold-fitted structural variant feature transformation."""
    ids = [sample_id]
    recurrent_labels = sv_state["recurrent_labels"]
    selected = sv[sv["SV_Label"].isin(recurrent_labels)] if not sv.empty else sv
    if selected.empty or not recurrent_labels:
        frame = pd.DataFrame(0.0, index=ids, columns=recurrent_labels, dtype="float32")
    else:
        frame = pd.crosstab(selected["Sample_Id"], selected["SV_Label"])
        frame = frame.reindex(index=ids, columns=recurrent_labels, fill_value=0).gt(0).astype("float32")

    if sv.empty:
        frame["SV_Total_Count"] = pd.Series(0.0, index=ids, dtype="float32")
    else:
        frame["SV_Total_Count"] = (
            sv.groupby("Sample_Id").size().reindex(ids, fill_value=0).astype("float32")
        )
    frame["has_observed_SV"] = frame["SV_Total_Count"].gt(0).astype("float32")
    frame.index = pd.Index(ids, name="SAMPLE_ID")
    return frame


def extract_spectrum_features(maf: pd.DataFrame, sample_id: str) -> pd.DataFrame:
    """Exact V20 Step 03 SBS-6 mutational spectrum & Ti/Tv extraction."""
    ids = [sample_id]
    valid_bases = {"A", "C", "G", "T"}
    complement = {"A": "T", "T": "A", "C": "G", "G": "C"}

    if maf.empty:
        counts = pd.DataFrame(0.0, index=ids, columns=SBS_CLASSES, dtype="float32")
    else:
        snps = maf[maf["Variant_Type"].isin(["SNP", "SNV"])].copy()
        if not snps.empty:
            ref_up = snps["Reference_Allele"].astype(str).str.upper()
            alt_up = snps["Tumor_Seq_Allele2"].astype(str).str.upper()
            snps = snps.loc[ref_up.isin(valid_bases) & alt_up.isin(valid_bases)].copy()
        if snps.empty:
            counts = pd.DataFrame(0.0, index=ids, columns=SBS_CLASSES, dtype="float32")
        else:
            def classify(ref: str, alt: str) -> str:
                r, a = ref.upper(), alt.upper()
                if r in ("A", "G"):
                    r, a = complement[r], complement.get(a, a)
                return r + ">" + a

            snps["SBS_Class"] = [
                classify(r, a)
                for r, a in zip(snps["Reference_Allele"], snps["Tumor_Seq_Allele2"])
            ]
            counts = pd.crosstab(snps["Tumor_Sample_Barcode"], snps["SBS_Class"])
            counts = counts.reindex(index=ids, columns=SBS_CLASSES, fill_value=0).astype("float32")

    total = counts.sum(axis=1)
    transitions = counts["C>T"] + counts["T>C"]
    transversions = counts["C>A"] + counts["C>G"] + counts["T>A"] + counts["T>G"]

    result = pd.concat(
        [
            counts.rename(columns={c: c.replace(">", "_") + "_Count" for c in SBS_CLASSES}),
            counts.div(total.clip(lower=1), axis=0).rename(
                columns={c: c.replace(">", "_") + "_Fraction" for c in SBS_CLASSES}
            ),
            (transitions / transversions.clip(lower=1)).rename("Ti_Tv_Ratio"),
            total.rename("Spectrum_SNP_Count"),
        ],
        axis=1,
    )
    proportions = [c for c in result.columns if "Fraction" in c or "Ratio" in c]
    result.loc[total.lt(4), proportions] = np.nan
    result.index = pd.Index(ids, name="SAMPLE_ID")
    return result.astype("float32")


def extract_clinical_features(
    sample_id: str, age: Optional[float], sex: Optional[str]
) -> pd.DataFrame:
    """Build clinical raw columns (AGE_AT_SEQ_REPORT, SEX) matching V20 schema."""
    age_val = float(age) if age is not None and np.isfinite(age) else np.nan
    if sex == "Male":
        sex_val = 1.0
    elif sex == "Female":
        sex_val = 0.0
    else:
        sex_val = np.nan
    return pd.DataFrame(
        {"AGE_AT_SEQ_REPORT": [age_val], "SEX": [sex_val]},
        index=pd.Index([sample_id], name="SAMPLE_ID"),
        dtype="float64",
    )
