"""Typer CLI for V20 tissue-classifier."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from . import __version__
from .config import (
    FALLBACK_MODEL_DIR,
    ModelSpec,
    PRIMARY_MODEL_DIR,
    PipelineConfig,
    ReferenceData,
    REFERENCE_DIR,
)

app = typer.Typer(
    name="tissue-classifier",
    help="Single-sample v2.0 cross-architecture (2x RealMLP + 2x TabM) tissue-of-origin classification and Top-3 SHAP PDF reporting.",
    no_args_is_help=True,
)
console = Console()


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
        force=True,
    )


@app.command()
def predict(
    maf: Optional[Path] = typer.Option(None, "--maf", help="Path to somatic MAF file"),
    seg: Optional[Path] = typer.Option(None, "--seg", help="Path to CNA SEG file (hg19/hg38)"),
    sv: Optional[Path] = typer.Option(None, "--sv", help="Path to gene-level SV file"),
    age: Optional[float] = typer.Option(None, "--age", help="Patient age at sequencing (years)"),
    sex: Optional[str] = typer.Option(None, "--sex", help="Biological sex (Male/Female)"),
    genome: str = typer.Option("hg19", "--genome", help="Input genome build ('hg19' or 'hg38')"),
    allow_no_cna: bool = typer.Option(
        False,
        "--allow-no-cna",
        help="Fall back to retrained fallback_without_cna ensemble if CNA is missing or incompatible",
    ),
    model_dir: Path = typer.Option(PRIMARY_MODEL_DIR, "--model-dir", help="Primary model bundle directory"),
    output: Path = typer.Option(Path("./results"), "--output", "-o", help="Root output directory"),
    sample_id: str = typer.Option("SAMPLE", "--sample-id", help="Sample identifier"),
    config_file: Optional[Path] = typer.Option(None, "--config", help="YAML configuration file"),
    shap_permutations: int = typer.Option(
        4, "--shap-permutations", help="Antithetic PermutationSHAP cycles (default 4; 2 for rapid ~78s SHAP; 0 to skip)"
    ),
    shap_background: int = typer.Option(
        32, "--shap-background", help="Development background donors for SHAP (32 or 64)"
    ),
    no_shap: bool = typer.Option(False, "--no-shap", help="Skip SHAP attribution for rapid inference"),
    html: bool = typer.Option(False, "--html", help="Also write standalone report.html"),
    device: Optional[str] = typer.Option(None, "--device", help="Compute device ('mps' or 'cpu')"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """Classify a single tumor specimen's tissue of origin and generate a clinical PDF report."""
    _setup_logging(verbose)
    if config_file is not None:
        cfg = PipelineConfig.from_yaml(config_file)
    else:
        if maf is None:
            console.print("[bold red]Error:[/bold red] --maf is required when --config is not provided.")
            raise typer.Exit(code=2)
        cfg = PipelineConfig(
            maf_path=maf,
            seg_path=seg,
            sv_path=sv,
            age=age,
            sex=sex,
            genome=genome,  # type: ignore[arg-type]
            allow_no_cna=allow_no_cna,
            model_dir=model_dir,
            output_dir=output,
            sample_id=sample_id,
            compute_shap=(not no_shap and shap_permutations > 0),
            shap_permutations=0 if no_shap else shap_permutations,
            shap_background_size=shap_background,
            generate_html=html,
            device=device,
        )

    from .pipeline import run_pipeline

    out = run_pipeline(cfg)
    if out.status == "TECHNICAL_NO_CALL":
        console.print(
            f"\n[bold yellow]TECHNICAL NO-CALL[/bold yellow] ({out.compilation.no_call_reason}): "
            f"{out.compilation.diagnostic_message}"
        )
    elif out.prediction is not None:
        console.print(
            f"\n[bold green]Predicted Origin:[/bold green] "
            f"{out.prediction.predicted_display_name} ({out.prediction.predicted_class}) — "
            f"[bold]{out.prediction.confidence:.1%}[/bold] [{out.prediction.policy_tier}]"
        )
    console.print(f"[bold]PDF Report:[/bold] {out.report_pdf_path}")
    console.print(f"[bold]JSON Summary:[/bold] {out.prediction_json_path}")
    if out.report_html_path is not None:
        console.print(f"[bold]HTML Report:[/bold] {out.report_html_path}")


@app.command()
def validate(
    maf: Path = typer.Option(..., "--maf", help="Path to somatic MAF file"),
    seg: Optional[Path] = typer.Option(None, "--seg", help="Path to CNA SEG file"),
    sv: Optional[Path] = typer.Option(None, "--sv", help="Path to SV file"),
    sample_id: str = typer.Option("SAMPLE", "--sample-id", help="Sample identifier"),
    genome: str = typer.Option("hg19", "--genome", help="Input genome build ('hg19' or 'hg38')"),
) -> None:
    """Validate genomic input files against V20 shared panel & coordinate contracts."""
    _setup_logging()
    from .preprocessing.parsers import (
        parse_and_filter_maf,
        parse_and_filter_sv,
        parse_and_validate_seg,
    )

    ref = ReferenceData(REFERENCE_DIR)
    maf_res = parse_and_filter_maf(maf, sample_id, ref.useful_genes, ref.bed_lookup, genome=genome)
    console.print(
        f"[green]MAF valid:[/green] {len(maf_res.usable_maf)} usable panel variants "
        f"(from {maf_res.raw_count} raw rows)"
    )
    if sv is not None:
        sv_res = parse_and_filter_sv(sv, sample_id, ref.useful_genes)
        console.print(
            f"[green]SV valid:[/green] {len(sv_res.usable_sv)} usable panel SVs "
            f"(from {sv_res.raw_count} raw rows)"
        )
    if seg is not None:
        seg_res = parse_and_validate_seg(seg, sample_id, ref.chrom_sizes, genome=genome)
        if seg_res.valid:
            console.print(f"[green]SEG valid:[/green] {len(seg_res.segments)} non-overlapping segments")
        else:
            console.print(f"[red]SEG invalid:[/red] {seg_res.error_reason}")
            raise typer.Exit(code=1)


@app.command()
def info() -> None:
    """Display bundled v2.0 ensemble models, thresholds, and reference data."""
    _setup_logging()
    ref = ReferenceData(REFERENCE_DIR)
    table = Table(title=f"tissue-classifier v{__version__} — Bundled v2.0 Configuration")
    table.add_column("Property", style="bold")
    table.add_column("Value")

    table.add_row("Reference Directory", str(REFERENCE_DIR))
    table.add_row("Shared Target Genes", str(len(ref.useful_genes)))
    table.add_row("hg19 chr2 Length", f"{ref.chrom_sizes.get('chr2', 0):,} bp")
    table.add_row(
        "SHAP Background Archive",
        "Available (64 dev donors; default 32)"
        if (REFERENCE_DIR / "shap_background.npz").exists()
        else "Missing",
    )

    for label, mdir in [
        ("Primary Model", PRIMARY_MODEL_DIR),
        ("No-CNA Fallback Model", FALLBACK_MODEL_DIR),
    ]:
        if (mdir / "model_spec.json").exists():
            spec = ModelSpec.from_dir(mdir)
            thr = spec.thresholds
            n_nets = (len(spec.views) if spec.views else 1) * spec.n_folds * len(spec.arms)
            cal_str = "Vector Scaling (L2)" if spec.vector_scale_a is not None else f"T={spec.temperature:.4f}"
            table.add_row(
                label,
                f"{spec.display_name}\n"
                f"{n_nets} nets | 22 classes | α={spec.alpha_balanced} ({spec.pooling}) | {cal_str} | "
                f"High≥{thr['high_confidence_top1']:.2f}, Std≥{thr['standard_confidence_top1']:.2f}, "
                f"Top3≥{thr['differential_top3_sum']:.2f}",
            )
        else:
            table.add_row(label, f"Not found at {mdir}")

    console.print(table)


if __name__ == "__main__":
    app()
