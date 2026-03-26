"""Typer CLI for tissue-classifier."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from .config import DEFAULT_MODEL_DIR, REFERENCE_DIR, PipelineConfig

app = typer.Typer(name="tissue-classifier", help="Single-sample tissue-of-origin classification.", no_args_is_help=True)
console = Console()


def _setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(message)s", datefmt="[%X]",
                        handlers=[RichHandler(rich_tracebacks=True, show_path=False)])


@app.command()
def predict(
    maf: Path = typer.Option(..., "--maf", help="Path to MAF file"),
    seg: Optional[Path] = typer.Option(None, "--seg", help="Path to SEG file"),
    sv: Optional[Path] = typer.Option(None, "--sv", help="Path to SV file"),
    age: Optional[float] = typer.Option(None, "--age", help="Age at sequencing"),
    sex: Optional[str] = typer.Option(None, "--sex", help="Sex (Male/Female)"),
    genome: str = typer.Option("hg19", "--genome", help="Genome build"),
    model_dir: Path = typer.Option(DEFAULT_MODEL_DIR, "--model-dir"),
    output: Path = typer.Option(Path("./results"), "--output", "-o"),
    sample_id: str = typer.Option("SAMPLE", "--sample-id"),
    config_file: Optional[Path] = typer.Option(None, "--config"),
    shap_nsamples: int = typer.Option(500, "--shap-nsamples"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Classify a single sample's tissue of origin."""
    _setup_logging(verbose)
    if config_file:
        cfg = PipelineConfig.from_yaml(config_file)
    else:
        cfg = PipelineConfig(
            maf_path=maf, seg_path=seg, sv_path=sv, age=age, sex=sex,
            genome=genome, model_dir=model_dir,
            output_dir=output, sample_id=sample_id, shap_nsamples=shap_nsamples,
        )
    from .pipeline import run_pipeline
    report_path = run_pipeline(cfg)
    console.print(f"\n[bold green]Report saved to:[/bold green] {report_path}")


@app.command()
def validate(
    maf: Path = typer.Option(..., "--maf"),
    seg: Optional[Path] = typer.Option(None, "--seg"),
):
    """Validate input files."""
    _setup_logging()
    from .preprocessing.maf_parser import parse_maf
    from .preprocessing.seg_parser import parse_seg
    try:
        df = parse_maf(maf)
        console.print(f"[green]MAF valid:[/green] {len(df)} variants")
    except Exception as e:
        console.print(f"[red]MAF invalid:[/red] {e}")
    if seg:
        try:
            df = parse_seg(seg)
            console.print(f"[green]SEG valid:[/green] {len(df)} segments")
        except Exception as e:
            console.print(f"[red]SEG invalid:[/red] {e}")


@app.command()
def info():
    """Show model and reference data information."""
    _setup_logging()
    table = Table(title="Tissue Classifier Info")
    table.add_column("Property"); table.add_column("Value")
    from . import __version__
    table.add_row("Version", __version__)
    table.add_row("Model directory", str(DEFAULT_MODEL_DIR))
    table.add_row("Reference directory", str(REFERENCE_DIR))
    labels_path = REFERENCE_DIR / "class_labels.json"
    if labels_path.exists():
        with open(labels_path) as f: labels = json.load(f)
        table.add_row("Tumor types", str(len(labels)))
        table.add_row("Classes", ", ".join(sorted(labels.keys())))
    order_path = REFERENCE_DIR / "training_feature_order.json"
    if order_path.exists():
        with open(order_path) as f: features = json.load(f)
        table.add_row("Features", str(len(features)))
    table.add_row("SHAP background", "Available" if (REFERENCE_DIR / "shap_background.pkl").exists() else "Not found")
    table.add_row("Model available", "Yes" if DEFAULT_MODEL_DIR.exists() else "No")
    console.print(table)


if __name__ == "__main__":
    app()
