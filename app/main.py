import asyncio
import typer
import csv
import json
import logging
import signal
import sys
import os
import atexit
from typing import Optional, Set
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from .config import Config
from .types import AnalyzedBiomarker, BiomarkerEntry
from . import database as db
from . import loader
from . import llm
from . import agent as research_agent
from . import logic

app = typer.Typer(help="Open Blood Analysis CLI")
console = Console()

# Global set of files to clean up on exit
CLEANUP_PATHS: Set[str] = set()


def cleanup():
    """Final cleanup of temp files and locks."""
    for path in CLEANUP_PATHS:
        if os.path.exists(path):
            try:
                if os.path.isdir(path):
                    import shutil

                    shutil.rmtree(path)
                else:
                    os.remove(path)
            except Exception:
                pass


atexit.register(cleanup)


def signal_handler(sig, frame):
    """Handle termination signals."""
    cleanup()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


@app.command()
def analyze(
    file_path: str = typer.Argument(..., help="Path to the PDF or Image file"),
    output: Optional[str] = typer.Option(
        None, help="Path to save the result (JSON/CSV)"
    ),
    research: bool = typer.Option(
        True, help="Enable deep research for unknown biomarkers"
    ),
    debug: bool = typer.Option(False, help="Show verbose debug logs"),
    sex: Optional[str] = typer.Option(None, help="Biological sex (male/female)"),
    age: Optional[int] = typer.Option(None, help="Age in years"),
):
    """
    Analyze a blood test report and extract biomarkers.
    """
    asyncio.run(_analyze_flow(file_path, output, research, debug, sex=sex, age=age))


async def _analyze_flow(
    file_path: str,
    output: str | None,
    research_enabled: bool,
    debug: bool = False,
    sex: str | None = None,
    age: int | None = None,
):
    # Configure logging
    log_level = logging.DEBUG if debug else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Silence verbose 3rd-party loggers
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.debug(f"Starting analysis flow for: {file_path}")

    config = Config.load()

    # Track lock file for cleanup
    lock_path = config.biomarkers_path + ".lock"
    CLEANUP_PATHS.add(lock_path)

    # 0. Load DB State
    biomarkers_db = db.load_db(config.biomarkers_path)

    image_paths = []
    try:
        # 1. Pipeline: Ingestion
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task(description="Loading file...", total=None)
            try:
                image_paths = loader.load_file_as_images(file_path)
                for p in image_paths:
                    CLEANUP_PATHS.add(p)
                progress.update(task, completed=True)
                console.print(
                    f"[green]✓[/green] Loaded {len(image_paths)} pages/images."
                )
            except Exception as e:
                console.print(f"[red]Error loading file:[/red] {e}")
                return

            # 2. Pipeline: Extraction
            task = progress.add_task(
                description="Extracting data with AI...", total=None
            )
            extracted_items = await llm.extract_biomarkers(image_paths, config)
            progress.update(task, completed=True)
            console.print(
                f"[green]✓[/green] Extracted {len(extracted_items)} potential biomarkers."
            )

            # 3. Pipeline: Processing
            analyzed_results = []
            used_biomarker_ids: set[str] = set()  # Track used IDs to prevent duplicates
            task = progress.add_task(
                description="Matching and researching...", total=len(extracted_items)
            )

            for item in extracted_items:
                matched_entry: BiomarkerEntry | None = None

                # Phase 1: Exact lowercase match
                exact_match = db.find_exact_match(biomarkers_db, item.raw_name)
                if exact_match and exact_match.id not in used_biomarker_ids:
                    matched_entry = exact_match
                    logger.debug(
                        f"Exact match for '{item.raw_name}' -> {exact_match.id}"
                    )

                # Phase 2: Fuzzy candidates + AI disambiguation
                if not matched_entry:
                    # Get candidates, excluding already-used IDs
                    all_candidates = db.find_fuzzy_candidates(
                        biomarkers_db, item.raw_name
                    )
                    candidates = [
                        c for c in all_candidates if c[0].id not in used_biomarker_ids
                    ]

                    # Ask AI to decide
                    decision, entry = await research_agent.disambiguate_biomarker(
                        item.raw_name, candidates, config
                    )

                    if decision == "match" and entry:
                        matched_entry = entry
                        logger.debug(
                            f"AI confirmed match for '{item.raw_name}' -> {entry.id}"
                        )
                    elif decision == "unknown":
                        # Skip this item - it's not a real biomarker
                        progress.console.print(
                            f"[dim]  ~ Skipped '{item.raw_name}' (not a biomarker)[/dim]"
                        )
                        progress.advance(task)
                        continue
                    elif decision == "research" and research_enabled:
                        progress.console.print(
                            f"[yellow]?[/yellow] Researching: '{item.raw_name}'..."
                        )
                        new_entry = await research_agent.research_biomarker(
                            item.raw_name, config
                        )
                        if new_entry:
                            biomarkers_db = db.append_to_db(
                                config.biomarkers_path, biomarkers_db, new_entry
                            )
                            console.print(f"  [green]+[/green] Added: {new_entry.id}")
                            matched_entry = new_entry
                        else:
                            console.print(
                                f"  [red]x[/red] Could not identify '{item.raw_name}'"
                            )

                # Analyze if we have a match
                if matched_entry:
                    used_biomarker_ids.add(matched_entry.id)
                    analyzed = logic.analyze_value(
                        item.raw_name,
                        item.value,
                        item.unit,
                        matched_entry,
                        sex=sex,
                        age=age,
                    )
                    analyzed_results.append(analyzed)
                else:
                    # Include as unknown
                    analyzed_results.append(
                        AnalyzedBiomarker(
                            biomarker_id="unknown",
                            display_name=item.raw_name,
                            value=item.value,
                            unit=item.unit,
                            status="unknown",
                        )
                    )

                progress.advance(task)

        # 4. Output
        table = Table(title="Analysis Results")
        table.add_column("Biomarker", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_column("Unit", style="green")
        table.add_column("Status", style="bold")
        table.add_column("ID", style="dim")

        for res in analyzed_results:
            status_style = (
                "green"
                if res.status == "normal"
                else "red"
                if res.status in ["high", "low"]
                else "yellow"
            )
            table.add_row(
                res.display_name,
                str(res.value),
                res.unit,
                f"[{status_style}]{res.status}[/{status_style}]",
                res.biomarker_id,
            )

        console.print(table)

        # 5. Export
        if output:
            try:
                if output.lower().endswith(".json"):
                    with open(output, "w") as f:
                        data = [res.model_dump() for res in analyzed_results]
                        json.dump(data, f, indent=2)
                    console.print(f"[green]✓[/green] Saved JSON to {output}")
                elif output.lower().endswith(".csv"):
                    with open(output, "w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(
                            [
                                "biomarker_id",
                                "display_name",
                                "value",
                                "unit",
                                "status",
                                "notes",
                            ]
                        )
                        for res in analyzed_results:
                            writer.writerow(
                                [
                                    res.biomarker_id,
                                    res.display_name,
                                    res.value,
                                    res.unit,
                                    res.status,
                                    res.notes or "",
                                ]
                            )
                    console.print(f"[green]✓[/green] Saved CSV to {output}")
                else:
                    console.print(
                        "[red]Unknown output format. Please use .json or .csv[/red]"
                    )
            except Exception as e:
                console.print(f"[red]Error saving output:[/red] {e}")

    finally:
        if image_paths:
            loader.cleanup_images(image_paths)


if __name__ == "__main__":
    app()
