import asyncio
import typer
import csv
import json
import logging
import signal
import sys
import os
import atexit
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set
from rich.console import Console
from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
)

from .config import Config
from .ai_client import build_ai_client
from .types import AnalyzedBiomarker, BiomarkerEntry, ExtractedBiomarker, ReportMetadata
from . import database as db
from . import loader
from . import agent as research_agent
from . import computed as computed_logic
from .extraction import extract_report
from . import logic
from . import resolution
from .types import LearnedContextAlias, LearnedValueAlias

app = typer.Typer(help="Open Blood Analysis CLI")
console = Console()

# Global set of files to clean up on exit
CLEANUP_PATHS: Set[str] = set()


def cleanup():
    """Final cleanup of temp files created by this process."""
    for path in CLEANUP_PATHS:
        if os.path.exists(path):
            try:
                if os.path.isdir(path):
                    import shutil

                    shutil.rmtree(path)
                else:
                    os.remove(path)
            except Exception as exc:
                logging.getLogger(__name__).debug(
                    "Cleanup failed for '%s': %s", path, exc
                )


atexit.register(cleanup)


def signal_handler(sig, frame):
    """Handle termination signals."""
    cleanup()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def _format_range(min_val: float | None, max_val: float | None) -> str:
    if min_val is None and max_val is None:
        return "-"
    if min_val is None:
        return f"<= {max_val}"
    if max_val is None:
        return f">= {min_val}"
    return f"{min_val} - {max_val}"


def _format_peak(peak: float | None) -> str:
    if peak is None:
        return "-"
    return str(peak)


def _format_value(value: float | str | bool) -> str:
    if isinstance(value, bool):
        return "Positive" if value else "Negative"
    if isinstance(value, float):
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value)


def _status_style(status: str) -> str:
    if status == "optimal":
        return "bright_green"
    if status == "normal":
        return "green"
    if status == "moderate":
        return "yellow"
    if status == "elevated":
        return "orange3"
    if status in {"high", "low", "abnormal"}:
        return "red"
    if status == "unknown":
        return "grey70"
    return "white"


def _render_status_summary(results: list[AnalyzedBiomarker]):
    counts = Counter(res.status for res in results)
    ordered = [
        "optimal",
        "normal",
        "moderate",
        "elevated",
        "high",
        "low",
        "abnormal",
        "unknown",
    ]
    parts = []
    for status in ordered:
        count = counts.get(status, 0)
        if count:
            color = _status_style(status)
            parts.append(f"[{color}]{status}[/{color}]: {count}")

    if parts:
        console.print(
            Panel(
                "  •  ".join(parts),
                title="Run Summary",
                border_style="blue",
                padding=(0, 1),
            )
        )


def _fmt_optional(value: object) -> str:
    if value is None:
        return "-"
    text = str(value).strip()
    return text if text else "-"


def _render_metadata_summary(
    metadata: ReportMetadata,
    effective_sex: str | None,
    effective_age: int | None,
    sex_source: str,
    age_source: str,
):
    rows = [
        f"[bold]Patient age:[/bold] {_fmt_optional(metadata.patient.age)}",
        f"[bold]Patient gender:[/bold] {_fmt_optional(metadata.patient.gender)}",
        f"[bold]Lab company:[/bold] {_fmt_optional(metadata.lab.company_name)}",
        f"[bold]Lab location:[/bold] {_fmt_optional(metadata.lab.location)}",
        f"[bold]Collection date:[/bold] {_fmt_optional(metadata.blood_collection.date)}",
        f"[bold]Collection time:[/bold] {_fmt_optional(metadata.blood_collection.time)}",
        f"[bold]Collection datetime:[/bold] {_fmt_optional(metadata.blood_collection.datetime)}",
        f"[bold]Applied analysis age:[/bold] {_fmt_optional(effective_age)} [dim]({age_source})[/dim]",
        f"[bold]Applied analysis sex:[/bold] {_fmt_optional(effective_sex)} [dim]({sex_source})[/dim]",
    ]
    console.print(
        Panel(
            "\n".join(rows),
            title="Report Metadata",
            border_style="grey66",
            padding=(0, 1),
        )
    )


def _configure_logging(debug: bool):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    if debug:
        logging.getLogger("google_genai").setLevel(logging.INFO)
        logging.getLogger("google_genai.models").setLevel(logging.INFO)
    else:
        logging.getLogger("google_genai").setLevel(logging.WARNING)
        logging.getLogger("google_genai.models").setLevel(logging.WARNING)


def _build_config(biomarkers_path: str | None = None) -> Config:
    if biomarkers_path is None:
        return Config()
    return Config(biomarkers_path=str(Path(biomarkers_path).expanduser()))


def _upsert_biomarker_entry(
    entries: list[BiomarkerEntry],
    new_entry: BiomarkerEntry,
    matched_id: str | None = None,
) -> list[BiomarkerEntry]:
    ids_to_replace = {new_entry.id}
    if matched_id:
        ids_to_replace.add(matched_id)
    filtered = [entry for entry in entries if entry.id not in ids_to_replace]
    filtered.append(new_entry)
    return filtered


def _normalize_entry_for_compare(entry: BiomarkerEntry) -> dict:
    payload = entry.model_dump()

    for key in ["aliases", "enum_values", "normal_values"]:
        values = payload.get(key)
        if isinstance(values, list):
            payload[key] = sorted({str(v) for v in values})

    conversions = payload.get("conversions")
    if isinstance(conversions, dict):
        payload["conversions"] = {k: conversions[k] for k in sorted(conversions)}

    rules = payload.get("reference_rules")
    if isinstance(rules, list):
        payload["reference_rules"] = sorted(
            rules,
            key=lambda r: (
                str(r.get("condition", "")),
                int(r.get("priority", 0)),
                str(r.get("min_normal", "")),
                str(r.get("max_normal", "")),
            ),
        )

    return payload


def _entries_equivalent(a: BiomarkerEntry, b: BiomarkerEntry) -> bool:
    return _normalize_entry_for_compare(a) == _normalize_entry_for_compare(b)


def _fmt_diff_value(value: object, max_len: int = 120) -> str:
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        text = str(value)
    if len(text) > max_len:
        return text[: max_len - 1] + "…"
    return text


def _entry_diff_rows(
    old: BiomarkerEntry, new: BiomarkerEntry
) -> list[tuple[str, str, str]]:
    old_payload = _normalize_entry_for_compare(old)
    new_payload = _normalize_entry_for_compare(new)
    keys = sorted(set(old_payload.keys()) | set(new_payload.keys()))

    rows: list[tuple[str, str, str]] = []
    for key in keys:
        old_val = old_payload.get(key)
        new_val = new_payload.get(key)
        if old_val != new_val:
            rows.append((key, _fmt_diff_value(old_val), _fmt_diff_value(new_val)))
    return rows


def _entry_by_id(entries: list[BiomarkerEntry], entry_id: str) -> BiomarkerEntry | None:
    for entry in entries:
        if entry.id == entry_id:
            return entry
    return None


def _high_confidence_candidate(
    candidates: list[tuple[BiomarkerEntry, str, float]],
    min_score: float = 90.0,
    min_margin: float = 6.0,
) -> BiomarkerEntry | None:
    """
    Deterministically accept an obvious fuzzy candidate to avoid unnecessary LLM/research.
    """
    if not candidates:
        return None

    top_entry, _top_label, top_score = candidates[0]
    second_score = candidates[1][2] if len(candidates) > 1 else 0.0
    if top_score >= min_score and (top_score - second_score) >= min_margin:
        return top_entry

    return None


def _is_numeric_value(value: float | str | bool) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _has_unresolved_unit_mismatch(
    item: ExtractedBiomarker, entry: BiomarkerEntry
) -> bool:
    """
    True when a quantitative numeric value cannot be normalized to canonical unit.
    """
    if entry.value_type != "quantitative":
        return False
    if not _is_numeric_value(item.value):
        return False
    if not item.unit or not entry.canonical_unit:
        return False

    numeric_value = float(item.value)
    _converted_value, converted_unit = logic.convert_units(
        numeric_value,
        item.unit,
        entry.canonical_unit,
        entry,
    )
    return logic.normalize_unit(converted_unit) != logic.normalize_unit(
        entry.canonical_unit
    )


def _match_priority(source: str) -> int:
    priorities = {
        "exact_id": 3,
        "context_alias": 3,
        "exact_alias": 2,
        "fuzzy_high_confidence": 2,
        "research": 2,
        "ai": 1,
    }
    return priorities.get(source, 0)


def _can_use_exact_alias_match(raw_name: str) -> bool:
    return resolution.can_use_exact_alias_match(raw_name)


@dataclass
class _PendingResearchJob:
    index: int
    item: ExtractedBiomarker
    research_key: tuple[str, str, str, str, bool]


@dataclass
class _PendingResearchGroup:
    key: tuple[str, str, str, str, bool]
    item: ExtractedBiomarker
    task: asyncio.Task[BiomarkerEntry | None]
    jobs: list[_PendingResearchJob]


async def _run_research_job(
    item: ExtractedBiomarker,
    config: Config,
    client=None,
) -> BiomarkerEntry | None:
    entry: BiomarkerEntry | None = None
    try:
        entry = await research_agent.research_biomarker(
            item.raw_name,
            config,
            extracted_unit=item.unit,
            client=client,
            item=item,
        )
    except Exception:
        logging.getLogger(__name__).exception(
            "Unhandled research task failure for '%s'", item.raw_name
        )
    return entry


@app.command()
def analyze(
    file_path: Optional[str] = typer.Argument(
        None,
        help="Path to the PDF or Image file. Omit when using --reresearch-biomarker.",
    ),
    output: Optional[str] = typer.Option(
        None, help="Path to save the result (JSON/CSV)"
    ),
    research: bool = typer.Option(
        True, help="Enable deep research for unknown biomarkers"
    ),
    ask_before_research: bool = typer.Option(
        True,
        "--ask-before-research/--no-ask-before-research",
        help="Prompt before researching unknown biomarkers (auto-disabled when non-interactive).",
    ),
    debug: bool = typer.Option(False, help="Show verbose debug logs"),
    sex: Optional[str] = typer.Option(None, help="Biological sex (male/female)"),
    age: Optional[int] = typer.Option(None, help="Age in years"),
    save_raw: Optional[str] = typer.Option(
        None, "--save-raw", help="Save raw LLM response JSON to this file path"
    ),
    reresearch_biomarker: Optional[str] = typer.Option(
        None,
        "--reresearch-biomarker",
        help="Manually re-research and update one biomarker entry by ID/name.",
    ),
    reresearch_unit: Optional[str] = typer.Option(
        None,
        "--reresearch-unit",
        help="Optional unit context for manual re-research.",
    ),
    dry_run_reresearch: bool = typer.Option(
        False,
        "--dry-run-reresearch",
        help="Show researched JSON without writing the biomarkers DB file.",
    ),
    biomarkers_path: Optional[str] = typer.Option(
        None,
        "--biomarkers-path",
        help="Path to the biomarkers database JSON file.",
    ),
    show_skipped: bool = typer.Option(
        False,
        "--show-skipped",
        help="Include skipped/unmatched placeholder rows in report and exports.",
    ),
    review_decisions: bool = typer.Option(
        False,
        "--review-decisions/--no-review-decisions",
        help="Review AI decision recommendations interactively. Default is AI auto-decide.",
    ),
):
    """
    Analyze a blood test report, or manually re-research a biomarker entry.
    """
    if reresearch_biomarker:
        if file_path:
            raise typer.BadParameter(
                "Do not provide a file path when using --reresearch-biomarker."
            )
        asyncio.run(
            _reresearch_flow(
                reresearch_biomarker,
                debug=debug,
                extracted_unit=reresearch_unit,
                dry_run=dry_run_reresearch,
                biomarkers_path=biomarkers_path,
            )
        )
        return

    if not file_path:
        raise typer.BadParameter(
            "Missing FILE_PATH. Provide a report path or use --reresearch-biomarker."
        )

    asyncio.run(
        _analyze_flow(
            file_path,
            output,
            research,
            ask_before_research,
            debug,
            sex=sex,
            age=age,
            save_raw=save_raw,
            biomarkers_path=biomarkers_path,
            show_skipped=show_skipped,
            review_decisions=review_decisions,
        )
    )


async def _reresearch_flow(
    biomarker_query: str,
    debug: bool = False,
    extracted_unit: str | None = None,
    dry_run: bool = False,
    biomarkers_path: str | None = None,
):
    _configure_logging(debug)
    config = _build_config(biomarkers_path)
    entries = db.load_db(config.biomarkers_path)
    matched = db.find_exact_match(entries, biomarker_query)

    research_name = matched.id if matched else biomarker_query
    unit_hint = extracted_unit or (matched.canonical_unit if matched else None)

    console.print(f"[cyan]Researching:[/cyan] {research_name}")
    refreshed = await research_agent.research_biomarker(
        research_name, config, extracted_unit=unit_hint
    )
    if not refreshed:
        console.print(f"[red]x[/red] Could not research '{biomarker_query}'")
        return

    if dry_run:
        if matched:
            diff_rows = _entry_diff_rows(matched, refreshed)
            if diff_rows:
                diff_table = Table(
                    title=f"Dry-Run Diff • {matched.id} -> {refreshed.id}",
                    box=box.SIMPLE_HEAVY,
                    header_style="bold white",
                )
                diff_table.add_column("Field", style="cyan")
                diff_table.add_column("Current", style="yellow")
                diff_table.add_column("Proposed", style="green")
                for field, old_val, new_val in diff_rows:
                    diff_table.add_row(field, old_val, new_val)
                console.print(diff_table)
            else:
                console.print(
                    f"[yellow]~[/yellow] No effective changes detected for '{matched.id}'."
                )
        console.print_json(json.dumps(refreshed.model_dump(), indent=2))
        console.print("[yellow]Dry run:[/yellow] No DB changes written.")
        return

    if matched and _entries_equivalent(matched, refreshed):
        console.print(
            f"[yellow]~[/yellow] Re-research completed: no effective changes for '{matched.id}'."
        )
        return

    diff_rows: list[tuple[str, str, str]] = []
    if matched:
        diff_rows = _entry_diff_rows(matched, refreshed)

    updated_entries = _upsert_biomarker_entry(
        entries,
        refreshed,
        matched_id=matched.id if matched else None,
    )
    db.save_db(config.biomarkers_path, updated_entries)

    if matched:
        console.print(
            f"[green]✓[/green] Re-researched '{matched.id}' -> '{refreshed.id}' and updated DB."
        )
        if diff_rows:
            diff_table = Table(
                title=f"Updated Fields • {matched.id}",
                box=box.SIMPLE_HEAVY,
                header_style="bold white",
            )
            diff_table.add_column("Field", style="cyan")
            diff_table.add_column("Previous", style="yellow")
            diff_table.add_column("New", style="green")
            for field, old_val, new_val in diff_rows:
                diff_table.add_row(field, old_val, new_val)
            console.print(diff_table)
    else:
        console.print(
            f"[green]✓[/green] Researched '{biomarker_query}' and added '{refreshed.id}'."
        )


async def _analyze_flow(
    file_path: str,
    output: str | None,
    research_enabled: bool,
    ask_before_research: bool,
    debug: bool = False,
    sex: str | None = None,
    age: int | None = None,
    save_raw: str | None = None,
    biomarkers_path: str | None = None,
    show_skipped: bool = False,
    review_decisions: bool = False,
):
    """
    Analyze a blood test report and extract biomarkers.
    """
    _configure_logging(debug)

    logger = logging.getLogger(__name__)
    logger.debug(f"Starting analysis flow for: {file_path}")

    config = _build_config(biomarkers_path)
    ai_client = build_ai_client(config)

    # 0. Load DB State
    biomarkers_db = db.load_db(config.biomarkers_path)

    image_paths = []
    try:
        console.print(
            Panel(
                f"[bold]File:[/bold] {Path(file_path).name}",
                title="Blood Analysis",
                border_style="cyan",
                padding=(0, 1),
            )
        )

        # 1. Pipeline: Ingestion
        console.print("[bold cyan]1/4[/bold cyan] Loading report...")
        try:
            image_paths = loader.load_file_as_images(file_path)
            if image_paths:
                CLEANUP_PATHS.add(os.path.dirname(image_paths[0]))
            for p in image_paths:
                CLEANUP_PATHS.add(p)
            console.print(f"[green]✓[/green] Loaded {len(image_paths)} page(s).")
        except Exception as e:
            console.print(f"[red]Error loading file:[/red] {e}")
            return

        # 2. Pipeline: Extraction
        console.print("[bold cyan]2/4[/bold cyan] Extracting biomarkers...")
        try:
            extraction_result = await extract_report(
                image_paths=image_paths,
                config=config,
            )
            extracted_items = extraction_result.biomarkers
            extraction_notes = extraction_result.notes
            extraction_metadata = extraction_result.metadata
            raw_llm_response = extraction_result.raw_payload
        except Exception as e:
            console.print(
                f"[bold red]✗ Extraction failed:[/bold red] {e}\n"
                "[red]Please retry in a moment (model/provider is temporarily unavailable).[/red]"
            )
            return
        if save_raw and raw_llm_response:
            try:
                with open(save_raw, "w") as raw_f:
                    raw_data = json.loads(raw_llm_response)
                    json.dump(raw_data, raw_f, indent=2)
                console.print(f"[green]✓[/green] Saved raw LLM response to {save_raw}")
            except Exception:
                with open(save_raw, "w") as raw_f:
                    raw_f.write(raw_llm_response)
                console.print(f"[green]✓[/green] Saved raw LLM response to {save_raw}")
        if not extracted_items:
            console.print(
                "[bold red]✗ Extraction produced no biomarkers.[/bold red] "
                "The AI response could not be parsed into biomarker entries."
            )
            return
        console.print(
            f"[green]✓[/green] Extracted {len(extracted_items)} potential biomarker(s)."
        )
        if extraction_notes:
            console.print(
                f"[green]✓[/green] Captured {len(extraction_notes)} general observation(s)."
            )

        effective_age = age if age is not None else extraction_metadata.patient.age
        effective_sex = sex if sex else extraction_metadata.patient.gender
        age_source = (
            "cli"
            if age is not None
            else "extracted"
            if effective_age is not None
            else "none"
        )
        sex_source = "cli" if sex else "extracted" if effective_sex else "none"
        _render_metadata_summary(
            extraction_metadata,
            effective_sex=effective_sex,
            effective_age=effective_age,
            sex_source=sex_source,
            age_source=age_source,
        )

        # 3. Pipeline: Processing
        console.print("[bold cyan]3/4[/bold cyan] Matching and enrichment...")
        analyzed_results: list[AnalyzedBiomarker | None] = [None] * len(extracted_items)
        supplemental_results: list[AnalyzedBiomarker] = []
        id_assignments: dict[str, tuple[int, str, float]] = {}
        pending_research_groups: dict[tuple[str, str, str, str, bool], _PendingResearchGroup] = {}
        deferred_computed_items: list[tuple[int, ExtractedBiomarker]] = []
        skipped_unit_assist_keys: set[tuple[str, str]] = set()
        max_parallel_research = min(8, max(1, len(extracted_items)))
        research_semaphore = asyncio.Semaphore(max_parallel_research)
        with Progress(
            SpinnerColumn(style="cyan"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=28),
            MofNCompleteColumn(),
            TextColumn("[dim]{task.fields[status]}[/dim]"),
            TimeElapsedColumn(),
            transient=True,
            console=console,
        ) as progress:
            overall_task = progress.add_task(
                description="Processing biomarkers",
                total=len(extracted_items),
                status="resolving",
            )

            def _store_unknown_result(index: int, item: ExtractedBiomarker):
                analyzed_results[index] = AnalyzedBiomarker(
                    biomarker_id="unknown",
                    display_name=item.raw_name,
                    value=item.value,
                    unit=item.unit,
                    status="unknown",
                )

            def _store_matched_result(
                index: int,
                item: ExtractedBiomarker,
                matched_entry: BiomarkerEntry,
                match_source: str,
            ):
                nonlocal biomarkers_db
                if resolution.should_persist_match_alias(
                    matched_entry, item, match_source
                ):
                    biomarkers_db = db.add_alias_to_entry(
                        config.biomarkers_path,
                        biomarkers_db,
                        matched_entry.id,
                        item.raw_name,
                    )
                    biomarkers_db = db.add_context_alias_to_entry(
                        config.biomarkers_path,
                        biomarkers_db,
                        matched_entry.id,
                        LearnedContextAlias(
                            raw_name=item.raw_name,
                            raw_unit=item.unit or None,
                            specimen=item.specimen,
                            representation=resolution.observed_representation(item),
                        ),
                    )
                    refreshed_entry = _entry_by_id(biomarkers_db, matched_entry.id)
                    if refreshed_entry:
                        matched_entry = refreshed_entry

                analyzed = logic.analyze_value(
                    item.raw_name,
                    item.value,
                    item.unit,
                    matched_entry,
                    sex=effective_sex,
                    age=effective_age,
                    specimen=item.specimen,
                    semantic_value=item.semantic_value,
                    measurement_qualifier=item.measurement_qualifier,
                )
                if item.raw_value_text and analyzed.semantic_value:
                    biomarkers_db = db.add_value_alias_to_entry(
                        config.biomarkers_path,
                        biomarkers_db,
                        matched_entry.id,
                        LearnedValueAlias(
                            raw_value=item.raw_value_text,
                            semantic_value=analyzed.semantic_value,
                            measurement_qualifier=analyzed.measurement_qualifier,
                        ),
                    )
                analyzed_results[index] = analyzed

            def _assign_match(
                index: int,
                item: ExtractedBiomarker,
                matched_entry: BiomarkerEntry,
                match_source: str,
                match_score: float,
            ):
                existing = id_assignments.get(matched_entry.id)
                if existing and existing[0] != index:
                    existing_index, existing_source, existing_score = existing
                    new_rank = (_match_priority(match_source), match_score)
                    old_rank = (_match_priority(existing_source), existing_score)
                    if new_rank <= old_rank:
                        console.print(
                            f"[dim]~ Kept existing match for '{matched_entry.id}' and skipped duplicate from '{item.raw_name}'.[/dim]"
                        )
                        _store_unknown_result(index, item)
                        return

                    _store_unknown_result(
                        existing_index, extracted_items[existing_index]
                    )
                    console.print(
                        f"[dim]~ Reassigned '{matched_entry.id}' to '{item.raw_name}' (stronger match than '{extracted_items[existing_index].raw_name}').[/dim]"
                    )

                _store_matched_result(
                    index,
                    item,
                    matched_entry,
                    match_source=match_source,
                )
                id_assignments[matched_entry.id] = (
                    index,
                    match_source,
                    float(match_score),
                )

            async def _run_limited_research(
                item: ExtractedBiomarker,
            ) -> BiomarkerEntry | None:
                async with research_semaphore:
                    return await _run_research_job(
                        item, config, client=ai_client,
                    )

            async def _await_research_group(
                key: tuple[str, str, str, str, bool],
                group: _PendingResearchGroup,
            ) -> tuple[tuple[str, str, str, str, bool], BiomarkerEntry | None]:
                return key, await group.task

            async def _resolve_binary_decision(
                *,
                decision_name: str,
                question: str,
                context: dict[str, object],
                fallback_default: bool = False,
            ) -> bool:
                ai_default = fallback_default
                recommendation = await research_agent.recommend_binary_decision(
                    decision_name=decision_name,
                    question=question,
                    context=context,
                    config=config,
                    client=ai_client,
                )
                if recommendation:
                    ai_default = bool(recommendation.get("approved", False))
                    ai_label = "yes" if ai_default else "no"
                    ai_confidence = float(recommendation.get("confidence", 0.0))
                    ai_reason = str(recommendation.get("reason", "")).strip()
                    reason_suffix = f" ({ai_reason})" if ai_reason else ""
                    console.print(
                        f"[dim]~ AI decision [{decision_name}]: {ai_label} ({ai_confidence:.2f}){reason_suffix}[/dim]"
                    )
                else:
                    fallback_label = "yes" if ai_default else "no"
                    console.print(
                        f"[dim]~ AI decision [{decision_name}] unavailable; using fallback '{fallback_label}'.[/dim]"
                    )

                if not review_decisions:
                    return ai_default

                if not sys.stdin.isatty():
                    console.print(
                        f"[dim]~ Review requested for '{decision_name}' but non-interactive mode; using AI decision.[/dim]"
                    )
                    return ai_default

                progress.stop()
                try:
                    return typer.confirm(question, default=ai_default)
                finally:
                    progress.start()

            async def _maybe_assist_unit_conversion(
                item: ExtractedBiomarker, matched_entry: BiomarkerEntry
            ) -> BiomarkerEntry:
                nonlocal biomarkers_db

                unit_key = logic.normalize_unit(item.unit)
                cache_key = (matched_entry.id, unit_key)
                if cache_key in skipped_unit_assist_keys:
                    return matched_entry

                if not _has_unresolved_unit_mismatch(item, matched_entry):
                    return matched_entry

                spawn_question = (
                    f"Unit conversion failed for '{item.raw_name}' "
                    f"({item.unit} -> {matched_entry.canonical_unit}). "
                    "Spawn a thinking agent to propose conversion JSON?"
                )
                should_spawn = await _resolve_binary_decision(
                    decision_name="spawn_unit_conversion_agent",
                    question=spawn_question,
                    context={
                        "raw_name": item.raw_name,
                        "biomarker_id": matched_entry.id,
                        "input_unit": item.unit,
                        "canonical_unit": matched_entry.canonical_unit,
                        "value": item.value,
                        "has_existing_conversion": any(
                            logic.normalize_unit(k) == logic.normalize_unit(item.unit)
                            for k in matched_entry.conversions
                        ),
                    },
                    fallback_default=False,
                )

                if not should_spawn:
                    skipped_unit_assist_keys.add(cache_key)
                    return matched_entry

                console.print(
                    f"[cyan]~[/cyan] Running unit conversion thinking step for '{item.raw_name}'..."
                )
                proposal = await research_agent.think_unit_conversion(
                    biomarker_name=item.raw_name,
                    biomarker_id=matched_entry.id,
                    from_unit=item.unit,
                    canonical_unit=matched_entry.canonical_unit,
                    observed_value=item.value,
                    config=config,
                    client=ai_client,
                )
                if not proposal:
                    console.print(
                        f"[yellow]~[/yellow] No conversion proposal generated for '{item.raw_name}'."
                    )
                    skipped_unit_assist_keys.add(cache_key)
                    return matched_entry

                console.print(
                    Panel(
                        json.dumps(proposal, indent=2, ensure_ascii=False),
                        title=f"Unit Conversion Proposal • {item.raw_name}",
                        border_style="grey66",
                        padding=(0, 1),
                    )
                )

                action = str(proposal.get("action", "")).strip().lower()
                proposed_formula = str(proposal.get("formula", "")).strip()
                proposed_input_unit = str(
                    proposal.get("input_unit") or item.unit
                ).strip()
                if action != "add_conversion" or not proposed_formula:
                    console.print(
                        f"[dim]~ Conversion proposal declined for '{item.raw_name}' ({action or 'no action'}).[/dim]"
                    )
                    skipped_unit_assist_keys.add(cache_key)
                    return matched_entry

                updated_conversions = dict(matched_entry.conversions)
                existing_key = next(
                    (
                        k
                        for k in updated_conversions
                        if logic.normalize_unit(k)
                        == logic.normalize_unit(proposed_input_unit)
                    ),
                    None,
                )
                target_key = existing_key or proposed_input_unit
                old_formula = updated_conversions.get(target_key)
                updated_conversions[target_key] = proposed_formula

                candidate_entry = matched_entry.model_copy(
                    update={"conversions": updated_conversions}
                )
                preview_value, preview_unit = logic.convert_units(
                    float(item.value),
                    item.unit,
                    matched_entry.canonical_unit,
                    candidate_entry,
                )
                if logic.normalize_unit(preview_unit) != logic.normalize_unit(
                    matched_entry.canonical_unit
                ):
                    console.print(
                        f"[yellow]~[/yellow] Proposed formula did not resolve unit mismatch for '{item.raw_name}'."
                    )
                    skipped_unit_assist_keys.add(cache_key)
                    return matched_entry

                add_question = (
                    f"Add conversion for '{matched_entry.id}': "
                    f"{target_key} -> {matched_entry.canonical_unit} "
                    f"using formula '{proposed_formula}'?"
                )
                should_add = await _resolve_binary_decision(
                    decision_name="add_unit_conversion",
                    question=add_question,
                    context={
                        "raw_name": item.raw_name,
                        "biomarker_id": matched_entry.id,
                        "input_unit": target_key,
                        "canonical_unit": matched_entry.canonical_unit,
                        "formula": proposed_formula,
                        "proposal": proposal,
                        "preview_value": preview_value,
                        "preview_unit": preview_unit,
                        "current_formula": old_formula,
                    },
                    fallback_default=False,
                )

                if not should_add:
                    skipped_unit_assist_keys.add(cache_key)
                    return matched_entry

                biomarkers_db = _upsert_biomarker_entry(
                    biomarkers_db,
                    candidate_entry,
                    matched_id=matched_entry.id,
                )
                await db.asave_db(config.biomarkers_path, biomarkers_db)
                matched_entry = (
                    _entry_by_id(biomarkers_db, matched_entry.id) or candidate_entry
                )
                if old_formula and old_formula != proposed_formula:
                    console.print(
                        f"[green]✓[/green] Updated conversion for '{matched_entry.id}' ({target_key}: '{old_formula}' -> '{proposed_formula}')."
                    )
                else:
                    console.print(
                        f"[green]✓[/green] Added conversion for '{matched_entry.id}' ({target_key}: '{proposed_formula}')."
                    )
                console.print(
                    f"[dim]~ Preview conversion result: {_format_value(preview_value)} {matched_entry.canonical_unit}[/dim]"
                )
                return matched_entry

            async def _resolve_entry_for_item(
                item: ExtractedBiomarker,
                *,
                allow_computed: bool = False,
            ) -> tuple[str, BiomarkerEntry | None, str, float]:
                candidate_entries = [
                    entry
                    for entry in biomarkers_db
                    if (
                        (allow_computed and (entry.kind == "computed" or entry.computed_definition is not None))
                        or (not allow_computed and entry.kind != "computed")
                    )
                ]
                context_match = resolution.find_context_alias_match(candidate_entries, item)
                if context_match:
                    return ("match", context_match, "context_alias", 950.0)

                exact_match = db.find_exact_match(
                    candidate_entries,
                    item.raw_name,
                    include_aliases=False,
                )
                if exact_match and resolution.is_entry_compatible(
                    exact_match,
                    item,
                    allow_computed=allow_computed,
                ):
                    return ("match", exact_match, "exact_id", 1000.0)

                if allow_computed or _can_use_exact_alias_match(item.raw_name):
                    exact_alias_match = db.find_exact_match(
                        candidate_entries,
                        item.raw_name,
                        include_aliases=True,
                    )
                    if exact_alias_match and (
                        db.normalize_biomarker_name(exact_alias_match.id)
                        != db.normalize_biomarker_name(item.raw_name)
                    ) and resolution.is_entry_compatible(
                        exact_alias_match,
                        item,
                        allow_computed=allow_computed,
                    ):
                        return ("match", exact_alias_match, "exact_alias", 900.0)

                candidates = resolution.filter_candidates(
                    db.find_fuzzy_candidates(candidate_entries, item.raw_name),
                    item,
                    allow_computed=allow_computed,
                )
                if not candidates and not candidate_entries:
                    return ("research", None, "ai", 0.0)
                hc_entry = _high_confidence_candidate(candidates)
                if hc_entry:
                    return (
                        "match",
                        hc_entry,
                        "fuzzy_high_confidence",
                        candidates[0][2] if candidates else 0.0,
                    )

                decision, entry = await research_agent.disambiguate_biomarker(
                    item.raw_name,
                    candidates,
                    config,
                    client=ai_client,
                    item=item,
                    allow_computed=allow_computed,
                )
                if decision == "match" and entry:
                    score = next(
                        (
                            candidate_score
                            for candidate_entry, _label, candidate_score in candidates
                            if candidate_entry.id == entry.id
                        ),
                        0.0,
                    )
                    return ("match", entry, "ai", score)
                return (decision, None, "ai", 0.0)

            # Dedup only truly repeated extracted rows; do not collapse rows that
            # merely share the same label but differ in value/unit/specimen.
            seen_row_keys: dict[tuple[str, str, str, str, str, bool], int] = {}
            dedup_indices_to_skip: set[int] = set()
            for i, item in enumerate(extracted_items):
                dedup_key = resolution.extraction_dedup_key(item)
                if dedup_key in seen_row_keys:
                    dedup_indices_to_skip.add(i)
                    logger.debug(
                        "Skipping duplicate extraction row '%s' at index %d (first seen at %d)",
                        item.raw_name, i, seen_row_keys[dedup_key],
                    )
                else:
                    seen_row_keys[dedup_key] = i

            for index, item in enumerate(extracted_items):
                if index in dedup_indices_to_skip:
                    _store_unknown_result(index, item)
                    progress.advance(overall_task)
                    continue

                if resolution.is_computed_candidate(item):
                    deferred_computed_items.append((index, item))
                    continue

                decision, matched_entry, match_source, match_score = (
                    await _resolve_entry_for_item(item)
                )

                if decision == "unknown":
                    console.print(
                        f"[dim]~ Skipped '{item.raw_name}' (not a biomarker)[/dim]"
                    )
                    _store_unknown_result(index, item)
                    progress.advance(overall_task)
                    continue

                if decision == "research" and research_enabled:
                    should_research = True
                    if ask_before_research:
                        if not sys.stdin.isatty():
                            console.print(
                                f"[dim]~ Skipped '{item.raw_name}' (cannot prompt in non-interactive mode)[/dim]"
                            )
                            _store_unknown_result(index, item)
                            progress.advance(overall_task)
                            continue
                        progress.stop()
                        try:
                            should_research = typer.confirm(
                                (
                                    f"Research unknown biomarker '{item.raw_name}'"
                                    f" ({_format_value(item.value)} {item.unit})?"
                                ),
                                default=True,
                            )
                        finally:
                            progress.start()
                    if not should_research:
                        console.print(
                            f"[dim]~ Skipped '{item.raw_name}' (research declined)[/dim]"
                        )
                        _store_unknown_result(index, item)
                        progress.advance(overall_task)
                        continue

                    pending_key = resolution.research_key(item)
                    existing_group = pending_research_groups.get(pending_key)
                    if existing_group is not None:
                        logger.debug(
                            "Reusing in-flight research for '%s' at index %d",
                            item.raw_name,
                            index,
                        )
                        existing_group.jobs.append(
                            _PendingResearchJob(
                                index=index,
                                item=item,
                                research_key=pending_key,
                            )
                        )
                        continue

                    task = asyncio.create_task(
                        _run_limited_research(item)
                    )
                    pending_research_groups[pending_key] = _PendingResearchGroup(
                        key=pending_key,
                        item=item,
                        task=task,
                        jobs=[
                            _PendingResearchJob(
                                index=index,
                                item=item,
                                research_key=pending_key,
                            )
                        ],
                    )
                    continue

                if matched_entry and decision == "match":
                    matched_entry = await _maybe_assist_unit_conversion(
                        item, matched_entry
                    )
                    _assign_match(
                        index,
                        item,
                        matched_entry,
                        match_source=match_source,
                        match_score=match_score,
                    )
                else:
                    _store_unknown_result(index, item)

                progress.advance(overall_task)

            if pending_research_groups:
                research_remaining = len(pending_research_groups)
                progress.update(
                    overall_task,
                    status=f"awaiting {research_remaining} research result{'s' if research_remaining != 1 else ''}",
                )
                grouped_awaitables = [
                    _await_research_group(key, group)
                    for key, group in pending_research_groups.items()
                ]
                for completed_task in asyncio.as_completed(grouped_awaitables):
                    completed_key, new_entry = await completed_task
                    group = pending_research_groups[completed_key]
                    group_item = group.item
                    matched_entry: BiomarkerEntry | None = None

                    if new_entry:
                        existing_equivalent = db.find_match_for_entry(
                            biomarkers_db, new_entry
                        )
                        if existing_equivalent:
                            should_merge = False
                            merge_question = (
                                "Merge researched biomarker "
                                f"'{new_entry.id}' into existing "
                                f"'{existing_equivalent.id}'?"
                            )
                            merge_recommendation = (
                                await research_agent.recommend_merge_decision(
                                    new_entry=new_entry,
                                    existing_entry=existing_equivalent,
                                    observed_raw_name=group_item.raw_name,
                                    config=config,
                                    client=ai_client,
                                )
                            )
                            if merge_recommendation:
                                should_merge = bool(
                                    merge_recommendation.get("approved", False)
                                )
                                merge_label = "yes" if should_merge else "no"
                                merge_confidence = float(
                                    merge_recommendation.get("confidence", 0.0)
                                )
                                merge_reason = str(
                                    merge_recommendation.get("reason", "")
                                ).strip()
                                reason_suffix = (
                                    f" ({merge_reason})" if merge_reason else ""
                                )
                                console.print(
                                    f"[dim]~ AI decision [merge_researched_entry]: {merge_label} ({merge_confidence:.2f}){reason_suffix}[/dim]"
                                )
                            else:
                                console.print(
                                    "[dim]~ AI decision [merge_researched_entry] unavailable; using fallback 'no'.[/dim]"
                                )

                            if review_decisions:
                                if not sys.stdin.isatty():
                                    console.print(
                                        "[dim]~ Merge review requested but non-interactive mode; using AI decision.[/dim]"
                                    )
                                else:
                                    progress.stop()
                                    try:
                                        should_merge = typer.confirm(
                                            merge_question,
                                            default=should_merge,
                                        )
                                    finally:
                                        progress.start()

                            if should_merge:
                                biomarkers_db = await db.amerge_researched_entry(
                                    config.biomarkers_path,
                                    biomarkers_db,
                                    existing_equivalent.id,
                                    new_entry,
                                    group_item.raw_name,
                                )
                                console.print(
                                    f"[cyan]~[/cyan] Merged researched aliases into existing: {existing_equivalent.id}"
                                )
                                matched_entry = _entry_by_id(
                                    biomarkers_db, existing_equivalent.id
                                )
                            else:
                                biomarkers_db = await db.aappend_to_db(
                                    config.biomarkers_path, biomarkers_db, new_entry
                                )
                                console.print(
                                    f"[green]+[/green] Added as separate biomarker: {new_entry.id}"
                                )
                                matched_entry = (
                                    _entry_by_id(biomarkers_db, new_entry.id)
                                    or new_entry
                                )
                        else:
                            biomarkers_db = await db.aappend_to_db(
                                config.biomarkers_path, biomarkers_db, new_entry
                            )
                            console.print(f"[green]+[/green] Added: {new_entry.id}")
                            matched_entry = new_entry
                    else:
                        console.print(
                            f"[red]x[/red] Could not identify '{group_item.raw_name}'"
                        )

                    for job in group.jobs:
                        if matched_entry:
                            job_matched_entry = await _maybe_assist_unit_conversion(
                                job.item, matched_entry
                            )
                            _assign_match(
                                job.index,
                                job.item,
                                job_matched_entry,
                                match_source="research",
                                match_score=200.0,
                            )
                        else:
                            _store_unknown_result(job.index, job.item)
                        progress.advance(overall_task)
                    research_remaining -= 1
                    if research_remaining > 0:
                        progress.update(
                            overall_task,
                            status=f"awaiting {research_remaining} research result{'s' if research_remaining != 1 else ''}",
                        )

            if deferred_computed_items:
                progress.update(overall_task, status="computed biomarkers")
                for index, item in deferred_computed_items:
                    decision, matched_entry, _match_source, _match_score = (
                        await _resolve_entry_for_item(item, allow_computed=True)
                    )
                    if (
                        matched_entry is None
                        and decision == "research"
                        and research_enabled
                    ):
                        should_research = True
                        if ask_before_research and sys.stdin.isatty():
                            progress.stop()
                            try:
                                should_research = typer.confirm(
                                    (
                                        f"Research computed biomarker '{item.raw_name}'"
                                        f" ({_format_value(item.value)} {item.unit})?"
                                    ),
                                    default=True,
                                )
                            finally:
                                progress.start()
                        elif ask_before_research and not sys.stdin.isatty():
                            should_research = False

                        if should_research:
                            new_entry = await research_agent.research_biomarker(
                                item.raw_name,
                                config,
                                extracted_unit=item.unit,
                                client=ai_client,
                                item=item,
                                allow_computed=True,
                            )
                            if new_entry:
                                existing_equivalent = db.find_match_for_entry(
                                    biomarkers_db, new_entry
                                )
                                if existing_equivalent:
                                    biomarkers_db = await db.amerge_researched_entry(
                                        config.biomarkers_path,
                                        biomarkers_db,
                                        existing_equivalent.id,
                                        new_entry,
                                        item.raw_name,
                                    )
                                    matched_entry = _entry_by_id(
                                        biomarkers_db, existing_equivalent.id
                                    )
                                else:
                                    biomarkers_db = await db.aappend_to_db(
                                        config.biomarkers_path,
                                        biomarkers_db,
                                        new_entry,
                                    )
                                    matched_entry = (
                                        _entry_by_id(biomarkers_db, new_entry.id)
                                        or new_entry
                                    )

                    current_results = [
                        result
                        for result in [*analyzed_results, *supplemental_results]
                        if result is not None and result.biomarker_id != "unknown"
                    ]
                    if matched_entry:
                        outcome = computed_logic.compute_entry(
                            matched_entry,
                            current_results,
                        )
                        if outcome:
                            provenance = "computed"
                            notes = [
                                "Computed from " + ", ".join(outcome.dependencies)
                            ]
                            if computed_logic.values_match(
                                item.value,
                                outcome.value,
                                outcome.tolerance,
                            ):
                                provenance = "verified_computed"
                                notes.append("Observed report value verified")
                            analyzed = logic.analyze_value(
                                raw_name=item.raw_name,
                                raw_value=outcome.value,
                                raw_unit=matched_entry.canonical_unit,
                                entry=matched_entry,
                                sex=effective_sex,
                                age=effective_age,
                                specimen=item.specimen,
                                semantic_value=item.semantic_value,
                                measurement_qualifier=item.measurement_qualifier,
                                provenance=provenance,
                                derived_from=outcome.dependencies,
                            )
                            analyzed_results[index] = analyzed.model_copy(
                                update={"notes": "; ".join(notes)}
                            )
                        elif isinstance(item.value, (int, float)) and not isinstance(item.value, bool):
                            analyzed = logic.analyze_value(
                                raw_name=item.raw_name,
                                raw_value=item.value,
                                raw_unit=item.unit or matched_entry.canonical_unit,
                                entry=matched_entry,
                                sex=effective_sex,
                                age=effective_age,
                                specimen=item.specimen,
                                semantic_value=item.semantic_value,
                                measurement_qualifier=item.measurement_qualifier,
                                provenance="reported",
                            )
                            analyzed_results[index] = analyzed
                        else:
                            _store_unknown_result(index, item)
                    else:
                        _store_unknown_result(index, item)
                    progress.advance(overall_task)

            present_ids = {
                result.biomarker_id
                for result in [*analyzed_results, *supplemental_results]
                if result is not None and result.biomarker_id != "unknown"
            }
            computation_inputs = [
                result
                for result in [*analyzed_results, *supplemental_results]
                if result is not None and result.biomarker_id != "unknown"
            ]
            for entry in biomarkers_db:
                if entry.id in present_ids:
                    continue
                definition = entry.computed_definition
                if definition is None or not definition.compute_when_missing:
                    continue
                outcome = computed_logic.compute_entry(entry, computation_inputs)
                if outcome is None:
                    continue
                analyzed = logic.analyze_value(
                    raw_name=entry.id,
                    raw_value=outcome.value,
                    raw_unit=entry.canonical_unit,
                    entry=entry,
                    sex=effective_sex,
                    age=effective_age,
                    specimen=entry.specimen,
                    provenance="computed",
                    derived_from=outcome.dependencies,
                )
                analyzed = analyzed.model_copy(
                    update={
                        "notes": "Computed from " + ", ".join(outcome.dependencies),
                    }
                )
                supplemental_results.append(analyzed)
                computation_inputs.append(analyzed)
                present_ids.add(entry.id)

        analyzed_results = [
            result
            if result is not None
            else AnalyzedBiomarker(
                biomarker_id="unknown",
                display_name=extracted_items[index].raw_name,
                value=extracted_items[index].value,
                unit=extracted_items[index].unit,
                status="unknown",
            )
            for index, result in enumerate(analyzed_results)
        ]
        report_results = (
            analyzed_results + supplemental_results
            if show_skipped
            else [
                res
                for res in [*analyzed_results, *supplemental_results]
                if res.biomarker_id != "unknown"
            ]
        )
        skipped_count = len(analyzed_results) - len(
            [res for res in analyzed_results if res.biomarker_id != "unknown"]
        )

        # 4. Output
        console.print("[bold cyan]4/4[/bold cyan] Rendering report...")
        if skipped_count and not show_skipped:
            console.print(
                f"[dim]~ Hidden {skipped_count} skipped/unmatched row(s). Use --show-skipped to include them.[/dim]"
            )
        table = Table(
            title=f"Analysis Results • {Path(file_path).name}",
            box=box.SIMPLE_HEAVY,
            header_style="bold white",
            show_lines=False,
            row_styles=["", "dim"],
        )
        table.add_column("Biomarker", style="cyan")
        table.add_column("Value", style="magenta", justify="right")
        table.add_column("Unit", style="green")
        table.add_column("Reference", style="grey82")
        table.add_column("Optimal", style="grey82")
        table.add_column("Peak", style="grey70", justify="right")
        table.add_column("Status", style="bold", justify="center")
        table.add_column("ID", style="dim")

        for res in report_results:
            status_style = _status_style(res.status)
            table.add_row(
                res.display_name,
                _format_value(res.value),
                res.unit,
                _format_range(res.min_reference, res.max_reference),
                _format_range(res.min_optimal, res.max_optimal),
                _format_peak(res.peak_value),
                f"[{status_style}]{res.status}[/{status_style}]",
                res.biomarker_id,
            )

        console.print(table)
        _render_status_summary(report_results)

        if extraction_notes:
            notes_text = "\n".join([f"- {note}" for note in extraction_notes])
            console.print(
                Panel(
                    notes_text,
                    title="General Observations",
                    border_style="grey66",
                    padding=(0, 1),
                )
            )

        # 5. Export
        if output:
            try:
                if output.lower().endswith(".json"):
                    with open(output, "w") as f:
                        output_data = {
                            "biomarkers": [
                                {
                                    "biomarker_id": res.biomarker_id,
                                    "display_name": res.display_name,
                                    "value": res.value,
                                    "unit": res.unit,
                                    "status": res.status,
                                    "reference_status": res.reference_status,
                                    "optimal_status": res.optimal_status,
                                    "notes": res.notes,
                                    "min_reference": res.min_reference,
                                    "max_reference": res.max_reference,
                                    "min_optimal": res.min_optimal,
                                    "max_optimal": res.max_optimal,
                                    "peak_value": res.peak_value,
                                }
                                for res in report_results
                            ],
                            "notes": extraction_notes,
                            "metadata": extraction_metadata.model_dump(),
                        }
                        json.dump(output_data, f, indent=2)
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
                                "reference_status",
                                "optimal_status",
                                "notes",
                                "min_reference",
                                "max_reference",
                                "min_optimal",
                                "max_optimal",
                                "peak_value",
                            ]
                        )
                        for res in report_results:
                            writer.writerow(
                                [
                                    res.biomarker_id,
                                    res.display_name,
                                    res.value,
                                    res.unit,
                                    res.status,
                                    res.reference_status,
                                    res.optimal_status,
                                    res.notes or "",
                                    res.min_reference,
                                    res.max_reference,
                                    res.min_optimal,
                                    res.max_optimal,
                                    res.peak_value,
                                ]
                            )

                        writer.writerow([])
                        writer.writerow(["Metadata"])
                        writer.writerow(
                            ["patient.age", extraction_metadata.patient.age]
                        )
                        writer.writerow(
                            ["patient.gender", extraction_metadata.patient.gender]
                        )
                        writer.writerow(
                            ["lab.company_name", extraction_metadata.lab.company_name]
                        )
                        writer.writerow(
                            ["lab.location", extraction_metadata.lab.location]
                        )
                        writer.writerow(
                            [
                                "blood_collection.date",
                                extraction_metadata.blood_collection.date,
                            ]
                        )
                        writer.writerow(
                            [
                                "blood_collection.time",
                                extraction_metadata.blood_collection.time,
                            ]
                        )
                        writer.writerow(
                            [
                                "blood_collection.datetime",
                                extraction_metadata.blood_collection.datetime,
                            ]
                        )
                        writer.writerow(["analysis_context.age", effective_age])
                        writer.writerow(["analysis_context.sex", effective_sex])

                        if extraction_notes:
                            writer.writerow([])
                            writer.writerow(["General Notes"])
                            for note in extraction_notes:
                                writer.writerow([note])

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
