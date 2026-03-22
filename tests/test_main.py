import asyncio
from pathlib import Path
from typing import Awaitable

from typer.testing import CliRunner

from app.main import (
    app,
    _build_config,
    _can_use_exact_alias_match,
    _entries_equivalent,
    _entry_diff_rows,
    _has_unresolved_unit_mismatch,
    _high_confidence_candidate,
    _upsert_biomarker_entry,
)
from app.types import BiomarkerEntry, ExtractedBiomarker

runner = CliRunner()


def _entry(entry_id: str) -> BiomarkerEntry:
    return BiomarkerEntry(
        id=entry_id,
        aliases=[entry_id],
        canonical_unit="mg/dL",
        conversions={},
    )


def test_upsert_biomarker_entry_replaces_existing_match_and_dedupes_new_id():
    entries = [_entry("a"), _entry("b"), _entry("c")]
    refreshed = _entry("c")

    updated = _upsert_biomarker_entry(entries, refreshed, matched_id="b")

    ids = [item.id for item in updated]
    assert "b" not in ids
    assert ids.count("c") == 1
    assert ids.count("a") == 1


def test_upsert_biomarker_entry_appends_when_no_match():
    entries = [_entry("a")]
    refreshed = _entry("z")

    updated = _upsert_biomarker_entry(entries, refreshed)
    ids = [item.id for item in updated]

    assert ids == ["a", "z"]


def test_high_confidence_candidate_accepts_clear_winner():
    a = _entry("a")
    b = _entry("b")
    winner = _high_confidence_candidate(
        [
            (a, "alias-a", 97.0),
            (b, "alias-b", 88.0),
        ]
    )
    assert winner is not None
    assert winner.id == "a"


def test_high_confidence_candidate_rejects_close_scores():
    a = _entry("a")
    b = _entry("b")
    winner = _high_confidence_candidate(
        [
            (a, "alias-a", 96.0),
            (b, "alias-b", 94.0),
        ]
    )
    assert winner is None


def test_entries_equivalent_ignores_alias_order():
    a = BiomarkerEntry(
        id="hdl_cholesterol",
        aliases=["HDL-C", "HDL"],
        canonical_unit="mg/dL",
        conversions={},
    )
    b = BiomarkerEntry(
        id="hdl_cholesterol",
        aliases=["HDL", "HDL-C"],
        canonical_unit="mg/dL",
        conversions={},
    )

    assert _entries_equivalent(a, b)


def test_entry_diff_rows_reports_changed_fields():
    old = BiomarkerEntry(
        id="ldl_cholesterol",
        aliases=["LDL"],
        canonical_unit="mg/dL",
        max_normal=100.0,
        conversions={},
    )
    new = BiomarkerEntry(
        id="ldl_cholesterol",
        aliases=["LDL", "COLESTEROL LDL-PLUS"],
        canonical_unit="mg/dL",
        max_normal=90.0,
        conversions={},
    )

    rows = _entry_diff_rows(old, new)
    changed_fields = {field for field, _old, _new in rows}

    assert "aliases" in changed_fields
    assert "max_normal" in changed_fields


def test_has_unresolved_unit_mismatch_detects_unconvertible_quantitative_units():
    entry = BiomarkerEntry(
        id="thyroid_stimulating_hormone",
        aliases=["TSH"],
        canonical_unit="uIU/mL",
        min_normal=0.45,
        max_normal=4.5,
        conversions={},
    )
    extracted = ExtractedBiomarker(
        raw_name="TSH ULTRASENSIBLE, SANGRE",
        value=1.54,
        unit="uUI/mL",
    )

    assert _has_unresolved_unit_mismatch(extracted, entry) is True


def test_has_unresolved_unit_mismatch_is_false_when_formula_converts():
    entry = BiomarkerEntry(
        id="thyroid_stimulating_hormone",
        aliases=["TSH"],
        canonical_unit="uIU/mL",
        min_normal=0.45,
        max_normal=4.5,
        conversions={"uUI/mL": "x"},
    )
    extracted = ExtractedBiomarker(
        raw_name="TSH ULTRASENSIBLE, SANGRE",
        value=1.54,
        unit="uUI/mL",
    )

    assert _has_unresolved_unit_mismatch(extracted, entry) is False


def test_can_use_exact_alias_match_blocks_computed_labels():
    assert _can_use_exact_alias_match("AA/EPA #") is False
    assert _can_use_exact_alias_match("Omega 3 Index #") is False


def test_can_use_exact_alias_match_allows_direct_analyte_labels():
    assert _can_use_exact_alias_match("Omega 6 #") is True
    assert _can_use_exact_alias_match("Eicosapentaenoic Acid (EPA), C20:5 w3 #") is True


def test_build_config_expands_biomarkers_path_override():
    config = _build_config("~/custom-biomarkers.json")

    assert config.biomarkers_path == str(Path("~/custom-biomarkers.json").expanduser())


def test_cli_passes_biomarkers_path_to_analyze_flow(monkeypatch, tmp_path: Path):
    captured: dict[str, object] = {}
    biomarkers_path = tmp_path / "iterations" / "v2.json"

    async def fake_analyze_flow(
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
    ) -> None:
        captured["file_path"] = file_path
        captured["biomarkers_path"] = biomarkers_path

    def run_sync(coro: Awaitable[object]) -> None:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(coro)
        finally:
            loop.close()

    monkeypatch.setattr("app.main._analyze_flow", fake_analyze_flow)
    monkeypatch.setattr("app.main.asyncio.run", run_sync)

    result = runner.invoke(
        app,
        ["report.pdf", "--biomarkers-path", str(biomarkers_path)],
    )

    assert result.exit_code == 0
    assert captured == {
        "file_path": "report.pdf",
        "biomarkers_path": str(biomarkers_path),
    }


def test_cli_passes_biomarkers_path_to_reresearch_flow(monkeypatch, tmp_path: Path):
    captured: dict[str, object] = {}
    biomarkers_path = tmp_path / "iterations" / "v3.json"

    async def fake_reresearch_flow(
        biomarker_query: str,
        debug: bool = False,
        extracted_unit: str | None = None,
        dry_run: bool = False,
        biomarkers_path: str | None = None,
    ) -> None:
        captured["biomarker_query"] = biomarker_query
        captured["biomarkers_path"] = biomarkers_path

    def run_sync(coro: Awaitable[object]) -> None:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(coro)
        finally:
            loop.close()

    monkeypatch.setattr("app.main._reresearch_flow", fake_reresearch_flow)
    monkeypatch.setattr("app.main.asyncio.run", run_sync)

    result = runner.invoke(
        app,
        [
            "--reresearch-biomarker",
            "hdl_cholesterol",
            "--biomarkers-path",
            str(biomarkers_path),
        ],
    )

    assert result.exit_code == 0
    assert captured == {
        "biomarker_query": "hdl_cholesterol",
        "biomarkers_path": str(biomarkers_path),
    }
