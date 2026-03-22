from pathlib import Path

from app import database as db
from app.types import BiomarkerEntry, LearnedContextAlias, LearnedValueAlias


def _entry(entry_id: str, aliases: list[str]) -> BiomarkerEntry:
    return BiomarkerEntry(
        id=entry_id,
        aliases=aliases,
        canonical_unit="mg/dL",
        conversions={},
    )


def test_find_exact_match_is_accent_and_case_tolerant():
    entries = [_entry("triglycerides", ["triglicéridos", "TG"])]

    match = db.find_exact_match(entries, "TRIGLICERIDOS")

    assert match is not None
    assert match.id == "triglycerides"


def test_find_exact_match_can_ignore_aliases():
    entries = [_entry("eicosapentaenoic_acid", ["Eicosapentaenoic Acid (EPA), C20:5 w3 #"])]

    no_alias_match = db.find_exact_match(
        entries,
        "Eicosapentaenoic Acid (EPA), C20:5 w3 #",
        include_aliases=False,
    )
    with_alias_match = db.find_exact_match(
        entries,
        "Eicosapentaenoic Acid (EPA), C20:5 w3 #",
        include_aliases=True,
    )

    assert no_alias_match is None
    assert with_alias_match is not None
    assert with_alias_match.id == "eicosapentaenoic_acid"


def test_find_fuzzy_candidates_uses_normalized_labels():
    entries = [_entry("triglycerides", ["triglicéridos", "TG"])]

    candidates = db.find_fuzzy_candidates(
        entries, "TRIGLICERIDOS", top_n=3, min_score=70
    )

    assert candidates
    assert candidates[0][0].id == "triglycerides"


def test_add_alias_to_entry_updates_once_with_normalized_dedup(tmp_path: Path):
    path = tmp_path / "biomarkers.json"
    entries = [_entry("hdl_cholesterol", ["HDL"])]
    db.save_db(str(path), entries)

    updated = db.add_alias_to_entry(str(path), entries, "hdl_cholesterol", "hdl")
    still_single = next(e for e in updated if e.id == "hdl_cholesterol")
    assert still_single.aliases == ["HDL"]

    updated = db.add_alias_to_entry(
        str(path), updated, "hdl_cholesterol", "COLESTEROL HDL"
    )
    refreshed = next(e for e in updated if e.id == "hdl_cholesterol")
    assert "COLESTEROL HDL" in refreshed.aliases


def test_find_match_for_entry_prefers_existing_equivalent():
    existing = _entry("hdl_cholesterol", ["High-density lipoprotein cholesterol"])
    candidate = _entry(
        "high_density_lipoprotein",
        ["HDL", "High-density lipoprotein cholesterol"],
    )

    match = db.find_match_for_entry([existing], candidate)

    assert match is not None
    assert match.id == "hdl_cholesterol"


def test_merge_researched_entry_adds_aliases_instead_of_new_id(tmp_path: Path):
    path = tmp_path / "biomarkers.json"
    existing = _entry("hdl_cholesterol", ["HDL"])
    db.save_db(str(path), [existing])

    researched = _entry(
        "high_density_lipoprotein",
        ["COLESTEROL HDL", "High-density lipoprotein cholesterol"],
    )
    merged = db.merge_researched_entry(
        str(path),
        [existing],
        "hdl_cholesterol",
        researched,
        observed_raw_name="COLESTEROL HDL",
    )

    final = next(e for e in merged if e.id == "hdl_cholesterol")
    assert "COLESTEROL HDL" in final.aliases
    assert "high_density_lipoprotein" in final.aliases


def test_add_context_alias_to_entry_persists_specimen_and_unit(tmp_path: Path):
    path = tmp_path / "biomarkers.json"
    entries = [_entry("urine_bilirubin", ["Bilirubin"])]
    db.save_db(str(path), entries)

    updated = db.add_context_alias_to_entry(
        str(path),
        entries,
        "urine_bilirubin",
        LearnedContextAlias(
            raw_name="Bilirubin",
            raw_unit="",
            specimen="urine",
            representation="boolean",
        ),
    )

    refreshed = next(e for e in updated if e.id == "urine_bilirubin")
    assert refreshed.learned_context_aliases[0].specimen == "urine"


def test_add_value_alias_to_entry_persists_semantic_mapping(tmp_path: Path):
    path = tmp_path / "biomarkers.json"
    entries = [_entry("nitrite", ["Nitrite"])]
    db.save_db(str(path), entries)

    updated = db.add_value_alias_to_entry(
        str(path),
        entries,
        "nitrite",
        LearnedValueAlias(raw_value="NEGATIF", semantic_value="negative"),
    )

    refreshed = next(e for e in updated if e.id == "nitrite")
    assert refreshed.learned_value_aliases[0].semantic_value == "negative"


def test_save_db_creates_missing_parent_directories(tmp_path: Path):
    path = tmp_path / "iterations" / "candidate" / "biomarkers.v2.json"

    db.save_db(str(path), [_entry("hdl_cholesterol", ["HDL"])])

    assert path.exists()
