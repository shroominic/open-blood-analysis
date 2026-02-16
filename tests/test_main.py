from app.main import (
    _can_use_exact_alias_match,
    _entries_equivalent,
    _entry_diff_rows,
    _has_unresolved_unit_mismatch,
    _high_confidence_candidate,
    _upsert_biomarker_entry,
)
from app.types import BiomarkerEntry, ExtractedBiomarker


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
