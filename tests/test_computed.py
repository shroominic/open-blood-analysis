from app.computed import compute_entry, values_match
from app.types import AnalyzedBiomarker, BiomarkerEntry


def test_compute_entry_uses_dependencies_from_analyzed_results():
    entry = BiomarkerEntry(
        id="albumin_globulin_ratio",
        aliases=["A/G Ratio"],
        canonical_unit="ratio",
        kind="computed",
        representation="ratio",
        computed_definition={
            "dependencies": ["albumin", "globulin"],
            "formula": "albumin / globulin",
            "tolerance": 0.02,
        },
        conversions={},
    )
    analyzed = [
        AnalyzedBiomarker(
            biomarker_id="albumin",
            display_name="Albumin",
            value=4.7,
            unit="g/dL",
        ),
        AnalyzedBiomarker(
            biomarker_id="globulin",
            display_name="Globulin",
            value=2.3,
            unit="g/dL",
        ),
    ]

    outcome = compute_entry(entry, analyzed)

    assert outcome is not None
    assert outcome.value == 4.7 / 2.3
    assert outcome.dependencies == ["albumin", "globulin"]


def test_values_match_uses_relative_tolerance():
    assert values_match(1.81, 1.8, 0.02) is True
    assert values_match(2.2, 1.8, 0.02) is False

