from app import resolution
from app.types import BiomarkerEntry, ComputedDefinition, ExtractedBiomarker


def _entry(
    entry_id: str,
    *,
    specimen: str | None = None,
    representation: str | None = None,
    value_type: str = "quantitative",
    kind: str = "direct",
) -> BiomarkerEntry:
    return BiomarkerEntry(
        id=entry_id,
        aliases=[entry_id],
        canonical_unit="%" if representation == "percent" else "10^9/L",
        specimen=specimen,
        representation=representation,
        value_type=value_type,  # type: ignore[arg-type]
        kind=kind,  # type: ignore[arg-type]
        conversions={},
        computed_definition=(
            ComputedDefinition(
                dependencies=["a", "b"],
                formula="a / b",
            )
            if kind == "computed"
            else None
        ),
    )


def test_is_entry_compatible_rejects_percent_to_absolute_count():
    entry = _entry(
        "neutrophils_absolute_count",
        specimen="blood",
        representation="absolute_count",
    )
    item = ExtractedBiomarker(raw_name="NEUTROPHIL", value=57.3, unit="%")

    assert resolution.is_entry_compatible(entry, item) is False


def test_is_entry_compatible_rejects_qualitative_urine_to_serum_quantitative():
    entry = _entry(
        "serum_bilirubin",
        specimen="serum",
        representation="quantitative",
    )
    item = ExtractedBiomarker(
        raw_name="Bilirubin",
        value="NEGATIVE",
        unit="",
        specimen="urine",
    )

    assert resolution.is_entry_compatible(entry, item) is False


def test_is_computed_candidate_detects_ratio_style_names():
    item = ExtractedBiomarker(raw_name="A/G Ratio", value=1.8, unit="")

    assert resolution.is_computed_candidate(item) is True

