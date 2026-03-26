from app.agent import (
    _extract_json_payload,
    _fallback_biomarker_from_context,
    _parse_binary_decision_payload,
    _sanitize_research_payload,
)
from app.types import ExtractedBiomarker


def test_sanitize_research_payload_nulls_redundant_optimal_range():
    payload = {
        "id": "x",
        "canonical_unit": "mg/dL",
        "min_normal": 10.0,
        "max_normal": 20.0,
        "min_optimal": 10.0,
        "max_optimal": 20.0,
    }

    sanitized = _sanitize_research_payload(payload)

    assert sanitized["min_optimal"] is None
    assert sanitized["max_optimal"] is None


def test_extract_json_payload_unwraps_markdown_fence():
    payload = "```json\n{\"action\":\"add_conversion\"}\n```"
    assert _extract_json_payload(payload) == '{"action":"add_conversion"}'


def test_parse_binary_decision_payload_accepts_yes():
    parsed = _parse_binary_decision_payload(
        {
            "decision": "yes",
            "confidence": 1.2,
            "reason": "clear equivalence",
        }
    )
    assert parsed is not None
    assert parsed["approved"] is True
    assert parsed["decision"] == "yes"
    assert parsed["confidence"] == 1.0


def test_parse_binary_decision_payload_rejects_invalid_decision():
    parsed = _parse_binary_decision_payload(
        {
            "decision": "maybe",
            "confidence": 0.5,
            "reason": "uncertain",
        }
    )
    assert parsed is None


def test_sanitize_research_payload_salvages_qualitative_urine_entry():
    payload = {
        "id": "bilirubin",
        "canonical_unit": None,
        "value_type": "quantitative",
        "learned_value_aliases": [{"alias": "NEG", "value": "negative"}],
    }
    item = ExtractedBiomarker(
        raw_name="Bilirubin",
        value="NEGATIVE",
        unit="",
        specimen="urine",
    )

    sanitized = _sanitize_research_payload(payload, extracted_unit="", item=item)

    assert sanitized["id"] == "urine_bilirubin"
    assert sanitized["canonical_unit"] == ""
    assert sanitized["specimen"] == "urine"
    assert sanitized["value_type"] == "enum"
    assert sanitized["learned_value_aliases"][0]["raw_value"] == "NEG"
    assert sanitized["learned_value_aliases"][0]["semantic_value"] == "negative"


def test_sanitize_research_payload_forces_quantitative_interpretation_for_numeric_entries():
    payload = {
        "id": "hba1c_blood",
        "canonical_unit": "MMOL/MOL",
        "value_type": "quantitative",
        "interpretation": {
            "kind": "categorical_labels",
            "label_map": {"below_42": "Normal"},
            "ordered_values": ["below_42"],
        },
    }

    sanitized = _sanitize_research_payload(payload)

    assert sanitized["interpretation"]["kind"] == "quantitative_range"
    assert sanitized["interpretation"]["label_map"] == {}


def test_sanitize_research_payload_defaults_missing_value_type_for_qualitative_rows():
    item = ExtractedBiomarker(
        raw_name="Blood",
        value="NEGATIVE",
        unit="",
        specimen="urine",
    )
    payload = {
        "id": "blood",
        "canonical_unit": None,
        "value_type": None,
    }

    sanitized = _sanitize_research_payload(payload, item=item)

    assert sanitized["value_type"] == "enum"
    assert sanitized["id"] == "urine_blood"


def test_sanitize_research_payload_rewrites_blank_unit_ph_to_urine_ph():
    item = ExtractedBiomarker(
        raw_name="pH",
        value=7.5,
        unit="",
        specimen=None,
    )
    payload = {
        "id": "blood_ph",
        "canonical_unit": "pH",
        "value_type": "quantitative",
    }

    sanitized = _sanitize_research_payload(payload, item=item)

    assert sanitized["id"] == "urine_ph"
    assert sanitized["specimen"] == "urine"


def test_fallback_biomarker_from_context_creates_urine_ph_entry():
    item = ExtractedBiomarker(
        raw_name="pH",
        value=7.5,
        unit="",
        specimen="urine",
    )

    fallback = _fallback_biomarker_from_context("pH", item)

    assert fallback is not None
    assert fallback.id == "urine_ph"
    assert fallback.min_normal == 4.5
    assert fallback.max_normal == 8.0


def test_fallback_biomarker_from_context_creates_urine_bilirubin_entry():
    item = ExtractedBiomarker(
        raw_name="Bilirubin",
        value="NEGATIVE",
        unit="",
        specimen="urine",
    )

    fallback = _fallback_biomarker_from_context("Bilirubin", item)

    assert fallback is not None
    assert fallback.id == "urine_bilirubin"
    assert fallback.value_type == "enum"
