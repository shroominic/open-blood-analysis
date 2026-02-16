from app.agent import (
    _extract_json_payload,
    _parse_binary_decision_payload,
    _sanitize_research_payload,
)


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
