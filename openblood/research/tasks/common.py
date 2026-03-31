from __future__ import annotations

from typing import Any


def extract_json_payload(content: str) -> str:
    text = (content or "").strip()
    if "```json" in text:
        return text.split("```json", 1)[1].split("```", 1)[0].strip()
    if "```" in text:
        return text.split("```", 1)[1].split("```", 1)[0].strip()
    return text


def parse_binary_decision_payload(data: dict[str, Any]) -> dict[str, Any] | None:
    decision = str(data.get("decision", "")).strip().lower()
    if decision not in {"yes", "no"}:
        return None

    confidence_raw = data.get("confidence", 0.0)
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    reason = str(data.get("reason", "")).strip()
    return {
        "decision": decision,
        "approved": decision == "yes",
        "confidence": confidence,
        "reason": reason,
    }
