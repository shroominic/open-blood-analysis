from __future__ import annotations

import json
import logging

from app.ai_client import retry_async
from app.research.backends.base import ResearchBackend
from app.research.tasks.common import extract_json_payload

logger = logging.getLogger(__name__)


async def think_unit_conversion(
    *,
    backend: ResearchBackend,
    model: str,
    biomarker_name: str,
    biomarker_id: str,
    from_unit: str,
    canonical_unit: str,
    observed_value: float | str | bool,
) -> dict[str, object] | None:
    prompt = f"""
    You are a unit-conversion assistant for a blood biomarker pipeline.

    Context:
    - Raw biomarker label: "{biomarker_name}"
    - Biomarker ID: "{biomarker_id}"
    - Observed unit: "{from_unit}"
    - Canonical unit: "{canonical_unit}"
    - Example observed value: {observed_value!r}

    Task:
    Propose whether we should add a conversion mapping from observed unit to canonical unit.
    The formula must convert INPUT -> CANONICAL using variable x.

    Output JSON schema:
    {{
      "action": "add_conversion" | "no_conversion",
      "input_unit": "string",
      "canonical_unit": "string",
      "formula": "string using x or empty",
      "confidence": 0.0,
      "reason": "short reason"
    }}

    Rules:
    1. If units are equivalent notation variants (e.g. IU ordering variants), formula can be "x".
    2. If uncertain, use action "no_conversion" and keep formula empty.
    3. Keep reason short (one sentence).
    4. Return ONLY valid JSON.
    """

    try:
        content = extract_json_payload(
            await retry_async(
                backend.generate_json,
                model=model,
                prompt=prompt,
            )
            or ""
        )
        if not content:
            return None
        data = json.loads(content)
        if not isinstance(data, dict):
            return None
        return data
    except Exception as exc:
        logger.error("Failed unit conversion thinking step: %s", exc)
        return None
