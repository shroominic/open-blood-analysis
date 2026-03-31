from __future__ import annotations

import json
import logging
from typing import Any

from openblood.ai_client import retry_async
from openblood.research.backends.base import ResearchBackend
from openblood.research.tasks.common import extract_json_payload, parse_binary_decision_payload
from openblood.types import BiomarkerEntry

logger = logging.getLogger(__name__)


async def recommend_merge_decision(
    *,
    backend: ResearchBackend,
    model: str,
    new_entry: BiomarkerEntry,
    existing_entry: BiomarkerEntry,
    observed_raw_name: str,
) -> dict[str, Any] | None:
    context = {
        "observed_raw_name": observed_raw_name,
        "new_entry": new_entry.model_dump(),
        "existing_entry": existing_entry.model_dump(),
    }
    context_json = json.dumps(context, ensure_ascii=False, sort_keys=True)
    prompt = f"""
    You are a strict clinical ontology reviewer for blood biomarkers.
    Your job is to decide whether two biomarker entries are the SAME analyte and can be merged.

    You must be highly conservative:
    - If uncertain, respond NO.
    - Similar spelling is not enough.
    - Similar units are not enough.
    - Overlapping aliases are not enough unless semantics are clearly identical.

    Data:
    {context_json}

    Hard block rules (must be NO merge):
    1. HDL vs LDL are NEVER mergeable.
    2. ApoA1 vs ApoB are NEVER mergeable.
    3. Different analyte families/classes (lipid fraction vs liver enzyme vs hormone) are NEVER mergeable.
    4. Different canonical biological meaning despite lexical similarity (e.g. total vs free when distinct analytes) are NEVER mergeable.
    5. Ratios/indexes must not merge with direct measured analytes.

    Merge only if all are true:
    1. Canonical analyte meaning is identical.
    2. Aliases are true multilingual/synonym variants of the same analyte.
    3. Value type compatibility is consistent (quantitative/boolean/enum).
    4. Canonical unit difference is explainable as notation or valid conversion for same analyte.
    5. No contradiction in description or reference-range semantics.

    Respond with JSON only:
    {{
      "decision": "yes" | "no",
      "confidence": 0.0,
      "reason": "short reason focused on analyte identity"
    }}
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
        return parse_binary_decision_payload(data)
    except Exception as exc:
        logger.error("Failed merge recommendation: %s", exc)
        return None
