from __future__ import annotations

import json
import logging
from typing import Any

from app.ai_client import retry_async
from app.research.backends.base import ResearchBackend
from app.research.tasks.common import extract_json_payload, parse_binary_decision_payload

logger = logging.getLogger(__name__)


async def recommend_binary_decision(
    *,
    backend: ResearchBackend,
    model: str,
    decision_name: str,
    question: str,
    context: dict[str, Any],
) -> dict[str, Any] | None:
    context_json = json.dumps(context, ensure_ascii=False, sort_keys=True)
    prompt = f"""
    You are a cautious reviewer for a blood biomarker automation pipeline.

    Decision name: "{decision_name}"
    Question: "{question}"
    Context JSON:
    {context_json}

    Respond with JSON only using:
    {{
      "decision": "yes" | "no",
      "confidence": 0.0,
      "reason": "short reason"
    }}

    Rules:
    1. Choose "yes" only if context strongly supports it.
    2. If uncertain, choose "no".
    3. Keep reason to one sentence.
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
        logger.error(
            "Failed binary decision recommendation (%s): %s",
            decision_name,
            exc,
        )
        return None
