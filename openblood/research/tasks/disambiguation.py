from __future__ import annotations

import json
import logging
from typing import Literal

from openblood.ai_client import retry_async
from openblood.research.backends.base import ResearchBackend
from openblood.types import BiomarkerEntry, ExtractedBiomarker

logger = logging.getLogger(__name__)


async def disambiguate_biomarker(
    *,
    backend: ResearchBackend,
    model: str,
    raw_name: str,
    candidates: list[tuple[BiomarkerEntry, str, float]],
    item: ExtractedBiomarker | None = None,
    allow_computed: bool = False,
) -> tuple[Literal["match", "research", "unknown"], BiomarkerEntry | None]:
    candidates_text = "\n".join(
        [
            (
                f"  {index + 1}. ID: '{entry.id}' "
                f"(matched via '{match_str}', score={score:.1f}, "
                f"unit='{entry.canonical_unit}', type='{entry.value_type}', "
                f"kind='{entry.kind}', specimen='{entry.specimen}', representation='{entry.representation}', "
                f"aliases={entry.aliases[:6]})"
            )
            for index, (entry, match_str, score) in enumerate(candidates)
        ]
    )
    row_context: dict[str, object] = {}
    if item is not None:
        row_context = {
            "raw_name": item.raw_name,
            "value": item.value,
            "unit": item.unit,
            "specimen": item.specimen,
            "measurement_qualifier": item.measurement_qualifier,
            "semantic_value": item.semantic_value,
            "is_computed_candidate": item.is_computed_candidate,
        }

    prompt = f"""You are a strict clinical biomarker disambiguation reviewer.

Raw biomarker name from a lab report: "{raw_name}"
Row context JSON: {json.dumps(row_context, ensure_ascii=False, sort_keys=True)}

These are possible matches from our database:
{candidates_text if candidates else "  (No candidates found)"}

Decide the best action:
1. If one of the candidates is clearly the SAME biomarker (just different spelling/language), respond with: {{"action": "match", "index": <1-based index>}}
2. If none match and this is a real biomarker we should add to our database, respond with: {{"action": "research"}}
3. If this is NOT a biomarker (e.g., a date, patient name, company header), respond with: {{"action": "unknown"}}

Strict safety rules:
- If uncertain, choose "unknown" (do NOT force a match).
- Respect specimen and representation differences. Blood vs urine and percent vs absolute-count are NOT interchangeable.
- Qualitative urine analytes must not be matched to quantitative blood analytes.
- If the raw name is composite/ratio/index style (e.g., has "/" between markers, or contains words like ratio/index/risk), choose "unknown" unless a candidate is explicitly that same computed biomarker.
- Never map a composite label to a single direct analyte candidate (example: "AA/EPA" must not map to "eicosapentaenoic_acid").
- Short acronym overlap alone (like EPA in AA/EPA) is insufficient for a match.
- {"Computed biomarkers may be matched or researched when the row context indicates a computed candidate." if allow_computed else "Do NOT match or research computed biomarkers in this mode."}

Respond with ONLY valid JSON, no explanation."""

    try:
        content = await retry_async(
            backend.generate_json,
            model=model,
            prompt=prompt,
        )
        if not content:
            return ("research", None)

        data = json.loads(content)
        action = data.get("action", "research")

        if action == "match" and candidates:
            idx = data.get("index", 1) - 1
            if 0 <= idx < len(candidates):
                logger.debug("AI matched '%s' to '%s'", raw_name, candidates[idx][0].id)
                return ("match", candidates[idx][0])
            return ("research", None)
        if action == "unknown":
            logger.debug("AI marked '%s' as unknown/skip", raw_name)
            return ("unknown", None)

        logger.debug("AI requested research for '%s'", raw_name)
        return ("research", None)
    except Exception as exc:
        logger.error("Disambiguation failed: %s", exc)
        return ("research", None)
