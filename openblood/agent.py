from __future__ import annotations

from typing import Any, Literal

from .ai_client import AIClient
from .config import Config
from .research.orchestrator import ResearchOrchestrator
from .research.tasks.biomarker_research import (
    fallback_biomarker_from_context as _fallback_biomarker_from_context,
    sanitize_research_payload as _sanitize_research_payload,
)
from .research.tasks.common import (
    extract_json_payload as _extract_json_payload,
    parse_binary_decision_payload as _parse_binary_decision_payload,
)
from .types import BiomarkerEntry, ExtractedBiomarker


def _orchestrator(config: Config) -> ResearchOrchestrator:
    return ResearchOrchestrator(config=config)


async def disambiguate_biomarker(
    raw_name: str,
    candidates: list[tuple[BiomarkerEntry, str, float]],
    config: Config,
    client: AIClient | None = None,
    item: ExtractedBiomarker | None = None,
    allow_computed: bool = False,
) -> tuple[Literal["match", "research", "unknown"], BiomarkerEntry | None]:
    return await _orchestrator(config).disambiguate_biomarker(
        raw_name=raw_name,
        candidates=candidates,
        item=item,
        allow_computed=allow_computed,
        client=client,
    )


async def think_unit_conversion(
    biomarker_name: str,
    biomarker_id: str,
    from_unit: str,
    canonical_unit: str,
    observed_value: float | str | bool,
    config: Config,
    client: AIClient | None = None,
) -> dict[str, object] | None:
    return await _orchestrator(config).think_unit_conversion(
        biomarker_name=biomarker_name,
        biomarker_id=biomarker_id,
        from_unit=from_unit,
        canonical_unit=canonical_unit,
        observed_value=observed_value,
        client=client,
    )


async def recommend_binary_decision(
    *,
    decision_name: str,
    question: str,
    context: dict[str, Any],
    config: Config,
    client: AIClient | None = None,
) -> dict[str, Any] | None:
    return await _orchestrator(config).recommend_binary_decision(
        decision_name=decision_name,
        question=question,
        context=context,
        client=client,
    )


async def recommend_merge_decision(
    *,
    new_entry: BiomarkerEntry,
    existing_entry: BiomarkerEntry,
    observed_raw_name: str,
    config: Config,
    client: AIClient | None = None,
) -> dict[str, Any] | None:
    return await _orchestrator(config).recommend_merge_decision(
        new_entry=new_entry,
        existing_entry=existing_entry,
        observed_raw_name=observed_raw_name,
        client=client,
    )


async def research_biomarker(
    biomarker_name: str,
    config: Config,
    extracted_unit: str | None = None,
    client: AIClient | None = None,
    item: ExtractedBiomarker | None = None,
    allow_computed: bool = False,
) -> BiomarkerEntry | None:
    return await _orchestrator(config).research_biomarker(
        biomarker_name=biomarker_name,
        extracted_unit=extracted_unit,
        item=item,
        allow_computed=allow_computed,
        client=client,
    )
