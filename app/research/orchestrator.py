from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, TypeVar

from app.ai_client import AIClient
from app.config import Config, ResearchBackendSpec
from app.research.backends import (
    DelegatingAIClientResearchBackend,
    GeminiResearchBackend,
    OpenAICompatibleResearchBackend,
    PerplexityResearchBackend,
    ResearchBackend,
)
from app.research.tasks import (
    disambiguate_biomarker as disambiguate_task,
    recommend_binary_decision as binary_decision_task,
    recommend_merge_decision as merge_decision_task,
    research_biomarker as biomarker_research_task,
    think_unit_conversion as unit_conversion_task,
)
from app.types import BiomarkerEntry, ExtractedBiomarker

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _model_for_task(config: Config, task_name: str) -> str:
    if task_name == "biomarker_research":
        return config.research
    if task_name == "disambiguation":
        return config.ocr
    return config.thinking


def _build_backend_from_spec(spec: ResearchBackendSpec) -> ResearchBackend:
    backend_id = spec.resolved_id()
    api_key = spec.resolved_api_key()
    if not api_key:
        raise ValueError(f"API key is required for research backend '{backend_id}'.")

    if spec.type == "gemini":
        return GeminiResearchBackend(backend_id=backend_id, api_key=api_key)

    base_url = spec.resolved_base_url()
    if not base_url:
        raise ValueError(f"base_url is required for research backend '{backend_id}'.")

    if spec.type == "perplexity":
        return PerplexityResearchBackend(
            backend_id=backend_id,
            api_key=api_key,
            base_url=base_url,
            supports_web_search=spec.resolved_supports_web_search(),
        )

    return OpenAICompatibleResearchBackend(
        backend_id=backend_id,
        api_key=api_key,
        base_url=base_url,
        supports_web_search=spec.resolved_supports_web_search(),
    )


class ResearchOrchestrator:
    def __init__(
        self,
        *,
        config: Config,
        backends: list[ResearchBackend] | None = None,
    ) -> None:
        self._config = config
        self._backends = (
            list(backends) if backends is not None
            else [_build_backend_from_spec(s) for s in config.resolved_research_backends]
        )

    def _active_backends(
        self,
        *,
        client: AIClient | None = None,
        require_web_search: bool = False,
    ) -> list[ResearchBackend]:
        if client is not None:
            provider = self._config.ai_provider
            return [
                DelegatingAIClientResearchBackend(
                    backend_id=provider,
                    client=client,
                    supports_web_search=provider in {"gemini"},
                )
            ]

        backends = self._backends
        if require_web_search:
            search_backends = [b for b in backends if b.supports_web_search]
            if search_backends:
                return search_backends
        return backends

    async def _run_with_strategy(
        self,
        *,
        task_name: str,
        fn: Callable[[ResearchBackend, str], Awaitable[T]],
        client: AIClient | None = None,
        require_web_search: bool = False,
    ) -> T:
        backends = self._active_backends(
            client=client,
            require_web_search=require_web_search,
        )
        if not backends:
            raise ValueError("No research backends are configured.")

        model = _model_for_task(self._config, task_name)
        if self._config.research_strategy == "primary" or len(backends) == 1:
            return await fn(backends[0], model)

        last_exc: Exception | None = None
        for backend in backends:
            try:
                return await fn(backend, model)
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Research task '%s' failed on backend '%s': %s",
                    task_name,
                    backend.backend_id,
                    exc,
                )
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"Research task '{task_name}' did not return a result.")

    async def disambiguate_biomarker(
        self,
        *,
        raw_name: str,
        candidates: list[tuple[BiomarkerEntry, str, float]],
        item: ExtractedBiomarker | None = None,
        allow_computed: bool = False,
        client: AIClient | None = None,
    ) -> tuple[str, BiomarkerEntry | None]:
        return await self._run_with_strategy(
            task_name="disambiguation",
            client=client,
            fn=lambda backend, model: disambiguate_task(
                backend=backend, model=model,
                raw_name=raw_name, candidates=candidates,
                item=item, allow_computed=allow_computed,
            ),
        )

    async def think_unit_conversion(
        self,
        *,
        biomarker_name: str,
        biomarker_id: str,
        from_unit: str,
        canonical_unit: str,
        observed_value: float | str | bool,
        client: AIClient | None = None,
    ) -> dict[str, object] | None:
        return await self._run_with_strategy(
            task_name="unit_conversion",
            client=client,
            fn=lambda backend, model: unit_conversion_task(
                backend=backend, model=model,
                biomarker_name=biomarker_name, biomarker_id=biomarker_id,
                from_unit=from_unit, canonical_unit=canonical_unit,
                observed_value=observed_value,
            ),
        )

    async def recommend_binary_decision(
        self,
        *,
        decision_name: str,
        question: str,
        context: dict[str, Any],
        client: AIClient | None = None,
    ) -> dict[str, Any] | None:
        return await self._run_with_strategy(
            task_name="binary_decision",
            client=client,
            fn=lambda backend, model: binary_decision_task(
                backend=backend, model=model,
                decision_name=decision_name, question=question,
                context=context,
            ),
        )

    async def recommend_merge_decision(
        self,
        *,
        new_entry: BiomarkerEntry,
        existing_entry: BiomarkerEntry,
        observed_raw_name: str,
        client: AIClient | None = None,
    ) -> dict[str, Any] | None:
        return await self._run_with_strategy(
            task_name="merge_decision",
            client=client,
            fn=lambda backend, model: merge_decision_task(
                backend=backend, model=model,
                new_entry=new_entry, existing_entry=existing_entry,
                observed_raw_name=observed_raw_name,
            ),
        )

    async def research_biomarker(
        self,
        *,
        biomarker_name: str,
        extracted_unit: str | None = None,
        item: ExtractedBiomarker | None = None,
        allow_computed: bool = False,
        client: AIClient | None = None,
    ) -> BiomarkerEntry | None:
        return await self._run_with_strategy(
            task_name="biomarker_research",
            client=client,
            require_web_search=True,
            fn=lambda backend, model: biomarker_research_task(
                backend=backend, model=model,
                biomarker_name=biomarker_name, extracted_unit=extracted_unit,
                item=item, allow_computed=allow_computed,
            ),
        )
