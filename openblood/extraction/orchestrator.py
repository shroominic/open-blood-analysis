from __future__ import annotations

import asyncio

from app.config import Config
from app.extraction.engines.base import ExtractionEngine
from app.extraction.engines.gemini_vision import GeminiVisionEngine
from app.extraction.engines.liteparse_text import LiteParseTextEngine
from app.extraction.engines.openai_compatible_vision import (
    OpenAICompatibleVisionEngine,
)
from app.extraction.fusion import fuse_engine_results
from app.extraction.types import (
    EngineBiomarkerCandidate,
    EngineExtractionResult,
    ExtractionPipelineResult,
    PageArtifact,
)
from app.types import ReportMetadata


def _build_page_artifacts(image_paths: list[str]) -> list[PageArtifact]:
    return [
        PageArtifact(page_num=index + 1, image_path=image_path)
        for index, image_path in enumerate(image_paths)
    ]


def _build_engines(config: Config) -> list[ExtractionEngine]:
    engines: list[ExtractionEngine] = []
    for spec in config.resolved_extraction_engines:
        if spec.type == "gemini_vision":
            engines.append(
                GeminiVisionEngine(
                    engine_id=spec.resolved_id(),
                    execution_mode=spec.execution_mode,
                    model=spec.model or config.ocr,
                    weight=spec.weight,
                )
            )
            continue

        if spec.type == "openai_compatible_vision":
            api_key = spec.resolved_api_key()
            if not api_key:
                raise ValueError(
                    f"API key is required for extraction engine '{spec.resolved_id()}'."
                )
            if not spec.base_url:
                raise ValueError(
                    f"base_url is required for extraction engine '{spec.resolved_id()}'."
                )
            if not spec.model:
                raise ValueError(
                    f"model is required for extraction engine '{spec.resolved_id()}'."
                )
            engines.append(
                OpenAICompatibleVisionEngine(
                    engine_id=spec.resolved_id(),
                    execution_mode=spec.execution_mode,
                    model=spec.model,
                    base_url=spec.base_url,
                    api_key=api_key,
                    headers=spec.headers,
                    weight=spec.weight,
                )
            )
            continue

        if spec.type == "liteparse_text":
            engines.append(
                LiteParseTextEngine(
                    engine_id=spec.resolved_id(),
                    cli_path=spec.cli_path,
                    model=spec.model or config.ocr,
                    execution_mode=spec.execution_mode,
                    weight=spec.weight,
                )
            )
            continue

        raise ValueError(f"Unsupported extraction engine type: {spec.type}")

    return engines


def _merge_notes(results: list[EngineExtractionResult]) -> list[str]:
    merged_notes: list[str] = []
    seen_notes: set[str] = set()
    for result in results:
        for note in result.notes:
            normalized_note = note.strip()
            if not normalized_note or normalized_note in seen_notes:
                continue
            seen_notes.add(normalized_note)
            merged_notes.append(normalized_note)
    return merged_notes


def _pick_text(*values: str | None) -> str | None:
    for value in values:
        if value is None:
            continue
        normalized = value.strip()
        if normalized:
            return normalized
    return None


def _merge_metadata(results: list[EngineExtractionResult]) -> ReportMetadata:
    merged = results[0].metadata.model_copy(deep=True)
    for result in results[1:]:
        merged.patient.age = merged.patient.age or result.metadata.patient.age
        merged.patient.gender = _pick_text(
            merged.patient.gender,
            result.metadata.patient.gender,
        )
        merged.lab.company_name = _pick_text(
            merged.lab.company_name,
            result.metadata.lab.company_name,
        )
        merged.lab.location = _pick_text(
            merged.lab.location,
            result.metadata.lab.location,
        )
        merged.blood_collection.date = _pick_text(
            merged.blood_collection.date,
            result.metadata.blood_collection.date,
        )
        merged.blood_collection.time = _pick_text(
            merged.blood_collection.time,
            result.metadata.blood_collection.time,
        )
        merged.blood_collection.datetime = _pick_text(
            merged.blood_collection.datetime,
            result.metadata.blood_collection.datetime,
        )
    return merged


def _merge_raw_payloads(results: list[EngineExtractionResult]) -> str:
    payloads = [result.raw_payload for result in results if result.raw_payload]
    return "\n\n".join(payloads)


def _merge_engine_results(
    *,
    engine_id: str,
    results: list[EngineExtractionResult],
) -> EngineExtractionResult:
    if not results:
        raise ValueError("Cannot merge an empty set of engine results.")

    merged_biomarkers: list[EngineBiomarkerCandidate] = []
    for result in results:
        merged_biomarkers.extend(
            candidate.model_copy(deep=True) for candidate in result.biomarkers
        )

    return EngineExtractionResult(
        engine_id=engine_id,
        biomarkers=merged_biomarkers,
        notes=_merge_notes(results),
        metadata=_merge_metadata(results),
        raw_payload=_merge_raw_payloads(results),
    )


def _assign_page_number(
    result: EngineExtractionResult,
    page_num: int,
) -> EngineExtractionResult:
    return EngineExtractionResult(
        engine_id=result.engine_id,
        biomarkers=[
            candidate.model_copy(
                update={
                    "page_num": (
                        candidate.page_num
                        if candidate.page_num is not None
                        else page_num
                    )
                }
            )
            for candidate in result.biomarkers
        ],
        notes=list(result.notes),
        metadata=result.metadata.model_copy(deep=True),
        raw_payload=result.raw_payload,
    )


async def _run_engine(
    *,
    engine: ExtractionEngine,
    page_artifacts: list[PageArtifact],
    config: Config,
) -> EngineExtractionResult:
    if engine.execution_mode == "document":
        return await engine.extract(page_artifacts=page_artifacts, config=config)

    page_results: list[EngineExtractionResult] = []
    for page_artifact in page_artifacts:
        page_result = await engine.extract(
            page_artifacts=[page_artifact],
            config=config,
        )
        page_results.append(_assign_page_number(page_result, page_artifact.page_num))

    return _merge_engine_results(engine_id=engine.engine_id, results=page_results)


async def extract_report(
    *,
    image_paths: list[str],
    config: Config,
    engines: list[ExtractionEngine] | None = None,
) -> ExtractionPipelineResult:
    active_engines = list(engines) if engines is not None else _build_engines(config)
    if not active_engines:
        raise ValueError("No extraction engines are configured.")

    page_artifacts = _build_page_artifacts(image_paths)
    engine_results = await asyncio.gather(
        *[
            _run_engine(
                engine=engine,
                page_artifacts=page_artifacts,
                config=config,
            )
            for engine in active_engines
        ]
    )
    return fuse_engine_results(
        engine_results=engine_results,
        fusion_mode=config.extraction_fusion_mode,
    )
