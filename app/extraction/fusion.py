from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from typing import Any

from app.extraction.types import (
    EngineBiomarkerCandidate,
    EngineExtractionResult,
    ExtractionPipelineResult,
)
from app.types import ExtractedBiomarker, ReportMetadata

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def _normalize_label(value: str) -> str:
    return _NON_ALNUM_RE.sub(" ", value.casefold()).strip()


def _normalize_unit(value: str) -> str:
    return value.strip().casefold().replace("μ", "u").replace("µ", "u")


def _label_quality(value: str) -> float:
    if not value:
        return 0.0
    alpha_chars = sum(1 for char in value if char.isalpha())
    good_chars = sum(1 for char in value if char.isalnum() or char in " -_/(),.%")
    return (alpha_chars / len(value)) * 0.7 + (good_chars / len(value)) * 0.3


def _value_matches(left: float | str | bool, right: float | str | bool) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) <= 0.01
    return str(left).strip() == str(right).strip()


def _labels_match(left: str, right: str) -> bool:
    normalized_left = _normalize_label(left)
    normalized_right = _normalize_label(right)
    if not normalized_left or not normalized_right:
        return False
    if normalized_left == normalized_right:
        return True
    return (
        SequenceMatcher(None, normalized_left, normalized_right).ratio() >= 0.82
    )


def _candidate_matches(
    left: EngineBiomarkerCandidate,
    right: EngineBiomarkerCandidate,
) -> bool:
    if left.page_num is not None and right.page_num is not None and left.page_num != right.page_num:
        return False

    left_unit = _normalize_unit(left.unit)
    right_unit = _normalize_unit(right.unit)
    unit_matches = not left_unit or not right_unit or left_unit == right_unit

    if _labels_match(left.raw_name, right.raw_name) and unit_matches:
        return True

    return unit_matches and _value_matches(left.value, right.value)


def _candidate_priority(
    candidate: EngineBiomarkerCandidate,
    *,
    engine_weight: float,
    engine_order: int,
) -> tuple[float, float, float]:
    return (
        engine_weight,
        float(candidate.confidence),
        _label_quality(candidate.raw_name) - (engine_order * 0.0001),
    )


def _merge_notes(engine_results: list[EngineExtractionResult]) -> list[str]:
    merged_notes: list[str] = []
    seen_notes: set[str] = set()
    for result in engine_results:
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


def _merge_metadata(engine_results: list[EngineExtractionResult]) -> ReportMetadata:
    ordered_results = sorted(engine_results, key=lambda result: result.weight, reverse=True)
    merged = ordered_results[0].metadata.model_copy(deep=True)
    for result in ordered_results[1:]:
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


def _to_debug_payload(
    *,
    engine_results: list[EngineExtractionResult],
    biomarkers: list[ExtractedBiomarker],
    notes: list[str],
    metadata: ReportMetadata,
    fusion_mode: str,
) -> str:
    def _parse_raw_payload(payload: str) -> Any:
        if not payload:
            return None
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return payload

    return json.dumps(
        {
            "fusion_mode": fusion_mode,
            "data": [biomarker.model_dump() for biomarker in biomarkers],
            "notes": notes,
            "metadata": metadata.model_dump(),
            "engines": {
                result.engine_id: {
                    "weight": result.weight,
                    "data": [candidate.model_dump() for candidate in result.biomarkers],
                    "notes": result.notes,
                    "metadata": result.metadata.model_dump(),
                    "raw_payload": _parse_raw_payload(result.raw_payload),
                }
                for result in engine_results
            },
        },
        indent=2,
        ensure_ascii=False,
    )


def fuse_engine_results(
    *,
    engine_results: list[EngineExtractionResult],
    fusion_mode: str,
) -> ExtractionPipelineResult:
    if not engine_results:
        return ExtractionPipelineResult()

    if len(engine_results) == 1:
        result = engine_results[0]
        return ExtractionPipelineResult(
            biomarkers=[
                candidate.to_extracted_biomarker() for candidate in result.biomarkers
            ],
            notes=list(result.notes),
            metadata=result.metadata.model_copy(deep=True),
            raw_payload=_to_debug_payload(
                engine_results=engine_results,
                biomarkers=[
                    candidate.to_extracted_biomarker() for candidate in result.biomarkers
                ],
                notes=list(result.notes),
                metadata=result.metadata.model_copy(deep=True),
                fusion_mode=fusion_mode,
            ),
            engine_results=engine_results,
        )

    ordered_results = list(engine_results)
    primary_result = max(
        ordered_results,
        key=lambda result: (result.weight, -ordered_results.index(result)),
    )
    fused_candidates: list[EngineBiomarkerCandidate] = []
    engine_order_by_id = {result.engine_id: index for index, result in enumerate(ordered_results)}
    engine_weight_by_id = {result.engine_id: result.weight for result in ordered_results}

    for result in ordered_results:
        for candidate in result.biomarkers:
            matched_index = next(
                (
                    index
                    for index, existing in enumerate(fused_candidates)
                    if _candidate_matches(existing, candidate)
                ),
                None,
            )
            if matched_index is None:
                fused_candidates.append(candidate.model_copy(deep=True))
                continue

            existing = fused_candidates[matched_index]
            existing_priority = _candidate_priority(
                existing,
                engine_weight=engine_weight_by_id.get(existing.source_engine, 1.0),
                engine_order=engine_order_by_id.get(existing.source_engine, 0),
            )
            candidate_priority = _candidate_priority(
                candidate,
                engine_weight=engine_weight_by_id.get(candidate.source_engine, 1.0),
                engine_order=engine_order_by_id.get(candidate.source_engine, 0),
            )
            if candidate_priority > existing_priority:
                fused_candidates[matched_index] = candidate.model_copy(deep=True)

    if fusion_mode == "primary":
        fused_candidates = [
            candidate.model_copy(deep=True) for candidate in primary_result.biomarkers
        ]

    fused_biomarkers = [
        candidate.to_extracted_biomarker() for candidate in fused_candidates
    ]
    fused_notes = _merge_notes(ordered_results)
    fused_metadata = _merge_metadata(ordered_results)
    return ExtractionPipelineResult(
        biomarkers=fused_biomarkers,
        notes=fused_notes,
        metadata=fused_metadata,
        raw_payload=_to_debug_payload(
            engine_results=ordered_results,
            biomarkers=fused_biomarkers,
            notes=fused_notes,
            metadata=fused_metadata,
            fusion_mode=fusion_mode,
        ),
        engine_results=ordered_results,
    )
