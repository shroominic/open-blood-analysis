from __future__ import annotations

from typing import Literal

from app import llm
from app.config import Config
from app.extraction.types import (
    EngineBiomarkerCandidate,
    EngineExtractionResult,
    PageArtifact,
)


class GeminiVisionEngine:
    def __init__(
        self,
        *,
        engine_id: str = "gemini_vision",
        execution_mode: Literal["document", "page"] = "document",
        model: str | None = None,
        weight: float = 1.0,
    ) -> None:
        self.engine_id = engine_id
        self.execution_mode = execution_mode
        self._model = model
        self._weight = weight

    async def extract(
        self,
        *,
        page_artifacts: list[PageArtifact],
        config: Config,
    ) -> EngineExtractionResult:
        image_paths = [page.image_path for page in page_artifacts if page.image_path]
        if len(image_paths) != len(page_artifacts):
            raise ValueError("GeminiVisionEngine requires image-backed page artifacts.")

        biomarkers, notes, metadata, raw_payload = await llm.extract_biomarkers(
            image_paths,
            config,
            model=self._model,
        )

        return EngineExtractionResult(
            engine_id=self.engine_id,
            biomarkers=[
                EngineBiomarkerCandidate(
                    raw_name=biomarker.raw_name,
                    value=biomarker.value,
                    unit=biomarker.unit,
                    flags=list(biomarker.flags),
                    confidence=biomarker.confidence,
                    page_num=None,
                    source_engine=self.engine_id,
                )
                for biomarker in biomarkers
            ],
            notes=list(notes),
            metadata=metadata,
            raw_payload=raw_payload,
            weight=self._weight,
        )
