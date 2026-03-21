from __future__ import annotations

from typing import Literal, Protocol

from app.config import Config
from app.extraction.types import EngineExtractionResult, PageArtifact

ExecutionMode = Literal["document", "page"]


class ExtractionEngine(Protocol):
    engine_id: str
    execution_mode: ExecutionMode

    async def extract(
        self,
        *,
        page_artifacts: list[PageArtifact],
        config: Config,
    ) -> EngineExtractionResult:
        ...
