from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

from app import llm
from app.config import Config
from app.extraction.types import (
    EngineBiomarkerCandidate,
    EngineExtractionResult,
    PageArtifact,
)


class LiteParseTextEngine:
    def __init__(
        self,
        *,
        engine_id: str = "liteparse_text",
        cli_path: str | None = None,
        model: str | None = None,
        execution_mode: Literal["document", "page"] = "document",
        weight: float = 1.0,
    ) -> None:
        self.engine_id = engine_id
        self.execution_mode = execution_mode
        self._cli_path = cli_path or "liteparse"
        self._model = model
        self._weight = weight

    def _resolve_cli_path(self) -> str:
        resolved = shutil.which(self._cli_path)
        if resolved:
            return resolved
        if Path(self._cli_path).exists():
            return self._cli_path
        raise FileNotFoundError(
            f"LiteParse CLI not found: {self._cli_path}. Install @llamaindex/liteparse or configure cli_path."
        )

    def _run_parse(self, *, image_path: str) -> dict:
        cli_path = self._resolve_cli_path()
        with tempfile.NamedTemporaryFile(
            prefix="liteparse_",
            suffix=".json",
            delete=False,
        ) as temp_file:
            output_path = Path(temp_file.name)
        try:
            result = subprocess.run(
                [
                    cli_path,
                    "parse",
                    image_path,
                    "--format",
                    "json",
                    "-q",
                    "-o",
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr or result.stdout or "LiteParse failed.")
            return json.loads(output_path.read_text())
        finally:
            output_path.unlink(missing_ok=True)

    async def extract(
        self,
        *,
        page_artifacts: list[PageArtifact],
        config: Config,
    ) -> EngineExtractionResult:
        image_paths = [page.image_path for page in page_artifacts if page.image_path]
        if len(image_paths) != len(page_artifacts):
            raise ValueError("LiteParseTextEngine requires image-backed page artifacts.")

        parsed_documents = [
            await asyncio.to_thread(self._run_parse, image_path=image_path)
            for image_path in image_paths
        ]
        document_text_parts: list[str] = []
        for parsed_document in parsed_documents:
            for page in parsed_document.get("pages", []):
                page_num = page.get("page") or page.get("pageNum") or len(document_text_parts) + 1
                page_text = str(page.get("text", ""))
                document_text_parts.append(f"--- PAGE {page_num} ---\n{page_text}")

        document_text = "\n\n".join(document_text_parts)
        biomarkers, notes, metadata, raw_payload = await llm.extract_biomarkers_from_text(
            document_text,
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
