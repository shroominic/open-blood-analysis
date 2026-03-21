from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
from dataclasses import dataclass
from typing import Literal
from urllib import error, request

from app import llm
from app.config import Config
from app.extraction.types import (
    EngineBiomarkerCandidate,
    EngineExtractionResult,
    PageArtifact,
)


@dataclass(slots=True)
class OpenAICompatibleVisionClient:
    base_url: str
    api_key: str
    headers: dict[str, str]
    timeout_seconds: float = 120.0

    def _endpoint(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"

    def _make_request(
        self,
        *,
        model: str,
        system_instruction: str,
        prompt: str,
        image_paths: list[str],
    ) -> str:
        user_content: list[dict[str, object]] = [{"type": "text", "text": prompt}]
        for image_path in image_paths:
            with open(image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode("ascii")
            mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{encoded_image}",
                    },
                }
            )

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_content},
            ],
            "response_format": {"type": "json_object"},
        }
        req = request.Request(
            url=self._endpoint(),
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                **self.headers,
            },
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"OpenAI-compatible vision request failed with {exc.code}: {body}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(
                f"OpenAI-compatible vision request failed: {exc.reason}"
            ) from exc

        data = json.loads(body)
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("OpenAI-compatible vision response had no choices.")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = [
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ]
            return "".join(text_parts)
        raise RuntimeError("OpenAI-compatible vision response content was not text.")

    async def extract_report_json(
        self,
        *,
        model: str,
        system_instruction: str,
        prompt: str,
        image_paths: list[str],
    ) -> str:
        return await asyncio.to_thread(
            self._make_request,
            model=model,
            system_instruction=system_instruction,
            prompt=prompt,
            image_paths=image_paths,
        )


class OpenAICompatibleVisionEngine:
    def __init__(
        self,
        *,
        engine_id: str,
        model: str,
        base_url: str,
        api_key: str,
        headers: dict[str, str] | None = None,
        execution_mode: Literal["document", "page"] = "document",
        weight: float = 1.0,
    ) -> None:
        self.engine_id = engine_id
        self.execution_mode = execution_mode
        self._model = model
        self._client = OpenAICompatibleVisionClient(
            base_url=base_url,
            api_key=api_key,
            headers=dict(headers or {}),
        )
        self._weight = weight

    async def extract(
        self,
        *,
        page_artifacts: list[PageArtifact],
        config: Config,
    ) -> EngineExtractionResult:
        image_paths = [page.image_path for page in page_artifacts if page.image_path]
        if len(image_paths) != len(page_artifacts):
            raise ValueError(
                "OpenAICompatibleVisionEngine requires image-backed page artifacts."
            )

        raw_payload = await self._client.extract_report_json(
            model=self._model,
            system_instruction=llm.build_report_extraction_system_instruction(
                source_kind="images"
            ),
            prompt="Extract all biomarkers from these blood test report pages.",
            image_paths=image_paths,
        )
        biomarkers, notes, metadata = llm._parse_llm_response(raw_payload)
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
