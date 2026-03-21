from __future__ import annotations

import asyncio
import base64
import json
import logging
import mimetypes
from urllib import error, request
from typing import Callable, Protocol, TypeVar

from google import genai
from google.genai import types

from .config import Config

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def retry_async(
    fn: Callable[..., T],
    *args,
    max_retries: int = 2,
    base_delay: float = 1.0,
    **kwargs,
) -> T:
    """Retry an async callable with exponential backoff on transient errors."""
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return await fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "Retry %d/%d for %s after error: %s (waiting %.1fs)",
                    attempt + 1, max_retries, getattr(fn, "__name__", fn), exc, delay,
                )
                await asyncio.sleep(delay)
    raise last_exc  # type: ignore[misc]


class AIClient(Protocol):
    async def prompt_json(self, *, model: str, prompt: str) -> str | None:
        ...

    async def prompt_json_with_search(self, *, model: str, prompt: str) -> str | None:
        ...

    async def prompt_text(
        self, *, model: str, prompt: str, use_web_search: bool = False
    ) -> str | None:
        ...

    async def extract_report_json(
        self,
        *,
        model: str,
        system_instruction: str,
        prompt: str,
        image_paths: list[str],
    ) -> str | None:
        ...


class GeminiAIClient:
    def __init__(self, api_key: str):
        self._client = genai.Client(api_key=api_key)

    async def prompt_json(self, *, model: str, prompt: str) -> str | None:
        response = await self._client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )
        return response.text

    async def prompt_json_with_search(self, *, model: str, prompt: str) -> str | None:
        response = await self._client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                tools=[types.Tool(google_search=types.GoogleSearch())],
            ),
        )
        return response.text

    async def prompt_text(
        self, *, model: str, prompt: str, use_web_search: bool = False
    ) -> str | None:
        tools = [types.Tool(google_search=types.GoogleSearch())] if use_web_search else None
        response = await self._client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(tools=tools),
        )
        return response.text

    async def extract_report_json(
        self,
        *,
        model: str,
        system_instruction: str,
        prompt: str,
        image_paths: list[str],
    ) -> str | None:
        contents: list[str | types.Part] = [prompt]
        for path in image_paths:
            with open(path, "rb") as f:
                image_bytes = f.read()
            mime_type = mimetypes.guess_type(path)[0] or "image/jpeg"
            contents.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))

        response = await self._client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
            ),
        )
        return response.text


class OpenAIClient:
    def __init__(self, api_key: str, base_url: str):
        self._api_key = api_key
        self._base_url = base_url

    def _endpoint(self) -> str:
        base = self._base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"

    def _make_request(
        self,
        *,
        model: str,
        messages: list[dict[str, object]],
        json_output: bool,
    ) -> str:
        payload: dict[str, object] = {
            "model": model,
            "messages": messages,
        }
        if json_output:
            payload["response_format"] = {"type": "json_object"}

        req = request.Request(
            url=self._endpoint(),
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with request.urlopen(req, timeout=120.0) as response:
                body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"OpenAI-compatible request failed with {exc.code}: {body}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(
                f"OpenAI-compatible request failed: {exc.reason}"
            ) from exc

        data = json.loads(body)
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("OpenAI-compatible response had no choices.")
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
        raise RuntimeError("OpenAI-compatible response content was not text.")

    async def prompt_json(self, *, model: str, prompt: str) -> str | None:
        return await asyncio.to_thread(
            self._make_request,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            json_output=True,
        )

    async def prompt_json_with_search(self, *, model: str, prompt: str) -> str | None:
        logger.warning(
            "OpenAI-compatible provider does not support web search; falling back to prompt_json."
        )
        return await self.prompt_json(model=model, prompt=prompt)

    async def prompt_text(
        self, *, model: str, prompt: str, use_web_search: bool = False
    ) -> str | None:
        if use_web_search:
            logger.warning(
                "OpenAI-compatible provider does not support web search; falling back to prompt_text without search."
            )
        return await asyncio.to_thread(
            self._make_request,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            json_output=False,
        )

    async def extract_report_json(
        self,
        *,
        model: str,
        system_instruction: str,
        prompt: str,
        image_paths: list[str],
    ) -> str | None:
        user_content: list[dict[str, object]] = [{"type": "text", "text": prompt}]
        for path in image_paths:
            with open(path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode("ascii")
            mime_type = mimetypes.guess_type(path)[0] or "image/jpeg"
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{encoded_image}",
                    },
                }
            )
        return await asyncio.to_thread(
            self._make_request,
            model=model,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_content},
            ],
            json_output=True,
        )


def build_ai_client(config: Config) -> AIClient:
    if config.ai_provider == "gemini":
        if not config.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required when AI_PROVIDER=gemini.")
        return GeminiAIClient(api_key=config.gemini_api_key)

    if config.ai_provider == "openai":
        if not config.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when AI_PROVIDER=openai.")
        return OpenAIClient(
            api_key=config.openai_api_key,
            base_url=config.openai_base_url,
        )

    raise ValueError(f"Unsupported AI provider: {config.ai_provider}")
