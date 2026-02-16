from __future__ import annotations

from typing import Protocol

from google import genai
from google.genai import types

from .config import Config


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
            contents.append(types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"))

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
    def __init__(self, api_key: str):
        self._api_key = api_key

    async def prompt_json(self, *, model: str, prompt: str) -> str | None:
        raise NotImplementedError("OpenAI provider is not implemented yet.")

    async def prompt_json_with_search(self, *, model: str, prompt: str) -> str | None:
        raise NotImplementedError("OpenAI provider is not implemented yet.")

    async def prompt_text(
        self, *, model: str, prompt: str, use_web_search: bool = False
    ) -> str | None:
        raise NotImplementedError("OpenAI provider is not implemented yet.")

    async def extract_report_json(
        self,
        *,
        model: str,
        system_instruction: str,
        prompt: str,
        image_paths: list[str],
    ) -> str | None:
        raise NotImplementedError("OpenAI provider is not implemented yet.")


def build_ai_client(config: Config) -> AIClient:
    if config.ai_provider == "gemini":
        if not config.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required when AI_PROVIDER=gemini.")
        return GeminiAIClient(api_key=config.gemini_api_key)

    if config.ai_provider == "openai":
        if not config.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when AI_PROVIDER=openai.")
        return OpenAIClient(api_key=config.openai_api_key)

    raise ValueError(f"Unsupported AI provider: {config.ai_provider}")
