from __future__ import annotations

from typing import Protocol

from openblood.ai_client import AIClient


class ResearchBackend(Protocol):
    backend_id: str
    supports_web_search: bool

    async def generate_json(
        self,
        *,
        model: str,
        prompt: str,
        use_web_search: bool = False,
    ) -> str | None: ...


class DelegatingAIClientResearchBackend:
    def __init__(
        self,
        *,
        backend_id: str,
        client: AIClient,
        supports_web_search: bool,
    ) -> None:
        self.backend_id = backend_id
        self._client = client
        self.supports_web_search = supports_web_search

    async def generate_json(
        self,
        *,
        model: str,
        prompt: str,
        use_web_search: bool = False,
    ) -> str | None:
        if use_web_search and self.supports_web_search:
            return await self._client.prompt_json_with_search(model=model, prompt=prompt)
        return await self._client.prompt_json(model=model, prompt=prompt)
