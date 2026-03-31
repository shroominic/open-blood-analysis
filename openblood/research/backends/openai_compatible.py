from __future__ import annotations

from openblood.ai_client import OpenAIClient


class OpenAICompatibleResearchBackend:
    def __init__(
        self,
        *,
        backend_id: str,
        api_key: str,
        base_url: str,
        supports_web_search: bool = False,
    ) -> None:
        self.backend_id = backend_id
        self._client = OpenAIClient(api_key=api_key, base_url=base_url)
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
