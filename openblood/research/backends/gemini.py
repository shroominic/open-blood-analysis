from __future__ import annotations

from openblood.ai_client import GeminiAIClient


class GeminiResearchBackend:
    def __init__(self, *, backend_id: str, api_key: str) -> None:
        self.backend_id = backend_id
        self._client = GeminiAIClient(api_key=api_key)
        self.supports_web_search = True

    async def generate_json(
        self,
        *,
        model: str,
        prompt: str,
        use_web_search: bool = False,
    ) -> str | None:
        if use_web_search:
            return await self._client.prompt_json_with_search(model=model, prompt=prompt)
        return await self._client.prompt_json(model=model, prompt=prompt)
