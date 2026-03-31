from __future__ import annotations

from .openai_compatible import OpenAICompatibleResearchBackend


class PerplexityResearchBackend(OpenAICompatibleResearchBackend):
    def __init__(
        self,
        *,
        backend_id: str,
        api_key: str,
        base_url: str,
        supports_web_search: bool = True,
    ) -> None:
        super().__init__(
            backend_id=backend_id,
            api_key=api_key,
            base_url=base_url,
            supports_web_search=supports_web_search,
        )
