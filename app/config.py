import os
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import (  # ty:ignore[unresolved-import]
    BaseSettings,
    SettingsConfigDict,
)


class ExtractionEngineSpec(BaseModel):
    type: Literal[
        "gemini_vision",
        "openai_compatible_vision",
        "liteparse_text",
    ]
    id: str | None = None
    enabled: bool = True
    execution_mode: Literal["document", "page"] = "document"
    model: str | None = None
    weight: float = 1.0
    base_url: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    cli_path: str | None = None

    def resolved_id(self) -> str:
        return self.id or self.type

    def resolved_api_key(self) -> str | None:
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.getenv(self.api_key_env)
        return None


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ai_provider: Literal["gemini", "openai"] = "gemini"
    gemini_api_key: str | None = None
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    ai_model: str = "gemini-3.1-flash-lite-preview"

    # Task specific models (fallback to ai_model if None)
    ai_ocr_model: str | None = None
    ai_research_model: str | None = None
    ai_thinking_model: str | None = None

    biomarkers_path: str = "biomarkers.json"
    extraction_engines: list[ExtractionEngineSpec] = Field(default_factory=list)
    extraction_fusion_mode: Literal["primary", "union", "consensus"] = "primary"
    extraction_debug_dir: str | None = None

    @property
    def ocr(self) -> str:
        return self.ai_ocr_model or self.ai_model

    @property
    def research(self) -> str:
        return self.ai_research_model or self.ai_model

    @property
    def thinking(self) -> str:
        return self.ai_thinking_model or self.ai_model

    @property
    def resolved_extraction_engines(self) -> list[ExtractionEngineSpec]:
        if self.extraction_engines:
            return [spec for spec in self.extraction_engines if spec.enabled]

        if self.ai_provider == "gemini":
            return [
                ExtractionEngineSpec(
                    type="gemini_vision",
                    id="gemini_vision",
                    execution_mode="document",
                    model=self.ocr,
                    weight=1.0,
                )
            ]

        return [
            ExtractionEngineSpec(
                type="openai_compatible_vision",
                id="openai_compatible_vision",
                execution_mode="document",
                model=self.ocr,
                base_url=self.openai_base_url,
                api_key=self.openai_api_key,
                weight=1.0,
            )
        ]
