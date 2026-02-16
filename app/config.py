from pydantic_settings import (  # ty:ignore[unresolved-import]
    BaseSettings,
    SettingsConfigDict,
)
from typing import Literal, Optional


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ai_provider: Literal["gemini", "openai"] = "gemini"
    gemini_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    ai_model: str = "gemini-3-flash-preview"

    # Task specific models (fallback to ai_model if None)
    ai_ocr_model: Optional[str] = None
    ai_research_model: Optional[str] = None
    ai_thinking_model: Optional[str] = None

    biomarkers_path: str = "biomarkers.json"

    @property
    def ocr(self) -> str:
        return self.ai_ocr_model or self.ai_model

    @property
    def research(self) -> str:
        return self.ai_research_model or self.ai_model

    @property
    def thinking(self) -> str:
        return self.ai_thinking_model or self.ai_model
