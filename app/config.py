from pydantic import BaseModel
from typing import Optional


class AIConfig(BaseModel):
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-5-nano"

    # Task specific models (fallback to 'model' if None)
    ocr_model: Optional[str] = None
    research_model: Optional[str] = None
    thinking_model: Optional[str] = None

    @property
    def ocr(self) -> str:
        return self.ocr_model or self.model

    @property
    def research(self) -> str:
        return self.research_model or self.model

    @property
    def thinking(self) -> str:
        return self.thinking_model or self.model


class Config(BaseModel):
    ai: AIConfig
    biomarkers_path: str = "biomarkers.json"

    @classmethod
    def load(cls, path: str = "config.json") -> "Config":
        with open(path, "r") as f:
            return cls.model_validate_json(f.read())
