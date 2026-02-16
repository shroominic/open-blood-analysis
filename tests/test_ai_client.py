import pytest

from app.ai_client import GeminiAIClient, OpenAIClient, build_ai_client
from app.config import Config


def test_build_ai_client_returns_gemini_client_when_configured():
    client = build_ai_client(Config(ai_provider="gemini", gemini_api_key="test-key"))
    assert isinstance(client, GeminiAIClient)


def test_build_ai_client_requires_gemini_key():
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        build_ai_client(Config(ai_provider="gemini", gemini_api_key=None))


def test_build_ai_client_returns_openai_client_when_configured():
    client = build_ai_client(Config(ai_provider="openai", openai_api_key="test-key"))
    assert isinstance(client, OpenAIClient)


def test_build_ai_client_requires_openai_key():
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        build_ai_client(Config(ai_provider="openai", openai_api_key=None))
