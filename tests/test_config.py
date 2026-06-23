import pytest

from src.config.settings import LLMProvider, Settings, get_settings


def test_default_settings() -> None:
    settings = Settings()
    assert settings.llm_provider == LLMProvider.OPENAI
    assert settings.llm_model == "gpt-4o"
    assert settings.llm_temperature == 0.0


def test_get_settings_returns_settings_instance() -> None:
    settings = get_settings()
    assert isinstance(settings, Settings)


def test_provider_enum_values() -> None:
    assert LLMProvider.OPENAI == "openai"
    assert LLMProvider.ANTHROPIC == "anthropic"
    assert LLMProvider.OPENROUTER == "openrouter"


def test_optional_keys_are_none_by_default() -> None:
    settings = Settings()
    assert settings.openai_api_key is None
    assert settings.anthropic_api_key is None
    assert settings.openrouter_api_key is None
    assert settings.tavily_api_key is None
    assert settings.exa_api_key is None
    assert settings.firecrawl_api_key is None
    assert settings.database_url is None


def test_settings_override_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "claude-3-5-sonnet-20241022")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.5")

    settings = Settings()
    assert settings.llm_provider == LLMProvider.ANTHROPIC
    assert settings.llm_model == "claude-3-5-sonnet-20241022"
    assert settings.llm_temperature == 0.5
