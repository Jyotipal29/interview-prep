from enum import StrEnum

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM configuration
    llm_provider: LLMProvider = LLMProvider.OPENAI
    llm_model: str = "gpt-4o"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    # API keys — all optional; validated at use-time by the model factory
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    openrouter_api_key: str | None = None

    # Search & crawl tools
    tavily_api_key: str | None = None
    exa_api_key: str | None = None
    firecrawl_api_key: str | None = None

    # Storage — optional; graph falls back to MemorySaver when unset
    database_url: str | None = None


def get_settings() -> Settings:
    return Settings()
