from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ResearchConfig(BaseSettings):
    """
    Runtime behaviour configuration for the research graph.

    Distinct from ``Settings`` (infrastructure / API keys).  All fields can
    be overridden via environment variables prefixed with ``RESEARCH_``.

    Example:
        RESEARCH_MAX_SEARCH_RESULTS=20 python main.py
    """

    model_config = SettingsConfigDict(
        env_prefix="RESEARCH_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    max_search_results: int = Field(default=10, ge=1, description="Max results per search query")
    max_crawl_pages: int = Field(default=20, ge=1, description="Max pages to crawl per agent run")
    max_retry_count: int = Field(default=3, ge=0, description="Max retries on transient failures")
    max_gap_iterations: int = Field(default=2, ge=1, description="Max gap-filling iterations")
    max_parallel_agents: int = Field(default=10, ge=1, description="Max agents running in parallel")


def get_research_config() -> ResearchConfig:
    return ResearchConfig()
