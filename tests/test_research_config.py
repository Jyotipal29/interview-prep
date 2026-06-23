import pytest

from src.config.research_config import ResearchConfig, get_research_config


def test_default_values() -> None:
    cfg = ResearchConfig()
    assert cfg.max_search_results == 10
    assert cfg.max_crawl_pages == 20
    assert cfg.max_retry_count == 3
    assert cfg.max_gap_iterations == 2
    assert cfg.max_parallel_agents == 10


def test_get_research_config_returns_instance() -> None:
    cfg = get_research_config()
    assert isinstance(cfg, ResearchConfig)


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_MAX_SEARCH_RESULTS", "25")
    monkeypatch.setenv("RESEARCH_MAX_PARALLEL_AGENTS", "4")
    cfg = ResearchConfig()
    assert cfg.max_search_results == 25
    assert cfg.max_parallel_agents == 4


def test_max_search_results_minimum(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_MAX_SEARCH_RESULTS", "0")
    with pytest.raises(Exception):
        ResearchConfig()


def test_max_retry_count_allows_zero() -> None:
    cfg = ResearchConfig(max_retry_count=0)
    assert cfg.max_retry_count == 0


def test_max_parallel_agents_minimum() -> None:
    with pytest.raises(Exception):
        ResearchConfig(max_parallel_agents=0)
