from dataclasses import fields
from typing import Any
from unittest.mock import MagicMock

from src.config.research_config import ResearchConfig
from src.models.schemas import Evidence, Source
from src.shared.context import RuntimeContext
from src.storage.interfaces import EvidenceRepository, ResearchRepository
from src.storage.memory import InMemoryEvidenceRepository, InMemoryResearchRepository
from src.tools.interfaces import CrawlerTool, ExtractionTool, SearchTool


class _StubSearch(SearchTool):
    @property
    def name(self) -> str:
        return "stub_search"

    async def search(self, query: str, **kwargs: Any) -> list[Evidence]:
        return []


class _StubCrawler(CrawlerTool):
    @property
    def name(self) -> str:
        return "stub_crawler"

    async def crawl(self, url: str, **kwargs: Any) -> Evidence:
        return Evidence(content="", source=Source(url=url))


class _StubExtractor(ExtractionTool):
    @property
    def name(self) -> str:
        return "stub_extractor"

    async def extract(self, content: str, schema: type, **kwargs: Any) -> Any:
        return None


def _make_context(config: ResearchConfig | None = None) -> RuntimeContext:
    return RuntimeContext(
        llm=MagicMock(),
        search_tool=_StubSearch(),
        crawler_tool=_StubCrawler(),
        extraction_tool=_StubExtractor(),
        evidence_repo=InMemoryEvidenceRepository(),
        research_repo=InMemoryResearchRepository(),
        research_config=config or ResearchConfig(),
    )


def test_runtime_context_can_be_created() -> None:
    ctx = _make_context()
    assert ctx is not None


def test_runtime_context_field_names() -> None:
    expected = {
        "llm",
        "search_tool",
        "crawler_tool",
        "extraction_tool",
        "evidence_repo",
        "research_repo",
        "research_config",
    }
    actual = {f.name for f in fields(RuntimeContext)}
    assert actual == expected


def test_runtime_context_holds_references() -> None:
    ctx = _make_context()
    assert isinstance(ctx.search_tool, SearchTool)
    assert isinstance(ctx.crawler_tool, CrawlerTool)
    assert isinstance(ctx.extraction_tool, ExtractionTool)
    assert isinstance(ctx.evidence_repo, EvidenceRepository)
    assert isinstance(ctx.research_repo, ResearchRepository)
    assert isinstance(ctx.research_config, ResearchConfig)


def test_runtime_context_uses_slots() -> None:
    ctx = _make_context()
    assert not hasattr(ctx, "__dict__")


def test_runtime_context_research_config_defaults() -> None:
    ctx = _make_context()
    assert ctx.research_config.max_search_results == 10
    assert ctx.research_config.max_parallel_agents == 10


def test_runtime_context_accepts_custom_config() -> None:
    cfg = ResearchConfig(max_search_results=50, max_parallel_agents=2)
    ctx = _make_context(config=cfg)
    assert ctx.research_config.max_search_results == 50
    assert ctx.research_config.max_parallel_agents == 2
