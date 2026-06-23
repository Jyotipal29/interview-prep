import asyncio
import sys
from typing import Any

from src.config.research_config import ResearchConfig
from src.config.settings import get_settings
from src.graph.builder import build_graph
from src.graph.checkpointing import get_checkpointer
from src.models.schemas import Evidence, Source
from src.services.llm_factory import get_llm
from src.shared.context import RuntimeContext
from src.shared.logging import configure_logging, get_logger
from src.state.research_state import create_initial_state
from src.storage.memory import InMemoryEvidenceRepository, InMemoryResearchRepository
from src.tools.interfaces import CrawlerTool, ExtractionTool, SearchTool


# ── Stub tool implementations (replace with real providers when ready) ─────────

class _StubSearchTool(SearchTool):
    @property
    def name(self) -> str:
        return "stub_search"

    async def search(self, query: str, **kwargs: Any) -> list[Evidence]:
        return []


class _StubCrawlerTool(CrawlerTool):
    @property
    def name(self) -> str:
        return "stub_crawler"

    async def crawl(self, url: str, **kwargs: Any) -> Evidence:
        return Evidence(content="", source=Source(url=url))


class _StubExtractionTool(ExtractionTool):
    @property
    def name(self) -> str:
        return "stub_extraction"

    async def extract(self, content: str, schema: type, **kwargs: Any) -> Any:
        return None


# ── Entry point ────────────────────────────────────────────────────────────────

async def main() -> None:
    configure_logging()
    logger = get_logger(__name__)

    settings = get_settings()
    logger.info("settings: provider=%s model=%s", settings.llm_provider, settings.llm_model)

    runtime = RuntimeContext(
        llm=get_llm(settings),
        search_tool=_StubSearchTool(),
        crawler_tool=_StubCrawlerTool(),
        extraction_tool=_StubExtractionTool(),
        evidence_repo=InMemoryEvidenceRepository(),
        research_repo=InMemoryResearchRepository(),
        research_config=ResearchConfig(),
    )

    checkpointer = get_checkpointer(settings.database_url)
    graph = build_graph(checkpointer=checkpointer, runtime=runtime)

    state = create_initial_state("Stripe", user_query="Research Stripe compensation")
    config = {"configurable": {"thread_id": "run-1"}}

    logger.info("graph: invoking for company=%r", state["company_name"])
    result = await graph.ainvoke(state, config=config)

    plan = result.get("research_plan")
    if plan:
        logger.info(
            "graph: plan — tasks=%s",
            [t.name for t in plan.tasks],
        )
    logger.info(
        "graph: finished — completed=%s failed=%s",
        result.get("completed_tasks", []),
        result.get("failed_tasks", []),
    )


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
