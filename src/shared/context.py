from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel

from src.config.research_config import ResearchConfig
from src.storage.interfaces import EvidenceRepository, ResearchRepository
from src.tools.interfaces import CrawlerTool, ExtractionTool, SearchTool


@dataclass(slots=True)
class RuntimeContext:
    """
    Immutable dependency container injected into graph nodes at build time.

    Nodes receive a RuntimeContext via closure (see graph/builder.py) rather
    than through LangGraph state, keeping external dependencies out of the
    serialisable state snapshot.

    Example node signature (future agents):

        async def planner_node(
            state: ResearchState,
            runtime: RuntimeContext,
        ) -> Command | dict:
            result = await runtime.llm.ainvoke(...)
            evidence = await runtime.search_tool.search(query)
            cfg = runtime.research_config.max_search_results
            ...
    """

    llm: BaseChatModel
    search_tool: SearchTool
    crawler_tool: CrawlerTool
    extraction_tool: ExtractionTool
    evidence_repo: EvidenceRepository
    research_repo: ResearchRepository
    research_config: ResearchConfig
