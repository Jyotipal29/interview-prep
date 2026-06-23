import sys

from src.config.settings import get_settings
from src.graph.builder import build_graph
from src.graph.checkpointing import get_checkpointer
from src.shared.logging import configure_logging, get_logger
from src.state.research_state import create_initial_state


def main() -> None:
    configure_logging()
    logger = get_logger(__name__)

    settings = get_settings()
    logger.info("settings: provider=%s model=%s", settings.llm_provider, settings.llm_model)

    checkpointer = get_checkpointer(settings.database_url)
    graph = build_graph(checkpointer=checkpointer)

    initial_state = create_initial_state(company_name="Anthropic")
    config = {"configurable": {"thread_id": "example-run-1"}}

    logger.info("graph: invoking for company=%r", initial_state["company_name"])
    result = graph.invoke(initial_state, config=config)

    logger.info("graph: finished (state_keys=%s)", sorted(result.keys()))


if __name__ == "__main__":
    sys.exit(main())
