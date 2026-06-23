from src.shared.logging import get_logger
from src.state.research_state import ResearchState

logger = get_logger(__name__)


def placeholder_node(state: ResearchState) -> dict[str, object]:
    """
    Stub node that occupies the graph until real agents are wired in.

    Receives the full ResearchState and returns an empty update dict,
    leaving all state fields unchanged.
    """
    company = state.get("company_name", "unknown")
    logger.info("placeholder_node: received state for company=%r", company)
    return {}
