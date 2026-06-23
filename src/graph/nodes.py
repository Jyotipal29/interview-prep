from src.graph.types import Command
from src.shared.logging import get_logger
from src.state.research_state import ResearchState

logger = get_logger(__name__)


def placeholder_node(state: ResearchState) -> Command | dict[str, object]:
    """
    Stub node that occupies the graph until real agents are wired in.

    Returns a plain dict (no state changes).  Real agents will return either
    a plain dict or a Command(update={...}, goto="next_node") to combine a
    state update with conditional routing in a single return value.
    """
    company = state.get("company_name", "unknown")
    logger.info("placeholder_node: received state for company=%r", company)
    return {}
