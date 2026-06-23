from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.graph.nodes import placeholder_node
from src.shared.logging import get_logger
from src.state.research_state import ResearchState

logger = get_logger(__name__)


def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    """
    Assemble and compile the research graph.

    The graph currently routes START → placeholder_node → END.  Future
    agents are added by inserting add_node / add_edge / add_conditional_edges
    calls before the compile() invocation.

    Args:
        checkpointer: Optional checkpoint saver (Postgres or MemorySaver).
                      When None the graph runs stateless (no persistence).
    """
    builder: StateGraph = StateGraph(ResearchState)

    builder.add_node("placeholder_node", placeholder_node)

    builder.add_edge(START, "placeholder_node")
    builder.add_edge("placeholder_node", END)

    graph = builder.compile(checkpointer=checkpointer)
    logger.info("graph: compiled successfully (nodes=%s)", list(graph.nodes))
    return graph
