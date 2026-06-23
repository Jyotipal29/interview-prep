from collections.abc import Callable
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.graph.dispatcher import dispatcher_node
from src.graph.nodes import placeholder_node
from src.graph.research_subgraph import REAL_AGENT_NODES, RESEARCH_NODES
from src.graph.routing import AGENT_ROUTES
from src.shared.logging import get_logger
from src.state.research_state import ResearchState

logger = get_logger(__name__)


def _with_runtime(node_fn: Callable[..., Any], rt: object) -> Callable[[ResearchState], object]:
    """
    Wrap a node that expects (state, runtime) so LangGraph only sees (state).

    LangGraph ≥ 0.2 reserves the parameter name 'runtime' for its own
    internal execution-context injection.  Using functools.partial leaves the
    original parameter name visible via inspect; wrapping with a closure hides
    it, preventing the collision.
    """
    async def _wrapper(state: ResearchState) -> object:
        return await node_fn(state, rt)

    return _wrapper


def build_graph(
    checkpointer: BaseCheckpointSaver | None = None,
    runtime: object | None = None,
) -> CompiledStateGraph:
    """
    Assemble and compile the research graph.

    With *runtime* the full dispatch architecture is wired:

        START → planner → dispatcher ─┬→ company_profile → END
                                      ├→ leadership       → END
                                      ├→ …                → END
                                      └→ competitors      → END

        dispatcher returns Command(goto=[Send(...)]) which fans out all
        research nodes in parallel.  When the plan is empty or all tasks are
        invalid it falls through its static edge to END.

    Without *runtime* (tests / no API keys):
        START → placeholder_node → END

    Args:
        checkpointer: Optional checkpoint saver (Postgres or MemorySaver).
        runtime: RuntimeContext instance.  When supplied, real agents are used.
    """
    builder: StateGraph = StateGraph(ResearchState)

    if runtime is not None:
        from src.agents.planner.node import planner_node

        # ── Planner ────────────────────────────────────────────────────────
        builder.add_node("planner", _with_runtime(planner_node, runtime))
        builder.add_edge(START, "planner")

        # ── Dispatcher ─────────────────────────────────────────────────────
        builder.add_node("dispatcher", dispatcher_node)
        builder.add_edge("planner", "dispatcher")
        # Fallback: when dispatcher returns a plain dict (no sends), go to END
        builder.add_edge("dispatcher", END)

        # ── Research nodes (fan-out targets) ──────────────────────────────
        # REAL_AGENT_NODES overrides the stub for activated domains.
        for domain, stub_fn in RESEARCH_NODES.items():
            node_name = AGENT_ROUTES[domain]
            real_fn = REAL_AGENT_NODES.get(domain)
            node_fn = _with_runtime(real_fn, runtime) if real_fn is not None else stub_fn
            builder.add_node(node_name, node_fn)
            builder.add_edge(node_name, END)

    else:
        builder.add_node("placeholder_node", placeholder_node)
        builder.add_edge(START, "placeholder_node")
        builder.add_edge("placeholder_node", END)

    graph = builder.compile(checkpointer=checkpointer)
    logger.info("graph: compiled successfully (nodes=%s)", list(graph.nodes))
    return graph
