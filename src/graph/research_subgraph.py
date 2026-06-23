"""
Stub research nodes — one per supported domain.

Each stub is a no-op placeholder that records an AgentExecution and marks its
task completed.  Replace the corresponding entry in RESEARCH_NODES with a real
agent implementation to activate that domain.

The builder registers these callables as graph nodes via AGENT_ROUTES so the
dispatcher can target them with Send().
"""
from collections.abc import Callable
from typing import Any

from src.graph.routing import AGENT_ROUTES
from src.shared.logging import get_logger
from src.shared.tracking import track_execution
from src.state.research_state import ResearchState

logger = get_logger(__name__)


def _stub_node_for(domain: str) -> Callable[..., Any]:
    """Return an async stub node for *domain*."""

    async def stub(state: ResearchState) -> dict[str, object]:
        task = state.get("active_task")
        completed: list[str] = []

        async with track_execution(state, domain) as execution:
            if task is not None:
                completed.append(task.id)
                logger.debug("stub[%s]: no-op for task %s", domain, task.id)

        return {
            "completed_tasks": completed,
            "agent_executions": [execution],
        }

    stub.__name__ = f"{domain}_stub"
    stub.__qualname__ = f"{domain}_stub"
    return stub


# Domain → stub callable registry (state-only signature).
RESEARCH_NODES: dict[str, Callable[..., Any]] = {
    domain: _stub_node_for(domain) for domain in AGENT_ROUTES
}

# Domain → real agent registry ((state, runtime) signature).
# Builder wraps these with _with_runtime() and overrides the corresponding stub.
# Add an entry here when promoting a domain from stub to real agent.
def _build_real_agent_nodes() -> dict[str, Callable[..., Any]]:
    from src.agents.company_profile.node import company_profile_node

    return {
        "company_profile": company_profile_node,
    }


REAL_AGENT_NODES: dict[str, Callable[..., Any]] = _build_real_agent_nodes()
