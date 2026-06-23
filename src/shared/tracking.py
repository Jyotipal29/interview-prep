from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from src.models.schemas import AgentExecution, ExecutionStatus
from src.state.research_state import ResearchState


@asynccontextmanager
async def track_execution(
    state: ResearchState,
    agent_name: str,
) -> AsyncGenerator[AgentExecution, None]:
    """
    Async context manager that records timing and status for a node execution.

    Yields a mutable AgentExecution that is finalised (status, completed_at,
    duration_ms) when the block exits — successfully or with an exception.
    The caller includes the execution in the node's return dict so the
    append reducer adds it to state["agent_executions"].

    Usage::

        async def company_profile_node(
            state: ResearchState,
            runtime: RuntimeContext,
        ) -> dict:
            async with track_execution(state, "company_profile") as execution:
                data = await runtime.search_tool.search("...")
            return {
                "company_profile": data,
                "agent_executions": [execution],
            }

    On success:  execution.status = COMPLETED, duration_ms is set.
    On failure:  execution.status = FAILED, error_message is set, exception re-raised.
    """
    execution = AgentExecution(agent_name=agent_name, status=ExecutionStatus.RUNNING)

    try:
        yield execution
        now = datetime.now(UTC)
        execution.completed_at = now
        execution.duration_ms = int((now - execution.started_at).total_seconds() * 1000)
        execution.status = ExecutionStatus.COMPLETED
    except Exception as exc:
        now = datetime.now(UTC)
        execution.completed_at = now
        execution.duration_ms = int((now - execution.started_at).total_seconds() * 1000)
        execution.status = ExecutionStatus.FAILED
        execution.error_message = str(exc)
        raise
