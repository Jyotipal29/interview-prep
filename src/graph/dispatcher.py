from src.graph.routing import get_route
from src.graph.types import Command, Send
from src.shared.logging import get_logger
from src.shared.tracking import track_execution
from src.state.research_state import ResearchState

logger = get_logger(__name__)


async def dispatcher_node(state: ResearchState) -> Command | dict[str, object]:
    """
    Fan-out orchestrator: read research_plan and emit one Send() per valid domain.

    Returns Command(goto=[Send(...)]) when at least one valid route exists so
    LangGraph executes all research nodes in parallel.  Returns a plain dict
    (and falls through the static builder edge to END) when the plan is empty
    or contains only unknown agent_type values.

    Invalid agent_type values are added to failed_tasks without raising.
    """
    plan = state.get("research_plan")
    if plan is None or not plan.tasks:
        logger.info("dispatcher: no plan or empty plan — skipping fan-out")
        return {}

    sends: list[Send] = []
    pending: list[str] = []
    invalid: list[str] = []

    async with track_execution(state, "dispatcher") as execution:
        for task in plan.tasks:
            route = get_route(task.agent_type)
            if route is not None:
                sends.append(Send(route, {"active_task": task}))
                pending.append(task.id)
                logger.debug("dispatcher: queuing task %s → node %r", task.id, route)
            else:
                invalid.append(task.id)
                logger.warning(
                    "dispatcher: unknown agent_type %r for task %s — skipping",
                    task.agent_type,
                    task.id,
                )

    logger.info(
        "dispatcher: %d task(s) dispatched, %d invalid",
        len(sends),
        len(invalid),
    )

    update: dict[str, object] = {
        "pending_tasks": pending,
        "failed_tasks": invalid,
        "agent_executions": [execution],
    }

    if sends:
        return Command(update=update, goto=sends)

    # No valid sends — fall through to the static edge → END
    return update
