from src.agents.planner.service import PlannerService
from src.graph.types import Command
from src.shared.context import RuntimeContext
from src.shared.tracking import track_execution
from src.state.research_state import ResearchState


async def planner_node(
    state: ResearchState,
    runtime: RuntimeContext,
) -> Command | dict[str, object]:
    service = PlannerService(runtime.llm)

    async with track_execution(state, "planner") as execution:
        plan = await service.plan(
            company_name=state["company_name"],
            user_query=state.get("user_query", ""),
            config=runtime.research_config,
        )

    return {
        "research_plan": plan,
        "agent_executions": [execution],
    }
