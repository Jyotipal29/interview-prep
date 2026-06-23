from src.agents.company_profile.service import CompanyProfileService
from src.graph.types import Command
from src.shared.context import RuntimeContext
from src.shared.tracking import track_execution
from src.state.research_state import ResearchState


async def company_profile_node(
    state: ResearchState,
    runtime: RuntimeContext,
) -> Command | dict[str, object]:
    task = state.get("active_task")
    service = CompanyProfileService(runtime)

    try:
        async with track_execution(state, "company_profile") as execution:
            profile, raw_evidence, errors = await service.research(state["company_name"])
    except Exception:
        return {
            "agent_executions": [execution],
            "failed_tasks": [task.id] if task else [],
        }

    result = service.build_agent_result(profile, raw_evidence, errors)

    return {
        "company_profile": profile.model_dump(mode="json"),
        "raw_evidence": raw_evidence,
        "agent_results": [result],
        "agent_executions": [execution],
        "completed_tasks": [task.id] if task else [],
        "errors": errors,
    }
