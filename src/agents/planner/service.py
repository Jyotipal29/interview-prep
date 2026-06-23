from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

from src.agents.planner.prompt import PLANNER_PROMPT
from src.agents.planner.schemas import PlannerOutput
from src.config.research_config import ResearchConfig
from src.models.schemas import ResearchPlan, ResearchTask
from src.shared.logging import get_logger

logger = get_logger(__name__)


class PlannerService:
    def __init__(self, llm: BaseChatModel) -> None:
        structured_llm: Runnable = llm.with_structured_output(PlannerOutput)
        self._chain: Runnable = PLANNER_PROMPT | structured_llm

    async def plan(
        self,
        company_name: str,
        user_query: str,
        config: ResearchConfig,
    ) -> ResearchPlan:
        effective_query = user_query.strip() or f"Research {company_name}"

        logger.info(
            "planner: planning for company=%r query=%r", company_name, effective_query
        )

        output: PlannerOutput = await self._chain.ainvoke(
            {"company_name": company_name, "user_query": effective_query}
        )

        plan = self._to_research_plan(output)
        logger.info("planner: selected %d domains", len(plan.tasks))
        return plan

    def _to_research_plan(self, output: PlannerOutput) -> ResearchPlan:
        tasks = [
            ResearchTask(
                name=task.agent_name.value,
                description=task.reason,
                agent_type=task.agent_name.value,
                priority=task.priority,
            )
            for task in output.planned_tasks
        ]
        return ResearchPlan(tasks=tasks)
