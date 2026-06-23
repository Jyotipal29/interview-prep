"""Unit tests for the Planner agent — all LLM calls are mocked."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.planner.node import planner_node
from src.agents.planner.schemas import ALL_DOMAINS, PlannedTask, PlannerOutput, ResearchDomain
from src.agents.planner.service import PlannerService
from src.config.research_config import ResearchConfig
from src.models.schemas import ExecutionStatus, ResearchPlan
from src.state.research_state import create_initial_state

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_service(output: PlannerOutput) -> PlannerService:
    """Build a PlannerService whose chain.ainvoke is pre-canned."""
    svc = PlannerService.__new__(PlannerService)
    svc._chain = MagicMock()
    svc._chain.ainvoke = AsyncMock(return_value=output)
    return svc


def _make_output(*domains: ResearchDomain) -> PlannerOutput:
    return PlannerOutput(
        planned_tasks=[
            PlannedTask(agent_name=d, priority=i + 1, reason=f"reason for {d.value}")
            for i, d in enumerate(domains)
        ]
    )


_config = ResearchConfig()

# ── Service._to_research_plan ─────────────────────────────────────────────────


class TestToPlan:
    def test_maps_all_fields(self) -> None:
        output = _make_output(ResearchDomain.COMPENSATION, ResearchDomain.EMPLOYEE_SENTIMENT)
        svc = _make_service(output)
        plan = svc._to_research_plan(output)

        assert len(plan.tasks) == 2
        t = plan.tasks[0]
        assert t.name == "compensation"
        assert t.agent_type == "compensation"
        assert t.priority == 1
        assert t.description == "reason for compensation"

    def test_empty_output_gives_empty_plan(self) -> None:
        output = PlannerOutput(planned_tasks=[])
        svc = _make_service(output)
        plan = svc._to_research_plan(output)
        assert isinstance(plan, ResearchPlan)
        assert plan.tasks == []


# ── Service.plan — five core scenarios ───────────────────────────────────────


class TestPlannerService:
    async def test_full_company_research(self) -> None:
        """Generic 'Research <company>' query → all 9 domains selected."""
        output = _make_output(*ALL_DOMAINS)
        svc = _make_service(output)

        plan = await svc.plan("eBay", "Research eBay", _config)

        assert len(plan.tasks) == 9
        names = {t.name for t in plan.tasks}
        assert names == {d.value for d in ALL_DOMAINS}

    async def test_compensation_only_research(self) -> None:
        """Compensation query → compensation + employee_sentiment."""
        output = _make_output(ResearchDomain.COMPENSATION, ResearchDomain.EMPLOYEE_SENTIMENT)
        svc = _make_service(output)

        plan = await svc.plan("Stripe", "Research Stripe compensation", _config)

        assert len(plan.tasks) == 2
        names = {t.name for t in plan.tasks}
        assert names == {"compensation", "employee_sentiment"}

    async def test_leadership_only_research(self) -> None:
        """Leadership query → leadership domain only."""
        output = _make_output(ResearchDomain.LEADERSHIP)
        svc = _make_service(output)

        plan = await svc.plan("OpenAI", "Research OpenAI leadership", _config)

        assert len(plan.tasks) == 1
        assert plan.tasks[0].name == "leadership"

    async def test_interview_only_research(self) -> None:
        """Interview query → interviews domain only."""
        output = _make_output(ResearchDomain.INTERVIEWS)
        svc = _make_service(output)

        plan = await svc.plan("Airbnb", "Research Airbnb interview process", _config)

        assert len(plan.tasks) == 1
        assert plan.tasks[0].name == "interviews"

    async def test_empty_query_defaults_to_company_research(self) -> None:
        """Blank user_query is normalised to 'Research <company>' before the chain call."""
        output = _make_output(*ALL_DOMAINS)
        svc = _make_service(output)

        plan = await svc.plan("Datadog", "", _config)

        # Chain must have been invoked with the default query string
        svc._chain.ainvoke.assert_awaited_once()
        call_kwargs = svc._chain.ainvoke.call_args[0][0]
        assert call_kwargs["user_query"] == "Research Datadog"
        assert len(plan.tasks) == 9

    async def test_chain_receives_correct_company_and_query(self) -> None:
        """Chain is called with the exact company_name and user_query from plan()."""
        output = _make_output(ResearchDomain.NEWS)
        svc = _make_service(output)

        await svc.plan("Meta", "Research Meta news", _config)

        svc._chain.ainvoke.assert_awaited_once_with(
            {"company_name": "Meta", "user_query": "Research Meta news"}
        )


# ── Planner node ──────────────────────────────────────────────────────────────


class TestPlannerNode:
    def _make_runtime(self, plan: ResearchPlan) -> MagicMock:
        runtime = MagicMock()
        runtime.llm = MagicMock()
        runtime.research_config = _config
        # Patch PlannerService inside the node so no LLM call is made
        return runtime

    async def test_node_returns_plan_and_execution(self) -> None:
        state = create_initial_state("eBay", "Research eBay")
        expected_plan = ResearchPlan(tasks=[])

        with patch("src.agents.planner.node.PlannerService") as MockService:
            MockService.return_value.plan = AsyncMock(return_value=expected_plan)
            runtime = MagicMock(research_config=_config)

            result = await planner_node(state, runtime)

        assert result["research_plan"] is expected_plan
        assert len(result["agent_executions"]) == 1

    async def test_execution_status_is_completed(self) -> None:
        state = create_initial_state("eBay", "Research eBay")

        with patch("src.agents.planner.node.PlannerService") as MockService:
            MockService.return_value.plan = AsyncMock(return_value=ResearchPlan())
            runtime = MagicMock(research_config=_config)

            result = await planner_node(state, runtime)

        execution = result["agent_executions"][0]
        assert execution.agent_name == "planner"
        assert execution.status == ExecutionStatus.COMPLETED
        assert execution.duration_ms is not None

    async def test_execution_status_is_failed_on_error(self) -> None:
        state = create_initial_state("eBay", "Research eBay")

        with patch("src.agents.planner.node.PlannerService") as MockService:
            MockService.return_value.plan = AsyncMock(side_effect=RuntimeError("LLM error"))
            runtime = MagicMock(research_config=_config)

            with pytest.raises(RuntimeError, match="LLM error"):
                await planner_node(state, runtime)

    async def test_node_passes_user_query_from_state(self) -> None:
        state = create_initial_state("Datadog", "Research Datadog engineering culture")

        with patch("src.agents.planner.node.PlannerService") as MockService:
            mock_svc = MockService.return_value
            mock_svc.plan = AsyncMock(return_value=ResearchPlan())
            runtime = MagicMock(research_config=_config)

            await planner_node(state, runtime)

        mock_svc.plan.assert_awaited_once_with(
            company_name="Datadog",
            user_query="Research Datadog engineering culture",
            config=_config,
        )
