"""Tests for the dispatcher node and routing layer."""

import pytest

from src.agents.planner.schemas import ALL_DOMAINS, ResearchDomain
from src.graph.dispatcher import dispatcher_node
from src.graph.routing import AGENT_ROUTES, get_route
from src.graph.types import Command, Send
from src.models.schemas import ExecutionStatus, ResearchPlan, ResearchTask
from src.state.research_state import ResearchState, create_initial_state

# ── Helpers ───────────────────────────────────────────────────────────────────


def _task(agent_type: str) -> ResearchTask:
    return ResearchTask(name=agent_type, description="test", agent_type=agent_type)


def _state_with_plan(*agent_types: str) -> ResearchState:
    state = create_initial_state("Acme")
    state["research_plan"] = ResearchPlan(tasks=[_task(t) for t in agent_types])
    return state


def _sends(result: Command) -> list[Send]:
    goto = result.goto
    if isinstance(goto, list):
        return [g for g in goto if isinstance(g, Send)]
    return [goto] if isinstance(goto, Send) else []


# ── Routing ───────────────────────────────────────────────────────────────────


class TestRouting:
    def test_all_domains_have_routes(self) -> None:
        for domain in ALL_DOMAINS:
            assert domain in AGENT_ROUTES, f"missing route for {domain}"

    def test_route_count_matches_domain_count(self) -> None:
        assert len(AGENT_ROUTES) == len(ALL_DOMAINS)

    def test_get_route_known_domain(self) -> None:
        assert get_route("company_profile") == "company_profile"
        assert get_route("compensation") == "compensation"

    def test_get_route_unknown_domain_returns_none(self) -> None:
        assert get_route("nonexistent") is None
        assert get_route("") is None


# ── Dispatcher — happy paths ──────────────────────────────────────────────────


class TestDispatcherNode:
    async def test_single_task_dispatch(self) -> None:
        """One valid task produces one Send to the correct node."""
        task = _task("company_profile")
        state = create_initial_state("Acme")
        state["research_plan"] = ResearchPlan(tasks=[task])

        result = await dispatcher_node(state)

        assert isinstance(result, Command)
        sends = _sends(result)
        assert len(sends) == 1
        assert sends[0].node == "company_profile"
        assert result.update["pending_tasks"] == [task.id]
        assert result.update["failed_tasks"] == []

    async def test_multi_task_dispatch(self) -> None:
        """Three valid tasks produce three Sends."""
        domains = ["company_profile", "leadership", "compensation"]
        tasks = [_task(d) for d in domains]
        state = create_initial_state("Acme")
        state["research_plan"] = ResearchPlan(tasks=tasks)

        result = await dispatcher_node(state)

        assert isinstance(result, Command)
        sends = _sends(result)
        assert len(sends) == 3
        node_names = {s.node for s in sends}
        assert node_names == set(domains)
        assert set(result.update["pending_tasks"]) == {t.id for t in tasks}

    async def test_all_task_dispatch(self) -> None:
        """All 9 domains produce 9 Sends — one per AGENT_ROUTES entry."""
        state = _state_with_plan(*[d.value for d in ALL_DOMAINS])

        result = await dispatcher_node(state)

        assert isinstance(result, Command)
        sends = _sends(result)
        assert len(sends) == len(ALL_DOMAINS)
        node_names = {s.node for s in sends}
        assert node_names == set(AGENT_ROUTES.values())

    async def test_send_payload_contains_active_task(self) -> None:
        """The arg passed via Send must include the active_task for the stub."""
        task = _task("leadership")
        state = create_initial_state("Acme")
        state["research_plan"] = ResearchPlan(tasks=[task])

        result = await dispatcher_node(state)

        sends = _sends(result)
        assert len(sends) == 1
        assert sends[0].arg["active_task"] == task

    # ── Invalid task handling ─────────────────────────────────────────────

    async def test_invalid_task_skipped_returns_plain_dict(self) -> None:
        """All-invalid plan: no Sends emitted, result is a plain dict."""
        state = _state_with_plan("nonexistent_domain")

        result = await dispatcher_node(state)

        assert isinstance(result, dict)
        assert not isinstance(result, Command)

    async def test_invalid_task_id_in_failed_tasks(self) -> None:
        """Invalid agent_type is recorded in failed_tasks."""
        task = _task("nonexistent_domain")
        state = create_initial_state("Acme")
        state["research_plan"] = ResearchPlan(tasks=[task])

        result = await dispatcher_node(state)

        assert task.id in result["failed_tasks"]
        assert result["pending_tasks"] == []

    async def test_mixed_valid_and_invalid_tasks(self) -> None:
        """Mix: valid task dispatched, invalid task goes to failed_tasks."""
        valid = _task("news")
        invalid = _task("nonexistent_domain")
        state = create_initial_state("Acme")
        state["research_plan"] = ResearchPlan(tasks=[valid, invalid])

        result = await dispatcher_node(state)

        assert isinstance(result, Command)
        sends = _sends(result)
        assert len(sends) == 1
        assert sends[0].node == "news"
        assert valid.id in result.update["pending_tasks"]
        assert invalid.id in result.update["failed_tasks"]

    async def test_multiple_invalid_tasks_all_in_failed_tasks(self) -> None:
        """Every invalid task ends up in failed_tasks."""
        tasks = [_task("bad1"), _task("bad2"), _task("bad3")]
        state = create_initial_state("Acme")
        state["research_plan"] = ResearchPlan(tasks=tasks)

        result = await dispatcher_node(state)

        assert isinstance(result, dict)
        assert set(result["failed_tasks"]) == {t.id for t in tasks}

    # ── Edge cases ────────────────────────────────────────────────────────

    async def test_no_plan_returns_empty_dict(self) -> None:
        state = create_initial_state("Acme")  # research_plan is None

        result = await dispatcher_node(state)

        assert result == {}

    async def test_empty_plan_returns_empty_dict(self) -> None:
        state = create_initial_state("Acme")
        state["research_plan"] = ResearchPlan(tasks=[])

        result = await dispatcher_node(state)

        assert result == {}

    async def test_dispatcher_creates_execution_record(self) -> None:
        state = _state_with_plan("company_profile")

        result = await dispatcher_node(state)

        assert isinstance(result, Command)
        executions = result.update["agent_executions"]
        assert len(executions) == 1
        execution = executions[0]
        assert execution.agent_name == "dispatcher"
        assert execution.status == ExecutionStatus.COMPLETED
        assert execution.duration_ms is not None
