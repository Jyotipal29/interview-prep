import pytest

from src.graph.builder import build_graph
from src.state.research_state import create_initial_state


def test_graph_compiles() -> None:
    graph = build_graph()
    assert graph is not None


def test_graph_has_placeholder_node() -> None:
    graph = build_graph()
    assert "placeholder_node" in graph.nodes


def test_graph_invoke_returns_state(initial_state: dict[str, object]) -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": "test-no-checkpointer"}}
    result = graph.invoke(initial_state, config=config)
    assert result["company_name"] == "TestCorp"


def test_graph_preserves_list_fields(initial_state: dict[str, object]) -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": "test-lists"}}
    result = graph.invoke(initial_state, config=config)
    assert result["raw_evidence"] == []
    assert result["errors"] == []
    assert result["research_gaps"] == []
    assert result["agent_executions"] == []


def test_create_initial_state_has_required_keys() -> None:
    state = create_initial_state("Anthropic")
    required_keys = {
        "research_id",
        "company_name",
        "raw_evidence",
        "verified_evidence",
        "research_gaps",
        "agent_results",
        "agent_executions",
        "errors",
    }
    assert required_keys.issubset(state.keys())
    assert state["company_name"] == "Anthropic"


def test_create_initial_state_generates_research_id() -> None:
    state = create_initial_state("Anthropic")
    assert isinstance(state["research_id"], str)
    assert len(state["research_id"]) > 0


def test_each_run_gets_unique_research_id() -> None:
    ids = {create_initial_state("Corp")["research_id"] for _ in range(10)}
    assert len(ids) == 10


def test_command_is_importable_from_graph_types() -> None:
    from src.graph.types import Command, Send

    assert Command is not None
    assert Send is not None


def test_command_update_only() -> None:
    from src.graph.types import Command

    cmd = Command(update={"company_name": "Acme"})
    assert cmd.update == {"company_name": "Acme"}


def test_command_with_goto() -> None:
    from src.graph.types import Command

    cmd = Command(update={"errors": ["retry"]}, goto="planner_node")
    assert cmd.goto == "planner_node"


def test_placeholder_node_return_type_is_dict() -> None:
    from src.graph.nodes import placeholder_node
    from src.state.research_state import create_initial_state

    state = create_initial_state("TypeCo")
    result = placeholder_node(state)
    assert isinstance(result, dict)
