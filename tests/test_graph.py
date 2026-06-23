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


def test_create_initial_state_has_required_keys() -> None:
    state = create_initial_state("Anthropic")
    required_keys = {
        "company_name",
        "raw_evidence",
        "verified_evidence",
        "research_gaps",
        "agent_results",
        "errors",
    }
    assert required_keys.issubset(state.keys())
    assert state["company_name"] == "Anthropic"
