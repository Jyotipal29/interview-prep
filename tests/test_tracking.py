import pytest

from src.models.schemas import ExecutionStatus
from src.shared.tracking import track_execution
from src.state.research_state import create_initial_state


@pytest.fixture()
def state() -> dict[str, object]:
    return create_initial_state("TrackCorp")


async def test_successful_execution_sets_completed_status(state: dict[str, object]) -> None:
    async with track_execution(state, "planner") as execution:
        pass

    assert execution.status == ExecutionStatus.COMPLETED
    assert execution.completed_at is not None
    assert execution.duration_ms is not None
    assert execution.duration_ms >= 0
    assert execution.error_message is None


async def test_execution_starts_as_running(state: dict[str, object]) -> None:
    captured_status = None

    async with track_execution(state, "planner") as execution:
        captured_status = execution.status

    assert captured_status == ExecutionStatus.RUNNING
    assert execution.status == ExecutionStatus.COMPLETED


async def test_failed_execution_sets_failed_status(state: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="something broke"):
        async with track_execution(state, "leadership") as execution:
            raise ValueError("something broke")

    assert execution.status == ExecutionStatus.FAILED
    assert execution.error_message == "something broke"
    assert execution.completed_at is not None
    assert execution.duration_ms is not None


async def test_exception_is_re_raised(state: dict[str, object]) -> None:
    with pytest.raises(RuntimeError):
        async with track_execution(state, "news") as _:
            raise RuntimeError("propagated")


async def test_duration_is_non_negative(state: dict[str, object]) -> None:
    async with track_execution(state, "compensation") as execution:
        pass

    assert execution.duration_ms >= 0


async def test_agent_name_is_preserved(state: dict[str, object]) -> None:
    async with track_execution(state, "tech_stack") as execution:
        pass

    assert execution.agent_name == "tech_stack"


async def test_execution_can_be_collected_in_return_dict(state: dict[str, object]) -> None:
    async with track_execution(state, "competitor") as execution:
        work_result = {"data": "found"}

    node_update = {
        "competitor_data": work_result,
        "agent_executions": [execution],
    }

    assert len(node_update["agent_executions"]) == 1
    assert node_update["agent_executions"][0].status == ExecutionStatus.COMPLETED
