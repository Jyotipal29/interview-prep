import pytest

from src.state.research_state import create_initial_state


@pytest.fixture()
def initial_state() -> dict[str, object]:
    return create_initial_state("TestCorp")
