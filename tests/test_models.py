import pytest

from src.models.schemas import (
    AgentResult,
    Evidence,
    GapSeverity,
    ResearchGap,
    ResearchPlan,
    ResearchTask,
    Source,
    SourceType,
    TaskStatus,
)


class TestSource:
    def test_auto_id(self) -> None:
        source = Source()
        assert source.id != ""

    def test_unique_ids(self) -> None:
        assert Source().id != Source().id

    def test_default_source_type(self) -> None:
        assert Source().source_type == SourceType.WEB

    def test_url_and_title(self) -> None:
        source = Source(url="https://example.com", title="Example")
        assert source.url == "https://example.com"
        assert source.title == "Example"


class TestEvidence:
    def test_creation(self) -> None:
        evidence = Evidence(content="some content", source=Source())
        assert evidence.id != ""
        assert evidence.content == "some content"
        assert evidence.confidence == 1.0
        assert evidence.tags == []

    def test_confidence_bounds(self) -> None:
        with pytest.raises(ValueError):
            Evidence(content="x", source=Source(), confidence=1.5)
        with pytest.raises(ValueError):
            Evidence(content="x", source=Source(), confidence=-0.1)


class TestResearchTask:
    def test_defaults(self) -> None:
        task = ResearchTask(name="T", description="D", agent_type="planner")
        assert task.status == TaskStatus.PENDING
        assert task.priority == 5
        assert task.dependencies == []

    def test_priority_bounds(self) -> None:
        with pytest.raises(ValueError):
            ResearchTask(name="T", description="D", agent_type="a", priority=0)
        with pytest.raises(ValueError):
            ResearchTask(name="T", description="D", agent_type="a", priority=11)


class TestResearchPlan:
    def test_empty_plan(self) -> None:
        plan = ResearchPlan()
        assert plan.id != ""
        assert plan.tasks == []

    def test_plan_with_tasks(self) -> None:
        task = ResearchTask(name="T", description="D", agent_type="a")
        plan = ResearchPlan(tasks=[task])
        assert len(plan.tasks) == 1


class TestAgentResult:
    def test_success(self) -> None:
        result = AgentResult(agent_type="planner", success=True)
        assert result.success is True
        assert result.data == {}
        assert result.evidence == []
        assert result.errors == []

    def test_failure_with_errors(self) -> None:
        result = AgentResult(agent_type="planner", success=False, errors=["timeout"])
        assert result.success is False
        assert result.errors == ["timeout"]


class TestResearchGap:
    def test_defaults(self) -> None:
        gap = ResearchGap(topic="leadership", description="no data")
        assert gap.id != ""
        assert gap.severity == GapSeverity.MEDIUM
        assert gap.suggested_agents == []

    def test_high_severity(self) -> None:
        gap = ResearchGap(topic="financials", description="no data", severity=GapSeverity.HIGH)
        assert gap.severity == GapSeverity.HIGH
