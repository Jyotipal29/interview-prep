import pytest

from src.models.schemas import (
    AgentExecution,
    AgentResult,
    Evidence,
    ExecutionStatus,
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

    def test_default_confidence_score(self) -> None:
        assert Source().confidence_score == 1.0

    def test_confidence_score_bounds(self) -> None:
        with pytest.raises(ValueError):
            Source(confidence_score=1.1)
        with pytest.raises(ValueError):
            Source(confidence_score=-0.1)

    def test_confidence_score_boundaries(self) -> None:
        assert Source(confidence_score=0.0).confidence_score == 0.0
        assert Source(confidence_score=1.0).confidence_score == 1.0

    def test_domain_specific_source_types(self) -> None:
        for st in (
            SourceType.COMPANY_WEBSITE,
            SourceType.SEC_FILING,
            SourceType.GLASSDOOR,
            SourceType.REDDIT,
            SourceType.LINKEDIN,
            SourceType.NEWS,
            SourceType.BLOG,
            SourceType.JOB_POSTING,
        ):
            source = Source(source_type=st)
            assert source.source_type == st

    def test_legacy_source_types_preserved(self) -> None:
        for st in (SourceType.WEB, SourceType.DATABASE, SourceType.API, SourceType.DOCUMENT, SourceType.SOCIAL):
            assert Source(source_type=st).source_type == st


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


class TestAgentExecution:
    def test_running_by_default(self) -> None:
        ex = AgentExecution(agent_name="planner")
        assert ex.status == ExecutionStatus.RUNNING
        assert ex.started_at is not None
        assert ex.completed_at is None
        assert ex.duration_ms is None
        assert ex.error_message is None

    def test_completed_execution(self) -> None:
        from datetime import UTC, datetime, timedelta

        started = datetime.now(UTC)
        completed = started + timedelta(milliseconds=250)
        ex = AgentExecution(
            agent_name="company_profile",
            started_at=started,
            completed_at=completed,
            duration_ms=250,
            status=ExecutionStatus.COMPLETED,
        )
        assert ex.status == ExecutionStatus.COMPLETED
        assert ex.duration_ms == 250

    def test_failed_execution(self) -> None:
        ex = AgentExecution(
            agent_name="leadership",
            status=ExecutionStatus.FAILED,
            error_message="connection timeout",
        )
        assert ex.status == ExecutionStatus.FAILED
        assert ex.error_message == "connection timeout"

    def test_all_status_values(self) -> None:
        for status in ExecutionStatus:
            ex = AgentExecution(agent_name="test", status=status)
            assert ex.status == status
