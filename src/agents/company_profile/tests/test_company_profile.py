"""Unit tests for the Company Profile agent — all LLM and tool calls are mocked."""

from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.company_profile.extractor import CompanyProfileExtractor
from src.agents.company_profile.node import company_profile_node
from src.agents.company_profile.schemas import (
    CitedStr,
    CompanyFacts,
    CompanyProfile,
)
from src.agents.company_profile.service import CompanyProfileService
from src.config.research_config import ResearchConfig
from src.models.schemas import AgentResult, Evidence, ExecutionStatus, ResearchTask, Source
from src.state.research_state import create_initial_state

# ── Fixtures and helpers ──────────────────────────────────────────────────────


def _source(url: str = "https://example.com") -> Source:
    return Source(url=url)


def _evidence(content: str = "some content", url: str = "https://example.com") -> Evidence:
    return Evidence(content=content, source=_source(url))


def _task(agent_type: str = "company_profile") -> ResearchTask:
    return ResearchTask(name=agent_type, description="test", agent_type=agent_type)


def _make_runtime(
    search_results: list[Evidence] | None = None,
    crawl_evidence: Evidence | None = None,
) -> MagicMock:
    rt = MagicMock()
    rt.research_config = ResearchConfig()
    rt.search_tool.search = AsyncMock(return_value=search_results or [])
    rt.crawler_tool.crawl = AsyncMock(return_value=crawl_evidence or _evidence())
    rt.llm = MagicMock()
    return rt


def _make_extractor(facts: CompanyFacts) -> CompanyProfileExtractor:
    ext = CompanyProfileExtractor.__new__(CompanyProfileExtractor)
    ext._chain = MagicMock()
    ext._chain.ainvoke = AsyncMock(return_value=facts)
    return ext


# ── Query generation ──────────────────────────────────────────────────────────


class TestQueryGeneration:
    def test_generates_five_queries(self) -> None:
        rt = _make_runtime()
        svc = CompanyProfileService(rt)
        queries = svc.generate_queries("eBay")
        assert len(queries) == 5

    def test_all_queries_contain_company_name(self) -> None:
        rt = _make_runtime()
        svc = CompanyProfileService(rt)
        queries = svc.generate_queries("eBay")
        assert all("eBay" in q for q in queries)

    def test_queries_are_distinct(self) -> None:
        rt = _make_runtime()
        svc = CompanyProfileService(rt)
        queries = svc.generate_queries("eBay")
        assert len(set(queries)) == len(queries)


# ── Extraction mapping / merge logic ─────────────────────────────────────────


class TestMerge:
    def _svc(self) -> CompanyProfileService:
        return CompanyProfileService(_make_runtime())

    def test_single_facts_mapped_to_profile(self) -> None:
        svc = self._svc()
        source = _source("https://ebay.com/about")
        facts = CompanyFacts(
            name="eBay",
            description="An e-commerce marketplace.",
            industry="E-commerce",
            headquarters="San Jose, California",
            founded_year=1995,
            employee_count="~9,000",
            business_model="marketplace",
            products=["marketplace", "payment processing"],
            target_customers=["consumers", "small businesses"],
            markets=["North America", "Europe"],
        )

        profile = svc._merge([(facts, source)])

        assert profile.name is not None and profile.name.value == "eBay"
        assert profile.description is not None and "marketplace" in profile.description.value
        assert profile.industry is not None and profile.industry.value == "E-commerce"
        hq = profile.headquarters
        assert hq is not None and hq.value == "San Jose, California"
        assert profile.founded_year is not None and profile.founded_year.value == 1995
        assert profile.employee_count is not None and profile.employee_count.value == "~9,000"
        assert profile.business_model is not None and profile.business_model.value == "marketplace"
        assert profile.products is not None and "marketplace" in profile.products.values
        tc = profile.target_customers
        assert tc is not None and "consumers" in tc.values
        assert profile.markets is not None and "North America" in profile.markets.values

    def test_first_source_wins_for_scalar_fields(self) -> None:
        svc = self._svc()
        s1, s2 = _source("https://a.com"), _source("https://b.com")
        f1 = CompanyFacts(name="eBay Inc", headquarters="San Jose")
        f2 = CompanyFacts(name="eBay", headquarters="Silicon Valley")

        profile = svc._merge([(f1, s1), (f2, s2)])

        assert profile.name is not None and profile.name.value == "eBay Inc"
        assert profile.headquarters is not None and profile.headquarters.value == "San Jose"

    def test_list_fields_merged_and_deduplicated(self) -> None:
        svc = self._svc()
        s1, s2 = _source("https://a.com"), _source("https://b.com")
        f1 = CompanyFacts(products=["marketplace", "ads"])
        f2 = CompanyFacts(products=["ads", "payments"])

        profile = svc._merge([(f1, s1), (f2, s2)])

        assert profile.products is not None
        assert profile.products.values == ["marketplace", "ads", "payments"]

    def test_sources_attached_to_fields(self) -> None:
        svc = self._svc()
        source = _source("https://ebay.com")
        facts = CompanyFacts(name="eBay", industry="E-commerce")

        profile = svc._merge([(facts, source)])

        assert source in profile.name.sources  # type: ignore[union-attr]
        assert source in profile.industry.sources  # type: ignore[union-attr]

    def test_empty_input_returns_empty_profile(self) -> None:
        svc = self._svc()
        profile = svc._merge([])
        assert isinstance(profile, CompanyProfile)
        assert profile.name is None

    def test_facts_with_none_fields_skipped(self) -> None:
        svc = self._svc()
        source = _source()
        facts = CompanyFacts()  # all None

        profile = svc._merge([(facts, source)])

        assert profile.name is None
        assert profile.industry is None


# ── Evidence creation ─────────────────────────────────────────────────────────


class TestEvidenceCreation:
    async def test_search_evidence_included_in_raw_evidence(self) -> None:
        ev = _evidence(url="https://ebay.com/about")
        rt = _make_runtime(search_results=[ev])
        svc = CompanyProfileService(rt)
        svc._extractor = _make_extractor(CompanyFacts(name="eBay"))

        _, raw_evidence, _ = await svc.research("eBay")

        assert ev in raw_evidence

    async def test_crawled_evidence_included_in_raw_evidence(self) -> None:
        search_ev = _evidence(url="https://ebay.com/about")
        crawl_ev = _evidence(content="full page content", url="https://ebay.com/about")
        rt = _make_runtime(search_results=[search_ev], crawl_evidence=crawl_ev)
        svc = CompanyProfileService(rt)
        svc._extractor = _make_extractor(CompanyFacts(name="eBay"))

        _, raw_evidence, _ = await svc.research("eBay")

        assert crawl_ev in raw_evidence


# ── Successful research orchestration ────────────────────────────────────────


class TestServiceResearch:
    async def test_returns_profile_evidence_and_no_errors_on_success(self) -> None:
        search_ev = _evidence(url="https://ebay.com/about")
        crawl_ev = _evidence(content="eBay is a marketplace.", url="https://ebay.com/about")
        rt = _make_runtime(search_results=[search_ev], crawl_evidence=crawl_ev)
        svc = CompanyProfileService(rt)
        svc._extractor = _make_extractor(CompanyFacts(name="eBay", industry="E-commerce"))

        profile, evidence, errors = await svc.research("eBay")

        assert isinstance(profile, CompanyProfile)
        assert profile.name is not None and profile.name.value == "eBay"
        assert len(evidence) > 0
        assert errors == []

    async def test_deduplicates_crawl_urls(self) -> None:
        # Three search results pointing to the same URL
        ev1 = _evidence(url="https://ebay.com")
        ev2 = _evidence(url="https://ebay.com")
        ev3 = _evidence(url="https://ebay.com")
        rt = _make_runtime(search_results=[ev1, ev2, ev3])
        svc = CompanyProfileService(rt)
        svc._extractor = _make_extractor(CompanyFacts())

        await svc.research("eBay")

        # crawl called exactly once despite 3 results
        assert rt.crawler_tool.crawl.call_count == 1


# ── Error handling ────────────────────────────────────────────────────────────


class TestSearchFailure:
    async def test_search_error_recorded_in_errors_list(self) -> None:
        rt = _make_runtime()
        rt.search_tool.search = AsyncMock(side_effect=RuntimeError("network timeout"))
        svc = CompanyProfileService(rt)

        profile, _, errors = await svc.research("eBay")

        assert isinstance(profile, CompanyProfile)
        assert any("search failed" in e for e in errors)

    async def test_partial_search_failure_continues(self) -> None:
        # First call fails, subsequent succeed
        rt = _make_runtime()
        rt.search_tool.search = AsyncMock(
            side_effect=[RuntimeError("timeout"), [_evidence()], [], [], []]
        )
        svc = CompanyProfileService(rt)
        svc._extractor = _make_extractor(CompanyFacts(name="eBay"))

        _, _, errors = await svc.research("eBay")

        assert len(errors) == 1
        assert "search failed" in errors[0]


class TestCrawlFailure:
    async def test_crawl_error_recorded_in_errors_list(self) -> None:
        search_ev = _evidence(url="https://ebay.com")
        rt = _make_runtime(search_results=[search_ev])
        rt.crawler_tool.crawl = AsyncMock(side_effect=RuntimeError("403 Forbidden"))
        svc = CompanyProfileService(rt)

        profile, _, errors = await svc.research("eBay")

        assert isinstance(profile, CompanyProfile)
        assert any("crawl/extract failed" in e for e in errors)

    async def test_crawl_failure_returns_empty_profile(self) -> None:
        search_ev = _evidence(url="https://ebay.com")
        rt = _make_runtime(search_results=[search_ev])
        rt.crawler_tool.crawl = AsyncMock(side_effect=RuntimeError("timeout"))
        svc = CompanyProfileService(rt)

        profile, _, _ = await svc.research("eBay")

        assert profile.name is None


class TestExtractionFailure:
    async def test_extraction_error_recorded_in_errors_list(self) -> None:
        search_ev = _evidence(url="https://ebay.com")
        crawl_ev = _evidence(content="page content", url="https://ebay.com")
        rt = _make_runtime(search_results=[search_ev], crawl_evidence=crawl_ev)
        svc = CompanyProfileService(rt)
        svc._extractor = _make_extractor(CompanyFacts())
        svc._extractor._chain.ainvoke = AsyncMock(side_effect=RuntimeError("LLM error"))

        _, _, errors = await svc.research("eBay")

        assert any("crawl/extract failed" in e for e in errors)


# ── Node ──────────────────────────────────────────────────────────────────────


class TestCompanyProfileNode:
    def _make_node_runtime(
        self,
        profile: CompanyProfile,
        errors: list[str] | None = None,
    ) -> MagicMock:
        rt = MagicMock()
        rt.research_config = ResearchConfig()
        rt.llm = MagicMock()
        rt.search_tool.search = AsyncMock(return_value=[])
        rt.crawler_tool.crawl = AsyncMock(return_value=_evidence())
        return rt

    async def test_node_returns_required_state_keys(self) -> None:
        state = create_initial_state("eBay")
        state["active_task"] = _task()
        rt = _make_runtime()

        with patch("src.agents.company_profile.node.CompanyProfileService") as MockSvc:
            profile = CompanyProfile(name=CitedStr(value="eBay"))
            mock_svc = MockSvc.return_value
            mock_svc.research = AsyncMock(return_value=(profile, [], []))
            mock_svc.build_agent_result = MagicMock(
                return_value=AgentResult(agent_type="company_profile", success=True)
            )

            result = await company_profile_node(state, rt)

        assert "company_profile" in result
        assert "raw_evidence" in result
        assert "agent_results" in result
        assert "agent_executions" in result
        assert "completed_tasks" in result

    async def test_node_marks_task_completed(self) -> None:
        task = _task()
        state = create_initial_state("eBay")
        state["active_task"] = task
        rt = _make_runtime()

        with patch("src.agents.company_profile.node.CompanyProfileService") as MockSvc:
            profile = CompanyProfile()
            mock_svc = MockSvc.return_value
            mock_svc.research = AsyncMock(return_value=(profile, [], []))
            mock_svc.build_agent_result = MagicMock(
                return_value=AgentResult(agent_type="company_profile", success=True)
            )

            result = await company_profile_node(state, rt)

        assert task.id in result["completed_tasks"]

    async def test_node_execution_status_is_completed(self) -> None:
        state = create_initial_state("eBay")
        state["active_task"] = _task()
        rt = _make_runtime()

        with patch("src.agents.company_profile.node.CompanyProfileService") as MockSvc:
            mock_svc = MockSvc.return_value
            mock_svc.research = AsyncMock(return_value=(CompanyProfile(), [], []))
            mock_svc.build_agent_result = MagicMock(
                return_value=AgentResult(agent_type="company_profile", success=True)
            )

            result = await company_profile_node(state, rt)

        execution = result["agent_executions"][0]
        assert execution.status == ExecutionStatus.COMPLETED
        assert execution.duration_ms is not None

    async def test_node_marks_task_failed_on_exception(self) -> None:
        task = _task()
        state = create_initial_state("eBay")
        state["active_task"] = task
        rt = _make_runtime()

        with patch("src.agents.company_profile.node.CompanyProfileService") as MockSvc:
            mock_svc = MockSvc.return_value
            mock_svc.research = AsyncMock(side_effect=RuntimeError("catastrophic failure"))

            result = await company_profile_node(state, rt)

        assert task.id in result["failed_tasks"]
        assert result["agent_executions"][0].status == ExecutionStatus.FAILED

    async def test_node_handles_no_active_task(self) -> None:
        state = create_initial_state("eBay")  # no active_task
        rt = _make_runtime()

        with patch("src.agents.company_profile.node.CompanyProfileService") as MockSvc:
            mock_svc = MockSvc.return_value
            mock_svc.research = AsyncMock(return_value=(CompanyProfile(), [], []))
            mock_svc.build_agent_result = MagicMock(
                return_value=AgentResult(agent_type="company_profile", success=True)
            )

            result = await company_profile_node(state, rt)

        assert result["completed_tasks"] == []
