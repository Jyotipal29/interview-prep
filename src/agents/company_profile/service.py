from src.agents.company_profile.extractor import CompanyProfileExtractor
from src.agents.company_profile.schemas import (
    CitedInt,
    CitedList,
    CitedStr,
    CompanyFacts,
    CompanyProfile,
)
from src.models.schemas import AgentResult, Evidence, Source
from src.shared.context import RuntimeContext
from src.shared.logging import get_logger

logger = get_logger(__name__)

_QUERY_TEMPLATES = [
    "{company} company overview",
    "{company} headquarters location",
    "{company} number of employees",
    "{company} products and services",
    "{company} business model revenue",
]


class CompanyProfileService:
    """Orchestrates the full company profile research workflow."""

    def __init__(self, runtime: RuntimeContext) -> None:
        self._runtime = runtime
        self._extractor = CompanyProfileExtractor(runtime.llm)

    def generate_queries(self, company_name: str) -> list[str]:
        return [t.format(company=company_name) for t in _QUERY_TEMPLATES]

    async def research(
        self, company_name: str
    ) -> tuple[CompanyProfile, list[Evidence], list[str]]:
        """Run the full research workflow.

        Returns:
            (profile, raw_evidence, errors) — errors is non-empty when individual
            search/crawl/extraction steps fail but overall research still completes.
        """
        rt = self._runtime
        cfg = rt.research_config
        errors: list[str] = []
        all_evidence: list[Evidence] = []

        # Steps 1-2: Generate queries and search
        queries = self.generate_queries(company_name)
        search_evidence: list[Evidence] = []
        for query in queries:
            try:
                results = await rt.search_tool.search(query, max_results=cfg.max_search_results)
                search_evidence.extend(results)
                all_evidence.extend(results)
            except Exception as exc:
                errors.append(f"search failed for {query!r}: {exc}")
                logger.warning("service: search error for %r: %s", query, exc)

        # Step 3: Select best candidate URLs (deduplicate, cap at max_crawl_pages)
        seen: set[str] = set()
        candidate_urls: list[str] = []
        for ev in search_evidence:
            url = ev.source.url
            if url and url not in seen:
                seen.add(url)
                candidate_urls.append(url)
                if len(candidate_urls) >= cfg.max_crawl_pages:
                    break

        # Steps 4-5: Crawl each URL and extract facts
        facts_with_sources: list[tuple[CompanyFacts, Source]] = []
        for url in candidate_urls:
            try:
                page = await rt.crawler_tool.crawl(url)
                all_evidence.append(page)
                facts = await self._extractor.extract(company_name, page)
                facts_with_sources.append((facts, page.source))
                logger.debug("service: extracted facts from %s", url)
            except Exception as exc:
                errors.append(f"crawl/extract failed for {url}: {exc}")
                logger.warning("service: crawl/extract error for %s: %s", url, exc)

        # Steps 6-7: Merge extracted facts into a single cited profile
        profile = self._merge(facts_with_sources)
        logger.info(
            "service: company_profile done — sources=%d errors=%d",
            len(facts_with_sources),
            len(errors),
        )
        return profile, all_evidence, errors

    def _merge(self, facts_with_sources: list[tuple[CompanyFacts, Source]]) -> CompanyProfile:
        """Merge facts from multiple pages into one CompanyProfile with citations."""
        profile = CompanyProfile()

        for facts, source in facts_with_sources:
            if facts.name:
                if profile.name is None:
                    profile.name = CitedStr(value=facts.name, sources=[source])
                elif source not in profile.name.sources:
                    profile.name.sources.append(source)

            if facts.description and profile.description is None:
                profile.description = CitedStr(value=facts.description, sources=[source])

            if facts.industry and profile.industry is None:
                profile.industry = CitedStr(value=facts.industry, sources=[source])

            if facts.headquarters and profile.headquarters is None:
                profile.headquarters = CitedStr(value=facts.headquarters, sources=[source])

            if facts.founded_year is not None and profile.founded_year is None:
                profile.founded_year = CitedInt(value=facts.founded_year, sources=[source])

            if facts.employee_count and profile.employee_count is None:
                profile.employee_count = CitedStr(value=facts.employee_count, sources=[source])

            if facts.business_model and profile.business_model is None:
                profile.business_model = CitedStr(value=facts.business_model, sources=[source])

            if facts.products:
                if profile.products is None:
                    profile.products = CitedList(values=list(facts.products), sources=[source])
                else:
                    existing = set(profile.products.values)
                    new_vals = [v for v in facts.products if v not in existing]
                    if new_vals:
                        profile.products.values.extend(new_vals)
                        profile.products.sources.append(source)

            if facts.target_customers:
                if profile.target_customers is None:
                    profile.target_customers = CitedList(
                        values=list(facts.target_customers), sources=[source]
                    )
                else:
                    existing = set(profile.target_customers.values)
                    new_vals = [v for v in facts.target_customers if v not in existing]
                    if new_vals:
                        profile.target_customers.values.extend(new_vals)
                        profile.target_customers.sources.append(source)

            if facts.markets:
                if profile.markets is None:
                    profile.markets = CitedList(values=list(facts.markets), sources=[source])
                else:
                    existing = set(profile.markets.values)
                    new_vals = [v for v in facts.markets if v not in existing]
                    if new_vals:
                        profile.markets.values.extend(new_vals)
                        profile.markets.sources.append(source)

        return profile

    def build_agent_result(
        self,
        profile: CompanyProfile,
        evidence: list[Evidence],
        errors: list[str],
    ) -> AgentResult:
        return AgentResult(
            agent_type="company_profile",
            success=len(errors) == 0,
            data=profile.model_dump(mode="json"),
            evidence=evidence,
            errors=errors,
        )
