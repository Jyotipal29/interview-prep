from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from src.models.schemas import Evidence

from src.agents.company_profile.prompt import EXTRACTION_PROMPT
from src.agents.company_profile.schemas import CompanyFacts
from src.shared.logging import get_logger

logger = get_logger(__name__)

_MAX_CONTENT_CHARS = 8_000


class CompanyProfileExtractor:
    """Extracts structured company facts from a crawled page using LLM structured output."""

    def __init__(self, llm: BaseChatModel) -> None:
        structured_llm = llm.with_structured_output(CompanyFacts)
        self._chain = EXTRACTION_PROMPT | structured_llm

    async def extract(self, company_name: str, page: Evidence) -> CompanyFacts:
        content = page.content[:_MAX_CONTENT_CHARS]
        logger.debug("extractor: extracting from %s (%d chars)", page.source.url, len(content))
        facts: CompanyFacts = await self._chain.ainvoke(
            {"company_name": company_name, "content": content}
        )
        return facts
