from abc import ABC, abstractmethod
from typing import Any

from src.models.schemas import Evidence


class SearchTool(ABC):
    """Interface for web / document search providers (Tavily, Exa, …)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool identifier used for logging and registry lookup."""

    @abstractmethod
    async def search(self, query: str, **kwargs: Any) -> list[Evidence]:
        """
        Execute a search and return a ranked list of Evidence objects.

        Each Evidence item contains the raw content and a fully-populated
        Source with URL, title, and retrieval timestamp.
        """


class CrawlerTool(ABC):
    """Interface for web page crawling / scraping providers (Firecrawl, …)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool identifier used for logging and registry lookup."""

    @abstractmethod
    async def crawl(self, url: str, **kwargs: Any) -> Evidence:
        """
        Fetch and clean the content at *url*.

        Returns a single Evidence object whose Source records the crawled URL
        and the retrieval timestamp.
        """


class ExtractionTool(ABC):
    """Interface for structured-data extraction from unstructured text."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool identifier used for logging and registry lookup."""

    @abstractmethod
    async def extract(self, content: str, schema: type, **kwargs: Any) -> Any:
        """
        Parse *content* into an instance of *schema*.

        *schema* is a Pydantic model class; the implementation decides
        whether to call an LLM, use regex, or another extraction strategy.
        """
