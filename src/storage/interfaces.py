from abc import ABC, abstractmethod

from src.models.schemas import Evidence, ResearchPlan


class EvidenceRepository(ABC):
    """Persistence interface for raw and verified evidence."""

    @abstractmethod
    async def save(self, evidence: Evidence) -> None:
        """Persist a single Evidence item (upsert by id)."""

    @abstractmethod
    async def get_by_id(self, evidence_id: str) -> Evidence | None:
        """Return the Evidence with *evidence_id*, or None if not found."""

    @abstractmethod
    async def find_by_tags(self, tags: list[str]) -> list[Evidence]:
        """Return all Evidence items that carry at least one of *tags*."""


class ResearchRepository(ABC):
    """Persistence interface for research plans and run metadata."""

    @abstractmethod
    async def save_plan(self, plan: ResearchPlan) -> None:
        """Persist a ResearchPlan (upsert by id)."""

    @abstractmethod
    async def get_plan(self, plan_id: str) -> ResearchPlan | None:
        """Return the ResearchPlan with *plan_id*, or None if not found."""
