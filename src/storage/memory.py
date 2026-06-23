from src.models.schemas import Evidence, ResearchPlan
from src.storage.interfaces import EvidenceRepository, ResearchRepository


class InMemoryEvidenceRepository(EvidenceRepository):
    """Thread-unsafe in-process evidence store for development and testing."""

    def __init__(self) -> None:
        self._store: dict[str, Evidence] = {}

    async def save(self, evidence: Evidence) -> None:
        self._store[evidence.id] = evidence

    async def get_by_id(self, evidence_id: str) -> Evidence | None:
        return self._store.get(evidence_id)

    async def find_by_tags(self, tags: list[str]) -> list[Evidence]:
        tag_set = set(tags)
        return [e for e in self._store.values() if tag_set.intersection(e.tags)]


class InMemoryResearchRepository(ResearchRepository):
    """Thread-unsafe in-process plan store for development and testing."""

    def __init__(self) -> None:
        self._plans: dict[str, ResearchPlan] = {}

    async def save_plan(self, plan: ResearchPlan) -> None:
        self._plans[plan.id] = plan

    async def get_plan(self, plan_id: str) -> ResearchPlan | None:
        return self._plans.get(plan_id)
