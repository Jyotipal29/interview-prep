from enum import StrEnum

from pydantic import BaseModel, Field


class ResearchDomain(StrEnum):
    COMPANY_PROFILE = "company_profile"
    LEADERSHIP = "leadership"
    COMPENSATION = "compensation"
    EMPLOYEE_SENTIMENT = "employee_sentiment"
    TECH_STACK = "tech_stack"
    FINANCIALS = "financials"
    INTERVIEWS = "interviews"
    NEWS = "news"
    COMPETITORS = "competitors"


ALL_DOMAINS: tuple[ResearchDomain, ...] = tuple(ResearchDomain)


class PlannedTask(BaseModel):
    agent_name: ResearchDomain = Field(description="The research domain to investigate")
    priority: int = Field(ge=1, le=9, description="Execution priority — 1 is highest")
    reason: str = Field(description="Why this domain was selected for the given query")


class PlannerOutput(BaseModel):
    """Structured output from the LLM planner — consumed directly, no JSON parsing."""

    planned_tasks: list[PlannedTask] = Field(
        description="Research tasks to execute, ordered by decreasing importance"
    )
