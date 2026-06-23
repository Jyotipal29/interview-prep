import operator
from typing import Annotated, Any, TypedDict

from src.models.schemas import AgentResult, Evidence, ResearchGap, ResearchPlan


class ResearchState(TypedDict, total=False):
    """
    LangGraph state schema for the company research graph.

    Fields typed as Annotated[list, operator.add] use append-semantics —
    node updates are concatenated onto the existing list rather than
    replacing it.  All other fields use replace-semantics (last write wins).

    total=False makes all keys optional so nodes can return partial updates
    without being required to supply every field.
    """

    # ── Input ──────────────────────────────────────────────────────────────
    company_name: str

    # ── Planning ───────────────────────────────────────────────────────────
    research_plan: ResearchPlan | None

    # ── Agent data (replace-semantics: latest agent output wins) ───────────
    company_profile: dict[str, Any] | None
    leadership_data: dict[str, Any] | None
    compensation_data: dict[str, Any] | None
    employee_sentiment_data: dict[str, Any] | None
    tech_stack_data: dict[str, Any] | None
    financial_data: dict[str, Any] | None
    interview_data: dict[str, Any] | None
    news_data: dict[str, Any] | None
    competitor_data: dict[str, Any] | None

    # ── Evidence (append-semantics: accumulate across all agents) ───────────
    raw_evidence: Annotated[list[Evidence], operator.add]
    verified_evidence: Annotated[list[Evidence], operator.add]

    # ── Analysis outputs (replace-semantics) ───────────────────────────────
    culture_analysis: dict[str, Any] | None
    compensation_analysis: dict[str, Any] | None
    risk_analysis: dict[str, Any] | None
    hiring_assessment: dict[str, Any] | None

    # ── Synthesis (append-semantics) ───────────────────────────────────────
    research_gaps: Annotated[list[ResearchGap], operator.add]
    agent_results: Annotated[list[AgentResult], operator.add]
    errors: Annotated[list[str], operator.add]

    # ── Output ─────────────────────────────────────────────────────────────
    final_report: str | None


def create_initial_state(company_name: str) -> ResearchState:
    """Return a fully-initialised state dict for a new research run."""
    return ResearchState(
        company_name=company_name,
        research_plan=None,
        company_profile=None,
        leadership_data=None,
        compensation_data=None,
        employee_sentiment_data=None,
        tech_stack_data=None,
        financial_data=None,
        interview_data=None,
        news_data=None,
        competitor_data=None,
        raw_evidence=[],
        verified_evidence=[],
        culture_analysis=None,
        compensation_analysis=None,
        risk_analysis=None,
        hiring_assessment=None,
        research_gaps=[],
        agent_results=[],
        errors=[],
        final_report=None,
    )
