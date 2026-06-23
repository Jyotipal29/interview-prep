from src.agents.planner.schemas import ResearchDomain

# Central mapping from domain name to the graph node that handles it.
# Future agents register themselves here by adding or replacing an entry.
AGENT_ROUTES: dict[str, str] = {
    ResearchDomain.COMPANY_PROFILE: "company_profile",
    ResearchDomain.LEADERSHIP: "leadership",
    ResearchDomain.COMPENSATION: "compensation",
    ResearchDomain.EMPLOYEE_SENTIMENT: "employee_sentiment",
    ResearchDomain.TECH_STACK: "tech_stack",
    ResearchDomain.FINANCIALS: "financials",
    ResearchDomain.INTERVIEWS: "interviews",
    ResearchDomain.NEWS: "news",
    ResearchDomain.COMPETITORS: "competitors",
}


def get_route(agent_type: str) -> str | None:
    """Return the graph node name for *agent_type*, or None if unknown."""
    return AGENT_ROUTES.get(agent_type)
