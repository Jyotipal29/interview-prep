from pydantic import BaseModel, Field

from src.models.schemas import Source


class CitedStr(BaseModel):
    """A string value with the sources that support it."""
    value: str
    sources: list[Source] = Field(default_factory=list)


class CitedInt(BaseModel):
    """An integer value with the sources that support it."""
    value: int
    sources: list[Source] = Field(default_factory=list)


class CitedList(BaseModel):
    """A list of string values with the sources that support them."""
    values: list[str] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)


class CompanyProfile(BaseModel):
    """Fully-typed company profile with per-field source citations."""

    name: CitedStr | None = None
    description: CitedStr | None = None
    industry: CitedStr | None = None
    headquarters: CitedStr | None = None
    founded_year: CitedInt | None = None
    employee_count: CitedStr | None = None
    business_model: CitedStr | None = None
    products: CitedList | None = None
    target_customers: CitedList | None = None
    markets: CitedList | None = None


class CompanyFacts(BaseModel):
    """Raw facts extracted from a single page via LLM structured output.

    Used as the LLM extraction target; converted to CompanyProfile by the service.
    """

    name: str | None = None
    description: str | None = None
    industry: str | None = None
    headquarters: str | None = None
    founded_year: int | None = None
    employee_count: str | None = None
    business_model: str | None = None
    products: list[str] = Field(default_factory=list)
    target_customers: list[str] = Field(default_factory=list)
    markets: list[str] = Field(default_factory=list)
