import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class SourceType(StrEnum):
    WEB = "web"
    DATABASE = "database"
    API = "api"
    DOCUMENT = "document"
    SOCIAL = "social"


class Source(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    url: str | None = None
    title: str | None = None
    source_type: SourceType = SourceType.WEB
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class Evidence(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    source: Source
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ResearchTask(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    agent_type: str
    status: TaskStatus = TaskStatus.PENDING
    priority: int = Field(default=5, ge=1, le=10)
    dependencies: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResearchPlan(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tasks: list[ResearchTask] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentResult(BaseModel):
    agent_type: str
    success: bool
    data: dict[str, Any] = Field(default_factory=dict)
    evidence: list[Evidence] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GapSeverity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ResearchGap(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    topic: str
    description: str
    severity: GapSeverity = GapSeverity.MEDIUM
    suggested_agents: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
