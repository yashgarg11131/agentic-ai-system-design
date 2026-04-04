"""
api/schemas.py — Pydantic request and response models for the API layer.

Having explicit schemas serves three purposes:
  1. Auto-validation — FastAPI rejects malformed requests before any code runs.
  2. Auto-documentation — OpenAPI/Swagger UI is generated from these models.
  3. Contract clarity — anyone reading this file understands exactly what
     the API accepts and returns.
"""

from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator


# ── Request models ────────────────────────────────────────────────────────────

class ProcessRequest(BaseModel):
    """Body for POST /api/v1/process — the main workflow endpoint."""

    input: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="The user's query, task, or instruction.",
        examples=["Explain the benefits of a multi-agent AI architecture."],
    )
    session_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional session identifier for multi-turn conversations. "
            "If omitted, a new ephemeral session is created automatically."
        ),
        examples=["user-abc-123"],
    )

    @field_validator("input")
    @classmethod
    def input_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("input must contain non-whitespace characters.")
        return v.strip()

    model_config = {"json_schema_extra": {
        "example": {
            "input": "Analyse the trade-offs of multi-agent AI systems.",
            "session_id": "demo-session-001",
        }
    }}


# ── Response models ───────────────────────────────────────────────────────────

class EvaluationDimensions(BaseModel):
    relevance:    float
    completeness: float
    coherence:    float
    confidence:   float


class EvaluationScore(BaseModel):
    score:       float   = Field(..., description="Weighted composite score (0–1).")
    grade:       str     = Field(..., description="Human-readable quality grade.")
    explanation: str     = Field(..., description="One-line rationale for the score.")
    dimensions:  EvaluationDimensions
    passed:      bool    = Field(..., description="True if score ≥ acceptable threshold.")


class StepDetail(BaseModel):
    """Represents one step in the workflow execution trace."""
    agent:      str
    success:    bool
    output:     str
    confidence: Optional[float] = None
    tokens_used: Optional[int]  = None
    latency_ms: Optional[float] = None
    error:      Optional[str]   = None


class ResponseMeta(BaseModel):
    request_id:       str
    session_id:       str
    intent:           str
    total_latency_ms: float
    total_tokens:     int
    success:          bool
    error:            Optional[str] = None


class ProcessResponse(BaseModel):
    """Response body for POST /api/v1/process."""

    input:            str
    steps:            list[dict[str, Any]]
    output:           str
    evaluation_score: EvaluationScore
    _meta:            Optional[ResponseMeta] = None

    model_config = {"populate_by_name": True}


# ── Utility response models ───────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str


class SessionHistoryResponse(BaseModel):
    session_id:  str
    turn_count:  int
    turns:       list[dict[str, Any]]
    metadata:    dict[str, Any]


class MemoryStatsResponse(BaseModel):
    total_sessions:  int
    active_sessions: int
    max_sessions:    int
    total_turns:     int


class AgentStatsResponse(BaseModel):
    agents: list[dict[str, Any]]


class ErrorResponse(BaseModel):
    detail: str
    request_id: Optional[str] = None
