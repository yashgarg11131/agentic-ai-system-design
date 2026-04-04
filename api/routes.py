"""
api/routes.py — All HTTP route handlers for the Agentic AI System.

Routes are organised into three routers:
  • workflow_router  — core processing endpoint
  • session_router   — session / memory inspection and management
  • system_router    — health, stats, and observability

Keeping routes in a dedicated module (instead of main.py) makes the app
easy to extend: add new routers and include them in main.py with a prefix.
"""

from fastapi import APIRouter, HTTPException, status

from api.schemas import (
    AgentStatsResponse,
    ErrorResponse,
    HealthResponse,
    MemoryStatsResponse,
    ProcessRequest,
    ProcessResponse,
    SessionHistoryResponse,
)
from config import settings
from memory.memory import memory_store
from orchestrator.orchestrator import orchestrator
from utils.logger import get_logger

logger = get_logger(__name__)


# ── Routers ───────────────────────────────────────────────────────────────────

workflow_router = APIRouter(tags=["Workflow"])
session_router  = APIRouter(prefix="/sessions", tags=["Sessions"])
system_router   = APIRouter(prefix="/system",   tags=["System"])


# ── Workflow ──────────────────────────────────────────────────────────────────

@workflow_router.post(
    "/process",
    response_model=ProcessResponse,
    summary="Run multi-agent workflow",
    description=(
        "Submit a user input to the Orchestrator.  The system classifies intent, "
        "routes to the appropriate agent(s), evaluates the output, and returns a "
        "structured response with a full execution trace."
    ),
    responses={
        200: {"description": "Workflow completed (may still report internal errors in _meta)."},
        422: {"model": ErrorResponse, "description": "Request validation failed."},
        500: {"model": ErrorResponse, "description": "Unexpected server error."},
    },
)
async def process_input(request: ProcessRequest) -> dict:
    """
    Main entry point.  Delegates entirely to the Orchestrator and serialises
    the WorkflowResult into the standard response envelope.
    """
    logger.info(
        "POST /process | session_id=%s input_len=%d",
        request.session_id, len(request.input),
    )

    result = orchestrator.run(
        user_input=request.input,
        session_id=request.session_id,
    )

    return result.to_response_dict()


# ── Session management ────────────────────────────────────────────────────────

@session_router.get(
    "/{session_id}",
    response_model=SessionHistoryResponse,
    summary="Retrieve session history",
    description="Return the full turn history for a session.",
    responses={404: {"model": ErrorResponse, "description": "Session not found."}},
)
async def get_session(session_id: str) -> dict:
    session = memory_store.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found or has expired.",
        )
    data = session.to_dict()
    return {
        "session_id":  session_id,
        "turn_count":  data["turn_count"],
        "turns":       data["turns"],
        "metadata":    data["metadata"],
    }


@session_router.delete(
    "/{session_id}",
    summary="Delete a session",
    description="Permanently remove a session and all its history.",
    responses={404: {"model": ErrorResponse, "description": "Session not found."}},
)
async def delete_session(session_id: str) -> dict:
    deleted = memory_store.delete_session(session_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )
    return {"deleted": True, "session_id": session_id}


@session_router.get(
    "",
    summary="List active sessions",
    description="Return all session IDs currently held in memory.",
)
async def list_sessions() -> dict:
    return {"sessions": memory_store.list_sessions()}


# ── System / observability ────────────────────────────────────────────────────

@system_router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Liveness probe — returns 200 when the server is running.",
)
async def health() -> dict:
    return {
        "status":      "healthy",
        "version":     settings.version,
        "environment": settings.environment,
    }


@system_router.get(
    "/memory/stats",
    response_model=MemoryStatsResponse,
    summary="Memory store statistics",
    description="Returns current session counts and memory usage.",
)
async def memory_stats() -> dict:
    return memory_store.stats()


@system_router.get(
    "/agents/stats",
    response_model=AgentStatsResponse,
    summary="Agent statistics",
    description="Returns cumulative call counts and token usage per agent.",
)
async def agent_stats() -> dict:
    return {"agents": orchestrator.agent_stats()}
