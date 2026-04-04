"""
api/main.py — FastAPI application entry point.

Responsibilities of this file:
  • Create and configure the FastAPI application instance.
  • Register all routers under the configured API prefix.
  • Add middleware (CORS, request-ID injection).
  • Define startup/shutdown lifecycle hooks.
  • Provide a global exception handler that always returns a structured JSON error.

Run locally:
    uvicorn api.main:app --reload --port 8000
    # OR from repo root:
    python -m uvicorn api.main:app --reload
"""

import time
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import session_router, system_router, workflow_router
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """
    Build and return the configured FastAPI application.

    Using a factory function (instead of a module-level `app = FastAPI()`)
    makes it easy to create isolated instances for testing.
    """
    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description=(
            "A production-grade multi-agent AI system for multi-step decision "
            "workflows.  Sends user input through an Orchestrator that classifies "
            "intent, routes to specialised agents, evaluates output quality, and "
            "maintains session memory across turns."
        ),
        docs_url=f"{settings.api_prefix}/docs",
        redoc_url=f"{settings.api_prefix}/redoc",
        openapi_url=f"{settings.api_prefix}/openapi.json",
    )

    # ── Middleware ────────────────────────────────────────────────────────────

    # CORS — open for local development; tighten allow_origins in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_instrumentation(request: Request, call_next):
        """Attach a unique request-ID and log every inbound request + response."""
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        start = time.perf_counter()

        logger.info(
            "→ %s %s | request_id=%s",
            request.method, request.url.path, request_id,
        )

        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000

        response.headers["X-Request-ID"] = request_id
        logger.info(
            "← %s %s | status=%d latency_ms=%.1f",
            request.method, request.url.path, response.status_code, elapsed_ms,
        )
        return response

    # ── Global error handler ──────────────────────────────────────────────────

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", "unknown")
        logger.error(
            "Unhandled exception | request_id=%s path=%s error=%s",
            request_id, request.url.path, exc, exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={
                "detail":     "An unexpected error occurred.",
                "request_id": request_id,
            },
        )

    # ── Routers ───────────────────────────────────────────────────────────────

    app.include_router(workflow_router, prefix=settings.api_prefix)
    app.include_router(session_router,  prefix=settings.api_prefix)
    app.include_router(system_router,   prefix=settings.api_prefix)

    # ── Lifecycle hooks ───────────────────────────────────────────────────────

    @app.on_event("startup")
    async def on_startup():
        logger.info(
            "=== %s v%s starting up | env=%s ===",
            settings.app_name, settings.version, settings.environment,
        )

    @app.on_event("shutdown")
    async def on_shutdown():
        logger.info("=== %s shutting down ===", settings.app_name)

    # ── Root redirect ─────────────────────────────────────────────────────────

    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "service":     settings.app_name,
            "version":     settings.version,
            "docs":        f"{settings.api_prefix}/docs",
            "health":      f"{settings.api_prefix}/system/health",
        }

    return app


# ── Module-level app instance (used by uvicorn) ───────────────────────────────

app = create_app()
