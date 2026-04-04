"""
config.py — Centralised application settings.

All tuneable parameters live here so nothing is scattered across modules.
Change a value once and every module picks it up automatically.
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class MemoryConfig:
    """Controls how the in-memory session store behaves."""

    max_sessions: int = 500              # Hard cap on concurrent sessions
    max_turns_per_session: int = 50      # Rolling window; older turns are evicted
    session_ttl_seconds: int = 3600      # 1-hour TTL (enforced on access)


@dataclass(frozen=True)
class EvaluationConfig:
    """Weights and thresholds for the response evaluator."""

    relevance_weight: float = 0.35
    completeness_weight: float = 0.30
    coherence_weight: float = 0.20
    confidence_weight: float = 0.15

    # Score boundaries for human-readable labels
    excellent_threshold: float = 0.85
    good_threshold: float = 0.70
    acceptable_threshold: float = 0.50


@dataclass(frozen=True)
class OrchestratorConfig:
    """Controls workflow orchestration behaviour."""

    max_agent_retries: int = 2
    step_timeout_seconds: float = 30.0
    enable_parallel_agents: bool = False  # Future: run independent agents concurrently


@dataclass(frozen=True)
class AppConfig:
    """Root config object injected throughout the application."""

    app_name: str = "Agentic AI System"
    version: str = "1.0.0"
    environment: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"
    api_prefix: str = "/api/v1"

    memory: MemoryConfig = field(default_factory=MemoryConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)


# Singleton instance imported by all modules
settings = AppConfig()
