"""
agents/base_agent.py — Abstract base class for every agent in the system.

All agents share the same interface so the orchestrator can treat them
uniformly.  Concrete agents override `_run()` with their specialised logic;
the public `process()` method handles cross-cutting concerns (validation,
error handling, logging, timing) without each subclass having to repeat it.

This is the Strategy pattern: the orchestrator selects an agent at runtime
and calls the same interface regardless of which concrete class it holds.
"""

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class AgentResult:
    """
    Standardised output produced by any agent.

    Having a fixed schema means the orchestrator never needs to inspect
    what type of agent ran — it just reads these fields.
    """

    agent_name: str
    success: bool
    output: str
    confidence: float = 0.0
    tokens_used: int = 0
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_step_dict(self) -> dict[str, Any]:
        """Serialisable representation used in the API response's `steps` list."""
        return {
            "agent": self.agent_name,
            "success": self.success,
            "output": self.output,
            "confidence": round(self.confidence, 3),
            "tokens_used": self.tokens_used,
            "latency_ms": round(self.latency_ms, 2),
            **({"error": self.error} if self.error else {}),
        }


# ── Base class ────────────────────────────────────────────────────────────────

class BaseAgent(ABC):
    """
    Abstract agent.  Subclasses must implement `_run()`.

    Lifecycle of a single agent call:
        process(input, context)
            └─ validate_input(input)        ← raises ValueError if invalid
            └─ _run(input, context)         ← implemented by subclass
            └─ wrap result in AgentResult
            └─ log timing / tokens
    """

    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description
        self._call_count = 0
        self._total_tokens = 0
        logger.info("Agent ready | name=%s", name)

    # ── Public interface (do not override) ────────────────────────────────────

    def process(
        self,
        input_text: str,
        context: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> AgentResult:
        """
        Entry point called by the orchestrator.

        Wraps _run() with uniform error handling and instrumentation so
        individual agents never need to worry about those concerns.
        """
        run_id = run_id or str(uuid.uuid4())[:8]
        logger.info("Agent START | name=%s run_id=%s", self.name, run_id)
        start = time.perf_counter()

        try:
            self.validate_input(input_text)
            result = self._run(input_text, context)
        except ValueError as exc:
            elapsed = (time.perf_counter() - start) * 1000
            logger.warning("Agent validation failed | name=%s error=%s", self.name, exc)
            return AgentResult(
                agent_name=self.name,
                success=False,
                output="",
                latency_ms=round(elapsed, 2),
                error=str(exc),
            )
        except Exception as exc:  # noqa: BLE001
            elapsed = (time.perf_counter() - start) * 1000
            logger.error("Agent ERROR | name=%s error=%s", self.name, exc, exc_info=True)
            return AgentResult(
                agent_name=self.name,
                success=False,
                output="",
                latency_ms=round(elapsed, 2),
                error=f"Unexpected error: {exc}",
            )

        elapsed = (time.perf_counter() - start) * 1000
        result.latency_ms = round(elapsed, 2)

        self._call_count += 1
        self._total_tokens += result.tokens_used

        logger.info(
            "Agent DONE | name=%s run_id=%s latency_ms=%.1f confidence=%.2f",
            self.name, run_id, result.latency_ms, result.confidence,
        )
        return result

    # ── Contract for subclasses ───────────────────────────────────────────────

    @abstractmethod
    def _run(self, input_text: str, context: Optional[str]) -> AgentResult:
        """
        Core agent logic.  Must return an AgentResult.
        Raise ValueError for domain-level validation failures.
        """

    def validate_input(self, input_text: str) -> None:
        """
        Called before _run().  Override to add domain-specific checks.
        Default: reject empty or whitespace-only strings.
        """
        if not input_text or not input_text.strip():
            raise ValueError(f"{self.name} received empty input.")

    # ── Observability helpers ─────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        return {
            "agent": self.name,
            "total_calls": self._call_count,
            "total_tokens": self._total_tokens,
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
