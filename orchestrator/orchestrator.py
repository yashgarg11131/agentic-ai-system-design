"""
orchestrator/orchestrator.py — Central workflow coordinator.

The Orchestrator is the brain of the system.  It is the *only* component
that knows about all other components; every other module is unaware of
its siblings.  This keeps coupling low and makes each module independently
testable.

Key design: Dependency Injection
  All components (agent, memory, evaluator) are passed into __init__.
  This makes the Orchestrator easy to explain, easy to test, and easy
  to swap any part of.

Workflow (one request):
  ┌─────────────┐
  │  user input │
  └──────┬──────┘
         │ 1. classify intent
         ▼
  ┌──────────────────┐
  │ ClassifierAgent  │  → intent label + routing hint
  └──────┬───────────┘
         │ 2. get context from memory
         ▼
  ┌──────────────────┐
  │     Memory       │  → last-N turns injected into prompt
  └──────┬───────────┘
         │ 3. process task with agent
         ▼
  ┌──────────────────┐
  │    TaskAgent     │  → synthesised response
  └──────┬───────────┘
         │ 4. update memory
         ▼
  ┌──────────────────┐
  │     Memory       │  → store user + agent turns
  └──────┬───────────┘
         │ 5. evaluate output
         ▼
  ┌──────────────────┐
  │    Evaluator     │  → quality score + grade
  └──────┬───────────┘
         │ 6. return structured response
         ▼
  ┌──────────────────┐
  │  WorkflowResult  │
  └──────────────────┘
"""

import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from agents.base_agent import AgentResult, BaseAgent
from agents.classifier_agent import ClassifierAgent
from agents.summariser_agent import SummariserAgent
from agents.task_agent import TaskAgent
from evaluation.evaluator import EvaluationResult, Evaluator, evaluator as default_evaluator
from memory.memory import MemoryStore, Turn, memory_store as default_memory
from utils.logger import get_logger

logger = get_logger(__name__)


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class WorkflowResult:
    """
    The final structured payload returned to the API layer (and the user).

    Fields map directly to the required API response format:
      { "input", "steps", "output", "evaluation_score" }
    """

    request_id: str
    session_id: str
    user_input: str
    steps: list[dict[str, Any]]
    output: str
    evaluation: EvaluationResult
    intent: str
    total_latency_ms: float
    total_tokens: int
    success: bool
    error: Optional[str] = None

    def to_response_dict(self) -> dict[str, Any]:
        """Serialisable dict that becomes the HTTP response body."""
        return {
            "input": self.user_input,
            "steps": self.steps,
            "output": self.output,
            "evaluation_score": {
                "score": round(self.evaluation.overall_score, 3),
                "grade": self.evaluation.grade,
                "explanation": self.evaluation.explanation,
                "dimensions": {
                    "relevance":    round(self.evaluation.relevance, 3),
                    "completeness": round(self.evaluation.completeness, 3),
                    "coherence":    round(self.evaluation.coherence, 3),
                    "confidence":   round(self.evaluation.confidence, 3),
                },
                "passed": self.evaluation.passed,
            },
            "_meta": {
                "request_id":       self.request_id,
                "session_id":       self.session_id,
                "intent":           self.intent,
                "total_latency_ms": round(self.total_latency_ms, 2),
                "total_tokens":     self.total_tokens,
                "success":          self.success,
                **({"error": self.error} if self.error else {}),
            },
        }


# ── Orchestrator ──────────────────────────────────────────────────────────────

class Orchestrator:
    """
    Coordinates the full multi-agent workflow.

    Dependencies are injected via __init__, making this class:
      • Testable  — swap any component for a mock in unit tests
      • Explicit  — you can see exactly what it depends on at a glance
      • Flexible  — register additional agents at runtime via register_agent()

    The composition root (bottom of this file) wires everything together.
    """

    def __init__(
        self,
        agent: BaseAgent,
        memory: MemoryStore,
        evaluator: Evaluator,
        classifier: Optional[BaseAgent] = None,
    ) -> None:
        self.agent      = agent       # primary task-executing agent
        self.memory     = memory      # session context store
        self.evaluator  = evaluator   # response quality scorer
        self.classifier = classifier or ClassifierAgent()  # intent router

        # Registry for additional/specialist agents (extensible at runtime)
        self._agents: dict[str, BaseAgent] = {
            self.agent.name:      self.agent,
            self.classifier.name: self.classifier,
        }

        logger.info(
            "Orchestrator initialised | agent=%s classifier=%s",
            self.agent.name, self.classifier.name,
        )

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self, user_input: str, session_id: Optional[str] = None) -> WorkflowResult:
        """
        Execute the full workflow for *user_input* within *session_id*.

        The five steps below map directly to the class docstring diagram.
        Raises nothing — all errors are caught and returned in WorkflowResult.
        """
        request_id = str(uuid.uuid4())
        session_id = session_id or str(uuid.uuid4())
        start      = time.perf_counter()

        logger.info(
            "Workflow START | request_id=%s session_id=%s input_len=%d",
            request_id, session_id, len(user_input),
        )

        steps: list[dict[str, Any]] = []
        total_tokens = 0

        try:
            # ── Step 1: Classify intent ───────────────────────────────────────
            classifier_result = self.classifier.process(
                user_input, context=None, run_id=request_id
            )
            steps.append(classifier_result.to_step_dict())
            total_tokens += classifier_result.tokens_used

            intent       = classifier_result.metadata.get("intent", "general")
            routing_hint = classifier_result.metadata.get("routing_hint", [self.agent.name])

            # ── Step 2: Get context from memory ───────────────────────────────
            context = self.memory.get_context(session_id, n=5)
            if context:
                steps.append({
                    "agent":   "Memory",
                    "step":    "context_retrieval",
                    "output":  f"Injected {context.count(chr(10)) + 1} prior turns.",
                    "success": True,
                })
            self.memory.add_turn(
                session_id,
                Turn(role="user", content=user_input, agent_name="api_gateway"),
            )

            # ── Step 3: Process task with agent ───────────────────────────────
            primary_agent  = self._resolve_agent(routing_hint)
            result: AgentResult = primary_agent.process(
                user_input, context=context, run_id=request_id
            )
            steps.append(result.to_step_dict())
            total_tokens += result.tokens_used

            # ── Step 4: Update memory ─────────────────────────────────────────
            self.memory.add_turn(
                session_id,
                Turn(
                    role="agent",
                    content=result.output,
                    agent_name=result.agent_name,
                    metadata={"intent": intent},
                ),
            )

            # ── Step 5: Evaluate output ───────────────────────────────────────
            score = self.evaluator.evaluate(
                user_input=user_input,
                agent_output=result.output,
                agent_confidence=result.confidence,
                intent=intent,
            )
            steps.append({
                "agent":   "Evaluator",
                "step":    "quality_evaluation",
                "output":  score.explanation,
                "score":   round(score.overall_score, 3),
                "grade":   score.grade,
                "success": True,
            })

            elapsed = (time.perf_counter() - start) * 1000
            logger.info(
                "Workflow DONE | request_id=%s score=%.3f grade=%s latency_ms=%.1f",
                request_id, score.overall_score, score.grade, elapsed,
            )

            return WorkflowResult(
                request_id=request_id,
                session_id=session_id,
                user_input=user_input,
                steps=steps,
                output=result.output,
                evaluation=score,
                intent=intent,
                total_latency_ms=elapsed,
                total_tokens=total_tokens,
                success=True,
            )

        except Exception as exc:  # noqa: BLE001
            elapsed = (time.perf_counter() - start) * 1000
            logger.error(
                "Workflow ERROR | request_id=%s error=%s", request_id, exc, exc_info=True
            )
            degraded_eval = EvaluationResult(
                overall_score=0.0, grade="Needs Improvement",
                relevance=0.0, completeness=0.0, coherence=0.0, confidence=0.0,
                explanation="Workflow failed — see error field.",
                passed=False,
            )
            return WorkflowResult(
                request_id=request_id,
                session_id=session_id,
                user_input=user_input,
                steps=steps,
                output="",
                evaluation=degraded_eval,
                intent="unknown",
                total_latency_ms=elapsed,
                total_tokens=total_tokens,
                success=False,
                error=str(exc),
            )

    # ── Agent management ──────────────────────────────────────────────────────

    def register_agent(self, agent: BaseAgent) -> None:
        """
        Register an additional agent at runtime.
        No changes to this class are needed to add new capabilities.
        """
        self._agents[agent.name] = agent
        logger.info("Agent registered | name=%s", agent.name)

    def agent_stats(self) -> list[dict]:
        all_agents = list(self._agents.values()) + [self.evaluator]
        return [a.stats() for a in all_agents if hasattr(a, "stats")]

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _resolve_agent(self, routing_hint: list[str]) -> BaseAgent:
        """Return the first agent from routing_hint that is registered."""
        for name in routing_hint:
            if name in self._agents:
                logger.info("Dispatching to agent=%s", name)
                return self._agents[name]
        logger.warning("No routing match — falling back to primary agent")
        return self.agent


# ── Composition root ──────────────────────────────────────────────────────────
#
# Dependencies are wired together here, once.
# To use a different agent or memory backend, change only these lines.

orchestrator = Orchestrator(
    agent=TaskAgent(),
    memory=default_memory,
    evaluator=default_evaluator,
    # classifier defaults to ClassifierAgent() inside __init__
    # register SummariserAgent so the classifier can route to it
)
orchestrator.register_agent(SummariserAgent())
