"""
utils/llm_simulator.py — Deterministic LLM call simulator.

In a real deployment each method here would call an actual model API
(OpenAI, Anthropic, etc.).  The simulator lets the entire system run
and be tested locally with zero cost and zero external dependencies.

Design note:
  The responses are keyword-driven so the system behaves realistically
  across different input domains.  Confidence scores reflect the
  simulator's certainty — higher for well-understood patterns, lower
  for ambiguous or short inputs.
"""

import random
import re
import time
from dataclasses import dataclass
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)


# ── Response container ────────────────────────────────────────────────────────

@dataclass
class LLMResponse:
    """Encapsulates every field a real LLM API response would return."""

    content: str
    confidence: float          # 0.0 – 1.0, simulates model certainty
    tokens_used: int           # prompt + completion tokens
    latency_ms: float          # wall-clock round-trip time


# ── Intent classification payloads ───────────────────────────────────────────

_INTENT_PATTERNS: dict[str, list[str]] = {
    "question": [
        "what", "how", "why", "when", "where", "who", "which",
        "explain", "describe", "tell me",
    ],
    "analysis": [
        "analyse", "analyze", "evaluate", "assess", "compare",
        "review", "examine", "investigate", "breakdown",
    ],
    "summarisation": [
        "summarise", "summarize", "summary", "brief", "overview",
        "tldr", "condense", "shorten", "key points",
    ],
    "action": [
        "create", "build", "generate", "write", "make", "produce",
        "draft", "design", "implement", "develop",
    ],
    "data_query": [
        "list", "show", "find", "search", "get", "fetch",
        "retrieve", "lookup", "display",
    ],
}

# ── Canned responses keyed by intent ─────────────────────────────────────────

_RESPONSES: dict[str, list[str]] = {
    "question": [
        "Based on the available context, {topic} involves several interrelated "
        "components. The primary mechanism works by decomposing the problem into "
        "discrete, manageable steps — each handled by a specialised subsystem. "
        "Key considerations include scalability, fault tolerance, and maintaining "
        "a consistent state across the workflow.",

        "To address your question about {topic}: the underlying principle relies "
        "on modular design where each component has a single, well-defined "
        "responsibility. This enables independent testing, easier debugging, and "
        "the ability to swap implementations without affecting adjacent layers.",
    ],
    "analysis": [
        "Analysing {topic}: \n\n"
        "• Strengths — clear separation of concerns, high cohesion within modules, "
        "low coupling across boundaries.\n"
        "• Weaknesses — added orchestration overhead; latency accumulates across "
        "agent hops.\n"
        "• Opportunities — parallel execution of independent agents, caching "
        "repeated sub-tasks, adaptive routing based on confidence scores.\n"
        "• Risks — cascading failures if one agent produces low-quality output "
        "that propagates to downstream agents.",

        "A systematic evaluation of {topic} reveals three core themes: "
        "efficiency (how quickly the task completes), accuracy (how close the "
        "output is to the ground truth), and maintainability (how easily an "
        "engineer can extend or debug the system). Optimising all three "
        "simultaneously requires deliberate trade-off decisions at the design phase.",
    ],
    "summarisation": [
        "Summary of {topic}: \n\n"
        "The core idea is to decompose complex, multi-step tasks into a directed "
        "graph of simpler operations. An orchestrator manages this graph, "
        "delegating each node to a specialised agent. Results flow back upstream, "
        "are evaluated for quality, and are stored in session memory for use by "
        "subsequent steps.",

        "Key takeaways for {topic}: (1) modular agents improve testability, "
        "(2) centralised orchestration simplifies flow control, (3) persistent "
        "memory enables coherent multi-turn conversations, and (4) an evaluation "
        "layer surfaces quality regressions before they reach the end-user.",
    ],
    "action": [
        "To accomplish {topic}, I will proceed through the following steps:\n\n"
        "1. Validate inputs and pre-conditions.\n"
        "2. Decompose the goal into atomic sub-tasks.\n"
        "3. Execute each sub-task through the appropriate specialised agent.\n"
        "4. Aggregate and reconcile partial results.\n"
        "5. Validate the final output against the original success criteria.\n"
        "6. Persist the result and update session context.",

        "Executing {topic}: the workflow initiates with intent classification, "
        "routes to the relevant agents in dependency order, merges their outputs "
        "into a coherent result, and records the full execution trace in memory "
        "for auditability and future reference.",
    ],
    "data_query": [
        "Query results for {topic}: the dataset contains multiple relevant "
        "entries. High-confidence matches are surfaced first, followed by "
        "lower-confidence candidates ranked by semantic similarity. "
        "Pagination metadata is included in the response envelope.",

        "Fetching {topic}: retrieved 5 relevant records. Each record includes "
        "an identifier, a relevance score, and a brief descriptor. "
        "Results are ordered by descending confidence.",
    ],
    "general": [
        "Processing your request about {topic}: the system has identified the "
        "key entities and relationships in your input. The most relevant "
        "context from memory has been incorporated to ensure a consistent, "
        "coherent response aligned with your session history.",

        "Regarding {topic}: the agentic workflow completed successfully. "
        "All sub-tasks were executed within the expected parameters. "
        "The output has been validated and meets the quality threshold "
        "defined in the evaluation configuration.",
    ],
}


# ── Simulator class ───────────────────────────────────────────────────────────

class LLMSimulator:
    """
    Simulates the interface of a real LLM provider.

    Methods mirror what a real integration would expose so that swapping
    in an actual API client requires only changing this class — all
    call-sites remain untouched.
    """

    def __init__(self, base_latency_ms: float = 120.0, jitter_ms: float = 60.0):
        self._base_latency = base_latency_ms
        self._jitter = jitter_ms

    # ── Core completion ────────────────────────────────────────────────���──────

    def complete(
        self,
        prompt: str,
        intent: Optional[str] = None,
        context: Optional[str] = None,
    ) -> LLMResponse:
        """
        Simulate a single-turn LLM completion.

        Args:
            prompt:  The user prompt / instruction.
            intent:  Pre-classified intent; if None, classify internally.
            context: Optional memory context injected into the prompt.

        Returns:
            LLMResponse with content, confidence, token count, and latency.
        """
        start = time.perf_counter()

        resolved_intent = intent or self._classify_intent(prompt)
        topic = self._extract_topic(prompt)
        content = self._generate_response(resolved_intent, topic, context)
        confidence = self._compute_confidence(prompt, resolved_intent)

        elapsed_ms = (time.perf_counter() - start) * 1000
        simulated_latency = self._base_latency + random.uniform(0, self._jitter)

        # Approximate token count: ~4 chars per token
        tokens = (len(prompt) + len(content)) // 4

        logger.debug(
            "LLM complete | intent=%s topic=%s confidence=%.2f tokens=%d",
            resolved_intent, topic, confidence, tokens,
        )

        return LLMResponse(
            content=content,
            confidence=confidence,
            tokens_used=tokens,
            latency_ms=round(simulated_latency + elapsed_ms, 2),
        )

    def classify(self, text: str) -> tuple[str, float]:
        """
        Classify the intent of *text* and return (intent_label, confidence).
        Used by the ClassifierAgent independently of a full completion.
        """
        intent = self._classify_intent(text)
        confidence = self._compute_confidence(text, intent)
        return intent, confidence

    def summarise(self, text: str) -> LLMResponse:
        """Dedicated summarisation path — mirrors a real `summarize` endpoint."""
        return self.complete(text, intent="summarisation")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _classify_intent(self, text: str) -> str:
        text_lower = text.lower()
        scores: dict[str, int] = {}

        for intent, keywords in _INTENT_PATTERNS.items():
            scores[intent] = sum(1 for kw in keywords if kw in text_lower)

        best_intent = max(scores, key=lambda k: scores[k])
        return best_intent if scores[best_intent] > 0 else "general"

    def _extract_topic(self, prompt: str) -> str:
        """Pull the most salient noun phrase from the prompt (heuristic)."""
        # Strip common question words, keep the content
        cleaned = re.sub(
            r"^(what|how|why|when|where|who|which|explain|describe|"
            r"analyse|analyze|summarise|summarize|create|build|generate|"
            r"tell me about|list|show|find)\s+",
            "",
            prompt.lower(),
        )
        # Capitalise and truncate to keep it readable in templates
        topic = cleaned.strip().capitalize()
        return topic[:60] + ("…" if len(topic) > 60 else "")

    def _generate_response(
        self, intent: str, topic: str, context: Optional[str]
    ) -> str:
        templates = _RESPONSES.get(intent, _RESPONSES["general"])
        template = random.choice(templates)
        response = template.format(topic=topic)

        if context:
            response += (
                f"\n\n[Context from memory applied: prior session established "
                f"that {context[:120]}{'…' if len(context) > 120 else ''}]"
            )
        return response

    def _compute_confidence(self, prompt: str, intent: str) -> float:
        """
        Heuristic confidence score based on prompt length and intent clarity.

        Short, ambiguous prompts → lower confidence.
        Longer, intent-rich prompts → higher confidence.
        """
        length_score = min(len(prompt) / 200, 1.0)  # saturates at 200 chars

        # Count how many intent keywords fired
        text_lower = prompt.lower()
        keyword_hits = sum(
            1
            for kw in _INTENT_PATTERNS.get(intent, [])
            if kw in text_lower
        )
        keyword_score = min(keyword_hits / 3, 1.0)  # saturates at 3 hits

        base = 0.55 + (0.25 * length_score) + (0.20 * keyword_score)
        # Add small random noise to simulate natural model variance
        noise = random.uniform(-0.03, 0.03)
        return round(min(max(base + noise, 0.0), 1.0), 3)


# ── Module-level singleton ────────────────────────────────────────────────────

llm = LLMSimulator()
