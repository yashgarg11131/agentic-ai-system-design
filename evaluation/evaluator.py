"""
evaluation/evaluator.py — Multi-dimensional response quality evaluator.

Design rationale:
  Evaluation is a first-class concern in any production AI system.  Rather
  than a single opaque "score", this module computes four orthogonal
  dimensions and combines them with configurable weights.  Each dimension
  can be replaced with a real model-based scorer without changing the
  aggregation logic.

Dimensions:
  • Relevance     — does the output address what was asked?
  • Completeness  — how much of the input surface area is covered?
  • Coherence     — is the output well-structured and readable?
  • Confidence    — how certain was the generating model?

The final score maps to a human-readable grade:
  ≥ 0.85 → "Excellent"
  ≥ 0.70 → "Good"
  ≥ 0.50 → "Acceptable"
  <  0.50 → "Needs Improvement"
"""

import re
from dataclasses import dataclass
from typing import Optional

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

cfg = settings.evaluation  # shorthand


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class EvaluationResult:
    """Full breakdown of a single evaluation run."""

    overall_score: float        # 0.0 – 1.0 weighted composite
    grade: str                  # human-readable label
    relevance: float
    completeness: float
    coherence: float
    confidence: float           # passed in from the agent result
    explanation: str            # one-line rationale for the score
    passed: bool                # True if score ≥ acceptable_threshold

    def to_dict(self) -> dict:
        return {
            "overall_score": round(self.overall_score, 3),
            "grade": self.grade,
            "dimensions": {
                "relevance":    round(self.relevance, 3),
                "completeness": round(self.completeness, 3),
                "coherence":    round(self.coherence, 3),
                "confidence":   round(self.confidence, 3),
            },
            "explanation": self.explanation,
            "passed": self.passed,
        }


# ── Evaluator ─────────────────────────────────────────────────────────────────

class Evaluator:
    """
    Scores agent outputs against the original user input.

    Each scoring method is independent and heuristic-based here.
    In production, replace each method with a dedicated LLM call or
    a fine-tuned reward model without touching the rest of the system.
    """

    # ── Public entry point ────────────────────────────────────────────────────

    def evaluate(
        self,
        user_input: str,
        agent_output: str,
        agent_confidence: float = 0.0,
        intent: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Score *agent_output* relative to *user_input*.

        Args:
            user_input:        The original request from the user.
            agent_output:      The final synthesised response.
            agent_confidence:  Confidence reported by the generating agent.
            intent:            Classified intent (adjusts scoring weights).

        Returns:
            EvaluationResult with per-dimension scores and an aggregate grade.
        """
        relevance    = self._score_relevance(user_input, agent_output)
        completeness = self._score_completeness(user_input, agent_output)
        coherence    = self._score_coherence(agent_output)
        confidence   = max(0.0, min(agent_confidence, 1.0))

        overall = (
            relevance    * cfg.relevance_weight
            + completeness * cfg.completeness_weight
            + coherence    * cfg.coherence_weight
            + confidence   * cfg.confidence_weight
        )
        overall = round(min(max(overall, 0.0), 1.0), 3)

        grade   = self._grade(overall)
        passed  = overall >= cfg.acceptable_threshold
        explanation = self._explain(overall, relevance, completeness, coherence, confidence)

        logger.info(
            "Evaluation | score=%.3f grade=%s relevance=%.2f "
            "completeness=%.2f coherence=%.2f confidence=%.2f",
            overall, grade, relevance, completeness, coherence, confidence,
        )

        return EvaluationResult(
            overall_score=overall,
            grade=grade,
            relevance=relevance,
            completeness=completeness,
            coherence=coherence,
            confidence=confidence,
            explanation=explanation,
            passed=passed,
        )

    # ── Dimension scorers ─────────────────────────────────────────────────────

    @staticmethod
    def _score_relevance(user_input: str, output: str) -> float:
        """
        Keyword-overlap relevance: what fraction of meaningful input tokens
        appear in the output?

        Real implementation: cosine similarity of sentence embeddings.
        """
        input_tokens  = set(_tokenise(user_input))
        output_tokens = set(_tokenise(output))

        if not input_tokens:
            return 0.5  # can't assess, return neutral

        overlap = input_tokens & output_tokens
        jaccard = len(overlap) / len(input_tokens | output_tokens)

        # Scale Jaccard (usually 0–0.3 for natural text) to 0.4–1.0
        return round(0.40 + jaccard * 2.0, 3)

    @staticmethod
    def _score_completeness(user_input: str, output: str) -> float:
        """
        Completeness proxy: ratio of output length to input length,
        clipped at a sensible ceiling.

        The assumption is that an adequate answer is 2–5× the length of the
        question.  Very short outputs are penalised; very long ones plateau.
        """
        ratio = len(output) / max(len(user_input), 1)
        # Ideal ratio is around 3.0; score peaks there and tapers off
        if ratio < 0.5:
            return round(0.3 + ratio * 0.4, 3)          # too short
        elif ratio < 5.0:
            return round(min(0.5 + (ratio - 0.5) / 9, 1.0), 3)  # grows
        else:
            return 0.95  # long enough — minor cap for conciseness

    @staticmethod
    def _score_coherence(output: str) -> float:
        """
        Structural coherence heuristics:
          • Penalise very short outputs (< 50 chars)
          • Reward presence of structured elements (bullets, numbers, newlines)
          • Reward sentences that end properly
        """
        if len(output) < 50:
            return 0.30

        score = 0.55  # baseline for non-empty output

        # Structure indicators
        if re.search(r"(\n[-•*]|\n\d+\.)", output):
            score += 0.15  # has bullet points or numbered list
        if output.count("\n") >= 2:
            score += 0.10  # multi-paragraph
        if re.search(r"[.!?]$", output.strip()):
            score += 0.08  # ends with proper punctuation
        if len(re.findall(r"[A-Z][^.!?]*[.!?]", output)) >= 3:
            score += 0.07  # multiple complete sentences

        return round(min(score, 1.0), 3)

    # ── Grade and explanation ─────────────────────────────────────���───────────

    @staticmethod
    def _grade(score: float) -> str:
        if score >= cfg.excellent_threshold:
            return "Excellent"
        if score >= cfg.good_threshold:
            return "Good"
        if score >= cfg.acceptable_threshold:
            return "Acceptable"
        return "Needs Improvement"

    @staticmethod
    def _explain(
        overall: float,
        relevance: float,
        completeness: float,
        coherence: float,
        confidence: float,
    ) -> str:
        weakest_dim = min(
            [("relevance", relevance), ("completeness", completeness),
             ("coherence", coherence), ("confidence", confidence)],
            key=lambda x: x[1],
        )
        if overall >= cfg.excellent_threshold:
            return (
                f"High-quality response across all dimensions "
                f"(overall {overall:.0%})."
            )
        return (
            f"Score {overall:.0%} — primary drag is {weakest_dim[0]} "
            f"({weakest_dim[1]:.0%}). Consider enriching output or "
            f"providing more context."
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tokenise(text: str) -> list[str]:
    """
    Lowercase, remove punctuation, split on whitespace.
    Filters stop-words to keep overlap signal meaningful.
    """
    _STOP_WORDS = {
        "a", "an", "the", "is", "it", "in", "on", "at", "to", "for",
        "of", "and", "or", "but", "with", "as", "by", "this", "that",
        "i", "you", "we", "they", "he", "she", "do", "does", "did",
    }
    tokens = re.sub(r"[^\w\s]", "", text.lower()).split()
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 2]


# ── Module-level singleton ────────────────────────────────────────────────────

evaluator = Evaluator()
