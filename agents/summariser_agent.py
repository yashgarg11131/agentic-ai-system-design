"""
agents/summariser_agent.py — Text summarisation specialist agent.

Responsibility:
  Receives the output of an earlier agent (or the raw user input for
  summarisation-intent tasks) and condenses it to key points with an
  estimated reading-time saving.

Why a dedicated SummariserAgent?
  Summarisation has different quality metrics to task execution (brevity,
  coverage, no new information introduced).  Keeping it isolated lets you
  swap in a fine-tuned summarisation model independently of the task model,
  and lets the evaluator apply summarisation-specific scoring criteria.
"""

from typing import Optional

from agents.base_agent import AgentResult, BaseAgent
from utils.llm_simulator import llm
from utils.logger import get_logger

logger = get_logger(__name__)

# Approximate words per minute for an adult reader
_WORDS_PER_MINUTE = 200


class SummariserAgent(BaseAgent):
    """
    Summarisation agent.

    Produces a condensed version of the input and includes metadata about
    compression ratio and estimated reading-time saved.
    """

    def __init__(self) -> None:
        super().__init__(
            name="SummariserAgent",
            description="Condenses input or prior agent output into concise key points.",
        )

    def _run(self, input_text: str, context: Optional[str]) -> AgentResult:
        response = llm.summarise(input_text)

        # Compute simple compression metrics
        input_word_count = len(input_text.split())
        output_word_count = len(response.content.split())
        compression_ratio = (
            round(1 - output_word_count / max(input_word_count, 1), 3)
            if input_word_count > 0 else 0.0
        )
        time_saved_seconds = max(
            (input_word_count - output_word_count) / _WORDS_PER_MINUTE * 60, 0
        )

        logger.debug(
            "SummariserAgent | input_words=%d output_words=%d compression=%.0f%%",
            input_word_count, output_word_count, compression_ratio * 100,
        )

        return AgentResult(
            agent_name=self.name,
            success=True,
            output=response.content,
            confidence=response.confidence,
            tokens_used=response.tokens_used,
            metadata={
                "compression_ratio": compression_ratio,
                "input_word_count": input_word_count,
                "output_word_count": output_word_count,
                "estimated_reading_time_saved_seconds": round(time_saved_seconds, 1),
            },
        )
