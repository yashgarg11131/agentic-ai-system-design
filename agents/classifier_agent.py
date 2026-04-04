"""
agents/classifier_agent.py — Intent classification and routing agent.

Responsibility:
  Determines what the user actually wants (question / analysis /
  summarisation / action / data_query / general) and produces a
  structured classification result.

Why a dedicated ClassifierAgent?
  Routing decisions should be explicit and traceable, not buried inside
  the orchestrator's if-else tree.  Separating classification into its own
  agent makes the decision auditable (you can see the confidence score and
  the label in the `steps` list of every API response) and replaceable
  (swap a keyword classifier for a fine-tuned model without touching the
  orchestrator).
"""

from typing import Optional

from agents.base_agent import AgentResult, BaseAgent
from utils.llm_simulator import llm
from utils.logger import get_logger

logger = get_logger(__name__)

# Human-readable descriptions returned alongside each label.
_INTENT_DESCRIPTIONS: dict[str, str] = {
    "question":       "User is seeking factual or explanatory information.",
    "analysis":       "User wants a structured breakdown or evaluation.",
    "summarisation":  "User wants the content condensed into key points.",
    "action":         "User wants the system to produce or execute something.",
    "data_query":     "User wants records or items retrieved.",
    "general":        "Intent is broad or does not match a specific category.",
}


class ClassifierAgent(BaseAgent):
    """
    Intent classification agent.

    Classifies the input, computes a routing recommendation, and returns
    structured metadata that the orchestrator uses to pick downstream agents.
    """

    def __init__(self) -> None:
        super().__init__(
            name="ClassifierAgent",
            description="Classifies user intent to guide orchestrator routing decisions.",
        )

    def _run(self, input_text: str, context: Optional[str]) -> AgentResult:
        intent, confidence = llm.classify(input_text)
        description = _INTENT_DESCRIPTIONS.get(intent, _INTENT_DESCRIPTIONS["general"])

        # Build a concise, human-readable classification summary
        output = (
            f"Classified intent: '{intent}' (confidence {confidence:.0%}).\n"
            f"{description}"
        )

        logger.debug(
            "ClassifierAgent | intent=%s confidence=%.2f", intent, confidence
        )

        return AgentResult(
            agent_name=self.name,
            success=True,
            output=output,
            confidence=confidence,
            tokens_used=len(input_text) // 4,  # classifier is lightweight
            metadata={
                "intent": intent,
                "intent_description": description,
                "routing_hint": self._routing_hint(intent),
            },
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _routing_hint(intent: str) -> list[str]:
        """
        Suggest which agents should handle this intent.
        The orchestrator uses this as a soft recommendation, not a hard rule.
        """
        routing_map: dict[str, list[str]] = {
            "question":      ["TaskAgent"],
            "analysis":      ["TaskAgent"],
            "summarisation": ["SummariserAgent"],
            "action":        ["TaskAgent"],
            "data_query":    ["TaskAgent"],
            "general":       ["TaskAgent"],
        }
        return routing_map.get(intent, ["TaskAgent"])
