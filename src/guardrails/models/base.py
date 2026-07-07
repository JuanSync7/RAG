# @summary
# Backend-agnostic contract for "guardian" judge models that classify text
# against safety risk dimensions. One guardian instance can be reused by many
# rails (toxicity, injection, topic_safety, faithfulness).
# Exports: GuardianRisk, GuardianVerdict, GuardianModel, GuardianUnavailable
# Deps: dataclasses, enum, abc, typing
# @end-summary
"""Guardian model contract.

A "guardian" is a classifier (LLM-based or otherwise) that answers a single
question about a piece of text: *is this safe along risk dimension X?*

Rails consume guardians instead of inlining provider-specific calls. This
keeps Granite Guardian / Llama Guard / ShieldGemma / self-check prompts
behind one stable interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Literal, Optional


class GuardianRisk(Enum):
    """Risk dimension a guardian can be asked to classify.

    These names map to provider-specific risk identifiers inside each
    guardian implementation (e.g. Granite Guardian's ``risk_name`` field).
    """

    # Umbrella unsafe-content check (covers most provider "harm" defaults)
    HARM = "harm"
    # Prompt-injection / jailbreak attempts
    JAILBREAK = "jailbreak"
    # Specific harm subcategories
    VIOLENCE = "violence"
    SEXUAL = "sexual"
    HATE = "hate"
    PROFANITY = "profanity"
    # Privacy
    PII = "pii"
    # Output-only: answer grounded in provided context
    GROUNDEDNESS = "groundedness"
    # Output-only: function/tool call abuse
    FUNCTION_CALL = "function_call"


Direction = Literal["input", "output"]


@dataclass(frozen=True)
class GuardianVerdict:
    """One guardian's answer for a single (text, risk) classification.

    Attributes:
        safe: True if the text is judged safe for the requested risk.
        risk: The risk dimension classified (echoed for caller convenience).
        score: Confidence that the text is *unsafe*, in ``[0.0, 1.0]``.
            Implementations that cannot produce a probability return
            ``0.9`` for unsafe and ``0.1`` for safe by convention.
        reason: Optional human-readable rationale (rarely populated; most
            classifiers return only yes/no).
        raw: Provider-native payload for debugging / telemetry.
    """

    safe: bool
    risk: GuardianRisk
    score: float
    reason: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


class GuardianUnavailable(RuntimeError):
    """Raised when a guardian cannot complete a classification.

    Rails treat this as "guardian missed"; the rail then falls through to
    its deterministic fallback (e.g. regex). This is *not* an error the
    caller should propagate — it represents a graceful degradation signal.
    """


class GuardianModel(ABC):
    """Abstract guardian classifier.

    Subclasses declare which risks they support via ``supported_risks`` and
    implement :meth:`classify`. Rails check ``supports(risk)`` before
    calling so an unsupported risk routes to the next-best judge instead
    of raising.

    Implementations MUST be safe to call concurrently from multiple
    threads. Stateful clients (e.g. local transformers) should serialize
    inference internally.
    """

    name: ClassVar[str] = "guardian"
    supported_risks: ClassVar[frozenset[GuardianRisk]] = frozenset()

    def supports(self, risk: GuardianRisk) -> bool:
        """Return True if this guardian can answer about ``risk``."""
        return risk in self.supported_risks

    @abstractmethod
    def classify(
        self,
        text: str,
        *,
        risk: GuardianRisk,
        context: Optional[list[str]] = None,
        direction: Direction = "input",
    ) -> GuardianVerdict:
        """Classify ``text`` along the given risk dimension.

        Args:
            text: The text to judge (user query for input direction;
                generated answer for output direction).
            risk: Which risk dimension to evaluate.
            context: Optional supporting context. Required for
                :attr:`GuardianRisk.GROUNDEDNESS`; ignored otherwise unless
                the provider supports context-aware classification.
            direction: ``"input"`` for user queries, ``"output"`` for bot
                responses. Lets providers select role-appropriate prompts.

        Returns:
            ``GuardianVerdict`` with ``safe`` set and ``score`` in [0, 1].

        Raises:
            GuardianUnavailable: When the underlying client/model cannot
                produce a verdict (network error, model not loaded, etc.).
                Rails interpret this as a graceful miss.
            ValueError: When ``risk`` is not in ``supported_risks``.
        """
