# @summary
# Toxicity detection rail. Delegates LLM-based judging to a GuardianModel
# (Granite Guardian, self-check, etc.) and keeps a deterministic regex
# fallback floor that runs even when no guardian is available.
# Exports: ToxicityFilter, ToxicityResult
# Deps: src.guardrails.common.schemas, src.guardrails.models, logging, re
# @end-summary
"""Toxicity filtering rail.

This rail rejects unsafe input and filters unsafe output. Decision tree:

1. If a :class:`GuardianModel` was injected and supports :attr:`HARM`,
   ask it. Trust its verdict on success.
2. If the guardian misses (raises ``GuardianUnavailable``) or wasn't
   provided, fall through to the regex keyword floor — the only path
   that always works, even with all external deps offline.

The legacy ``runtime`` argument is preserved for callers that pre-date
the guardian abstraction; when supplied without a ``guardian`` we wrap
it in a :class:`SelfCheckGuardian` so all judges share one code path.

Requirements references (from internal docs): REQ-401 through REQ-404.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from src.guardrails.common import RailVerdict
from src.guardrails.models import (
    GuardianModel,
    GuardianRisk,
    GuardianUnavailable,
)

logger = logging.getLogger("rag.guardrails.toxicity")

REJECTION_MESSAGE = (
    "Your query contains content that violates our usage policy. Please rephrase."
)

# Keyword-based fallback for deterministic detection when no guardian is available.
_TOXIC_KEYWORD_PATTERNS = [
    re.compile(
        r"\b(?:kill|murder|attack|bomb|shoot)\s+(?:you|them|people|everyone)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:hate|despise)\s+(?:all|every)\s+\w+\b", re.IGNORECASE),
    re.compile(r"\b(?:die|death\s+to)\b", re.IGNORECASE),
]


@dataclass
class ToxicityResult:
    """Result of toxicity detection.

    Attributes:
        verdict: PASS/REJECT/MODIFY verdict.
        score: Optional normalized score in [0.0, 1.0] when available.
        message: Optional human-facing message for rejections.
    """

    verdict: RailVerdict
    score: float = 0.0
    message: str = ""


class ToxicityFilter:
    """Detect toxic content via an injected guardian with regex fallback.

    Args:
        threshold: Confidence cutoff used by the regex fallback's score
            attribution; rarely needs tuning.
        guardian: Optional :class:`GuardianModel` to call for HARM
            classification. Pass ``None`` to use only the regex floor.
        runtime: Deprecated. When provided without ``guardian``, a
            :class:`SelfCheckGuardian` is constructed internally so the
            existing yes/no LLM behaviour is preserved verbatim.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        guardian: Optional[GuardianModel] = None,
        runtime=None,
    ) -> None:
        self._threshold = threshold

        if guardian is None and runtime is not None:
            # Back-compat: callers wiring ``runtime=`` get the old self-check
            # behaviour, but routed through the guardian interface.
            try:
                if runtime.initialized and runtime.rails is not None:
                    from src.guardrails.models import SelfCheckGuardian
                    guardian = SelfCheckGuardian(threshold=threshold)
            except AttributeError:
                # Unrecognised runtime shape — leave guardian unset
                pass

        self._guardian = guardian

    def check(self, text: str) -> ToxicityResult:
        """Check text for toxic content (input direction).

        Args:
            text: Input text to classify.

        Returns:
            `ToxicityResult` containing the verdict and optional metadata.
        """
        return self._classify(text, direction="input")

    def filter_output(self, text: str) -> str:
        """Filter toxic content from output text, replacing with placeholder.

        Args:
            text: Output text to filter.

        Returns:
            Original text if safe, otherwise ``"[CONTENT_FILTERED]"``.
        """
        result = self._classify(text, direction="output")
        if result.verdict == RailVerdict.REJECT:
            return "[CONTENT_FILTERED]"
        return text

    # ── internal ──────────────────────────────────────────────────────────

    def _classify(self, text: str, *, direction: str) -> ToxicityResult:
        """Run guardian → regex fallback in order.

        Guardian misses (network errors, empty responses) downgrade to the
        regex floor so the rail always returns a verdict.
        """
        if self._guardian is not None and self._guardian.supports(GuardianRisk.HARM):
            try:
                verdict = self._guardian.classify(
                    text, risk=GuardianRisk.HARM, direction=direction  # type: ignore[arg-type]
                )
                if not verdict.safe:
                    logger.info(
                        "Toxicity REJECT via %s (score=%.2f, dir=%s)",
                        self._guardian.name, verdict.score, direction,
                    )
                    return ToxicityResult(
                        verdict=RailVerdict.REJECT,
                        score=verdict.score,
                        message=REJECTION_MESSAGE,
                    )
                return ToxicityResult(verdict=RailVerdict.PASS, score=verdict.score)
            except GuardianUnavailable as e:
                logger.warning(
                    "Guardian %s unavailable (%s) — regex fallback",
                    self._guardian.name, e,
                )
            except Exception as e:
                # Defensive: guardians should raise GuardianUnavailable, but
                # if a subclass leaks an unexpected error we still want a
                # verdict rather than propagating up to the rail executor.
                logger.warning(
                    "Guardian %s raised %s — regex fallback",
                    self._guardian.name, type(e).__name__,
                )

        return self._check_with_keywords(text)

    def _check_with_keywords(self, text: str) -> ToxicityResult:
        """Detect toxic content using deterministic keyword patterns."""
        for pattern in _TOXIC_KEYWORD_PATTERNS:
            if pattern.search(text):
                logger.info("Toxicity detected via keyword pattern")
                return ToxicityResult(
                    verdict=RailVerdict.REJECT,
                    score=0.9,
                    message=REJECTION_MESSAGE,
                )
        return ToxicityResult(verdict=RailVerdict.PASS, score=0.0)
