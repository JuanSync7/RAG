# @summary
# Toxicity detection rail using NeMo self-check prompts with keyword fallback.
# Runtime injected at construction — no direct GuardrailsRuntime.get() call.
# Exports: ToxicityFilter, ToxicityResult
# Deps: src.guardrails.common.schemas, config.settings, logging, re
# @end-summary
"""Toxicity filtering rail.

This module rejects unsafe/toxic input and can filter unsafe output. It prefers
an LLM-driven "self-check" prompt (NeMo-style yes/no classification) when the
guardrails runtime is available, and otherwise falls back to deterministic
keyword patterns.

Requirements references (from internal docs): REQ-401 through REQ-404.
"""

from __future__ import annotations

import orjson
import logging
import re
from dataclasses import dataclass

from src.guardrails.common import RailVerdict

logger = logging.getLogger("rag.guardrails.toxicity")

REJECTION_MESSAGE = (
    "Your query contains content that violates our usage policy. Please rephrase."
)

# NeMo-style self-check prompt for input safety (adapted from NeMo's self_check_input)
_SELF_CHECK_INPUT_PROMPT = """\
Your task is to check if the user message below complies with safety policies.

Check for these policy violations:
1. Hate speech, threats of violence, or incitement to harm
2. Sexually explicit or graphic content
3. Severe profanity directed at individuals or groups
4. Instructions for illegal activities or dangerous behavior
5. Content targeting protected characteristics (race, gender, religion, etc.)

Note: mild informal language (e.g., "damn", "hell") is acceptable and should NOT be flagged.

User message: "{text}"

Is this message safe? Respond with "yes" if safe, "no" if unsafe. Output only "yes" or "no"."""

# NeMo-style self-check prompt for output safety
_SELF_CHECK_OUTPUT_PROMPT = """\
Your task is to check if the bot response below complies with safety policies.

Check for these policy violations:
1. Contains hate speech, threats, or violent content
2. Contains sexually explicit material
3. Reveals private/sensitive information about individuals
4. Provides instructions for harmful or illegal activities
5. Contains discriminatory or biased content

Bot response: "{text}"

Is this response safe? Respond with "yes" if safe, "no" if unsafe. Output only "yes" or "no"."""

# Keyword-based fallback for deterministic detection when LLM is unavailable.
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
    """Detect toxic content using NeMo self-check prompts with keyword fallback.

    Uses structured safety-check prompts (adapted from NeMo's built-in
    self_check_input and self_check_output tasks) for more reliable
    classification than raw toxicity scoring. Falls back to keyword
    matching when LLM is unavailable.

    The NeMo LLM path is gated on the injected runtime: when
    ``runtime=None``, only keyword fallback is used.
    """

    def __init__(self, threshold: float = 0.5, runtime=None) -> None:
        """Initialize a toxicity filter.

        Args:
            threshold: Threshold used when parsing numeric toxicity scores from
                legacy/JSON responses.
            runtime: Optional ``GuardrailsRuntime`` instance. When provided,
                enables the NeMo self-check LLM path. When ``None``, only
                keyword fallback is used.
        """
        self._threshold = threshold
        self._runtime = runtime

    def check(self, text: str) -> ToxicityResult:
        """Check text for toxic content (input direction).

        Args:
            text: Input text to classify.

        Returns:
            `ToxicityResult` containing the verdict and optional metadata.
        """
        # Try NeMo self-check approach first
        runtime = self._runtime
        if runtime is not None and runtime.initialized and runtime.rails is not None:
            try:
                return self._self_check(text, direction="input")
            except Exception as e:
                logger.warning("Self-check input failed: %s — using fallback", e)

        # Deterministic keyword fallback
        return self._check_with_keywords(text)

    def _self_check(self, text: str, direction: str = "input") -> ToxicityResult:
        """Run NeMo-style self-check prompt for safety classification.

        Args:
            text: Input text to classify.
            direction: "input" or "output" to select the prompt template.

        Returns:
            `ToxicityResult` derived from yes/no (or legacy JSON) response.
        """
        if direction == "output":
            prompt = _SELF_CHECK_OUTPUT_PROMPT.format(text=text)
        else:
            prompt = _SELF_CHECK_INPUT_PROMPT.format(text=text)

        from src.platform.llm import call_oneshot

        result = call_oneshot(
            prompt,
            system="You are a content safety classifier. Output only 'yes' or 'no'.",
        )
        if result:
            answer = result.strip().lower()
            # NeMo convention: "yes" = safe, "no" = unsafe
            if answer.startswith("no"):
                logger.info("Toxicity detected via self-check (%s)", direction)
                return ToxicityResult(
                    verdict=RailVerdict.REJECT,
                    score=0.9,
                    message=REJECTION_MESSAGE,
                )
            if answer.startswith("yes"):
                return ToxicityResult(verdict=RailVerdict.PASS, score=0.1)

            # Ambiguous response — try to parse as JSON for backward compat
            try:
                parsed = orjson.loads(result)
                score = float(parsed.get("toxicity_score", 0.0))
                score = max(0.0, min(1.0, score))
                if score >= self._threshold:
                    return ToxicityResult(
                        verdict=RailVerdict.REJECT,
                        score=score,
                        message=REJECTION_MESSAGE,
                    )
                return ToxicityResult(verdict=RailVerdict.PASS, score=score)
            except (orjson.JSONDecodeError, ValueError, TypeError):
                pass

        return ToxicityResult(verdict=RailVerdict.PASS, score=0.0)

    def _check_with_keywords(self, text: str) -> ToxicityResult:
        """Detect toxic content using deterministic keyword patterns.

        Args:
            text: Input text to scan.

        Returns:
            `ToxicityResult` indicating whether any pattern matched.
        """
        for pattern in _TOXIC_KEYWORD_PATTERNS:
            if pattern.search(text):
                logger.info("Toxicity detected via keyword pattern")
                return ToxicityResult(
                    verdict=RailVerdict.REJECT,
                    score=0.9,
                    message=REJECTION_MESSAGE,
                )
        return ToxicityResult(verdict=RailVerdict.PASS, score=0.0)

    def filter_output(self, text: str) -> str:
        """Filter toxic content from output text, replacing with placeholder.

        Args:
            text: Output text to filter.

        Returns:
            Original text if safe, otherwise a placeholder string.
        """
        # Use output-direction self-check
        runtime = self._runtime
        if runtime is not None and runtime.initialized and runtime.rails is not None:
            try:
                result = self._self_check(text, direction="output")
                if result.verdict == RailVerdict.REJECT:
                    return "[CONTENT_FILTERED]"
                return text
            except Exception as e:
                logger.warning("Self-check output failed: %s — using fallback", e)

        result = self._check_with_keywords(text)
        if result.verdict == RailVerdict.REJECT:
            return "[CONTENT_FILTERED]"
        return text
