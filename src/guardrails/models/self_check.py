# @summary
# Legacy "self-check" guardian: yes/no LLM prompt over the project's call_oneshot.
# Wraps the prompts that previously lived inline in shared/toxicity.py so
# rails can address all judges through one interface.
# Exports: SelfCheckGuardian
# Deps: src.guardrails.models.base, src.platform.llm, orjson, logging
# @end-summary
"""Self-check LLM guardian (legacy fallback).

Implements :class:`GuardianModel` over the project's general-purpose LLM via
:func:`src.platform.llm.call_oneshot`. Uses NeMo-style yes/no safety prompts
and supports two directions (input vs. output) and two risks
(``HARM``, ``GROUNDEDNESS``).

This guardian is the path used when no dedicated classifier model (e.g.
Granite Guardian) is configured. It is intentionally limited in risk
coverage; specialised guardians should be preferred for production.
"""

from __future__ import annotations

import logging
from typing import Optional

import orjson

from src.guardrails.models.base import (
    Direction,
    GuardianModel,
    GuardianRisk,
    GuardianUnavailable,
    GuardianVerdict,
)

logger = logging.getLogger("rag.guardrails.models.self_check")


_INPUT_PROMPT = """\
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

_OUTPUT_PROMPT = """\
Your task is to check if the bot response below complies with safety policies.

Check for these policy violations:
1. Contains hate speech, threats, or violent content
2. Contains sexually explicit material
3. Reveals private/sensitive information about individuals
4. Provides instructions for harmful or illegal activities
5. Contains discriminatory or biased content

Bot response: "{text}"

Is this response safe? Respond with "yes" if safe, "no" if unsafe. Output only "yes" or "no"."""

_GROUNDED_PROMPT = """\
You are checking whether a bot response is fully grounded in the provided context.

Context:
{context}

Response: "{text}"

Is the response fully supported by the context? Respond with "yes" if grounded, "no" if it contains unsupported claims. Output only "yes" or "no"."""

_SYSTEM_INSTR = "You are a content safety classifier. Output only 'yes' or 'no'."


class SelfCheckGuardian(GuardianModel):
    """Yes/No LLM guardian using ``call_oneshot``.

    Supports :attr:`GuardianRisk.HARM` for both directions and
    :attr:`GuardianRisk.GROUNDEDNESS` for output direction. Other risks
    raise ``ValueError``.

    Args:
        threshold: Confidence cutoff used when the LLM unexpectedly returns
            a JSON ``{"toxicity_score": ...}`` payload (legacy compat).
    """

    name = "self_check"
    supported_risks = frozenset(
        {GuardianRisk.HARM, GuardianRisk.GROUNDEDNESS}
    )

    def __init__(self, threshold: float = 0.5) -> None:
        self._threshold = threshold

    def classify(
        self,
        text: str,
        *,
        risk: GuardianRisk,
        context: Optional[list[str]] = None,
        direction: Direction = "input",
    ) -> GuardianVerdict:
        if risk not in self.supported_risks:
            raise ValueError(
                f"SelfCheckGuardian does not support risk {risk!r}; "
                f"supported: {sorted(r.value for r in self.supported_risks)}"
            )

        if risk == GuardianRisk.GROUNDEDNESS:
            ctx = "\n---\n".join(context or [])
            prompt = _GROUNDED_PROMPT.format(context=ctx or "(no context)", text=text)
        elif direction == "output":
            prompt = _OUTPUT_PROMPT.format(text=text)
        else:
            prompt = _INPUT_PROMPT.format(text=text)

        from src.platform.llm import call_oneshot

        try:
            result = call_oneshot(prompt, system=_SYSTEM_INSTR)
        except Exception as e:  # call_oneshot already logs; treat as miss
            raise GuardianUnavailable(str(e)) from e

        if not result:
            raise GuardianUnavailable("empty response from call_oneshot")

        answer = result.strip().lower()
        if answer.startswith("yes"):
            return GuardianVerdict(
                safe=True, risk=risk, score=0.1, raw={"answer": answer}
            )
        if answer.startswith("no"):
            return GuardianVerdict(
                safe=False, risk=risk, score=0.9, raw={"answer": answer}
            )

        # Legacy JSON payload compat (some older configs returned scores)
        try:
            parsed = orjson.loads(result)
            score = float(parsed.get("toxicity_score", 0.0))
            score = max(0.0, min(1.0, score))
            return GuardianVerdict(
                safe=score < self._threshold,
                risk=risk,
                score=score,
                raw=parsed,
            )
        except (orjson.JSONDecodeError, ValueError, TypeError):
            pass

        # Ambiguous response — treat as unavailable so the rail falls back
        raise GuardianUnavailable(f"unparseable self-check response: {result!r}")
