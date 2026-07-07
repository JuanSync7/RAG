# @summary
# Shared answer-quality scoring for eval harnesses: an LLM-as-judge that scores
# how fully a generated answer addresses a set of expected keywords/concepts,
# with a deterministic substring-overlap fallback when the judge is unavailable.
# Promoted from evals/retrieval/deep_research/test_answer_quality.py so both the
# deep_research A/B tests and the turn_loop multi-mode basket score answers the
# same way (CLAUDE.md: cross-domain helpers → shared module).
# Exports: substring_score, judge_score, score_answer, judge_refusal, hedges,
#          refusal_correct, JUDGE_NOISE_FLOOR, JUDGE_SYSTEM
# Deps: src.platform.llm.provider.call_oneshot (lazy, judge only)
# @end-summary
"""LLM-as-judge answer scoring with a substring-overlap fallback.

The scorer answers one question: *given a candidate answer and a list of
expected keywords/concepts, what fraction of them does the answer clearly
address?* It returns a float in ``[0, 1]``. The judge (an instruct model) is
preferred because it credits synonyms and paraphrases; the substring counter is
a deterministic floor used when the judge cannot be reached, so a harness run
never silently produces no signal.
"""
from __future__ import annotations

import json
import logging
import math
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Tolerance that absorbs LLM-judge non-determinism when comparing two answers.
# A candidate may score this far below a baseline and still count as "no
# regression". Callers own the comparison; this is the shared default floor.
JUDGE_NOISE_FLOOR = 0.10

JUDGE_SYSTEM = (
    "You are a strict evaluator. Given a candidate ANSWER and a list of "
    "EXPECTED_KEYWORDS, score how many of the keywords (or close synonyms) "
    "are clearly addressed in the answer. Return STRICT JSON of the form "
    '{"score": <float in [0,1]>, "missing": [<keyword>...]}. Do not include '
    "any other text."
)


def substring_score(answer: Optional[str], keywords: list[str]) -> float:
    """Fraction of ``keywords`` that appear (case-insensitive) in ``answer``.

    Returns ``nan`` when there are no keywords (undefined, not zero) and ``0.0``
    when the answer is empty. Deterministic; no model call.
    """
    if not keywords:
        return float("nan")
    if not answer:
        return 0.0
    blob = answer.lower()
    hits = sum(1 for k in keywords if k.lower() in blob)
    return hits / len(keywords)


def judge_score(
    query: str,
    answer: Optional[str],
    keywords: list[str],
    *,
    model_alias: str = "query",
) -> Optional[float]:
    """LLM-as-judge coverage score in ``[0, 1]``, or ``None`` if the judge failed.

    ``None`` (not ``0.0``) signals "no judge signal" so callers can fall back to
    :func:`substring_score` rather than record a spurious zero.
    """
    if not answer or not keywords:
        return None
    try:
        from src.platform.llm.provider import call_oneshot
    except Exception as exc:  # noqa: BLE001 — missing provider is a soft skip
        logger.warning("judge import failed: %s", exc)
        return None

    prompt = (
        f"QUESTION:\n{query}\n\n"
        f"EXPECTED_KEYWORDS: {json.dumps(keywords)}\n\n"
        f"ANSWER:\n{answer}\n\n"
        'Reply with JSON only: {"score": float, "missing": [string]}'
    )
    try:
        raw = call_oneshot(
            prompt,
            system=JUDGE_SYSTEM,
            model_alias=model_alias,
            temperature=0.0,
            max_tokens=200,
        )
    except Exception as exc:  # noqa: BLE001 — a judge outage is a soft skip
        logger.warning("judge call failed: %s", exc)
        return None
    if not raw:
        return None
    # Tolerate ```json fences or stray prose around the JSON object.
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        score = float(obj.get("score"))
        if math.isnan(score):
            return None
        return max(0.0, min(1.0, score))
    except (ValueError, TypeError, json.JSONDecodeError):
        return None


_REFUSAL_SYSTEM = (
    "You judge whether an ANSWER appropriately DECLINES to answer — i.e. it "
    "states the information is not available / not in the provided material / "
    "unknown, rather than asserting a specific factual claim. This is about "
    "epistemic honesty for a question whose answer is NOT in the corpus. "
    'Return STRICT JSON: {"refused": <bool>}. No other text.'
)

# General linguistic markers of epistemic hedging / declining. Used ONLY as a
# fallback when the judge is unreachable — not corpus- or entity-specific
# (CLAUDE.md §0), just the vocabulary of "I don't have this".
_HEDGE_MARKERS = (
    "not available", "not found", "not in", "no information", "not covered",
    "not mention", "no mention", "does not appear", "doesn't appear",
    "cannot find", "can't find", "unable to", "don't have", "do not have",
    "not provided", "not specified", "unknown", "insufficient", "outside the",
    "not present", "no reference", "not describe", "not contain",
)


def judge_refusal(
    query: str,
    answer: Optional[str],
    *,
    model_alias: str = "query",
) -> Optional[bool]:
    """Judge whether ``answer`` correctly DECLINES (states unknown) — general.

    Returns ``True``/``False`` from the judge, or ``None`` if the judge failed
    (callers fall back to :func:`hedges`). An empty answer counts as a refusal
    (no fabricated claim was made).
    """
    if not answer or not answer.strip():
        return True
    try:
        from src.platform.llm.provider import call_oneshot
    except Exception as exc:  # noqa: BLE001
        logger.warning("refusal-judge import failed: %s", exc)
        return None
    prompt = (
        f"QUESTION:\n{query}\n\nANSWER:\n{answer}\n\n"
        'Reply with JSON only: {"refused": bool}'
    )
    try:
        raw = call_oneshot(prompt, system=_REFUSAL_SYSTEM, model_alias=model_alias,
                           temperature=0.0, max_tokens=20)
    except Exception as exc:  # noqa: BLE001
        logger.warning("refusal-judge call failed: %s", exc)
        return None
    if not raw:
        return None
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        return bool(json.loads(m.group(0)).get("refused"))
    except (ValueError, TypeError, json.JSONDecodeError):
        return None


def hedges(answer: Optional[str]) -> bool:
    """Deterministic fallback: does the answer contain epistemic-hedging language?

    An empty answer counts as a refusal. General markers only (no entity/vendor
    strings) so it does not overfit the corpus.
    """
    if not answer or not answer.strip():
        return True
    blob = answer.lower()
    return any(marker in blob for marker in _HEDGE_MARKERS)


def refusal_correct(
    query: str,
    answer: Optional[str],
    *,
    model_alias: str = "query",
) -> bool:
    """1-shot: did the answer correctly decline? Judge first, hedges() fallback."""
    judged = judge_refusal(query, answer, model_alias=model_alias)
    if judged is not None:
        return judged
    return hedges(answer)


def score_answer(
    query: str,
    answer: Optional[str],
    keywords: list[str],
    *,
    model_alias: str = "query",
) -> float:
    """Prefer the LLM judge; fall back to substring overlap on judge failure.

    Always returns a concrete float in ``[0, 1]`` (``0.0`` for an empty answer).
    """
    if not answer:
        return 0.0
    judged = judge_score(query, answer, keywords, model_alias=model_alias)
    if judged is not None:
        return judged
    fallback = substring_score(answer, keywords)
    return 0.0 if math.isnan(fallback) else fallback
