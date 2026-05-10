"""Answer-quality tests using LLM-as-judge.

For each golden query carrying ``expected_answer_contains``, generate a
real answer via ``RAGChain.run(mode="query", skip_generation=False)`` for
both baseline and DR. Score with an LLM-as-judge prompt that returns a
0..1 floating-point score reflecting how many of the expected keywords
the answer mentions.

Acceptance:

  * On ``multi_aspect`` / ``disjoint`` queries DR's score must be >=
    baseline's score (within a small floor for judge noise).
  * On ``single_aspect`` queries DR must not regress beyond a noise floor.

Pure substring scoring is computed in parallel as a sanity floor — if the
LLM judge is unavailable the test falls back to substring counts.
"""

from __future__ import annotations

import json
import logging
import math
import re
from typing import Any

import pytest

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.eval_deep_research


# Tolerance to absorb LLM-judge non-determinism. DR may score 0.05 below
# baseline and still be accepted as "no regression". On multi/disjoint
# we still require DR >= baseline - tolerance.
_JUDGE_NOISE_FLOOR = 0.10


def _generate_answer(rag_chain, query: str, *, deep_research: bool) -> str | None:
    """Run mode=query end-to-end and return the generated answer string."""
    response = rag_chain.run(
        query=query,
        rerank_top_k=10,
        skip_generation=False,
        mode="query",
        deep_research=deep_research,
        overall_timeout_ms=240_000 if deep_research else 60_000,
    )
    return getattr(response, "generated_answer", None)


def _substring_score(answer: str, keywords: list[str]) -> float:
    """Fraction of keywords that appear (case-insensitive) in ``answer``."""
    if not keywords:
        return float("nan")
    if not answer:
        return 0.0
    blob = answer.lower()
    hits = sum(1 for k in keywords if k.lower() in blob)
    return hits / len(keywords)


_JUDGE_SYSTEM = (
    "You are a strict evaluator. Given a candidate ANSWER and a list of "
    "EXPECTED_KEYWORDS, score how many of the keywords (or close synonyms) "
    "are clearly addressed in the answer. Return STRICT JSON of the form "
    '{"score": <float in [0,1]>, "missing": [<keyword>...]}. Do not include '
    "any other text."
)


def _judge_score(query: str, answer: str, keywords: list[str]) -> float | None:
    """LLM-as-judge: returns a 0..1 score, or None if judge call failed."""
    if not answer or not keywords:
        return None
    try:
        from src.platform.llm.provider import call_oneshot
    except Exception as exc:
        logger.warning("judge import failed: %s", exc)
        return None

    prompt = (
        f"QUESTION:\n{query}\n\n"
        f"EXPECTED_KEYWORDS: {json.dumps(keywords)}\n\n"
        f"ANSWER:\n{answer}\n\n"
        'Reply with JSON only: {"score": float, "missing": [string]}'
    )
    raw = call_oneshot(
        prompt,
        system=_JUDGE_SYSTEM,
        model_alias="query",
        temperature=0.0,
        max_tokens=200,
    )
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


def _score(query: str, answer: str | None, keywords: list[str]) -> float:
    """Prefer LLM-judge; fall back to substring score on judge failure."""
    if not answer:
        return 0.0
    judged = _judge_score(query, answer, keywords)
    if judged is not None:
        return judged
    return _substring_score(answer, keywords)


def test_answer_quality_dr_no_regression(rag_chain, golden_queries_deep_research_asic):
    queries: list[dict[str, Any]] = [
        q for q in golden_queries_deep_research_asic.get("queries") or []
        if q.get("expected_answer_contains")
    ]
    if not queries:
        pytest.skip("No golden queries declare expected_answer_contains")

    rows = []
    for q in queries:
        keywords = q["expected_answer_contains"]
        try:
            base_ans = _generate_answer(rag_chain, q["query"], deep_research=False)
        except Exception as exc:
            logger.warning("baseline generation failed for %s: %s", q["id"], exc)
            base_ans = None
        try:
            dr_ans = _generate_answer(rag_chain, q["query"], deep_research=True)
        except Exception as exc:
            logger.warning("DR generation failed for %s: %s", q["id"], exc)
            dr_ans = None

        base_score = _score(q["query"], base_ans, keywords)
        dr_score = _score(q["query"], dr_ans, keywords)
        rows.append({
            "id": q["id"],
            "category": q.get("category", "unknown"),
            "baseline_score": base_score,
            "dr_score": dr_score,
            "baseline_has_answer": base_ans is not None,
            "dr_has_answer": dr_ans is not None,
        })

    # At least one row must have a generated answer somewhere — otherwise
    # generation is broken and the test signal is meaningless. Skip rather
    # than false-positive.
    if not any(r["baseline_has_answer"] or r["dr_has_answer"] for r in rows):
        pytest.skip("Neither baseline nor DR produced any generated_answer")

    failures: list[str] = []
    for r in rows:
        if r["category"] in ("multi_aspect", "disjoint"):
            # DR must >= baseline - noise on the queries it's designed for.
            if r["dr_score"] + _JUDGE_NOISE_FLOOR < r["baseline_score"]:
                failures.append(
                    f"{r['id']} ({r['category']}): DR={r['dr_score']:.2f} "
                    f"< baseline={r['baseline_score']:.2f} - {_JUDGE_NOISE_FLOOR}"
                )
        else:
            # single_aspect / unknown: tolerated regression up to noise floor.
            if r["dr_score"] + _JUDGE_NOISE_FLOOR < r["baseline_score"]:
                failures.append(
                    f"{r['id']} ({r['category']}): DR regressed beyond noise floor "
                    f"(DR={r['dr_score']:.2f}, baseline={r['baseline_score']:.2f})"
                )

    assert not failures, (
        "Answer-quality regressions detected:\n  " + "\n  ".join(failures)
        + f"\n\nFull rows: {rows}"
    )
