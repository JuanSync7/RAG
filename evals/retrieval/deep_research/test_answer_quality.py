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

The scorer itself lives in ``evals.common.answer_quality`` (shared with the
turn_loop multi-mode basket); this module keeps thin private aliases for
backward compatibility.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from evals.common.answer_quality import (
    JUDGE_NOISE_FLOOR as _JUDGE_NOISE_FLOOR,
    judge_score as _judge_score,
    score_answer as _score,
    substring_score as _substring_score,
)

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.eval_deep_research


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


# Back-compat: keep the private names importable for any external caller that
# reached into this module before the scorer moved to evals.common.
__all__ = ["_generate_answer", "_score", "_judge_score", "_substring_score",
           "_JUDGE_NOISE_FLOOR", "test_answer_quality_dr_no_regression"]
