# @summary
# Tests for P7e Mean Reciprocal Rank (MRR) metric over RetrievalResults.
# Covers reciprocal_rank (pure rank-sensitive fn: empty-expected raises,
# empty/miss → 0.0, first-hit-rank → 1/rank) with mutation-resistant data,
# aggregate_mrr_by_qtype (mean RR per qtype, mirrors recall aggregation
# rules), and an end-to-end build_eval_report population check.
# Mutation B target: test_reciprocal_rank_uses_first_hit_rank
# Mutation D target: test_build_eval_report_populates_mrr_by_qtype
# Exports: test_reciprocal_rank_raises_on_empty_expected,
#          test_reciprocal_rank_empty_retrieved,
#          test_reciprocal_rank_no_match,
#          test_reciprocal_rank_uses_first_hit_rank,
#          test_reciprocal_rank_first_position,
#          test_aggregate_mrr_by_qtype,
#          test_build_eval_report_populates_mrr_by_qtype
# Deps: src.eval.runner (reciprocal_rank, aggregate_mrr_by_qtype,
#       build_eval_report), src.eval.pack.schema, src.eval.runner.retrieve
# @end-summary
"""P7e — Mean Reciprocal Rank (MRR) metric tests."""
from __future__ import annotations

import pytest

from src.eval.runner import (
    RetrievalResults,
    aggregate_mrr_by_qtype,
    reciprocal_rank,
)


def test_reciprocal_rank_raises_on_empty_expected() -> None:
    """reciprocal_rank mirrors recall_at_k: empty expected is a ValueError."""
    with pytest.raises(ValueError):
        reciprocal_rank(["docs/a.md"], [])


def test_reciprocal_rank_empty_retrieved() -> None:
    """No retrieval is a real outcome → RR is 0.0, not an error."""
    assert reciprocal_rank([], ["docs/a.md"]) == pytest.approx(0.0)


def test_reciprocal_rank_no_match() -> None:
    """No retrieved source in the expected set → RR is 0.0."""
    assert reciprocal_rank(
        ["docs/x.md", "docs/y.md"], ["docs/a.md"]
    ) == pytest.approx(0.0)


def test_reciprocal_rank_uses_first_hit_rank() -> None:
    """RR is 1 / (1-based rank of the FIRST relevant retrieved source).

    MUTATION B TARGET: data is engineered so the correct RR (1/3) is
    distinct from every plausible wrong implementation:
      - set-overlap / recall style → 1.0
      - 0-based 1/i (i=2) → 0.5
      - 1/len(retrieved) → 0.25
      - 1/len(expected) → 1.0
    First relevant ("docs/d2.md") sits at index 2 → rank 3 → RR = 1/3.
    """
    rr = reciprocal_rank(
        ["docs/d0.md", "docs/d1.md", "docs/d2.md", "docs/d3.md"],
        ["docs/d2.md"],
    )
    assert rr == pytest.approx(1.0 / 3.0)
    # Distinctness guards (these would pass under the wrong impls listed above).
    assert rr != pytest.approx(1.0)
    assert rr != pytest.approx(0.5)
    assert rr != pytest.approx(0.25)


def test_reciprocal_rank_first_position() -> None:
    """First relevant at rank 1 → RR = 1.0; only the FIRST hit counts."""
    rr = reciprocal_rank(
        ["docs/a.md", "docs/b.md"], ["docs/a.md", "docs/b.md"]
    )
    assert rr == pytest.approx(1.0)


def test_aggregate_mrr_by_qtype() -> None:
    """Mean RR per qtype over scoreable goldens; empty-gold qtypes omitted.

    Mirrors aggregate_recall_by_qtype's filtering exactly but ranks matter.
    """
    from src.eval.pack.schema import Golden
    from src.eval.runner.retrieve import QueryRetrievalResult

    goldens = {
        "factoid": [
            Golden(
                qid="f1",
                qtype="factoid",
                query="q1",
                expected_answer_span="x",
                expected_source_docs=["docs/a.md"],
            ),
            Golden(
                qid="f2",
                qtype="factoid",
                query="q2",
                expected_answer_span="y",
                expected_source_docs=["docs/b.md"],
            ),
        ],
        "qfs": [
            Golden(
                qid="q1",
                qtype="qfs",
                query="q3",
                expected_source_docs=["docs/c.md"],
            ),
        ],
        "adversarial": [
            Golden(
                qid="a1",
                qtype="adversarial",
                query="q4",
                expected_source_docs=[],
            ),
        ],
    }
    per_query = {
        # f1: relevant at rank 1 → RR 1.0
        "f1": QueryRetrievalResult(
            qid="f1",
            qtype="factoid",
            query="q1",
            retrieved_sources=("docs/a.md", "docs/z.md"),
            k=5,
        ),
        # f2: relevant ("docs/b.md") at rank 2 → RR 0.5
        "f2": QueryRetrievalResult(
            qid="f2",
            qtype="factoid",
            query="q2",
            retrieved_sources=("docs/x.md", "docs/b.md"),
            k=5,
        ),
        # q1: relevant ("docs/c.md") at rank 3 → RR 1/3
        "q1": QueryRetrievalResult(
            qid="q1",
            qtype="qfs",
            query="q3",
            retrieved_sources=("docs/m.md", "docs/n.md", "docs/c.md"),
            k=5,
        ),
        "a1": QueryRetrievalResult(
            qid="a1",
            qtype="adversarial",
            query="q4",
            retrieved_sources=("docs/whatever.md",),
            k=5,
        ),
    }
    results = RetrievalResults(collection_name="C", k=5, per_query=per_query)

    out = aggregate_mrr_by_qtype(results, goldens)

    # mean(1.0, 0.5) = 0.75 — distinct from recall's 0.5 mean for this data.
    assert out["factoid"] == pytest.approx(0.75)
    assert out["qfs"] == pytest.approx(1.0 / 3.0)
    assert "adversarial" not in out


def test_aggregate_mrr_skips_missing_query_results() -> None:
    """A golden with no matching per_query entry is skipped, not zero-scored."""
    from src.eval.pack.schema import Golden
    from src.eval.runner.retrieve import QueryRetrievalResult

    goldens = {
        "factoid": [
            Golden(
                qid="present",
                qtype="factoid",
                query="q1",
                expected_answer_span="x",
                expected_source_docs=["docs/a.md"],
            ),
            Golden(
                qid="missing",
                qtype="factoid",
                query="q2",
                expected_answer_span="y",
                expected_source_docs=["docs/b.md"],
            ),
        ],
    }
    per_query = {
        "present": QueryRetrievalResult(
            qid="present",
            qtype="factoid",
            query="q1",
            retrieved_sources=("docs/a.md",),
            k=5,
        ),
    }
    results = RetrievalResults(collection_name="C", k=5, per_query=per_query)

    out = aggregate_mrr_by_qtype(results, goldens)

    # Only "present" is scoreable → mean over a single RR of 1.0.
    assert out["factoid"] == pytest.approx(1.0)


def test_build_eval_report_populates_mrr_by_qtype() -> None:
    """build_eval_report threads the mrr_by_qtype kwarg onto the report.

    MUTATION D TARGET: ignoring the kwarg (always {}) reds this.
    """
    from src.eval.runner import build_eval_report
    from src.eval.runner.faithfulness import FaithfulnessResults
    from src.eval.runner.retrieve import QueryRetrievalResult, RetrievalResults

    retrieval = RetrievalResults(
        collection_name="C",
        k=5,
        per_query={
            "f1": QueryRetrievalResult(
                qid="f1",
                qtype="factoid",
                query="q1",
                retrieved_sources=("docs/a.md",),
                k=5,
            )
        },
    )
    faith = FaithfulnessResults(
        collection_name="C",
        per_query={},
        total_queries_scored=0,
        total_queries_skipped=0,
    )

    report = build_eval_report(
        retrieval,
        faith,
        recall_by_qtype={"factoid": 1.0},
        per_query_recall={"f1": 1.0},
        mrr_by_qtype={"factoid": 0.5},
    )

    assert report.mrr_by_qtype == {"factoid": 0.5}


def test_build_eval_report_defaults_mrr_to_empty() -> None:
    """Omitting mrr_by_qtype yields an empty mapping (additive default)."""
    from src.eval.runner import build_eval_report
    from src.eval.runner.faithfulness import FaithfulnessResults
    from src.eval.runner.retrieve import RetrievalResults

    retrieval = RetrievalResults(collection_name="C", k=5, per_query={})
    faith = FaithfulnessResults(
        collection_name="C",
        per_query={},
        total_queries_scored=0,
        total_queries_skipped=0,
    )

    report = build_eval_report(
        retrieval,
        faith,
        recall_by_qtype={},
        per_query_recall={},
    )

    assert report.mrr_by_qtype == {}
