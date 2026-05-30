# @summary
# Pure retrieval metrics + per-qtype aggregators over RetrievalResults:
# recall@k (set-based, P4) and reciprocal rank / MRR (rank-sensitive, P7e).
# Exports: recall_at_k, aggregate_recall_by_qtype, reciprocal_rank,
#          aggregate_mrr_by_qtype
# Deps: stdlib (statistics, typing); src.eval.runner.retrieve.RetrievalResults;
#       src.eval.pack.schema.Golden
# @end-summary
"""P4 — recall@k metric + per-qtype aggregation; P7e — MRR.

``recall_at_k`` is set-based: |expected ∩ retrieved| / |expected|. Empty
``expected`` raises ValueError (caller must filter empty-gold goldens
first); empty ``retrieved`` returns 0.0. The asymmetry is intentional —
empty retrieval is a real outcome, empty expectation is a categorization
mismatch.

``reciprocal_rank`` is rank-sensitive: ``1 / rank`` of the FIRST retrieved
source that is in the expected set (1-based), else 0.0. It shares
recall's relevance predicate (exact source-path match) and its
empty-expected/empty-retrieved asymmetry. ``aggregate_mrr_by_qtype``
mirrors ``aggregate_recall_by_qtype`` exactly, taking the mean reciprocal
rank per qtype ("MRR").
"""
from __future__ import annotations

import statistics
from typing import Mapping, Sequence

from src.eval.pack.schema import Golden

from .retrieve import RetrievalResults


def recall_at_k(
    retrieved_sources: Sequence[str],
    expected_sources: Sequence[str],
) -> float:
    """Compute set-recall of expected sources within retrieved top-k.

    Args:
        retrieved_sources: Top-k retrieved source paths (rank order ignored).
        expected_sources: Gold source paths for this query.

    Returns:
        ``|expected ∩ retrieved| / |expected|``.

    Raises:
        ValueError: if ``expected_sources`` is empty (caller must filter).
    """
    if not expected_sources:
        raise ValueError("recall_at_k requires non-empty expected_sources")
    if not retrieved_sources:
        return 0.0
    expected_set = set(expected_sources)
    retrieved_set = set(retrieved_sources)
    return len(expected_set & retrieved_set) / len(expected_set)


def aggregate_recall_by_qtype(
    results: RetrievalResults,
    goldens: Mapping[str, list[Golden]],
) -> dict[str, float]:
    """Mean recall@k per qtype over scoreable goldens.

    Qtypes whose goldens ALL have empty ``expected_source_docs`` are
    omitted entirely from the returned dict (no nan, no zero).
    """
    out: dict[str, float] = {}
    for qtype, golden_list in goldens.items():
        recalls: list[float] = []
        for golden in golden_list:
            if not golden.expected_source_docs:
                continue
            qresult = results.per_query.get(golden.qid)
            if qresult is None:
                continue
            recalls.append(
                recall_at_k(
                    qresult.retrieved_sources, golden.expected_source_docs
                )
            )
        if recalls:
            out[qtype] = statistics.mean(recalls)
    return out


def reciprocal_rank(
    retrieved_sources: Sequence[str],
    expected_sources: Sequence[str],
) -> float:
    """Reciprocal rank of the first relevant source in rank-ordered retrieval.

    Uses the SAME relevance predicate as :func:`recall_at_k` (exact source
    -path membership in the expected set), but it is rank-sensitive rather
    than set-based: only the position of the FIRST relevant hit matters.

    Args:
        retrieved_sources: Retrieved source paths, rank-ordered best-first
            (index 0 == rank 1).
        expected_sources: Gold source paths for this query.

    Returns:
        ``1.0 / rank`` where ``rank`` is the 1-based position of the first
        retrieved source in ``set(expected_sources)``; ``0.0`` if retrieval
        is empty or no retrieved source is relevant.

    Raises:
        ValueError: if ``expected_sources`` is empty (caller must filter).
    """
    if not expected_sources:
        raise ValueError("reciprocal_rank requires non-empty expected_sources")
    if not retrieved_sources:
        return 0.0
    expected_set = set(expected_sources)
    for index, source in enumerate(retrieved_sources):
        if source in expected_set:
            return 1.0 / (index + 1)
    return 0.0


def aggregate_mrr_by_qtype(
    results: RetrievalResults,
    goldens: Mapping[str, list[Golden]],
) -> dict[str, float]:
    """Mean reciprocal rank (MRR) per qtype over scoreable goldens.

    Mirrors :func:`aggregate_recall_by_qtype` exactly — same filtering
    (skip goldens with empty ``expected_source_docs``, skip goldens with no
    matching query result, omit qtypes with no scoreable goldens) — but
    scores each golden with :func:`reciprocal_rank` and takes the per-qtype
    mean. "MRR" is the mean of these reciprocal ranks.
    """
    out: dict[str, float] = {}
    for qtype, golden_list in goldens.items():
        ranks: list[float] = []
        for golden in golden_list:
            if not golden.expected_source_docs:
                continue
            qresult = results.per_query.get(golden.qid)
            if qresult is None:
                continue
            ranks.append(
                reciprocal_rank(
                    qresult.retrieved_sources, golden.expected_source_docs
                )
            )
        if ranks:
            out[qtype] = statistics.mean(ranks)
    return out
