# @summary
# Pure recall@k metric + per-qtype aggregator over RetrievalResults.
# Exports: recall_at_k, aggregate_recall_by_qtype
# Deps: stdlib (statistics, typing); src.eval.runner.retrieve.RetrievalResults;
#       src.eval.pack.schema.Golden
# @end-summary
"""P4 — recall@k metric and per-qtype aggregation.

``recall_at_k`` is set-based: |expected ∩ retrieved| / |expected|. Empty
``expected`` raises ValueError (caller must filter empty-gold goldens
first); empty ``retrieved`` returns 0.0. The asymmetry is intentional —
empty retrieval is a real outcome, empty expectation is a categorization
mismatch.
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
