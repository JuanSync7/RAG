"""Retrieval quality metrics — deterministic, dependency-free.

recall@k:  fraction of gold items present in the top-k retrieved IDs.
ndcg@k:    binary-relevance nDCG over the top-k cut.
mean_metric: simple arithmetic mean over per-query scores (returns 0.0 on empty).
"""
# @summary
# Deterministic retrieval-quality metrics: recall@k, nDCG@k, mean over queries.
# Exports: recall_at_k, ndcg_at_k, mean_metric
# Deps: math (stdlib)
# @end-summary
from __future__ import annotations

import math
from typing import Iterable, Sequence


def recall_at_k(retrieved: Sequence[str], gold: Iterable[str], k: int) -> float:
    """Fraction of |gold| present in the first ``k`` retrieved IDs.

    Returns 0.0 when gold is empty (avoids ZeroDivision; equivalent to "no signal").
    """
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    top = list(retrieved)[: max(0, k)]
    hits = sum(1 for item in top if item in gold_set)
    return hits / len(gold_set)


def ndcg_at_k(retrieved: Sequence[str], gold: Iterable[str], k: int) -> float:
    """Binary-relevance nDCG@k.

    Uses the standard IR formulation: gain = 1 if id ∈ gold else 0;
    discount = 1 / log2(rank + 1) where rank is 1-indexed.
    IDCG is computed assuming as many relevant items as possible (capped at k)
    are placed at the top.
    """
    gold_set = set(gold)
    if not gold_set or k <= 0:
        return 0.0
    top = list(retrieved)[:k]
    dcg = 0.0
    for idx, item in enumerate(top, start=1):
        if item in gold_set:
            dcg += 1.0 / math.log2(idx + 1)
    ideal_hits = min(len(gold_set), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def mean_metric(per_query_scores: Sequence[float]) -> float:
    """Arithmetic mean of per-query scores; 0.0 on empty input."""
    if not per_query_scores:
        return 0.0
    return sum(per_query_scores) / len(per_query_scores)
