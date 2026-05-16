# @summary
# Fusion-scoring layer applied on top of cross-encoder rerank scores.
# Combines three signals: BM25 reciprocal-rank fusion, heading-path lexical
# match, and (tree-mode only) anchor-confidence. Pure functions — no I/O,
# no LLM calls — so the rerank stage stays cheap.
# Exports: compute_bm25_rrf, heading_match_score, anchor_confidence,
#          fuse_scores, RerankFusionWeights, content_tokens
# Deps: stdlib only (re).
# @end-summary
"""Rerank-score fusion helpers.

The cross-encoder reranker produces a single relevance score per candidate.
Three additional signals are fused on top of that score to address known
weaknesses:

1.  **BM25-RRF** — reintroduces lexical signal at rerank time using
    Reciprocal Rank Fusion. The cross-encoder is trained on dense semantic
    matching; pure-lexical hits (rare-token queries, acronyms) can drop in
    its ranking even when they're the right answer.

2.  **Heading-path match** — a chunk whose ``heading_path`` overlaps with
    the query's content tokens is structurally relevant. This compensates
    for cross-encoders that pay attention to body text but not to the
    surrounding heading hierarchy.

3.  **Anchor confidence** (tree-retrieval only) — leaves brought in via
    section-descent or sibling-lift carry an ``_anchor_rank`` indicating
    how confidently the structural step found their parent section. We
    boost leaves whose anchor was a high-confidence hit, decaying with the
    rank using ``1 / (anchor_k + anchor_rank)``.

All four entry points (``compute_bm25_rrf``, ``heading_match_score``,
``anchor_confidence``, ``fuse_scores``) are deterministic pure functions
and side-effect-free — easy to unit-test, easy to compose.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence


# ─── Stopwords ─────────────────────────────────────────────────────────────


# Small built-in stopword set — keeps heading-match deterministic and avoids
# pulling NLTK at runtime. Covers the common English function-word vocabulary
# encountered in technical-doc queries.
_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "could",
    "did", "do", "does", "for", "from", "had", "has", "have", "how", "i",
    "if", "in", "into", "is", "it", "its", "may", "me", "much", "must",
    "my", "no", "not", "of", "on", "or", "should", "so", "than", "that",
    "the", "their", "there", "these", "they", "this", "those", "to", "was",
    "we", "were", "what", "when", "where", "which", "who", "why", "will",
    "with", "would", "you",
})


_TOKEN = re.compile(r"\w+")


def content_tokens(text: str) -> list[str]:
    """Lowercase tokenize and strip stopwords + 1-char tokens.

    Used by ``heading_match_score`` so callers can introspect the same set
    of "content" tokens (or override matching behavior) if needed. Returns
    a deduplicated list preserving first-occurrence order.
    """
    seen: set[str] = set()
    out: list[str] = []
    for raw in _TOKEN.findall(text or ""):
        tok = raw.lower()
        if len(tok) <= 1:
            continue
        if tok in _STOPWORDS:
            continue
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return out


# ─── Reciprocal Rank Fusion ────────────────────────────────────────────────


def compute_bm25_rrf(
    ce_ranks: Sequence[int],
    bm25_ranks: Sequence[Optional[int]],
    k_rrf: int = 60,
) -> list[float]:
    """Compute RRF scores from two parallel rank lists.

    The classic RRF formula is::

        score = sum_over_lists( 1 / (k + rank_in_list) )

    For each candidate we have a CE rank (always present — every candidate
    was scored by the cross-encoder) and an optional BM25 rank. When the
    BM25 rank is ``None`` (descent/lift candidates that never appeared in
    the BM25-only Stage-2 pool), we substitute the pool size so it
    contributes the smallest possible amount — they can still rank well on
    CE alone.

    Args:
        ce_ranks: Cross-encoder rank per candidate, 0-indexed.
        bm25_ranks: BM25-only rank per candidate, 0-indexed; ``None`` for
            candidates absent from the BM25 list.
        k_rrf: RRF dampening constant. 60 is the canonical default.

    Returns:
        A list of float RRF scores in the same order as the inputs.
    """
    if len(ce_ranks) != len(bm25_ranks):
        raise ValueError(
            f"compute_bm25_rrf: ce_ranks ({len(ce_ranks)}) and "
            f"bm25_ranks ({len(bm25_ranks)}) length mismatch"
        )
    pool_size = len(ce_ranks)
    out: list[float] = []
    for ce_r, bm_r in zip(ce_ranks, bm25_ranks):
        bm_resolved = pool_size if bm_r is None else bm_r
        out.append(1.0 / (k_rrf + ce_r) + 1.0 / (k_rrf + bm_resolved))
    return out


# ─── Heading-path match ────────────────────────────────────────────────────


def heading_match_score(query: str, heading_path: Optional[Iterable[str]]) -> float:
    """Fraction of query content tokens that appear inside any heading segment.

    Tokens are matched as case-insensitive substrings (so "interrupt"
    matches "interrupts", and "uart" matches "UART_TX"). Stopwords are
    filtered out of the query side to avoid trivial matches on words like
    "the" or "how".

    Args:
        query: The user query.
        heading_path: Iterable of heading segments (typically a list of
            strings, deepest last). May be ``None`` or empty — returns 0.

    Returns:
        Float in ``[0.0, 1.0]``.
    """
    if not heading_path:
        return 0.0
    qtoks = content_tokens(query)
    if not qtoks:
        return 0.0
    haystack = " ".join(str(seg) for seg in heading_path).lower()
    if not haystack:
        return 0.0
    hits = sum(1 for t in qtoks if t in haystack)
    return hits / len(qtoks)


# ─── Anchor confidence ─────────────────────────────────────────────────────


def anchor_confidence(anchor_rank: Optional[int], anchor_k: int = 10) -> float:
    """Confidence boost for a tree-retrieved leaf, decaying with anchor rank.

    Args:
        anchor_rank: 0-indexed rank of the section/parent that supplied
            this leaf. ``None`` for Stage-4 leaves (no anchor).
        anchor_k: Dampening constant (similar role to ``k_rrf``).

    Returns:
        ``1 / (anchor_k + anchor_rank)`` for tree leaves, ``0.0`` otherwise.
    """
    if anchor_rank is None:
        return 0.0
    return 1.0 / (anchor_k + anchor_rank)


# ─── Combined fusion ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class RerankFusionWeights:
    """Convenience container for fusion weights.

    Mirrors the keys ``fuse_scores`` accepts in its ``weights`` mapping.
    Provided as a typed contract so callers can pass either a dict or a
    structured value. ``rrf=0`` disables BM25-RRF, ``heading=0`` disables
    heading-match, ``anchor=0`` disables anchor confidence.
    """

    rrf: float = 1.0
    heading: float = 0.15
    anchor: float = 0.10


def _weight(weights: Mapping[str, float] | RerankFusionWeights, key: str) -> float:
    if isinstance(weights, RerankFusionWeights):
        return float(getattr(weights, key))
    return float(weights.get(key, 0.0))


def fuse_scores(
    *,
    ce_scores: Sequence[float],
    bm25_ranks: Sequence[Optional[int]],
    heading_scores: Sequence[float],
    anchor_confs: Sequence[float],
    weights: Mapping[str, float] | RerankFusionWeights,
    k_rrf: int = 60,
) -> list[float]:
    """Combine cross-encoder + RRF + heading + anchor signals into one score.

    The output is in input order — callers typically zip the result back
    onto the candidate list and re-sort. CE ranks are derived from
    ``ce_scores`` (descending; stable for ties).

    Args:
        ce_scores: Cross-encoder scores per candidate (sigmoid-bounded
            for the local BGE reranker, ``[0, 1]`` typically).
        bm25_ranks: 0-indexed BM25-only ranks per candidate; ``None`` when
            the candidate was not in the BM25-only list.
        heading_scores: Heading-match score per candidate, in ``[0, 1]``.
        anchor_confs: Anchor-confidence per candidate, in ``[0, 1]`` (or
            0 for non-tree candidates).
        weights: Mapping with keys ``"rrf"``, ``"heading"``, ``"anchor"``,
            or a ``RerankFusionWeights`` instance.
        k_rrf: Dampening constant for the BM25-RRF component.

    Returns:
        Fused scores, same length and order as the inputs.

    Raises:
        ValueError: if the parallel lists disagree in length.
    """
    n = len(ce_scores)
    if not (len(bm25_ranks) == len(heading_scores) == len(anchor_confs) == n):
        raise ValueError(
            f"fuse_scores: input length mismatch ce={n} bm25={len(bm25_ranks)} "
            f"heading={len(heading_scores)} anchor={len(anchor_confs)}"
        )

    w_rrf = _weight(weights, "rrf")
    w_head = _weight(weights, "heading")
    w_anchor = _weight(weights, "anchor")

    # CE ranks: dense ranking of ce_scores descending. Stable on ties so
    # that the position in the input list is the tiebreaker (matches the
    # behavior callers see when they zip+sort).
    indexed = sorted(range(n), key=lambda i: (-ce_scores[i], i))
    ce_ranks: list[int] = [0] * n
    for rank, idx in enumerate(indexed):
        ce_ranks[idx] = rank

    rrf = (
        compute_bm25_rrf(ce_ranks, bm25_ranks, k_rrf=k_rrf)
        if w_rrf != 0.0
        else [0.0] * n
    )

    out: list[float] = []
    for i in range(n):
        out.append(
            float(ce_scores[i])
            + w_rrf * rrf[i]
            + w_head * float(heading_scores[i])
            + w_anchor * float(anchor_confs[i])
        )
    return out


__all__ = [
    "RerankFusionWeights",
    "anchor_confidence",
    "compute_bm25_rrf",
    "content_tokens",
    "fuse_scores",
    "heading_match_score",
]
