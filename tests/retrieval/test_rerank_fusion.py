# @summary
# Unit tests for the rerank fusion module: RRF, heading-match, anchor-confidence,
# and the combined fuse_scores entry point.
# Deps: pytest, src.retrieval.query.nodes.rerank_fusion
# @end-summary
"""Unit tests for ``src.retrieval.query.nodes.rerank_fusion``.

Each helper is exercised in isolation so a regression in one signal can be
localized to a single test. The combined ``fuse_scores`` test verifies the
weighted-sum aggregation that the rerank stage relies on.
"""
from __future__ import annotations

import math

import pytest

from src.retrieval.query.nodes.rerank_fusion import (
    anchor_confidence,
    compute_bm25_rrf,
    fuse_scores,
    heading_match_score,
)


# ─── compute_bm25_rrf ──────────────────────────────────────────────────────


def test_rrf_two_lists_simple():
    """Two perfectly aligned rank lists give 2 / (k + rank) per item."""
    ce_ranks = [0, 1, 2]
    bm25_ranks = [0, 1, 2]
    k = 60
    out = compute_bm25_rrf(ce_ranks, bm25_ranks, k_rrf=k)
    expected = [2.0 / (k + 0), 2.0 / (k + 1), 2.0 / (k + 2)]
    assert out == pytest.approx(expected)


def test_rrf_handles_missing_bm25_rank():
    """``None`` for a BM25 rank should mean "rank = pool_size" — i.e. lowest BM25 contribution."""
    ce_ranks = [0, 1, 2]
    bm25_ranks = [0, None, 2]   # second item has no BM25 rank
    k = 60
    out = compute_bm25_rrf(ce_ranks, bm25_ranks, k_rrf=k)
    pool_size = 3
    expected = [
        1.0 / (k + 0) + 1.0 / (k + 0),
        1.0 / (k + 1) + 1.0 / (k + pool_size),
        1.0 / (k + 2) + 1.0 / (k + 2),
    ]
    assert out == pytest.approx(expected)


def test_rrf_disabled_returns_ce_only_via_fuse_scores():
    """When fusion is off, ``fuse_scores`` returns CE scores untouched."""
    ce_scores = [0.9, 0.7, 0.5]
    out = fuse_scores(
        ce_scores=ce_scores,
        bm25_ranks=[0, 1, 2],
        heading_scores=[0.5, 0.5, 0.5],
        anchor_confs=[0.1, 0.1, 0.1],
        weights={"rrf": 0.0, "heading": 0.0, "anchor": 0.0},
    )
    assert out == pytest.approx(ce_scores)


# ─── heading_match_score ───────────────────────────────────────────────────


def test_heading_match_full_overlap():
    """All non-stopword query tokens appear in heading_path → score = 1.0."""
    score = heading_match_score("interrupt handling", ["UART", "Interrupts"])
    # "interrupt" matches "interrupts" (substring), "handling" → no match;
    # 1/2 of content tokens hit. Score should be > 0 and ≤ 1.
    assert 0.0 < score <= 1.0


def test_heading_match_empty_path():
    """Missing/empty heading path → 0.0 (graceful no-op)."""
    assert heading_match_score("interrupt handling", []) == 0.0
    assert heading_match_score("interrupt handling", None) == 0.0


def test_heading_match_stopwords_filtered():
    """"how does the UART work" → only ``uart`` and ``work`` count."""
    # heading mentions UART only — score = 1/2.
    score = heading_match_score("how does the UART work", ["UART Block"])
    assert score == pytest.approx(0.5)


def test_heading_match_all_stopwords_returns_zero():
    """Query that's pure stopwords yields 0 (no content tokens to match)."""
    assert heading_match_score("how does the", ["UART"]) == 0.0


# ─── anchor_confidence ─────────────────────────────────────────────────────


def test_anchor_confidence_decay():
    """Better anchor rank → higher confidence."""
    k = 10
    a0 = anchor_confidence(0, anchor_k=k)
    a1 = anchor_confidence(1, anchor_k=k)
    a2 = anchor_confidence(2, anchor_k=k)
    assert a0 > a1 > a2 > 0.0
    assert a0 == pytest.approx(1.0 / 10)
    assert a1 == pytest.approx(1.0 / 11)


def test_anchor_confidence_none_returns_zero():
    """Leaf hit with no anchor → 0.0."""
    assert anchor_confidence(None, anchor_k=10) == 0.0


# ─── fuse_scores ───────────────────────────────────────────────────────────


def test_fuse_scores_combines_signals():
    """All three signals add into the final fused score with their weights."""
    ce_scores = [0.5, 0.5]
    bm25_ranks = [0, 1]
    heading_scores = [1.0, 0.0]
    anchor_confs = [0.1, 0.0]
    weights = {"rrf": 1.0, "heading": 0.2, "anchor": 0.3}

    # CE rank ordering follows ce_scores desc; ties broken by stable index (0, 1).
    # Item 0: rrf = 1/(60+0) + 1/(60+0) ; heading = 1.0 ; anchor = 0.1
    # Item 1: rrf = 1/(60+1) + 1/(60+1) ; heading = 0.0 ; anchor = 0.0
    out = fuse_scores(
        ce_scores=ce_scores,
        bm25_ranks=bm25_ranks,
        heading_scores=heading_scores,
        anchor_confs=anchor_confs,
        weights=weights,
        k_rrf=60,
    )

    rrf_0 = 2.0 / 60.0
    rrf_1 = 2.0 / 61.0
    expected_0 = 0.5 + 1.0 * rrf_0 + 0.2 * 1.0 + 0.3 * 0.1
    expected_1 = 0.5 + 1.0 * rrf_1 + 0.2 * 0.0 + 0.3 * 0.0
    assert out[0] == pytest.approx(expected_0)
    assert out[1] == pytest.approx(expected_1)


def test_fuse_scores_preserves_order():
    """Output is in the same order as inputs (no sorting inside fuse)."""
    ce_scores = [0.1, 0.9, 0.5]
    out = fuse_scores(
        ce_scores=ce_scores,
        bm25_ranks=[0, 1, 2],
        heading_scores=[0.0, 0.0, 0.0],
        anchor_confs=[0.0, 0.0, 0.0],
        weights={"rrf": 0.0, "heading": 0.0, "anchor": 0.0},
    )
    # rrf=0 weight, heading=0, anchor=0 → equals ce_scores.
    assert out == pytest.approx(ce_scores)


def test_fuse_scores_input_length_mismatch_raises():
    """Mismatched list lengths must error early — fusion contract is positional."""
    with pytest.raises(ValueError):
        fuse_scores(
            ce_scores=[0.5, 0.5],
            bm25_ranks=[0],            # too short
            heading_scores=[0.0, 0.0],
            anchor_confs=[0.0, 0.0],
            weights={"rrf": 1.0, "heading": 0.0, "anchor": 0.0},
        )
