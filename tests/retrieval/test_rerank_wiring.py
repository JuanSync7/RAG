# @summary
# Wiring tests: confirm RAGChain integrates the rerank-fusion module
# correctly. Stubs the reranker + BM25 search at the boundary; asserts on
# fused score ordering and bit-identical legacy behavior when fusion is off.
# Deps: pytest, src.retrieval.pipeline.rag_chain.RAGChain
# @end-summary
"""Wiring tests for ``RAGChain``'s rerank-fusion integration.

These tests exercise ``_apply_rerank_fusion`` and ``_collect_bm25_ranks``
on a bare ``RAGChain`` — no Weaviate, no real reranker. Fast, deterministic.
"""
from __future__ import annotations

import types
from collections import OrderedDict
from typing import Any

import pytest

from src.retrieval.common import RankedResult
from src.retrieval.pipeline.rag_chain import RAGChain


class _DummySpan:
    def set_attribute(self, *_a, **_kw): return None
    def end(self, *_a, **_kw): return None
    def __enter__(self): return self
    def __exit__(self, *_): return False


class _DummyTracer:
    def span(self, *_a, **_kw): return _DummySpan()
    def start_span(self, *_a, **_kw): return _DummySpan()


def _make_chain() -> RAGChain:
    chain = object.__new__(RAGChain)
    chain.tracer = _DummyTracer()
    chain.retry_provider = None
    chain.retry_policy = object()
    chain._persistent_weaviate = False
    chain._weaviate_client = None
    chain._embedding_cache = OrderedDict()
    chain._embedding_cache_max = 128
    return chain


def _hit(text: str, *, chunk_id: str, heading_path=None, anchor_rank=None,
         score: float = 0.5) -> Any:
    meta: dict = {
        "chunk_id": chunk_id,
        "node_kind": "chunk",
        "document_id": "doc-A",
        "heading_path": heading_path or [],
    }
    if anchor_rank is not None:
        meta["_anchor_rank"] = anchor_rank
    return types.SimpleNamespace(
        text=text,
        score=score,
        metadata=meta,
        collection="default",
        uuid=chunk_id,
    )


def _ranked(item, score: float) -> RankedResult:
    return RankedResult(text=item.text, score=score, metadata=item.metadata)


# ─── Fusion off ⇒ legacy CE behavior ───────────────────────────────────────


def test_fusion_disabled_matches_pure_ce(monkeypatch):
    """When ``RAG_RERANK_FUSION_ENABLED`` is False, ``_apply_rerank_fusion``
    is a passthrough — bit-identical to today's pure-CE behaviour."""
    import src.retrieval.pipeline.rag_chain as rc

    monkeypatch.setattr(rc, "RAG_RERANK_FUSION_ENABLED", False)

    chain = _make_chain()
    cands = [_hit(f"text-{i}", chunk_id=f"c{i}") for i in range(3)]
    reranked = [
        _ranked(cands[0], 0.9),
        _ranked(cands[1], 0.5),
        _ranked(cands[2], 0.1),
    ]
    out = chain._apply_rerank_fusion(
        query="anything",
        candidates=cands,
        reranked=reranked,
        bm25_ranks=[0, 1, 2],
        rerank_top_k=10,
    )
    assert out is reranked  # exact same object → no copy/resort
    assert [r.score for r in out] == [0.9, 0.5, 0.1]


# ─── Fusion on ⇒ ordering changes when BM25 outranks CE ────────────────────


def test_fusion_changes_top_k_ordering(monkeypatch):
    """A gold candidate with weak CE but strong BM25 + heading match should
    move into the top-K when fusion is enabled."""
    import src.retrieval.pipeline.rag_chain as rc

    monkeypatch.setattr(rc, "RAG_RERANK_FUSION_ENABLED", True)
    monkeypatch.setattr(rc, "RAG_RERANK_RRF_LAMBDA", 5.0)
    monkeypatch.setattr(rc, "RAG_RERANK_HEADING_LAMBDA", 1.0)
    monkeypatch.setattr(rc, "RAG_RERANK_ANCHOR_LAMBDA", 0.0)
    monkeypatch.setattr(rc, "RAG_RERANK_RRF_K", 10)
    monkeypatch.setattr(rc, "RAG_RERANK_ANCHOR_K", 10)

    chain = _make_chain()
    # Two candidates. "ce_winner" — high CE, no BM25 hit, no heading match.
    # "bm25_winner" — low CE but #1 BM25 + heading-overlap with the query.
    ce_winner = _hit("noise", chunk_id="ce", heading_path=["Misc"])
    bm25_winner = _hit("interrupt source", chunk_id="bm25",
                       heading_path=["UART", "Interrupts"])

    reranked = [
        _ranked(ce_winner, 0.80),
        _ranked(bm25_winner, 0.50),
    ]
    cands = [bm25_winner, ce_winner]   # Stage-4 BM25 saw bm25_winner first
    bm25_ranks = [0, None]             # bm25_winner=rank0, ce_winner=absent

    out = chain._apply_rerank_fusion(
        query="interrupt handling",
        candidates=cands,
        reranked=reranked,
        bm25_ranks=bm25_ranks,
        rerank_top_k=2,
    )
    # bm25_winner should have moved to rank 0.
    assert out[0].metadata["chunk_id"] == "bm25"
    assert out[1].metadata["chunk_id"] == "ce"


# ─── Anchor metadata threaded ─────────────────────────────────────────────


def test_anchor_confidence_boost_lifts_tree_leaf(monkeypatch):
    """An anchored leaf (with ``_anchor_rank`` metadata) should beat a
    non-anchored peer when CE scores are tied and anchor lambda is high."""
    import src.retrieval.pipeline.rag_chain as rc

    monkeypatch.setattr(rc, "RAG_RERANK_FUSION_ENABLED", True)
    monkeypatch.setattr(rc, "RAG_RERANK_RRF_LAMBDA", 0.0)
    monkeypatch.setattr(rc, "RAG_RERANK_HEADING_LAMBDA", 0.0)
    monkeypatch.setattr(rc, "RAG_RERANK_ANCHOR_LAMBDA", 1.0)
    monkeypatch.setattr(rc, "RAG_RERANK_ANCHOR_K", 10)

    chain = _make_chain()
    anchored = _hit("a", chunk_id="anchored", anchor_rank=0)
    plain = _hit("b", chunk_id="plain", anchor_rank=None)

    reranked = [
        _ranked(plain, 0.6),
        _ranked(anchored, 0.6),
    ]
    out = chain._apply_rerank_fusion(
        query="q",
        candidates=[plain, anchored],
        reranked=reranked,
        bm25_ranks=[None, None],
        rerank_top_k=2,
    )
    assert out[0].metadata["chunk_id"] == "anchored"


# ─── BM25-rank collection ──────────────────────────────────────────────────


def test_collect_bm25_ranks_returns_aligned_list():
    """Aligns BM25 0-indexed rank to candidates by chunk_id; ``None`` when
    a candidate is absent from the BM25 result set."""
    chain = _make_chain()

    bm25_pool = [
        _hit("first", chunk_id="A"),
        _hit("second", chunk_id="B"),
        _hit("third", chunk_id="C"),
    ]

    def fake_search(bm25_query, query_emb, alpha, search_limit, filters):
        assert alpha == 0.0  # BM25-only
        return bm25_pool

    chain._do_search = fake_search

    candidates = [
        _hit("first", chunk_id="A"),
        _hit("missing", chunk_id="Z"),
        _hit("third", chunk_id="C"),
    ]
    out = chain._collect_bm25_ranks(
        bm25_query="q",
        query_embedding=None,
        search_limit=10,
        base_filters=[],
        candidates=candidates,
    )
    assert out == [0, None, 2]


def test_collect_bm25_ranks_failure_returns_all_none():
    """If the BM25 search fails, fallback is all-None (graceful degrade)."""
    chain = _make_chain()

    def boom(*_a, **_kw):
        raise RuntimeError("weaviate offline")

    chain._do_search = boom

    cands = [_hit("x", chunk_id="X")]
    out = chain._collect_bm25_ranks(
        bm25_query="q",
        query_embedding=None,
        search_limit=10,
        base_filters=[],
        candidates=cands,
    )
    assert out == [None]


# ─── R1.3 anchor metadata threading ────────────────────────────────────────


def test_anchor_metadata_threaded_through_descent():
    """Descent leaves should carry an ``_anchor_rank`` matching the section
    they were expanded from (0-indexed)."""
    chain = _make_chain()

    section_hits = [
        types.SimpleNamespace(
            text="sec0", score=0.9,
            metadata={"node_kind": "section", "chunk_id": "p0",
                      "document_id": "doc-A", "heading_path": ["A"]},
            collection="default",
        ),
        types.SimpleNamespace(
            text="sec1", score=0.7,
            metadata={"node_kind": "section", "chunk_id": "p1",
                      "document_id": "doc-A", "heading_path": ["B"]},
            collection="default",
        ),
    ]
    l0 = _hit("leaf0", chunk_id="L0")
    l0.metadata["parent_section_id"] = "p0"
    l1 = _hit("leaf1", chunk_id="L1")
    l1.metadata["parent_section_id"] = "p1"
    leaves_by_parent = {"p0": [l0], "p1": [l1]}

    def fake_search(bm25_query, qe, alpha, limit, filters):
        if any(getattr(f, "value", None) == "section" for f in (filters or [])):
            return section_hits
        return []

    def fake_expand(parent_ids, per_section_limit, base_filters):
        out = []
        for pid in parent_ids:
            for c in leaves_by_parent.get(pid, []):
                out.append(c)
        return out

    chain._do_search = fake_search
    chain._expand_sections_to_leaves = fake_expand
    chain._fetch_siblings = fake_expand

    leaves = chain._run_tree_descent(
        bm25_query="q",
        query_embedding=None,
        alpha=0.5,
        descent_top_k=2,
        leaves_per_section=1,
        base_filters=[],
    )
    metas = [l.metadata for l in leaves]
    assert metas[0].get("_anchor_rank") == 0
    assert metas[1].get("_anchor_rank") == 1


def test_anchor_metadata_threaded_through_lift():
    """Lift siblings should carry the anchor rank of the seed that supplied
    their parent section."""
    chain = _make_chain()

    seeds = [
        _hit("seedA", chunk_id="A0"),
        _hit("seedB", chunk_id="B0"),
    ]
    seeds[0].metadata["parent_section_id"] = "pA"
    seeds[1].metadata["parent_section_id"] = "pB"

    def fake_expand(parent_ids, per_group_limit, base_filters):
        out = []
        for pid in parent_ids:
            sib = _hit(f"sib-{pid}", chunk_id=f"sib-{pid}")
            sib.metadata["parent_section_id"] = pid
            out.append(sib)
        return out

    chain._expand_sections_to_leaves = fake_expand
    chain._fetch_siblings = fake_expand

    siblings = chain._run_tree_lift(
        seed_results=seeds,
        seed_top_k=2,
        siblings_per_group=1,
        base_filters=[],
    )
    metas = [s.metadata for s in siblings]
    assert metas[0].get("_anchor_rank") == 0   # supplied by seeds[0]
    assert metas[1].get("_anchor_rank") == 1   # supplied by seeds[1]


