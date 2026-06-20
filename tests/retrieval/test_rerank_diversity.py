# @summary
# Unit tests for RAGChain._apply_doc_diversity — the per-document diversity cap
# that stops one document (e.g. a spreadsheet split into many near-duplicate row
# chunks) from monopolising the reranked top-K.
# Deps: pytest, src.retrieval.pipeline.rag_chain, src.retrieval.common.schemas
# @end-summary
"""Unit tests for the rerank-time per-document diversity cap."""
from __future__ import annotations

from collections import Counter

import src.retrieval.pipeline.rag_chain as rc
from src.retrieval.common.schemas import RankedResult
from src.retrieval.pipeline.rag_chain import RAGChain


def _r(doc: str, score: float) -> RankedResult:
    return RankedResult(text=f"{doc}:{score}", score=score, metadata={"document_id": doc})


def _docs(out):
    return Counter(r.metadata["document_id"] for r in out)


def test_cap_breaks_single_document_monopoly(monkeypatch):
    """A dominant document is capped at ceil(top_k*fraction); freed slots go to others."""
    monkeypatch.setattr(rc, "RAG_RERANK_DIVERSITY_ENABLED", True)
    monkeypatch.setattr(rc, "RAG_RERANK_MAX_DOC_FRACTION", 0.6)
    # docA wins every raw slot (10 high scores); B/C only appear lower down.
    ranked = [_r("A", 1.0 - i * 0.01) for i in range(10)]
    ranked += [_r("B", 0.5), _r("C", 0.49), _r("B", 0.48)]

    out = RAGChain._apply_doc_diversity(ranked, top_k=10)
    counts = _docs(out)
    assert len(out) == 10
    # ceil(10 * 0.6) == 6 capped; 1 backfilled (B=2, C=1 fill only 3 of 4 freed).
    assert counts["A"] == 7
    assert counts["B"] == 2 and counts["C"] == 1
    # diversity achieved: pure top-10-by-score would have been all "A".
    assert set(counts) == {"A", "B", "C"}
    # result stays score-sorted
    assert [r.score for r in out] == sorted((r.score for r in out), reverse=True)


def test_backfill_preserves_count_when_no_diversity(monkeypatch):
    """A genuinely single-source pool is never starved: backfill restores top_k."""
    monkeypatch.setattr(rc, "RAG_RERANK_DIVERSITY_ENABLED", True)
    monkeypatch.setattr(rc, "RAG_RERANK_MAX_DOC_FRACTION", 0.6)
    ranked = [_r("A", 1.0 - i * 0.01) for i in range(15)]
    out = RAGChain._apply_doc_diversity(ranked, top_k=10)
    assert len(out) == 10
    assert _docs(out)["A"] == 10  # no other doc to diversify to


def test_disabled_is_passthrough(monkeypatch):
    monkeypatch.setattr(rc, "RAG_RERANK_DIVERSITY_ENABLED", False)
    monkeypatch.setattr(rc, "RAG_RERANK_MAX_DOC_FRACTION", 0.6)
    ranked = [_r("A", 1.0 - i * 0.01) for i in range(10)] + [_r("B", 0.4)]
    out = RAGChain._apply_doc_diversity(ranked, top_k=10)
    assert _docs(out)["A"] == 10  # cap not applied → monopoly preserved


def test_fraction_one_is_passthrough(monkeypatch):
    monkeypatch.setattr(rc, "RAG_RERANK_DIVERSITY_ENABLED", True)
    monkeypatch.setattr(rc, "RAG_RERANK_MAX_DOC_FRACTION", 1.0)
    ranked = [_r("A", 1.0 - i * 0.01) for i in range(10)] + [_r("B", 0.4)]
    out = RAGChain._apply_doc_diversity(ranked, top_k=10)
    assert _docs(out)["A"] == 10


def test_small_pool_is_noop(monkeypatch):
    monkeypatch.setattr(rc, "RAG_RERANK_DIVERSITY_ENABLED", True)
    monkeypatch.setattr(rc, "RAG_RERANK_MAX_DOC_FRACTION", 0.6)
    ranked = [_r("A", 0.9), _r("A", 0.8)]
    out = RAGChain._apply_doc_diversity(ranked, top_k=10)
    assert len(out) == 2  # pool already <= top_k → returned as-is
