# @summary
# Unit tests for the Stage-1 card-index router (``route_documents``, slice B1).
# Pins the B1 contract from DOCUMENT_ROUTING_DESIGN.md §3 (ROUTE = embed query →
# search the CARD index → top-N doc IDs as a SOFT hint) and §7 (top-N never
# top-1; soft; FALL BACK to pure flat when confidence is low / routed set empty /
# collection missing / any error). The router must NEVER raise — every failure
# path degrades to RoutingResult([], {}, used=False). The vector_db ``search``
# facade is always MOCKED (monkeypatched as imported into router.py) — no live
# vector store, no embeddings.
# Deps: pytest, src.retrieval.routing.router, src.vector_db.common.schemas,
#       config.settings
# @end-summary
"""Unit tests for ``src.retrieval.routing.router.route_documents``.

The router embeds (caller pre-embeds) a query and issues a vector-first search
against the small document-card collection, then returns up to ``top_n`` routed
``document_id`` values as a *soft* hint (``RoutingResult``). It is a pure
read: it filters nothing itself and it never raises.

Mocking strategy mirrors the decomposition tests: ``route_documents`` resolves
the search facade via ``search`` *as imported into the router module*, so these
tests monkeypatch ``src.retrieval.routing.router.search`` with a fake that
returns canned :class:`SearchResult` objects (or raises, to simulate a missing
collection / backend error).
"""
from __future__ import annotations

import pytest

from config import settings
from src.retrieval.routing import router
from src.retrieval.routing.router import route_documents
from src.retrieval.routing.schemas import RoutingResult
from src.vector_db.common.schemas import SearchResult


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _card(document_id: str, score: float) -> SearchResult:
    """Build a canned card SearchResult (document_id lives in metadata)."""
    return SearchResult(
        text=f"card for {document_id}",
        score=score,
        metadata={"document_id": document_id, "title": f"Doc {document_id}"},
        object_id=f"obj-{document_id}",
        collection=settings.RAG_DOCUMENT_CARD_COLLECTION,
    )


def _install_search(monkeypatch, results=None, *, exc: Exception | None = None):
    """Monkeypatch ``search`` (as imported in router) and capture call args."""
    calls: list[dict] = []

    def _fake_search(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        if exc is not None:
            raise exc
        return list(results or [])

    monkeypatch.setattr(router, "search", _fake_search)
    return calls


# A 3-dim stand-in embedding; the router never inspects its contents.
_EMB = [0.1, 0.2, 0.3]


# ─── NORMAL: distinct hits above min_score → used=True, rank order ───────────


def test_normal_three_hits_routes_in_rank_order(monkeypatch):
    _install_search(
        monkeypatch,
        [_card("doc-a", 0.9), _card("doc-b", 0.8), _card("doc-c", 0.7)],
    )
    result = route_documents(_EMB, min_score=0.0)

    assert isinstance(result, RoutingResult)
    assert result.used is True
    assert result.doc_ids == ["doc-a", "doc-b", "doc-c"]
    assert result.scores == {"doc-a": 0.9, "doc-b": 0.8, "doc-c": 0.7}
    assert len(result.doc_ids) <= settings.RAG_DOCUMENT_ROUTING_TOP_N


# ─── TOP-N CAP: more hits than top_n → only top_n doc_ids ────────────────────


def test_top_n_caps_doc_ids(monkeypatch):
    hits = [_card(f"doc-{i}", 0.9 - i * 0.05) for i in range(10)]
    _install_search(monkeypatch, hits)
    result = route_documents(_EMB, top_n=3, min_score=0.0)

    assert result.used is True
    assert result.doc_ids == ["doc-0", "doc-1", "doc-2"]
    assert len(result.doc_ids) == 3


# ─── DEDUPE: two hits same document_id → one doc_id, rank preserved ──────────


def test_dedupe_same_document_id(monkeypatch):
    _install_search(
        monkeypatch,
        [_card("doc-a", 0.9), _card("doc-b", 0.85), _card("doc-a", 0.7)],
    )
    result = route_documents(_EMB, min_score=0.0)

    assert result.used is True
    assert result.doc_ids == ["doc-a", "doc-b"]
    # First (highest-rank) occurrence wins for the score.
    assert result.scores["doc-a"] == 0.9


# ─── LOW CONFIDENCE: top hit below min_score → used=False, empty ─────────────


def test_low_confidence_below_min_score(monkeypatch):
    _install_search(
        monkeypatch,
        [_card("doc-a", 0.1), _card("doc-b", 0.05)],
    )
    result = route_documents(_EMB, min_score=0.5)

    assert result.used is False
    assert result.doc_ids == []
    assert result.scores == {}


# ─── EMPTY RESULT: no hits → used=False, empty ───────────────────────────────


def test_empty_result(monkeypatch):
    _install_search(monkeypatch, [])
    result = route_documents(_EMB)

    assert result.used is False
    assert result.doc_ids == []
    assert result.scores == {}


# ─── SEARCH RAISES: collection missing / backend down → used=False, no raise ─


def test_search_raises_returns_unused(monkeypatch):
    _install_search(monkeypatch, exc=RuntimeError("collection does not exist"))
    # Must NOT propagate the exception.
    result = route_documents(_EMB)

    assert result.used is False
    assert result.doc_ids == []
    assert result.scores == {}


def test_search_raises_value_error_returns_unused(monkeypatch):
    _install_search(monkeypatch, exc=ValueError("backend down"))
    result = route_documents(_EMB)
    assert result.used is False
    assert result.doc_ids == []


# ─── DEFAULTS: omitting args uses settings values for limit + collection ─────


def test_defaults_use_settings(monkeypatch):
    calls = _install_search(monkeypatch, [_card("doc-a", 0.9)])
    route_documents(_EMB)

    assert len(calls) == 1
    kwargs = calls[0]["kwargs"]
    assert kwargs["limit"] == settings.RAG_DOCUMENT_ROUTING_TOP_N
    assert kwargs["collection"] == settings.RAG_DOCUMENT_CARD_COLLECTION
    # Vector-first: pure (or vector-dominant) similarity over the cards.
    assert kwargs["alpha"] == 1.0
    # The pre-computed query embedding is forwarded verbatim.
    assert kwargs["query_embedding"] is _EMB


def test_explicit_args_override_defaults(monkeypatch):
    calls = _install_search(monkeypatch, [_card("doc-a", 0.9)])
    route_documents(
        _EMB, top_n=4, min_score=0.0, collection="CustomCards", query_text="axi"
    )

    kwargs = calls[0]["kwargs"]
    assert kwargs["limit"] == 4
    assert kwargs["collection"] == "CustomCards"
    # query_text is forwarded as the keyword/bm25 query component.
    assert kwargs["query"] == "axi"


# ─── ROBUSTNESS: a hit missing document_id is skipped, not fatal ─────────────


def test_hit_without_document_id_is_skipped(monkeypatch):
    bad = SearchResult(text="no id", score=0.95, metadata={}, object_id="x")
    _install_search(monkeypatch, [bad, _card("doc-b", 0.8)])
    result = route_documents(_EMB, min_score=0.0)

    # The first hit (top score) has no document_id → skipped; doc-b survives.
    assert result.used is True
    assert result.doc_ids == ["doc-b"]
