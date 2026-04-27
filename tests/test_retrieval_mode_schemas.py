"""Schema-level checks for the mode=retrieval API surface (S3).

These are deliberately narrow: they confirm the new QueryRequest /
QueryResponse fields exist with the expected types and defaults, and that
the activity-layer ``mode == "retrieval"`` override forces skip_generation.
End-to-end behaviour (workflow → RAGChain → response) is exercised in
integration tests once the worker stack is available.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from server.schemas import QueryRequest, QueryResponse


def test_query_request_defaults_to_query_mode():
    req = QueryRequest(query="hello")
    assert req.mode == "query"
    assert req.retrieval_sub_mode == "auto"
    assert req.extra_processing is False


def test_query_request_accepts_retrieval_mode():
    req = QueryRequest(query="docs about photosynthesis", mode="retrieval")
    assert req.mode == "retrieval"


def test_query_request_accepts_hard_sub_mode():
    req = QueryRequest(
        query="literal text",
        mode="retrieval",
        retrieval_sub_mode="hard",
    )
    assert req.retrieval_sub_mode == "hard"


def test_query_request_rejects_unknown_mode():
    with pytest.raises(ValidationError):
        QueryRequest(query="x", mode="bogus")


def test_query_request_rejects_unknown_sub_mode():
    with pytest.raises(ValidationError):
        QueryRequest(query="x", mode="retrieval", retrieval_sub_mode="extreme")


def test_query_response_history_fields_default_sane():
    resp = QueryResponse(
        query="x", processed_query="x", query_confidence=0.5, action="search",
    )
    assert resp.history_decision is None
    assert resp.history_turns_used == 0
    assert resp.relevant_doc_ids == []
    assert resp.ignored_doc_ids == []
    assert resp.seen_doc_ids == []


def test_query_response_round_trips_history_fields():
    resp = QueryResponse(
        query="x",
        processed_query="x",
        query_confidence=0.5,
        action="search",
        history_decision="partial_history",
        history_turns_used=3,
        relevant_doc_ids=["d1"],
        ignored_doc_ids=["d2"],
        seen_doc_ids=["d1", "d2"],
    )
    assert resp.history_decision == "partial_history"
    assert resp.history_turns_used == 3


def test_activity_forces_skip_generation_in_retrieval_mode(monkeypatch):
    """The execute_rag_query activity must override skip_generation=False
    when mode=retrieval, regardless of caller intent."""
    from server import activities

    captured = {}

    class _FakeRAGChain:
        def run(self, **kwargs):
            captured.update(kwargs)
            from src.retrieval.common.schemas import RAGResponse
            return RAGResponse(
                query=kwargs["query"],
                processed_query=kwargs["query"],
                query_confidence=1.0,
                action="search",
                results=[],
            )

    monkeypatch.setattr(activities, "get_rag_chain", lambda: _FakeRAGChain())
    # Bypass the activity-result cache so the run actually fires.
    monkeypatch.setattr(activities._cache, "get", lambda *a, **kw: None)
    monkeypatch.setattr(activities._cache, "set", lambda *a, **kw: None)
    # Stub Temporal activity context dependencies for unit-test invocation.
    import logging as _logging
    monkeypatch.setattr(
        activities.activity, "logger", _logging.getLogger("test"), raising=False,
    )

    activities.execute_rag_query({
        "query": "find me docs",
        "mode": "retrieval",
        "retrieval_sub_mode": "hard",
        "skip_generation": False,  # caller says don't skip — activity must override.
    })

    assert captured["skip_generation"] is True
    assert captured["mode"] == "retrieval"
    assert captured["retrieval_sub_mode"] == "hard"


def test_activity_preserves_skip_generation_for_query_mode(monkeypatch):
    """In query mode the activity must not flip skip_generation behind the caller."""
    from server import activities

    captured = {}

    class _FakeRAGChain:
        def run(self, **kwargs):
            captured.update(kwargs)
            from src.retrieval.common.schemas import RAGResponse
            return RAGResponse(
                query=kwargs["query"],
                processed_query=kwargs["query"],
                query_confidence=1.0,
                action="search",
                results=[],
            )

    monkeypatch.setattr(activities, "get_rag_chain", lambda: _FakeRAGChain())
    monkeypatch.setattr(activities._cache, "get", lambda *a, **kw: None)
    monkeypatch.setattr(activities._cache, "set", lambda *a, **kw: None)
    import logging as _logging
    monkeypatch.setattr(
        activities.activity, "logger", _logging.getLogger("test"), raising=False,
    )

    activities.execute_rag_query({
        "query": "explain RAG",
        "mode": "query",
        "skip_generation": False,
    })

    assert captured["skip_generation"] is False
    assert captured["mode"] == "query"
