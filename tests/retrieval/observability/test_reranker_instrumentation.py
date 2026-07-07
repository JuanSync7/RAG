"""OTel instrumentation contract for reranker callsites.

Local BGE reranker test is skipped (heavy torch deps); we test TEI by mocking
its httpx client.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from src.retrieval.query.nodes.reranker import TEIReranker
from src.vector_db.common.schemas import SearchResult


def _spans_named(exporter, name):
    return [s for s in exporter.get_finished_spans() if s.name == name]


def _mk_doc(i: int) -> SearchResult:
    return SearchResult(text=f"doc {i}", score=0.5, metadata={"i": i})


def test_tei_rerank_emits_retrieval_rerank_tei_span(otel_capture):
    rr = TEIReranker(base_url="http://tei.test", model="rerank", timeout=5)
    fake_resp = MagicMock()
    fake_resp.raise_for_status.return_value = None
    fake_resp.json.return_value = [
        {"index": 0, "score": 0.9},
        {"index": 1, "score": 0.4},
    ]
    rr._client = MagicMock()
    rr._client.post.return_value = fake_resp

    docs = [_mk_doc(0), _mk_doc(1)]
    out = rr.rerank("q", docs, top_k=2)
    assert len(out) == 2
    assert out[0].score == 0.9

    spans = _spans_named(otel_capture, "retrieval.rerank.tei")
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert attrs["candidate_count"] == 2
    assert attrs["top_k"] == 2
    assert attrs["endpoint"] == "http://tei.test"
    assert attrs["output_count"] == 2
    assert attrs["score_max"] == 0.9


def test_tei_rerank_empty_docs_emits_no_span(otel_capture):
    rr = TEIReranker(base_url="http://tei.test", model="rerank", timeout=5)
    rr._client = MagicMock()  # should not be called
    out = rr.rerank("q", [], top_k=5)
    assert out == []
    assert _spans_named(otel_capture, "retrieval.rerank.tei") == []
