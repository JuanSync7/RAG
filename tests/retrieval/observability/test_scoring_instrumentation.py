"""OTel instrumentation contract for confidence scoring."""
from __future__ import annotations

from src.retrieval.generation.confidence.scoring import compute_composite_confidence


def _spans_named(exporter, name):
    return [s for s in exporter.get_finished_spans() if s.name == name]


def test_compute_composite_confidence_emits_span(otel_capture):
    bd = compute_composite_confidence(
        reranker_scores=[0.9, 0.5],
        llm_confidence_text="high",
        answer="something something",
        retrieved_texts=["something something more"],
    )
    spans = _spans_named(otel_capture, "retrieval.confidence.score")
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert attrs["reranker_score_count"] == 2
    assert attrs["retrieved_doc_count"] == 1
    assert attrs["answer_len"] == len("something something")
    assert 0.0 <= attrs["composite"] <= 1.0
    assert abs(attrs["composite"] - bd.composite) < 1e-9
