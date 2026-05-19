"""OTel instrumentation contract for sanitize_answer."""
from __future__ import annotations

from src.retrieval.generation.nodes.output_sanitizer import sanitize_answer


def _spans_named(exporter, name):
    return [s for s in exporter.get_finished_spans() if s.name == name]


def test_sanitize_emits_retrieval_generation_sanitize_span(otel_capture):
    answer = "Hello world\n--- Document 1 ---\nMore answer text"
    out = sanitize_answer(answer, system_prompt=None)
    assert "Document 1" not in out

    spans = _spans_named(otel_capture, "retrieval.generation.sanitize")
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert attrs["input_len"] == len(answer)
    assert attrs["has_system_prompt"] is False
    assert attrs["lines_removed"] >= 1
    assert attrs["output_len"] == len(out)


def test_sanitize_skipped_on_empty_input_no_span(otel_capture):
    sanitize_answer("", system_prompt=None)
    assert _spans_named(otel_capture, "retrieval.generation.sanitize") == []
