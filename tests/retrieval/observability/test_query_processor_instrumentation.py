"""OTel instrumentation contract for query_processor.process_query."""
from __future__ import annotations

from unittest.mock import patch, MagicMock

from src.retrieval.query.nodes.query_processor import process_query


def _spans_named(exporter, name):
    return [s for s in exporter.get_finished_spans() if s.name == name]


def test_process_query_emits_retrieval_query_process(otel_capture):
    """Heuristic-only path (LLM unavailable) still opens the root span."""
    fake_state = {
        "current_query": "hello",
        "confidence": 1.0,
        "iteration": 0,
        "action": "",
        "clarification_message": "",
        "standalone_query": "hello",
    }
    fake_graph = MagicMock()
    fake_graph.invoke.return_value = fake_state
    with patch(
        "src.retrieval.query.nodes.query_processor._check_llm_available",
        return_value=False,
    ), patch(
        "src.retrieval.query.nodes.query_processor._get_compiled_graph",
        return_value=fake_graph,
    ):
        result = process_query("hello", mode="query")
    assert result.processed_query == "hello"

    spans = _spans_named(otel_capture, "retrieval.query.process")
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert attrs["raw_query_len"] == len("hello")
    assert "action" in attrs
    assert "confidence" in attrs


def test_process_query_retrieval_mode_short_circuits_no_span(otel_capture):
    """The retrieval-mode short-circuit must not open the root span."""
    result = process_query("hello", mode="retrieval", retrieval_sub_mode="hard")
    assert result.processed_query == "hello"
    assert _spans_named(otel_capture, "retrieval.query.process") == []
