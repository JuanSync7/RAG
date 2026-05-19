"""Rail-internal span nesting under the guardrails parent span.

Regression test for the ThreadPoolExecutor context-loss bug: prior to
``submit_with_context``, any ``tracer.span(...)`` opened inside a rail's
``.classify`` (running on a worker thread) became a fresh trace root.
After the fix, rail-internal spans inherit the caller's ambient OTel
context and nest under the ``guardrails.input`` parent span.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from src.guardrails.common import RailVerdict
from src.guardrails.nemo_guardrails.executor import InputRailExecutor
from src.guardrails.shared.injection import InjectionResult
from src.guardrails.shared.intent import IntentResult
from src.platform.observability import get_tracer


def test_rail_internal_span_nests_under_guardrails_input(otel_capture):
    """Spans opened inside a rail worker nest under guardrails.input."""

    def classify_with_internal_span(query):
        tracer = get_tracer()
        with tracer.span("test.rail_internal", {"q": "hash"}):
            return IntentResult(intent="rag_search", confidence=0.9)

    intent = MagicMock()
    intent.classify.side_effect = classify_with_internal_span

    injection = MagicMock()
    injection.check.return_value = InjectionResult(verdict=RailVerdict.PASS)

    ex = InputRailExecutor(
        intent_classifier=intent,
        injection_detector=injection,
        pii_detector=None,
        toxicity_filter=None,
        topic_safety_checker=None,
        timeout_seconds=5,
    )
    ex.execute("hello world", tenant_id="t1")

    spans = {s.name: s for s in otel_capture.get_finished_spans()}
    assert "guardrails.input" in spans
    assert "test.rail_internal" in spans

    parent = spans["guardrails.input"]
    inner = spans["test.rail_internal"]

    # Same trace, and inner is a descendant of guardrails.input.
    assert inner.context.trace_id == parent.context.trace_id
    assert inner.parent is not None
    # Parent linkage: inner span's parent is the guardrails.input span
    # (rails run their work on a worker; the per-rail child span is only
    # opened in the collector thread *after* the worker finishes).
    assert inner.parent.span_id == parent.context.span_id
