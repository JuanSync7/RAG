"""OTel instrumentation contract for InputRailExecutor / OutputRailExecutor.

Verifies:
- ``guardrails.input`` / ``guardrails.output`` parent spans with rail_count,
  timeout_ms, verdict attrs.
- ``guardrails.rail.<name>`` child spans with rail_name, verdict, and
  rail-specific attrs.
- Parent/child nesting via ambient OTel context.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from src.guardrails.common import RailVerdict
from src.guardrails.nemo_guardrails.executor import (
    InputRailExecutor,
    OutputRailExecutor,
)
from src.guardrails.shared.faithfulness import FaithfulnessResult
from src.guardrails.shared.injection import InjectionResult
from src.guardrails.shared.intent import IntentResult
from src.guardrails.shared.topic_safety import TopicSafetyResult
from src.guardrails.shared.toxicity import ToxicityResult


def _spans_named(exporter, name):
    return [s for s in exporter.get_finished_spans() if s.name == name]


def _by_id(exporter):
    return {s.context.span_id: s for s in exporter.get_finished_spans()}


def test_input_executor_emits_guardrails_input_span(otel_capture):
    intent = MagicMock()
    intent.classify.return_value = IntentResult(intent="rag_search", confidence=0.9)
    injection = MagicMock()
    injection.check.return_value = InjectionResult(verdict=RailVerdict.PASS)
    pii = MagicMock()
    pii.redact.return_value = ("hello", [])
    toxicity = MagicMock()
    toxicity.check.return_value = ToxicityResult(verdict=RailVerdict.PASS, score=0.1)
    topic = MagicMock()
    topic.check.return_value = TopicSafetyResult(verdict=RailVerdict.PASS, on_topic=True)

    ex = InputRailExecutor(
        intent_classifier=intent,
        injection_detector=injection,
        pii_detector=pii,
        toxicity_filter=toxicity,
        topic_safety_checker=topic,
        timeout_seconds=5,
    )
    result = ex.execute("hello world", tenant_id="t1")

    parents = _spans_named(otel_capture, "guardrails.input")
    assert len(parents) == 1
    attrs = parents[0].attributes
    assert attrs["rail_count"] == 5
    assert attrs["timeout_ms"] == 5000
    assert attrs["verdict"] == "pass"

    # Per-rail child spans
    expected = {"intent", "injection", "pii", "toxicity", "topic_safety"}
    rail_spans = _spans_named(otel_capture, "guardrails.rail.intent") \
        + _spans_named(otel_capture, "guardrails.rail.injection") \
        + _spans_named(otel_capture, "guardrails.rail.pii") \
        + _spans_named(otel_capture, "guardrails.rail.toxicity") \
        + _spans_named(otel_capture, "guardrails.rail.topic_safety")
    assert {s.attributes["rail_name"] for s in rail_spans} == expected

    # Verify nesting: every per-rail span has the guardrails.input span as parent
    parent_id = parents[0].context.span_id
    for s in rail_spans:
        assert s.parent is not None
        assert s.parent.span_id == parent_id

    # Spot-check rail-specific attrs
    intent_span = _spans_named(otel_capture, "guardrails.rail.intent")[0]
    assert intent_span.attributes["intent"] == "rag_search"
    assert intent_span.attributes["confidence"] == 0.9
    inj_span = _spans_named(otel_capture, "guardrails.rail.injection")[0]
    assert inj_span.attributes["verdict"] == "pass"
    tox_span = _spans_named(otel_capture, "guardrails.rail.toxicity")[0]
    assert tox_span.attributes["score"] == 0.1
    ts_span = _spans_named(otel_capture, "guardrails.rail.topic_safety")[0]
    assert ts_span.attributes["on_topic"] is True


def test_input_executor_pii_detection_marks_modify(otel_capture):
    pii = MagicMock()
    detection = MagicMock()
    detection.pii_type = "EMAIL"
    pii.redact.return_value = ("[EMAIL_REDACTED]", [detection, detection])

    ex = InputRailExecutor(pii_detector=pii, timeout_seconds=3)
    ex.execute("a@b.com c@d.com")

    pii_spans = _spans_named(otel_capture, "guardrails.rail.pii")
    assert len(pii_spans) == 1
    assert pii_spans[0].attributes["verdict"] == "modify"
    assert pii_spans[0].attributes["entity_count"] == 2

    parents = _spans_named(otel_capture, "guardrails.input")
    # PII detections roll the aggregated outer verdict up to "modify"
    assert parents[0].attributes["verdict"] == "modify"


def test_input_executor_rail_exception_records_error(otel_capture):
    injection = MagicMock()
    injection.check.side_effect = RuntimeError("boom")
    ex = InputRailExecutor(injection_detector=injection, timeout_seconds=2)
    ex.execute("q")

    inj_spans = _spans_named(otel_capture, "guardrails.rail.injection")
    assert len(inj_spans) == 1
    # OTel status code: 2 == ERROR
    assert inj_spans[0].status.status_code.value == 2


def test_output_executor_emits_guardrails_output_span(otel_capture):
    faith = MagicMock()
    faith.check.return_value = FaithfulnessResult(
        overall_score=0.82, verdict=RailVerdict.PASS,
    )
    pii = MagicMock()
    pii.redact.return_value = ("answer", [])
    tox = MagicMock()
    tox.filter_output.return_value = "answer"

    ex = OutputRailExecutor(
        faithfulness_checker=faith,
        pii_detector=pii,
        toxicity_filter=tox,
        timeout_seconds=7,
    )
    ex.execute("answer", ["ctx1", "ctx2"])

    parents = _spans_named(otel_capture, "guardrails.output")
    assert len(parents) == 1
    attrs = parents[0].attributes
    assert attrs["rail_count"] == 3
    assert attrs["timeout_ms"] == 7000
    assert attrs["verdict"] == "pass"

    expected = {"faithfulness", "pii", "toxicity"}
    rail_spans = (
        _spans_named(otel_capture, "guardrails.rail.faithfulness")
        + _spans_named(otel_capture, "guardrails.rail.pii")
        + _spans_named(otel_capture, "guardrails.rail.toxicity")
    )
    assert {s.attributes["rail_name"] for s in rail_spans} == expected

    parent_id = parents[0].context.span_id
    for s in rail_spans:
        assert s.parent is not None and s.parent.span_id == parent_id

    f_span = _spans_named(otel_capture, "guardrails.rail.faithfulness")[0]
    assert f_span.attributes["score"] == 0.82


def test_output_executor_faithfulness_reject_marks_parent(otel_capture):
    faith = MagicMock()
    faith.check.return_value = FaithfulnessResult(
        overall_score=0.1,
        verdict=RailVerdict.REJECT,
        fallback_message="nope",
    )
    ex = OutputRailExecutor(faithfulness_checker=faith, timeout_seconds=4)
    result = ex.execute("hallucinated", ["ctx"])
    assert result.final_answer == "nope"

    parents = _spans_named(otel_capture, "guardrails.output")
    assert parents[0].attributes["verdict"] == "reject"
    f_span = _spans_named(otel_capture, "guardrails.rail.faithfulness")[0]
    assert f_span.attributes["verdict"] == "reject"
