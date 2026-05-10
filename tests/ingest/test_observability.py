"""Tests for the stage_span observability helper.

Contract:
- Emits a single structured JSON log event on exit.
- Event shape follows OpenTelemetry conventions:
  trace_id, span_id, parent_span_id, stage_name, duration_ms, status,
  attributes, optional error.
- Status is "ok" on clean exit, "error" on exception (and exception re-raised).
- duration_ms is a non-negative float.
- span_id is a fresh UUID per invocation; trace_id is passed through verbatim.
"""

from __future__ import annotations

import json
import logging

import pytest


def _capture(caplog, level=logging.INFO):
    caplog.set_level(level, logger="rag.ingest.obs")
    return caplog


def _events(caplog):
    out = []
    for rec in caplog.records:
        msg = rec.getMessage()
        try:
            payload = json.loads(msg)
        except (ValueError, TypeError):
            continue
        if isinstance(payload, dict) and payload.get("event") == "stage_span":
            out.append(payload)
    return out


def test_stage_span_emits_otel_shaped_event(caplog):
    from src.ingest.common.observability import stage_span

    _capture(caplog)
    with stage_span("document_ingestion", trace_id="trace-abc", source_key="docs/a.md"):
        pass

    events = _events(caplog)
    assert len(events) == 1
    ev = events[0]
    assert ev["trace_id"] == "trace-abc"
    assert ev["stage_name"] == "document_ingestion"
    assert ev["status"] == "ok"
    assert ev["span_id"]  # non-empty
    assert "parent_span_id" in ev
    assert isinstance(ev["duration_ms"], (int, float))
    assert ev["duration_ms"] >= 0
    assert ev["attributes"]["source_key"] == "docs/a.md"


def test_stage_span_records_error_and_reraises(caplog):
    from src.ingest.common.observability import stage_span

    _capture(caplog)
    with pytest.raises(ValueError, match="boom"):
        with stage_span("structure_detection", trace_id="t1"):
            raise ValueError("boom")

    events = _events(caplog)
    assert len(events) == 1
    assert events[0]["status"] == "error"
    assert "boom" in events[0]["error"]
    assert events[0]["stage_name"] == "structure_detection"


def test_stage_span_unique_span_ids(caplog):
    from src.ingest.common.observability import stage_span

    _capture(caplog)
    with stage_span("a", trace_id="t"):
        pass
    with stage_span("b", trace_id="t"):
        pass

    events = _events(caplog)
    assert len(events) == 2
    assert events[0]["span_id"] != events[1]["span_id"]


def test_stage_span_supports_parent_span(caplog):
    from src.ingest.common.observability import stage_span

    _capture(caplog)
    with stage_span("phase1", trace_id="t") as parent:
        with stage_span("nested", trace_id="t", parent_span_id=parent["span_id"]):
            pass

    events = _events(caplog)
    # inner finishes first
    inner = events[0]
    outer = events[1]
    assert inner["stage_name"] == "nested"
    assert inner["parent_span_id"] == outer["span_id"]


def test_stage_span_yields_context_with_span_id():
    from src.ingest.common.observability import stage_span

    with stage_span("x", trace_id="abc") as ctx:
        assert ctx["trace_id"] == "abc"
        assert isinstance(ctx["span_id"], str) and ctx["span_id"]


def test_node_span_decorator_pulls_trace_id_from_state(caplog):
    from src.ingest.common.observability import node_span

    @node_span("chunking")
    def my_node(state):
        return {"ok": True}

    _capture(caplog)
    out = my_node({"trace_id": "trace-99", "source_key": "k"})
    assert out == {"ok": True}
    ev = _events(caplog)[0]
    assert ev["trace_id"] == "trace-99"
    assert ev["stage_name"] == "chunking"
    assert ev["attributes"]["source_key"] == "k"
    assert ev["status"] == "ok"


def test_node_span_decorator_records_error(caplog):
    from src.ingest.common.observability import node_span

    @node_span("boom_node")
    def my_node(state):
        raise ValueError("kaboom")

    _capture(caplog)
    try:
        my_node({"trace_id": "t"})
    except ValueError:
        pass
    ev = _events(caplog)[0]
    assert ev["status"] == "error"
    assert "kaboom" in ev["error"]


def test_stage_span_attributes_preserved_on_error(caplog):
    from src.ingest.common.observability import stage_span

    _capture(caplog)
    with pytest.raises(RuntimeError):
        with stage_span("text_cleaning", trace_id="t", chunks=42, model="x"):
            raise RuntimeError("x")
    ev = _events(caplog)[0]
    assert ev["attributes"] == {"chunks": 42, "model": "x"}
