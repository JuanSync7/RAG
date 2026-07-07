# @summary
# Stage span helper for OpenTelemetry-shaped structured logging.
# Exports: stage_span
# Deps: stdlib only
# @end-summary

"""OpenTelemetry-shaped stage spans for the ingestion pipeline.

`stage_span(stage_name, trace_id=..., **attrs)` is a context manager that emits
a single structured JSON log event on exit with the following shape:

    {
      "event": "stage_span",
      "trace_id": "...",
      "span_id": "...",
      "parent_span_id": "...",
      "stage_name": "...",
      "duration_ms": 12.34,
      "status": "ok" | "error",
      "attributes": {...},
      "error": "..."   # only on status=="error"
    }

The yielded value is a dict carrying ``span_id`` and ``trace_id`` so callers
can pass ``parent_span_id`` to nested spans.

Logger name: ``rag.ingest.obs``. Format is JSON-encoded message text so
existing logging handlers don't need restructuring.
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
import uuid
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Iterator

from src.platform.observability import get_tracer

_logger = logging.getLogger("rag.ingest.obs")


@contextmanager
def stage_span(
    stage_name: str,
    *,
    trace_id: str = "",
    parent_span_id: str = "",
    **attributes: Any,
) -> Iterator[dict]:
    """Context manager that emits an OTel-shaped JSON log event on exit.

    Args:
        stage_name: Pipeline stage / node name.
        trace_id: Per-document trace ID. Pass through verbatim (use the
            Temporal staging_batch_id when invoked from a Temporal activity,
            so a single ID flows from MinIO write to Weaviate insert).
        parent_span_id: Parent span ID for nested spans. Empty for roots.
        **attributes: Free-form attributes recorded under "attributes".
    """
    span_id = uuid.uuid4().hex
    start = time.perf_counter()
    status = "ok"
    err: BaseException | None = None
    try:
        yield {"span_id": span_id, "trace_id": trace_id}
    except BaseException as exc:
        status = "error"
        err = exc
        raise
    finally:
        duration_ms = round((time.perf_counter() - start) * 1000.0, 3)
        event: dict[str, Any] = {
            "event": "stage_span",
            "trace_id": trace_id,
            "span_id": span_id,
            "parent_span_id": parent_span_id,
            "stage_name": stage_name,
            "duration_ms": duration_ms,
            "status": status,
            "attributes": dict(attributes),
        }
        if err is not None:
            event["error"] = f"{type(err).__name__}: {err}"
        try:
            payload = json.dumps(event, default=str)
        except (TypeError, ValueError):
            payload = json.dumps({**event, "attributes": {"_repr": repr(attributes)}}, default=str)
        _logger.info(payload)


def node_span(stage_name: str) -> Callable:
    """Decorator wrapping a LangGraph node fn (state dict in, partial state out).

    Pulls ``trace_id`` from the input state and emits a stage_span event with
    ``source_key`` recorded as an attribute. Errors are recorded but re-raised
    so LangGraph error handling stays intact.
    """

    def _decorate(fn: Callable) -> Callable:
        @wraps(fn)
        def _wrapped(state: dict, *args: Any, **kwargs: Any):
            trace_id = state.get("trace_id", "") if isinstance(state, dict) else ""
            source_key = state.get("source_key", "") if isinstance(state, dict) else ""

            # Open a real OTel span around the node so nested
            # get_tracer().span(...) calls inside the body nest under it via
            # ambient OTel context. Lazy lookup of the tracer is critical —
            # tests swap the backend singleton post-import, and capturing it
            # at module-import time defeats that. The whole construction is
            # wrapped fail-open so a busted backend never breaks ingest.
            try:
                otel_cm = get_tracer().span(
                    f"node.{stage_name}",
                    attributes={
                        "stage_name": stage_name,
                        "trace_id": trace_id,
                        "source_key": source_key,
                    },
                )
            except Exception as exc:  # pragma: no cover — defensive
                _logger.warning("node_span OTel span construction failed: %s", exc)
                otel_cm = contextlib.nullcontext()

            # OTel span is outer so its lifetime brackets the JSON log too;
            # stage_span is inner so the existing log shape stays untouched.
            with otel_cm:
                with stage_span(stage_name, trace_id=trace_id, source_key=source_key):
                    return fn(state, *args, **kwargs)

        return _wrapped

    return _decorate
