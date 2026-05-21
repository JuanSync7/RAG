# @summary
# OpenTelemetry counters for adaptive table chunking gate decisions.
# Exports: increment_row_chunks_emitted, increment_summary_only_small,
#          increment_summary_only_uniform, increment_summary_text_truncated,
#          get_counters
# Deps: opentelemetry.metrics
# @end-summary
"""OTel-native counters around the adaptive table chunker's gate decisions.

The adaptive table chunker (``_apply_adaptive_table_chunking``) makes two
independent gate decisions per table that determine whether row chunks are
emitted alongside the table-summary chunk:

* small-table gate — does the table fit within ``max_table_rows_for_row_chunks``
  / ``max_table_cols_for_row_chunks``?
* uniform-row gate — are all body rows the same width as the header row?

We want production signal for how often each gate fires versus how often row
chunks actually land. Counters are registered lazily against the global OTel
``MeterProvider`` so they no-op when no SDK provider is configured (the default
in tests per ``project_observability_otel.md``), and route through OTLP to
Langfuse/Phoenix/Braintrust when one is.

Counter names use the ``ingest.table.*`` namespace so they nest cleanly under
the broader ingest telemetry hierarchy on the receiving backend.
"""
from __future__ import annotations

from threading import Lock
from typing import Any

_INSTRUMENTATION_NAME = "ragweave.ingest.table"
_INSTRUMENTATION_VERSION = "1.0.0"

_counters: dict[str, Any] | None = None
_init_lock = Lock()


def _build_counters() -> dict[str, Any]:
    """Construct the four counter instruments against the global meter.

    Called at most once per process; results are cached. If the OTel API is
    unavailable for any reason we fall back to a dict of no-op shims so call
    sites can stay unconditional.
    """
    try:
        from opentelemetry import metrics
    except Exception:  # pragma: no cover — opentelemetry is a transitive dep
        return {
            "row_chunks_emitted": _NoopCounter(),
            "summary_only_due_to_small": _NoopCounter(),
            "summary_only_due_to_uniform": _NoopCounter(),
            "summary_text_truncated": _NoopCounter(),
        }

    meter = metrics.get_meter(_INSTRUMENTATION_NAME, _INSTRUMENTATION_VERSION)
    return {
        "row_chunks_emitted": meter.create_counter(
            name="ingest.table.row_chunks_emitted",
            description=(
                "Number of tables for which row chunks were emitted in "
                "addition to the summary chunk."
            ),
            unit="1",
        ),
        "summary_only_due_to_small": meter.create_counter(
            name="ingest.table.summary_only_due_to_small",
            description=(
                "Number of tables that were emitted as summary-only because "
                "they exceeded max_table_rows_for_row_chunks / "
                "max_table_cols_for_row_chunks or lacked a usable header."
            ),
            unit="1",
        ),
        "summary_only_due_to_uniform": meter.create_counter(
            name="ingest.table.summary_only_due_to_uniform",
            description=(
                "Number of tables that were emitted as summary-only because "
                "body row widths did not match the header (ragged / uniform "
                "gate)."
            ),
            unit="1",
        ),
        "summary_text_truncated": meter.create_counter(
            name="ingest.table.summary_text_truncated",
            description=(
                "Number of table summary texts that were truncated by the "
                "configured table_summary_max_chars cap."
            ),
            unit="1",
        ),
    }


class _NoopCounter:
    """Last-resort counter shim used only when the OTel API import fails."""

    def add(self, amount: int, attributes: dict | None = None) -> None:
        return None


def get_counters() -> dict[str, Any]:
    """Return the lazily-initialized counter registry for this module."""
    global _counters
    if _counters is not None:
        return _counters
    with _init_lock:
        if _counters is None:
            _counters = _build_counters()
    return _counters


def _safe_add(name: str, amount: int = 1) -> None:
    try:
        get_counters()[name].add(amount)
    except Exception:  # pragma: no cover — fail-open contract
        return None


def increment_row_chunks_emitted(amount: int = 1) -> None:
    """Record that row chunks were emitted for ``amount`` tables (default 1)."""
    _safe_add("row_chunks_emitted", amount)


def increment_summary_only_small(amount: int = 1) -> None:
    """Record that the small-table gate fired for ``amount`` tables."""
    _safe_add("summary_only_due_to_small", amount)


def increment_summary_only_uniform(amount: int = 1) -> None:
    """Record that the uniform-row gate fired for ``amount`` tables."""
    _safe_add("summary_only_due_to_uniform", amount)


def increment_summary_text_truncated(amount: int = 1) -> None:
    """Record that ``amount`` table summary texts were truncated."""
    _safe_add("summary_text_truncated", amount)


def _reset_for_tests() -> None:
    """Drop the cached counter registry. Test-only — never call in production."""
    global _counters
    with _init_lock:
        _counters = None


__all__ = [
    "get_counters",
    "increment_row_chunks_emitted",
    "increment_summary_only_small",
    "increment_summary_only_uniform",
    "increment_summary_text_truncated",
]
