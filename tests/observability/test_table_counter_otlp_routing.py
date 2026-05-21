# @summary
# Routing-verification tests for the four ingest.table.* OTel counters introduced
# in PR #100. Validates that counter records actually reach an OTLP-shaped
# exporter (the wire-format path used in production for Langfuse / Phoenix /
# Braintrust) and not merely an InMemoryMetricReader (which the DD unit tests
# already cover at tests/ingest/test_docling_adaptive_table_chunking.py).
# Exports: TestTableCounterOTLPRouting
# Deps: opentelemetry.sdk.metrics, src.ingest.support.table_metrics
# @end-summary

"""OTLP-shape routing verification for ``ragweave.ingest.table`` counters.

The DD unit tests proved counter increments via ``InMemoryMetricReader`` —
which short-circuits the SDK's export path. This module installs a custom
``MetricExporter`` subclass behind a ``PeriodicExportingMetricReader`` so the
same data flow the real OTLP exporter receives is exercised end-to-end, then
asserts that all four production counter names and the meter scope name make
it through ``MeterProvider.force_flush``.

The exporter is purely in-process: per ``project_observability_otel.md`` we
must never connect to a live collector from CI.
"""
from __future__ import annotations

from typing import Any

import pytest

from opentelemetry import metrics as otel_metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    AggregationTemporality,
    MetricExporter,
    MetricExportResult,
    PeriodicExportingMetricReader,
)

from src.ingest.support import table_metrics as _tm
from src.ingest.common.types import IngestionConfig


# Reuse the fixture helpers from the adaptive-table-chunking test module so we
# drive the exact same code path that the production chunker takes.
from tests.ingest.test_docling_adaptive_table_chunking import (  # noqa: E402
    _make_raw_chunk,
    _make_table_artifact,
    _run_chunk,
)


_EXPECTED_COUNTER_NAMES = {
    "ingest.table.row_chunks_emitted",
    "ingest.table.summary_only_due_to_small",
    "ingest.table.summary_only_due_to_uniform",
    "ingest.table.summary_text_truncated",
}
_EXPECTED_SCOPE = "ragweave.ingest.table"


class _CapturingOTLPShapeExporter(MetricExporter):
    """In-process exporter that mirrors the OTLP exporter's public surface.

    Real OTLP exporters (``opentelemetry.exporter.otlp.proto.http`` /
    ``opentelemetry.exporter.otlp.proto.grpc``) subclass ``MetricExporter`` and
    receive a fully materialised ``MetricsData`` payload on each ``export``
    call. By capturing that payload here we verify the counters reach the
    same boundary they would on the wire — without depending on a collector.
    """

    def __init__(self) -> None:
        # Default to CUMULATIVE temporality (matches Langfuse / Phoenix
        # ingestion conventions for counters).
        super().__init__(
            preferred_temporality={},
            preferred_aggregation={},
        )
        self.exported: list[Any] = []

    def export(self, metrics_data: Any, timeout_millis: float = 10_000, **kwargs: Any) -> MetricExportResult:  # type: ignore[override]
        self.exported.append(metrics_data)
        return MetricExportResult.SUCCESS

    def force_flush(self, timeout_millis: float = 10_000) -> bool:  # type: ignore[override]
        return True

    def shutdown(self, timeout_millis: float = 30_000, **kwargs: Any) -> None:  # type: ignore[override]
        return None

    # Convenience helpers ---------------------------------------------------
    def all_metric_names(self) -> set[str]:
        names: set[str] = set()
        for payload in self.exported:
            for rm in payload.resource_metrics:
                for sm in rm.scope_metrics:
                    for metric in sm.metrics:
                        names.add(metric.name)
        return names

    def all_scope_names(self) -> set[str]:
        scopes: set[str] = set()
        for payload in self.exported:
            for rm in payload.resource_metrics:
                for sm in rm.scope_metrics:
                    scopes.add(sm.scope.name)
        return scopes


@pytest.fixture()
def otlp_capture_provider():
    """Install a fresh ``MeterProvider`` whose only reader is our capturing exporter.

    The reset path mirrors how the live OTel adapter (``OTelBackend``)
    installs the global provider — but uses an in-process exporter instead of
    OTLP. ``table_metrics`` is reset before AND after so its lazy singleton
    rebinds against this provider's meter (rather than any stale one left by
    sibling tests).
    """
    exporter = _CapturingOTLPShapeExporter()
    # Periodic reader with a long interval — we drive flushes manually so
    # tests stay deterministic and don't depend on wall-clock timing.
    reader = PeriodicExportingMetricReader(
        exporter,
        export_interval_millis=60_000,
        export_timeout_millis=5_000,
    )
    provider = MeterProvider(metric_readers=[reader])

    # ``set_meter_provider`` is single-shot per process; clear the proxy
    # provider so this SDK provider actually takes effect. This mirrors the
    # workaround used by tests/ingest/test_docling_adaptive_table_chunking.py.
    # Reset the internal state of opentelemetry.metrics so this provider
    # actually replaces whatever was installed by a sibling test. The
    # ``_METER_PROVIDER_SET_ONCE`` Once gate is the load-bearing one — without
    # resetting it ``set_meter_provider`` silently no-ops with a warning, and
    # our counters keep emitting to the previous provider's exporter.
    from opentelemetry.metrics import _internal as _otel_metrics_internal
    from opentelemetry.util._once import Once

    prior_meter = getattr(_otel_metrics_internal, "_METER_PROVIDER", None)
    prior_once = getattr(_otel_metrics_internal, "_METER_PROVIDER_SET_ONCE", None)
    _otel_metrics_internal._METER_PROVIDER = None  # type: ignore[attr-defined]
    _otel_metrics_internal._METER_PROVIDER_SET_ONCE = Once()  # type: ignore[attr-defined]
    otel_metrics.set_meter_provider(provider)

    # Drop any cached counter registry so it rebinds against THIS provider.
    _tm._reset_for_tests()
    try:
        yield provider, exporter
    finally:
        try:
            provider.force_flush(timeout_millis=1_000)
            provider.shutdown()
        except Exception:
            pass
        # Restore prior provider state and reset the counter singleton so we
        # don't leak this MeterProvider into sibling tests.
        _otel_metrics_internal._METER_PROVIDER = prior_meter  # type: ignore[attr-defined]
        if prior_once is not None:
            _otel_metrics_internal._METER_PROVIDER_SET_ONCE = prior_once  # type: ignore[attr-defined]
        _tm._reset_for_tests()


class TestTableCounterOTLPRouting:
    """End-to-end verification that the four counters reach an OTLP-shape exporter."""

    def test_three_synthetic_tables_export_all_four_counter_names(
        self, otlp_capture_provider
    ):
        """Driving the chunker over the DD fixture must surface every counter name through the exporter."""
        provider, exporter = otlp_capture_provider

        # 1) small + uniform → row_chunks_emitted
        tbl_ok = _make_table_artifact(
            table_id="ok",
            cells=[["H1", "H2"], ["a", "b"], ["c", "d"]],
            section_path="A",
        )
        tbl_ok.self_ref = "#/tables/0"

        # 2) too many rows → small_gate
        big_cells = [["Col A", "Col B"]] + [[f"a{i}", f"b{i}"] for i in range(40)]
        tbl_big = _make_table_artifact(
            table_id="big", cells=big_cells, section_path="B"
        )
        tbl_big.self_ref = "#/tables/1"

        # 3) ragged rows → uniform_gate
        tbl_ragged = _make_table_artifact(
            table_id="ragged",
            cells=[["H1", "H2", "H3"], ["a", "b", "c"], ["d", "e"]],
            section_path="C",
        )
        tbl_ragged.self_ref = "#/tables/2"

        # 4) wide table under a tight cap → summary_text_truncated
        headers = [f"col_{i:03d}_descriptor" for i in range(120)]
        wide_cells = [headers] + [[f"v{i}" for i in range(120)] for _ in range(5)]
        tbl_wide = _make_table_artifact(
            table_id="wide", cells=wide_cells, caption="Wide"
        )
        tbl_wide.self_ref = "#/tables/3"

        chunks_first = [
            _make_raw_chunk(tbl_ok.markdown),
            _make_raw_chunk(tbl_big.markdown),
            _make_raw_chunk(tbl_ragged.markdown),
        ]
        _run_chunk(chunks_first, [tbl_ok, tbl_big, tbl_ragged])
        _run_chunk(
            [_make_raw_chunk(tbl_wide.markdown)],
            [tbl_wide],
            config=IngestionConfig(table_summary_max_chars=256),
        )

        # Flush so the periodic reader hands the payload to the exporter.
        assert provider.force_flush(timeout_millis=1_000) is True

        names = exporter.all_metric_names()
        missing = _EXPECTED_COUNTER_NAMES - names
        assert not missing, (
            f"Counters never reached the OTLP-shape exporter: {sorted(missing)}. "
            f"Saw only: {sorted(names)}."
        )

    def test_exported_records_carry_ragweave_ingest_table_scope(
        self, otlp_capture_provider
    ):
        """At least one exported scope must be ``ragweave.ingest.table`` (meter naming contract)."""
        provider, exporter = otlp_capture_provider

        tbl = _make_table_artifact(
            cells=[["H1", "H2"], ["a", "b"], ["c", "d"]]
        )
        _run_chunk([_make_raw_chunk(tbl.markdown)], [tbl])

        assert provider.force_flush(timeout_millis=1_000) is True
        scopes = exporter.all_scope_names()
        assert _EXPECTED_SCOPE in scopes, (
            f"Expected meter scope {_EXPECTED_SCOPE!r} on at least one exported "
            f"record; got scopes={sorted(scopes)}."
        )
