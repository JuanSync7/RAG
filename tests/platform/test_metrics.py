"""Contract tests for the shared Prometheus metrics registry.

These tests pin the *exposition contract* of :mod:`src.platform.metrics`:
the exact exported metric names and label-name sets for every metric, plus
the shape and live-registry wiring of :func:`render_metrics`. Dashboards and
scrapers depend on the exact names and labels, so any rename or label-set
change here is a breaking change and must red a test.

Name-derivation note (verified experimentally against prometheus_client):
``Counter`` objects store ``_name`` *without* the ``_total`` suffix (e.g. a
Counter declared ``rag_cache_hits_total`` has ``_name == 'rag_cache_hits'``)
and the exposition format re-appends ``_total`` on export. ``Gauge`` and
``Histogram`` keep their full declared name in ``_name``. The exported name is
therefore derived as ``_name + '_total'`` for counters and ``_name`` otherwise,
matching what a scraper actually sees in the rendered payload.
"""

from __future__ import annotations

import pytest
from prometheus_client import CONTENT_TYPE_LATEST

from src.platform import metrics as m

# --------------------------------------------------------------------------
# The contract spec table: (object, exported_name, label-name set).
# This table IS the dashboard/scraper contract. Editing a row here is the
# only sanctioned way to change a metric's public exposition surface.
# --------------------------------------------------------------------------
METRIC_CONTRACT = [
    (m.REQUESTS_TOTAL, "rag_api_requests_total", {"endpoint", "method", "status"}),
    (m.REQUEST_LATENCY_MS, "rag_api_request_latency_ms", {"endpoint", "method"}),
    (m.RATE_LIMIT_REJECTS, "rag_rate_limit_rejections_total", {"endpoint"}),
    (m.OVERLOAD_REJECTS, "rag_api_overload_rejections_total", {"endpoint"}),
    (m.INFLIGHT_REQUESTS, "rag_api_inflight_requests", set()),
    (m.PIPELINE_STAGE_MS, "rag_pipeline_stage_ms", {"stage", "bucket"}),
    (m.CACHE_HITS, "rag_cache_hits_total", {"layer"}),
    (m.CACHE_MISSES, "rag_cache_misses_total", {"layer"}),
    (m.MEMORY_OP_MS, "rag_memory_operation_ms", {"operation"}),
    (m.MEMORY_SUMMARY_TRIGGERS, "rag_memory_summary_triggers_total", {"reason"}),
]


def _exported_name(metric) -> str:
    """Derive the name a scraper sees from a prometheus_client metric object.

    Counters expose ``_name`` without the ``_total`` suffix; the exposition
    format re-appends it. All other metric types keep their full declared name.
    """
    base = metric._name
    if metric._type == "counter":
        return base + "_total"
    return base


# --------------------------------------------------------------------------
# Metric-name + label-set contract (the load-bearing tests)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "metric, expected_name, expected_labels",
    METRIC_CONTRACT,
    ids=[row[1] for row in METRIC_CONTRACT],
)
def test_metric_exported_name_matches_contract(metric, expected_name, expected_labels):
    """Each metric's exported name matches the documented dashboard contract."""
    assert _exported_name(metric) == expected_name


@pytest.mark.parametrize(
    "metric, expected_name, expected_labels",
    METRIC_CONTRACT,
    ids=[row[1] for row in METRIC_CONTRACT],
)
def test_metric_label_set_matches_contract(metric, expected_name, expected_labels):
    """Each metric's label-name set matches the contract exactly.

    Asserting set equality means an extra or missing label both fail: the
    label set is a hard contract, not a superset/subset relationship.
    """
    assert set(metric._labelnames) == expected_labels


# --------------------------------------------------------------------------
# render_metrics shape
# --------------------------------------------------------------------------


def test_render_metrics_returns_bytes_payload_and_content_type():
    """render_metrics returns a 2-tuple of (bytes payload, prometheus content type).

    The content type is compared against the imported ``CONTENT_TYPE_LATEST``
    rather than a hardcoded string, so a prometheus_client version bump that
    changes the canonical content type does not silently pass.
    """
    result = m.render_metrics()
    assert isinstance(result, tuple)
    assert len(result) == 2
    payload, content_type = result
    assert isinstance(payload, bytes)
    assert content_type == CONTENT_TYPE_LATEST


# --------------------------------------------------------------------------
# Registry -> payload wiring (teeth: proves live state is exported)
# --------------------------------------------------------------------------


def test_render_metrics_exports_live_counter_state():
    """Incrementing a labeled counter shows up as a series line in the payload.

    Uses a deliberately distinct label triple so collisions with other tests
    or process-global state cannot mask the increment, and asserts ``>= 1.0``
    (not ``== 1.0``) for the same reason. This proves render_metrics exports
    the live default registry, not a constant.
    """
    m.REQUESTS_TOTAL.labels(
        endpoint="/contract-probe", method="POST", status="200"
    ).inc()

    payload, _ = m.render_metrics()
    text = payload.decode("utf-8")

    series = [
        line
        for line in text.splitlines()
        if line.startswith("rag_api_requests_total{")
        and 'endpoint="/contract-probe"' in line
        and 'method="POST"' in line
        and 'status="200"' in line
    ]
    assert len(series) == 1, f"expected exactly one matching series, got: {series}"

    value = float(series[0].rsplit(" ", 1)[1])
    assert value >= 1.0


# --------------------------------------------------------------------------
# Histogram family presence
# --------------------------------------------------------------------------


def test_histogram_emits_bucket_count_sum_families():
    """A Histogram observation produces _bucket, _count and _sum families.

    Exercising rag_pipeline_stage_ms proves histograms are wired into the same
    rendered payload as counters/gauges.
    """
    m.PIPELINE_STAGE_MS.labels(stage="contract-probe", bucket="probe").observe(123.0)

    payload, _ = m.render_metrics()
    text = payload.decode("utf-8")

    # Label rendering order is not declaration order, so match the family
    # prefix plus the label fragments rather than a fixed brace string.
    def _family_line(prefix: str) -> str:
        for line in text.splitlines():
            if (
                line.startswith(prefix)
                and 'stage="contract-probe"' in line
                and 'bucket="probe"' in line
            ):
                return line
        raise AssertionError(f"no {prefix} line found for the probe labels")

    assert "rag_pipeline_stage_ms_bucket{" in text
    count_line = _family_line("rag_pipeline_stage_ms_count{")
    sum_line = _family_line("rag_pipeline_stage_ms_sum{")
    assert float(count_line.rsplit(" ", 1)[1]) >= 1.0
    assert float(sum_line.rsplit(" ", 1)[1]) >= 123.0
