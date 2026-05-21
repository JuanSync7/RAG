<!-- @summary
Operational reference for the four ingest.table.* OTel counters that track
adaptive-table-chunker gate decisions in the Docling ingest path.
@end-summary -->

# Ingest Table Counters

Production OTel counters emitted by `src/ingest/support/table_metrics.py` to
observe how the adaptive table chunker decides between emitting per-row chunks
and emitting summary-only chunks. Counters are routed through the standard
OTel adapter (`src/platform/observability/otel/backend.py`) to whichever
OTLP-compatible backend `OTEL_EXPORTER_OTLP_ENDPOINT` points at — Langfuse,
Phoenix, Braintrust, or any other OTLP collector.

## Counter contract

All four counters live under the OTel meter scope `ragweave.ingest.table`
(instrumentation version `1.0.0`). The names below are the production wire
format — DO NOT rename.

| Counter name | Type | Unit | Dimensions | Meaning |
| --- | --- | --- | --- | --- |
| `ingest.table.row_chunks_emitted` | Counter (monotonic, int) | `1` | none today | Number of tables for which per-row chunks were emitted alongside the summary chunk. |
| `ingest.table.summary_only_due_to_small` | Counter (monotonic, int) | `1` | none today | Tables that became summary-only because they exceeded `max_table_rows_for_row_chunks` / `max_table_cols_for_row_chunks`, or had no usable header. |
| `ingest.table.summary_only_due_to_uniform` | Counter (monotonic, int) | `1` | none today | Tables that became summary-only because body row widths did not match the header (ragged rows). |
| `ingest.table.summary_text_truncated` | Counter (monotonic, int) | `1` | none today | Table summary texts truncated by `table_summary_max_chars`. |

The four counters are increment-only, registered lazily against the global
`MeterProvider` (so they no-op cleanly when only `NoopBackend` is in play —
the default for tests per `project_observability_otel.md`).

## Dashboard queries

### Phoenix

Phoenix exposes OTel metrics through its OpenInference views. Add a panel per
counter using the metrics explorer:

```text
metric.name == "ingest.table.row_chunks_emitted" AND scope.name == "ragweave.ingest.table"
```

For the gate-ratio panel (summary-only share over the last hour):

```text
rate(ingest.table.summary_only_due_to_small[1h]) + rate(ingest.table.summary_only_due_to_uniform[1h])
  / rate(ingest.table.row_chunks_emitted[1h])
```

### Langfuse

Langfuse self-hosted v3 ingests OTel through `/api/public/otel/v1/metrics`.
Counters surface under Observations -> Metrics; filter by `name` (the OTel
metric name, not a Langfuse-specific alias):

```text
name:"ingest.table.row_chunks_emitted" scope:"ragweave.ingest.table"
```

Note: Langfuse stores the counters as cumulative aggregates — use a delta
window in the chart panel to see per-ingest movement.

## Verify in dashboard runbook

```text
1. Trigger an ingest of a table-heavy document (e.g. one of the synthetic
   fixtures under tests/fixtures/ingest/tables/).
2. Wait ~30s for the OTel periodic exporter to flush (default 60s; force a
   flush with `MeterProvider.force_flush()` if running ad-hoc).
3. In Phoenix/Langfuse, search for "ingest.table.row_chunks_emitted".
4. Expect at least one increment for any uniform, small-enough table; if the
   document contains an oversized or ragged table, the corresponding
   summary_only_due_to_* counter should also tick.
```

## Alerts worth considering

- **Silent regression — counters at 0 after a known-table-heavy ingest:**
  `ingest.table.row_chunks_emitted == 0 AND ingest.table.summary_only_due_to_small == 0 AND ingest.table.summary_only_due_to_uniform == 0` for a 1h window with ingest activity → likely the adaptive-chunking path was disabled or broken upstream.
- **All tables routed away from row chunks:** sudden spike in
  `summary_only_due_to_small` or `summary_only_due_to_uniform` without a
  matching `row_chunks_emitted` increment → likely a header-extraction
  regression or a configuration drift in `max_table_rows_for_row_chunks`.
- **Truncation creep:** sustained `ingest.table.summary_text_truncated` rate
  rising → the configured `table_summary_max_chars` may be too tight for the
  current document corpus.

## Test coverage

- **Unit:** `tests/ingest/test_docling_adaptive_table_chunking.py::TestAdaptiveTableChunkingMetrics` — verifies increments via `InMemoryMetricReader`.
- **Routing:** `tests/observability/test_table_counter_otlp_routing.py::TestTableCounterOTLPRouting` — verifies the same counters reach an OTLP-shape `MetricExporter` (the wire-format boundary the live OTLP exporters subclass).
