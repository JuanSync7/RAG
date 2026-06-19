# @summary
# Tests for adaptive table chunking inside DoclingParser.chunk() — Workstream B.
# Covers: table-summary chunk emission, row-with-headers chunks for small uniform
# tables, large/headerless/ragged-table gating, disable flag pass-through,
# prose-mention false positives, chunk_index re-sequencing, metadata propagation.
# Exports: TestAdaptiveTableChunking
# Deps: src.ingest.support.docling, src.ingest.support.parser_base,
#       src.ingest.common.types, unittest.mock, pytest
# @end-summary

"""Adaptive table chunking tests for ``DoclingParser.chunk()``.

HybridChunker and the tokenizer loader are stubbed via ``patch.dict`` on
``sys.modules`` and ``patch.object`` on the module-local helper so these tests
run without docling installed. Inputs are constructed as lightweight doubles
of ``DoclingDocument`` + ``TableArtifact`` so the post-processing logic is
exercised in isolation.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.ingest.common.types import IngestionConfig
from src.ingest.support import docling as _docling_mod
from src.ingest.support.docling import DoclingParser
from src.ingest.support.parser_base import ParseResult, TableArtifact, PageRef


def _make_raw_chunk(text: str, headings: list[str] | None = None) -> MagicMock:
    chunk = MagicMock()
    chunk.text = text
    chunk.meta = MagicMock()
    chunk.meta.headings = list(headings or [])
    # Avoid any auto-spec'd page provenance attributes that might mislead
    # _page_ref_from_chunk_meta; explicit empty values.
    chunk.meta.doc_items = []
    return chunk


def _markdown_for(cells: list[list[str]]) -> str:
    width = max(len(r) for r in cells)
    norm = [r + [""] * (width - len(r)) for r in cells]
    header = "| " + " | ".join(norm[0]) + " |"
    sep = "| " + " | ".join(["---"] * width) + " |"
    body = "\n".join("| " + " | ".join(r) + " |" for r in norm[1:])
    return "\n".join([header, sep, body]).strip()


def _make_table_artifact(
    *,
    table_id: str = "table-1",
    cells: list[list[str]] | None = None,
    has_header: bool = True,
    caption: str = "Demo Caption",
    section_path: str = "Top > Sub",
    page_no: int = 3,
) -> TableArtifact:
    if cells is None:
        cells = [["H1", "H2"], ["a", "b"], ["c", "d"]]
    md = _markdown_for(cells)
    return TableArtifact(
        table_id=table_id,
        markdown=md,
        cells=cells,
        num_rows=len(cells),
        num_cols=max(len(r) for r in cells),
        has_header=has_header,
        section_path=section_path,
        caption=caption,
        page_ref=PageRef(page_no=page_no, page_label=str(page_no)),
    )


def _run_chunk(
    raw_chunks: list[Any],
    tables: list[TableArtifact],
    *,
    config: IngestionConfig | None = None,
) -> list[Any]:
    """Drive DoclingParser.chunk() with stubbed HybridChunker + tokenizer."""
    if config is None:
        config = IngestionConfig()
    mock_chunker = MagicMock()
    mock_chunker.chunk = MagicMock(return_value=raw_chunks)
    mock_hc_cls = MagicMock(return_value=mock_chunker)

    with patch.dict(
        "sys.modules",
        {
            "docling_core.transforms.chunker": MagicMock(HybridChunker=mock_hc_cls),
        },
    ), patch.object(
        _docling_mod, "_get_or_build_tokenizer", return_value=MagicMock()
    ):
        parser = DoclingParser()
        parser._docling_document = MagicMock()
        parser._max_tokens = 512
        parser._config = config
        return parser.chunk(
            ParseResult(
                markdown="",
                headings=[],
                has_figures=False,
                page_count=0,
                tables=tables,
            )
        )


class TestAdaptiveTableChunking:
    """Behavior of DoclingParser.chunk() table post-processing."""

    def test_small_table_emits_summary_and_row_chunks(self):
        """A small table with header yields 1 summary + N row chunks; original dropped."""
        cells = [["Col A", "Col B"], ["x1", "y1"], ["x2", "y2"], ["x3", "y3"]]
        tbl = _make_table_artifact(cells=cells)
        prose = _make_raw_chunk("Some intro prose.", headings=["Top", "Sub"])
        table_chunk = _make_raw_chunk(tbl.markdown, headings=["Top", "Sub"])

        out = _run_chunk([prose, table_chunk], [tbl])
        types = [c.extra_metadata.get("chunk_type") for c in out]
        assert types.count("table_summary") == 1
        assert types.count("table_row") == 3
        # original table-dominant chunk dropped → prose + summary + 3 rows = 5
        assert len(out) == 5
        # summary text composition
        summary = next(c for c in out if c.extra_metadata.get("chunk_type") == "table_summary")
        assert "Table: Demo Caption" in summary.text
        assert "Columns: Col A | Col B" in summary.text
        assert "Rows: 3" in summary.text
        # row chunk shape
        row0 = next(
            c for c in out
            if c.extra_metadata.get("chunk_type") == "table_row"
            and c.extra_metadata.get("table_row_index") == 0
        )
        assert "Col A: x1" in row0.text
        assert "Col B: y1" in row0.text

    def test_large_table_emits_only_summary(self):
        """Tables exceeding max_table_rows_for_row_chunks emit only the summary."""
        header = ["Col A", "Col B"]
        body = [[f"a{i}", f"b{i}"] for i in range(40)]
        cells = [header] + body
        tbl = _make_table_artifact(cells=cells)
        cfg = IngestionConfig()  # default max_table_rows_for_row_chunks = 32
        table_chunk = _make_raw_chunk(tbl.markdown)
        out = _run_chunk([table_chunk], [tbl], config=cfg)
        types = [c.extra_metadata.get("chunk_type") for c in out]
        assert types.count("table_summary") == 1
        assert types.count("table_row") == 0

    def test_headerless_table_emits_only_summary(self):
        """A table without a header row never emits row chunks."""
        cells = [["a", "b"], ["c", "d"]]
        tbl = _make_table_artifact(cells=cells, has_header=False, caption="")
        table_chunk = _make_raw_chunk(tbl.markdown)
        out = _run_chunk([table_chunk], [tbl])
        types = [c.extra_metadata.get("chunk_type") for c in out]
        assert types.count("table_summary") == 1
        assert types.count("table_row") == 0
        # caption omitted from summary text when empty
        summary = next(c for c in out if c.extra_metadata.get("chunk_type") == "table_summary")
        assert not summary.text.startswith("Table:")
        assert "Columns:" in summary.text

    def test_ragged_rows_emit_only_summary(self):
        """A table whose body rows have unequal length never emits row chunks."""
        cells = [["H1", "H2", "H3"], ["a", "b", "c"], ["d", "e"]]
        tbl = _make_table_artifact(cells=cells)
        table_chunk = _make_raw_chunk(tbl.markdown)
        out = _run_chunk([table_chunk], [tbl])
        types = [c.extra_metadata.get("chunk_type") for c in out]
        assert types.count("table_summary") == 1
        assert types.count("table_row") == 0

    def test_disabled_flag_preserves_legacy_output(self):
        """enable_adaptive_table_chunking=False leaves HybridChunker output untouched."""
        cells = [["H1", "H2"], ["a", "b"], ["c", "d"]]
        tbl = _make_table_artifact(cells=cells)
        cfg = IngestionConfig(enable_adaptive_table_chunking=False)
        prose = _make_raw_chunk("intro")
        table_chunk = _make_raw_chunk(tbl.markdown)
        out = _run_chunk([prose, table_chunk], [tbl], config=cfg)
        assert len(out) == 2
        assert all(
            c.extra_metadata.get("chunk_type") not in ("table_summary", "table_row")
            for c in out
        )

    def test_prose_mention_of_caption_is_not_removed(self):
        """Chunks that only mention the caption (no signature row) are preserved."""
        cells = [["H1", "H2"], ["alpha-key", "beta-val"]]
        tbl = _make_table_artifact(cells=cells, caption="Pricing")
        # signature row begins with "| H1 | H2 |" — prose mentions caption only
        prose = _make_raw_chunk("See Pricing table below for details.")
        table_chunk = _make_raw_chunk(tbl.markdown)
        out = _run_chunk([prose, table_chunk], [tbl])
        # prose preserved
        assert any(c.text == "See Pricing table below for details." for c in out)
        # table-dominant chunk dropped
        assert not any(c.text == tbl.markdown for c in out)

    def test_chunk_index_is_contiguous_and_monotonic(self):
        """Final chunk_index values are 0..N-1 in order."""
        cells = [["H1", "H2"], ["a", "b"], ["c", "d"]]
        tbl = _make_table_artifact(cells=cells)
        prose1 = _make_raw_chunk("first")
        table_chunk = _make_raw_chunk(tbl.markdown)
        prose2 = _make_raw_chunk("last")
        out = _run_chunk([prose1, table_chunk, prose2], [tbl])
        indices = [c.chunk_index for c in out]
        assert indices == list(range(len(out)))

    def test_table_group_id_joins_summary_and_rows(self):
        """All chunks from one table share table_group_id; distinct tables get distinct ids."""
        cells_a = [["H1", "H2"], ["a", "b"], ["c", "d"]]
        cells_b = [["X", "Y"], ["1", "2"]]
        tbl_a = _make_table_artifact(
            table_id="table-1", cells=cells_a, section_path="A"
        )
        tbl_a.self_ref = "#/tables/0"
        tbl_b = _make_table_artifact(
            table_id="table-2", cells=cells_b, section_path="B"
        )
        tbl_b.self_ref = "#/tables/1"

        chunk_a = _make_raw_chunk(tbl_a.markdown)
        chunk_b = _make_raw_chunk(tbl_b.markdown)
        out = _run_chunk([chunk_a, chunk_b], [tbl_a, tbl_b])

        groups_a = {
            c.extra_metadata.get("table_group_id")
            for c in out
            if c.extra_metadata.get("table_id") == "table-1"
        }
        groups_b = {
            c.extra_metadata.get("table_group_id")
            for c in out
            if c.extra_metadata.get("table_id") == "table-2"
        }
        # Each table's chunks share one non-empty group id, and the two
        # tables' group ids differ.
        assert groups_a == {"#/tables/0"}
        assert groups_b == {"#/tables/1"}
        # Every adaptive chunk carries the field.
        for c in out:
            if c.extra_metadata.get("chunk_type") in ("table_summary", "table_row"):
                assert c.extra_metadata.get("table_group_id")

    def test_table_group_id_falls_back_to_table_id_when_self_ref_missing(self):
        """When parser has no self_ref, group_id falls back to table_id (still stable within doc)."""
        cells = [["H1", "H2"], ["a", "b"]]
        tbl = _make_table_artifact(cells=cells)
        # default TableArtifact.self_ref == ""
        assert tbl.self_ref == ""
        table_chunk = _make_raw_chunk(tbl.markdown)
        out = _run_chunk([table_chunk], [tbl])
        for c in out:
            if c.extra_metadata.get("chunk_type") in ("table_summary", "table_row"):
                assert c.extra_metadata.get("table_group_id") == "table-1"

    def test_oversized_table_summary_text_is_truncated_under_cap(self):
        """When summary text would exceed table_summary_max_chars, it is capped
        with a clear truncation marker; full markdown is preserved on metadata.

        Guards against 200+ row datasheet tables (or pathologically wide
        column headers) blowing past the embedder's max-input limit.
        """
        # Build many wide headers so the "Columns: ..." line dominates the
        # summary length and forces truncation under a tight cap.
        headers = [f"col_{i:03d}_descriptor" for i in range(120)]
        body = [[f"v{i}" for i in range(120)] for _ in range(250)]
        cells = [headers] + body
        tbl = _make_table_artifact(
            cells=cells, caption="Wide datasheet table"
        )
        cap = 512
        cfg = IngestionConfig(table_summary_max_chars=cap)
        table_chunk = _make_raw_chunk(tbl.markdown)
        out = _run_chunk([table_chunk], [tbl], config=cfg)
        summary = next(
            c for c in out if c.extra_metadata.get("chunk_type") == "table_summary"
        )
        # Embedded text capped.
        assert len(summary.text) <= cap
        # Clear truncation marker present.
        assert "[truncated" in summary.text
        assert summary.text.rstrip().endswith("chars]")
        # table_markdown on metadata is unchanged (full markdown intact).
        assert summary.extra_metadata["table_markdown"] == tbl.markdown
        # Sanity: full markdown is much larger than the cap.
        assert len(tbl.markdown) > cap

    def test_small_summary_not_truncated_when_under_cap(self):
        """Summaries that fit under the cap are returned byte-for-byte
        unchanged — no false truncation, no marker appended."""
        cells = [["Col A", "Col B"], ["x", "y"]]
        tbl = _make_table_artifact(cells=cells, caption="Tiny")
        cfg = IngestionConfig(table_summary_max_chars=4000)
        table_chunk = _make_raw_chunk(tbl.markdown)
        out = _run_chunk([table_chunk], [tbl], config=cfg)
        summary = next(
            c for c in out if c.extra_metadata.get("chunk_type") == "table_summary"
        )
        # No truncation marker.
        assert "[truncated" not in summary.text
        # Exact byte-for-byte composition. The heading breadcrumb ("Top\nSub")
        # is prepended to the embedded text (parity with prose contextualize);
        # this is a small, header-bearing table so its body is NOT folded in.
        expected = "Top\nSub\nTable: Tiny\nColumns: Col A | Col B\nRows: 1"
        assert summary.text == expected

    def test_summary_truncation_disabled_when_cap_zero(self):
        """A cap of 0 disables truncation entirely — long summaries pass through."""
        headers = [f"hdr_{i:03d}" for i in range(200)]
        cells = [headers, [f"v{i}" for i in range(200)]]
        tbl = _make_table_artifact(cells=cells, caption="No cap")
        cfg = IngestionConfig(table_summary_max_chars=0)
        table_chunk = _make_raw_chunk(tbl.markdown)
        out = _run_chunk([table_chunk], [tbl], config=cfg)
        summary = next(
            c for c in out if c.extra_metadata.get("chunk_type") == "table_summary"
        )
        assert "[truncated" not in summary.text
        # The Columns: line alone is well over 1000 chars; verify pass-through.
        assert len(summary.text) > 1000

    def test_truncation_is_deterministic_for_chunk_id_idempotency(self):
        """Two runs over the same table produce byte-identical summary text,
        so build_table_chunk_id (which hashes structural keys, not text) and
        any text-hashing downstream both stay stable across re-ingest."""
        headers = [f"col_{i:03d}" for i in range(100)]
        cells = [headers, [f"v{i}" for i in range(100)]]
        tbl_a = _make_table_artifact(cells=cells, caption="Det")
        tbl_b = _make_table_artifact(cells=cells, caption="Det")
        cfg = IngestionConfig(table_summary_max_chars=256)
        out_a = _run_chunk([_make_raw_chunk(tbl_a.markdown)], [tbl_a], config=cfg)
        out_b = _run_chunk([_make_raw_chunk(tbl_b.markdown)], [tbl_b], config=cfg)
        sum_a = next(
            c for c in out_a if c.extra_metadata.get("chunk_type") == "table_summary"
        )
        sum_b = next(
            c for c in out_b if c.extra_metadata.get("chunk_type") == "table_summary"
        )
        assert sum_a.text == sum_b.text
        assert "[truncated" in sum_a.text

    def test_metadata_propagates_to_summary_and_row_chunks(self):
        """table_id, page_ref, heading_path appear on both summary and row chunks."""
        cells = [["H1", "H2"], ["a", "b"]]
        tbl = _make_table_artifact(
            cells=cells, section_path="Alpha > Beta", page_no=7
        )
        table_chunk = _make_raw_chunk(tbl.markdown)
        out = _run_chunk([table_chunk], [tbl])
        target = [c for c in out if c.extra_metadata.get("chunk_type") in
                  ("table_summary", "table_row")]
        assert len(target) == 2  # 1 summary + 1 row
        for c in target:
            assert c.extra_metadata.get("table_id") == "table-1"
            assert c.heading_path == ["Alpha", "Beta"]
            assert c.section_path == "Alpha > Beta"
            assert c.page_ref is not None
            assert c.page_ref.page_no == 7


# ---------------------------------------------------------------------------
# Observability: OTel counter coverage for the gate decisions
# ---------------------------------------------------------------------------


class _DeltaReader:
    """Wrap an InMemoryMetricReader to expose post-baseline deltas.

    OTel SDK only allows one global MeterProvider per process, so we install
    a single module-scoped provider and snapshot a baseline at the start of
    each test. ``get_metrics_data`` here returns cumulative state; the
    helper below subtracts the baseline so each test sees only its own
    contribution.
    """

    def __init__(self, inner: Any, baseline: dict[str, int]) -> None:
        self._inner = inner
        self._baseline = baseline

    def collect(self) -> dict[str, int]:
        current = _drain_counters(self._inner)
        return {k: v - self._baseline.get(k, 0) for k, v in current.items()}


def _drain_counters(reader: Any) -> dict[str, int]:
    """Read cumulative ingest.table.* counter totals from an InMemoryMetricReader."""
    data = reader.get_metrics_data()
    out: dict[str, int] = {}
    if data is None:
        return out
    for resource_metric in data.resource_metrics:
        for scope_metric in resource_metric.scope_metrics:
            for metric in scope_metric.metrics:
                if not metric.name.startswith("ingest.table."):
                    continue
                total = 0
                metric_data = getattr(metric, "data", None)
                points = (
                    getattr(metric_data, "data_points", []) if metric_data else []
                )
                for point in points:
                    total += int(getattr(point, "value", 0))
                out[metric.name] = out.get(metric.name, 0) + total
    return out


@pytest.fixture(scope="module")
def _otel_module_reader():
    """Install a single MeterProvider + InMemoryMetricReader for this module.

    OTel forbids replacing the global MeterProvider, so the provider is
    installed once per module and shared across tests via the per-test
    ``otel_metric_reader`` fixture which snapshots a baseline.
    """
    from opentelemetry import metrics as otel_metrics
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    from src.ingest.support import table_metrics as _tm

    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    # ``set_meter_provider`` only honours the first call per process; clear the
    # internal flag so the SDK provider replaces any proxy already installed.
    otel_metrics._PROXY_METER_PROVIDER = None  # type: ignore[attr-defined]
    otel_metrics.set_meter_provider(provider)
    _tm._reset_for_tests()
    try:
        yield reader
    finally:
        try:
            provider.shutdown()
        except Exception:
            pass
        _tm._reset_for_tests()


@pytest.fixture
def otel_metric_reader(_otel_module_reader):
    """Return a delta-collecting wrapper so each test sees only its own counter activity."""
    baseline = _drain_counters(_otel_module_reader)
    return _DeltaReader(_otel_module_reader, baseline)


class TestAdaptiveTableChunkingMetrics:
    """OTel counter coverage for the small / uniform gate decisions."""

    def test_large_varied_table_increments_row_chunks_emitted(self, otel_metric_reader):
        """A small-enough uniform table fires the row_chunks_emitted counter exactly once."""
        cells = [["H1", "H2"], ["a", "b"], ["c", "d"], ["e", "f"]]
        tbl = _make_table_artifact(cells=cells)
        table_chunk = _make_raw_chunk(tbl.markdown)
        _run_chunk([table_chunk], [tbl])

        values = otel_metric_reader.collect()
        assert values.get("ingest.table.row_chunks_emitted") == 1
        assert values.get("ingest.table.summary_only_due_to_small", 0) == 0
        assert values.get("ingest.table.summary_only_due_to_uniform", 0) == 0

    def test_oversized_table_increments_small_gate(self, otel_metric_reader):
        """A table over max_table_rows_for_row_chunks fires the small-gate counter."""
        header = ["Col A", "Col B"]
        body = [[f"a{i}", f"b{i}"] for i in range(40)]
        cells = [header] + body
        tbl = _make_table_artifact(cells=cells)
        table_chunk = _make_raw_chunk(tbl.markdown)
        _run_chunk([table_chunk], [tbl])

        values = otel_metric_reader.collect()
        assert values.get("ingest.table.summary_only_due_to_small") == 1
        assert values.get("ingest.table.row_chunks_emitted", 0) == 0
        assert values.get("ingest.table.summary_only_due_to_uniform", 0) == 0

    def test_ragged_table_increments_uniform_gate(self, otel_metric_reader):
        """A ragged-row table fires the uniform-gate counter (not the small gate)."""
        cells = [["H1", "H2", "H3"], ["a", "b", "c"], ["d", "e"]]
        tbl = _make_table_artifact(cells=cells)
        table_chunk = _make_raw_chunk(tbl.markdown)
        _run_chunk([table_chunk], [tbl])

        values = otel_metric_reader.collect()
        assert values.get("ingest.table.summary_only_due_to_uniform") == 1
        assert values.get("ingest.table.summary_only_due_to_small", 0) == 0
        assert values.get("ingest.table.row_chunks_emitted", 0) == 0

    def test_three_synthetic_tables_each_increment_their_own_counter(
        self, otel_metric_reader
    ):
        """One large+varied, one oversized, one ragged → each counter +1 exactly."""
        # 1) small + uniform → row_chunks_emitted
        cells_ok = [["H1", "H2"], ["a", "b"], ["c", "d"]]
        tbl_ok = _make_table_artifact(
            table_id="ok", cells=cells_ok, section_path="A"
        )
        tbl_ok.self_ref = "#/tables/0"

        # 2) too many rows → small_gate
        header = ["Col A", "Col B"]
        body = [[f"a{i}", f"b{i}"] for i in range(40)]
        cells_big = [header] + body
        tbl_big = _make_table_artifact(
            table_id="big", cells=cells_big, section_path="B"
        )
        tbl_big.self_ref = "#/tables/1"

        # 3) ragged rows → uniform_gate
        cells_ragged = [["H1", "H2", "H3"], ["a", "b", "c"], ["d", "e"]]
        tbl_ragged = _make_table_artifact(
            table_id="ragged", cells=cells_ragged, section_path="C"
        )
        tbl_ragged.self_ref = "#/tables/2"

        chunks = [
            _make_raw_chunk(tbl_ok.markdown),
            _make_raw_chunk(tbl_big.markdown),
            _make_raw_chunk(tbl_ragged.markdown),
        ]
        _run_chunk(chunks, [tbl_ok, tbl_big, tbl_ragged])

        values = otel_metric_reader.collect()
        assert values.get("ingest.table.row_chunks_emitted") == 1
        assert values.get("ingest.table.summary_only_due_to_small") == 1
        assert values.get("ingest.table.summary_only_due_to_uniform") == 1

    def test_summary_text_truncated_counter_fires_on_truncation(
        self, otel_metric_reader
    ):
        """When a summary is capped, the truncated counter increments; otherwise it stays at 0."""
        # First: a table whose summary fits comfortably → no truncation.
        cells_small = [["Col A", "Col B"], ["x", "y"]]
        tbl_small = _make_table_artifact(cells=cells_small, caption="Tiny")
        cfg_loose = IngestionConfig(table_summary_max_chars=4000)
        _run_chunk(
            [_make_raw_chunk(tbl_small.markdown)], [tbl_small], config=cfg_loose
        )

        # Second: a wide table under a tight cap → truncation fires.
        headers = [f"col_{i:03d}_descriptor" for i in range(120)]
        body = [[f"v{i}" for i in range(120)] for _ in range(5)]
        cells_wide = [headers] + body
        tbl_wide = _make_table_artifact(
            table_id="wide", cells=cells_wide, caption="Wide"
        )
        cfg_tight = IngestionConfig(table_summary_max_chars=256)
        _run_chunk(
            [_make_raw_chunk(tbl_wide.markdown)], [tbl_wide], config=cfg_tight
        )

        values = otel_metric_reader.collect()
        assert values.get("ingest.table.summary_text_truncated") == 1

    def test_counters_are_noop_without_sdk_provider(self):
        """With no SDK MeterProvider installed (the default), increments must not crash."""
        from opentelemetry import metrics as otel_metrics

        from src.ingest.support import table_metrics as _tm

        prior_provider = otel_metrics.get_meter_provider()
        # Force the proxy / default no-op provider by clearing any SDK provider.
        otel_metrics._METER_PROVIDER = None  # type: ignore[attr-defined]
        _tm._reset_for_tests()
        try:
            cells = [["H1", "H2"], ["a", "b"], ["c", "d"]]
            tbl = _make_table_artifact(cells=cells)
            # Should not raise, regardless of routing.
            _run_chunk([_make_raw_chunk(tbl.markdown)], [tbl])
        finally:
            otel_metrics._METER_PROVIDER = prior_provider  # type: ignore[attr-defined]
            _tm._reset_for_tests()
