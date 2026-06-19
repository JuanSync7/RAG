# @summary
# Tests for table/figure embedded-text breadcrumb prepend (parity with prose
# contextualize) and the summary-only table body fold (cell values embedded).
# Covers: summary/row/figure breadcrumb, wide-table markdown fold, small-table
# no-fold, config toggles, and chunk_body_text round-trip for the gate.
# @end-summary

"""Tests for RAG_INGESTION_TABLE_EMBED_PREPEND_SECTION_PATH and
RAG_INGESTION_TABLE_SUMMARY_INCLUDE_BODY behaviour.

Reuses the stubbed-HybridChunker harness from test_docling_adaptive_table_chunking.
"""

from types import SimpleNamespace

import pytest

from src.ingest.common.types import IngestionConfig
from src.ingest.common.shared import chunk_body_text
from src.ingest.support.docling import _figure_artifacts_to_chunks
from tests.ingest.test_docling_adaptive_table_chunking import (
    _make_table_artifact,
    _make_raw_chunk,
    _run_chunk,
)


def _summary(out):
    return next(c for c in out if c.extra_metadata.get("chunk_type") == "table_summary")


def _rows(out):
    return [c for c in out if c.extra_metadata.get("chunk_type") == "table_row"]


# ---------------------------------------------------------------------------
# Breadcrumb prepend (parity with prose contextualize)
# ---------------------------------------------------------------------------

def test_summary_text_prepends_breadcrumb_by_default():
    tbl = _make_table_artifact(section_path="Chapter 3 > 3.1 Registers", caption="Map")
    out = _run_chunk([_make_raw_chunk(tbl.markdown)], [tbl])
    summary = _summary(out)
    assert summary.text.startswith("Chapter 3\n3.1 Registers\n")
    # The breadcrumb strips back to the structured summary body for the gate.
    body = chunk_body_text(summary.text, summary.heading_path)
    assert body.startswith("Table: Map")


def test_row_text_prepends_breadcrumb_by_default():
    # Small uniform table -> row chunks emitted; each carries the breadcrumb.
    cells = [["Field", "Value"], ["AWVALID", "1"], ["AWREADY", "0"]]
    tbl = _make_table_artifact(cells=cells, section_path="Ch5 > 5.2 Signals", caption="Sig")
    out = _run_chunk([_make_raw_chunk(tbl.markdown)], [tbl])
    rows = _rows(out)
    assert rows, "expected row chunks for a small uniform table"
    for r in rows:
        assert r.text.startswith("Ch5\n5.2 Signals\n")


def test_breadcrumb_disabled_by_flag():
    tbl = _make_table_artifact(section_path="Chapter 3 > 3.1 Registers", caption="Map")
    cfg = IngestionConfig(table_embed_prepend_section_path=False)
    out = _run_chunk([_make_raw_chunk(tbl.markdown)], [tbl], config=cfg)
    summary = _summary(out)
    assert summary.text.startswith("Table: Map")  # no breadcrumb


# ---------------------------------------------------------------------------
# Summary-only body fold (cell values embedded for wide/tall tables)
# ---------------------------------------------------------------------------

def _wide_table():
    # 40 body rows -> exceeds max_table_rows_for_row_chunks (32) -> summary-only.
    cells = [["Field", "Value"]] + [[f"REG{i}", f"0x{i:02X}"] for i in range(40)]
    return _make_table_artifact(cells=cells, section_path="Ch7 > 7.1 Map", caption="Registers")


def test_summary_only_table_folds_cell_values_by_default():
    tbl = _wide_table()
    out = _run_chunk([_make_raw_chunk(tbl.markdown)], [tbl])
    summary = _summary(out)
    assert not _rows(out), "wide table should be summary-only (no row chunks)"
    # A specific cell value must now be present in the embedded summary text.
    assert "REG37" in summary.text
    assert "0x25" in summary.text  # 37 == 0x25


def test_summary_only_fold_disabled_by_flag():
    tbl = _wide_table()
    cfg = IngestionConfig(table_summary_include_body=False)
    out = _run_chunk([_make_raw_chunk(tbl.markdown)], [tbl], config=cfg)
    summary = _summary(out)
    assert "REG37" not in summary.text  # no cell values folded in
    # Structured summary still present (breadcrumb + Columns/Rows).
    assert "Rows: 40" in summary.text


def test_small_table_summary_not_body_folded():
    # Small uniform table emits row chunks (cells already covered) -> summary stays
    # compact (no raw markdown folded in).
    cells = [["Field", "Value"], ["AWVALID", "1"], ["AWREADY", "0"]]
    tbl = _make_table_artifact(cells=cells, section_path="Ch5 > 5.2", caption="Sig")
    out = _run_chunk([_make_raw_chunk(tbl.markdown)], [tbl])
    summary = _summary(out)
    # A body cell value lives only in the markdown / row chunks; a row-chunked
    # table's summary must stay compact (no markdown fold), so it must not appear.
    assert "AWVALID" not in summary.text
    assert _rows(out), "small uniform table should still emit row chunks"


# ---------------------------------------------------------------------------
# Figure chunks
# ---------------------------------------------------------------------------

def _fig(**kw):
    base = dict(
        caption="Block diagram of the GPIO controller",
        caption_label="Figure 3-1",
        section_path="Chapter 3 GPIO > 3.2 Architecture",
        document_id="doc1",
        self_ref="#/pictures/0",
        page_no=12,
        image_uri="",
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_figure_chunk_prepends_breadcrumb_by_default():
    out = _figure_artifacts_to_chunks([_fig()])
    assert len(out) == 1
    assert out[0].text.startswith("Chapter 3 GPIO\n3.2 Architecture\n")
    body = chunk_body_text(out[0].text, out[0].heading_path)
    assert "GPIO controller" in body


def test_figure_chunk_breadcrumb_disabled():
    out = _figure_artifacts_to_chunks([_fig()], prepend_section_path=False)
    assert not out[0].text.startswith("Chapter 3 GPIO")
    assert out[0].text.startswith("Figure 3-1")
