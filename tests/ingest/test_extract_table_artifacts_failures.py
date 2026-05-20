# @summary
# Failure-mode contract tests for _extract_table_artifacts in
# src/ingest/support/docling.py. Pins the silent-skip behavior: a single
# table raising during extraction must not abort sibling-table extraction,
# and the failure must be logged with table_id context.
# Exports: TestExtractTableArtifactsFailures
# Deps: src.ingest.support.docling, pytest, unittest.mock
# @end-summary

"""Unit tests pinning the failure-mode contract of _extract_table_artifacts.

The function is the single entry point for converting Docling tables into
``TableArtifact`` rows. Because Docling's table grid / heading-stack APIs
are stringy and version-skewed, the contract is: any exception raised while
extracting one table is caught, logged with the synthesized ``table_id``
(and ``self_ref`` if available), and extraction proceeds with the next
sibling table. All-tables-fail returns ``[]`` rather than raising.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable
from unittest.mock import MagicMock

import pytest

from src.ingest.support.docling import _extract_table_artifacts


# ---------------------------------------------------------------------------
# Mock builders
# ---------------------------------------------------------------------------


def _make_table_mock(
    *,
    self_ref: str,
    cells: list[list[str]] | None = None,
    caption: str = "",
    page_no: int = 1,
) -> Any:
    cells = cells or [["a", "b"], ["1", "2"]]
    grid = []
    for r_idx, row in enumerate(cells):
        grid_row = []
        for c in row:
            cell = MagicMock()
            cell.text = c
            cell.column_header = r_idx == 0
            cell.row_header = False
            grid_row.append(cell)
        grid.append(grid_row)

    tbl = MagicMock()
    tbl.self_ref = self_ref
    tbl.label = "table"
    tbl.data = MagicMock()
    tbl.data.grid = grid
    tbl.export_to_markdown = MagicMock(return_value="| a | b |\n| --- | --- |\n| 1 | 2 |")
    tbl.caption_text = MagicMock(return_value=caption)
    prov = MagicMock()
    prov.page_no = page_no
    tbl.prov = [prov]
    return tbl


class _ExplodingIter:
    """A grid-row whose iteration raises a configured exception.

    Used to model malformed Docling table data where the row container
    looks list-like but blows up on element access during the cell
    materialization comprehension in ``_table_to_cells``.
    """

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def __iter__(self):
        raise self._exc


def _make_exploding_table(*, self_ref: str, exc: Exception) -> Any:
    """A table whose grid-row iteration raises ``exc``.

    Mirrors a realistic Docling failure path: ``_table_to_cells`` reaches
    the per-row materialization comprehension, and the underlying row
    object raises on iteration. This is the failure mode that escapes
    ``_table_to_cells`` to the outer ``except Exception`` in
    ``_extract_table_artifacts``.
    """
    tbl = MagicMock()
    tbl.self_ref = self_ref
    tbl.label = "table"
    data = MagicMock()
    # Grid contains one row that raises on iteration.
    data.grid = [_ExplodingIter(exc)]
    tbl.data = data
    prov = MagicMock()
    prov.page_no = 1
    tbl.prov = [prov]
    return tbl


def _make_doc(items: Iterable[Any], tables: list[Any]) -> Any:
    doc = MagicMock()
    seq = [(it, 1) for it in items]
    doc.iterate_items = MagicMock(return_value=iter(seq))
    doc.tables = tables
    doc.pictures = []
    doc.export_to_markdown.return_value = ""
    return doc


# ---------------------------------------------------------------------------
# Failure-mode contract tests
# ---------------------------------------------------------------------------


class TestExtractTableArtifactsFailures:
    """Pin the silent-skip-with-logging contract of _extract_table_artifacts."""

    def test_table_raising_attribute_error_on_grid_does_not_block_siblings(self):
        """An AttributeError mid-extraction is caught; sibling tables still extract.

        Models the realistic failure of a malformed Docling table whose
        internal grid attribute access raises. The healthy sibling must
        still produce a TableArtifact.
        """
        bad = _make_exploding_table(
            self_ref="#/tables/0",
            exc=AttributeError("grid"),
        )
        good = _make_table_mock(self_ref="#/tables/1")
        doc = _make_doc(items=[bad, good], tables=[bad, good])

        artifacts = _extract_table_artifacts(doc)

        # One survivor: the healthy sibling. Its synthesized id reflects its
        # source-list index (idx=1 -> "table-2"), proving the loop continued
        # past the failed table without renumbering.
        assert len(artifacts) == 1
        assert artifacts[0].table_id == "table-2"

    def test_table_missing_from_heading_stack_map_falls_back_to_empty_section_path(self):
        """When a table's self_ref isn't in the heading-stack map, section_path=''.

        The walker resolves heading breadcrumbs only for tables it observed
        via iterate_items(). A table present in ``docling_document.tables``
        but absent from the iterate stream must not raise — it falls back
        to section_path=''.
        """
        # Table is in .tables but the iterate stream is empty (so heading map
        # has no entry for this self_ref).
        tbl = _make_table_mock(self_ref="#/tables/orphan")
        doc = _make_doc(items=[], tables=[tbl])

        artifacts = _extract_table_artifacts(doc)

        assert len(artifacts) == 1
        assert artifacts[0].section_path == ""

    def test_failed_table_extraction_logs_table_id_context(self, caplog):
        """The warning log entry on failure must include the synthesized table_id."""
        bad = _make_exploding_table(
            self_ref="#/tables/bad",
            exc=AttributeError("grid"),
        )
        good = _make_table_mock(self_ref="#/tables/ok")
        doc = _make_doc(items=[bad, good], tables=[bad, good])

        caplog.set_level(logging.WARNING)
        # Force-propagate in case other tests have muted the module logger.
        from src.ingest.support import docling as _dmod
        _dmod.logger.propagate = True
        _dmod.logger.setLevel(logging.WARNING)
        artifacts = _extract_table_artifacts(doc)

        # Sibling still extracted.
        assert len(artifacts) == 1
        # Log payload mentions the synthesized table_id for the failed entry.
        joined = "\n".join(rec.getMessage() for rec in caplog.records)
        assert "table-1" in joined, (
            f"expected synthesized table_id 'table-1' in log payload, got: {joined!r}"
        )

    def test_all_tables_failing_returns_empty_list_and_does_not_raise(self):
        """If every table raises, the function returns [] rather than propagating."""
        bad_a = _make_exploding_table(self_ref="#/tables/a", exc=AttributeError("grid"))
        bad_b = _make_exploding_table(self_ref="#/tables/b", exc=RuntimeError("boom"))
        doc = _make_doc(items=[bad_a, bad_b], tables=[bad_a, bad_b])

        # Must not raise.
        artifacts = _extract_table_artifacts(doc)

        assert artifacts == []
