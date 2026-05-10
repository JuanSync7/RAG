"""Tests for TableArtifact extraction and chunk annotation (FR-3211).

Contracts:
- ParseResult.tables defaults to empty for parsers without table support.
- DoclingParser populates ParseResult.tables from DoclingDocument.tables.
- chunking_node marks chunks overlapping a table with chunk_type="table"
  and attaches structured cells.
- Non-table chunks default to chunk_type="text".
- chunking_node tolerates parsers that emit no tables (legacy compatibility).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.ingest.support.parser_base import ParseResult, TableArtifact


def test_table_artifact_dataclass_fields():
    t = TableArtifact(
        table_id="table-1",
        markdown="| a | b |\n|---|---|\n| 1 | 2 |",
        cells=[["a", "b"], ["1", "2"]],
        num_rows=2,
        num_cols=2,
        has_header=True,
        section_path="Results",
        caption="My caption",
    )
    assert t.table_id == "table-1"
    assert t.cells[1] == ["1", "2"]
    assert t.has_header is True
    assert t.caption == "My caption"


def test_parse_result_tables_default_empty():
    pr = ParseResult(markdown="x", headings=[], has_figures=False, page_count=0)
    assert pr.tables == []


def test_parse_result_tables_populated():
    tbl = TableArtifact(
        table_id="table-1", markdown="| a |", cells=[["a"]],
        num_rows=1, num_cols=1,
    )
    pr = ParseResult(
        markdown="x", headings=[], has_figures=False, page_count=0, tables=[tbl]
    )
    assert len(pr.tables) == 1
    assert pr.tables[0].table_id == "table-1"


def _fake_docling_table(cells, header=False, markdown=None, caption=""):
    """Build a duck-typed table object that mimics docling TableItem."""
    cell_objs = []
    for row in cells:
        row_cells = []
        for col_idx, text in enumerate(row):
            row_cells.append(SimpleNamespace(
                text=text,
                column_header=(header and row is cells[0]),
                row_header=False,
            ))
        cell_objs.append(row_cells)
    data = SimpleNamespace(grid=cell_objs)
    tbl = MagicMock()
    tbl.data = data
    if markdown is not None:
        tbl.export_to_markdown.return_value = markdown
    else:
        tbl.export_to_markdown.return_value = (
            "| " + " | ".join(cells[0]) + " |\n"
            + "| " + " | ".join(["---"] * len(cells[0])) + " |\n"
            + "\n".join("| " + " | ".join(r) + " |" for r in cells[1:])
        )
    tbl.caption_text.return_value = caption
    return tbl


def test_extract_table_artifacts_from_docling_document():
    from src.ingest.support.docling import _extract_table_artifacts

    fake_tbl = _fake_docling_table([["A", "B"], ["1", "2"]], header=True)
    doc = SimpleNamespace(tables=[fake_tbl])

    artifacts = _extract_table_artifacts(doc)
    assert len(artifacts) == 1
    a = artifacts[0]
    assert a.table_id == "table-1"
    assert a.num_rows == 2
    assert a.num_cols == 2
    assert a.has_header is True
    assert a.cells == [["A", "B"], ["1", "2"]]


def test_extract_table_artifacts_handles_none_document():
    from src.ingest.support.docling import _extract_table_artifacts
    assert _extract_table_artifacts(None) == []


def test_extract_table_artifacts_handles_no_tables_attr():
    from src.ingest.support.docling import _extract_table_artifacts
    assert _extract_table_artifacts(SimpleNamespace()) == []


def test_extract_table_artifacts_skips_failing_table():
    from src.ingest.support.docling import _extract_table_artifacts

    bad_tbl = MagicMock()
    bad_tbl.data = SimpleNamespace(grid=None)  # produces empty cells -> skipped
    good_tbl = _fake_docling_table([["x"]])
    doc = SimpleNamespace(tables=[bad_tbl, good_tbl])

    artifacts = _extract_table_artifacts(doc)
    assert len(artifacts) == 1
    assert artifacts[0].cells == [["x"]]


def test_match_table_artifact_signature_match():
    from src.ingest.embedding.nodes.chunking import _match_table_artifact

    tbl = TableArtifact(
        table_id="table-1",
        markdown="| Year | Count |\n|---|---|\n| 2024 | 42 |",
        cells=[["Year", "Count"], ["2024", "42"]],
        num_rows=2, num_cols=2, has_header=True,
    )
    chunk_text = "Some intro text.\n| Year | Count |\n|---|---|\n| 2024 | 42 |"
    matched = _match_table_artifact(chunk_text, [tbl])
    assert matched is tbl


def test_match_table_artifact_no_match():
    from src.ingest.embedding.nodes.chunking import _match_table_artifact

    tbl = TableArtifact(
        table_id="table-1",
        markdown="| Year | Count |\n|---|---|\n| 2024 | 42 |",
        cells=[["Year", "Count"]], num_rows=1, num_cols=2,
    )
    assert _match_table_artifact("plain prose", [tbl]) is None


def test_match_table_artifact_empty_table_list():
    from src.ingest.embedding.nodes.chunking import _match_table_artifact
    assert _match_table_artifact("anything", []) is None
