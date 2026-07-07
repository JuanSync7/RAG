"""Tests for TableArtifact.document_id propagation.

Contracts:
- ``TableArtifact`` accepts an optional ``document_id`` (defaults to "")
  so legacy callers keep working.
- ``_extract_table_artifacts`` stamps the caller-supplied ``document_id``
  on every emitted artifact.
- ``_make_table_chunks`` exposes ``document_id`` in both the summary chunk
  metadata and per-row chunk metadata so the value survives the chunk
  → Weaviate roundtrip.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.ingest.support.parser_base import TableArtifact


# ---------------------------------------------------------------------------
# Dataclass surface
# ---------------------------------------------------------------------------

def test_table_artifact_accepts_document_id():
    t = TableArtifact(
        table_id="table-1",
        markdown="| a |",
        cells=[["a"]],
        num_rows=1,
        num_cols=1,
        document_id="esp32-s3_datasheet",
    )
    assert t.document_id == "esp32-s3_datasheet"


def test_table_artifact_document_id_defaults_to_empty_string():
    # Backward compatibility: existing callers must not need to supply it.
    t = TableArtifact(
        table_id="table-1",
        markdown="| a |",
        cells=[["a"]],
        num_rows=1,
        num_cols=1,
    )
    assert t.document_id == ""


# ---------------------------------------------------------------------------
# Extraction propagation
# ---------------------------------------------------------------------------

def _fake_docling_table(cells):
    cell_objs = []
    for row in cells:
        cell_objs.append([
            SimpleNamespace(text=text, column_header=False, row_header=False)
            for text in row
        ])
    data = SimpleNamespace(grid=cell_objs)
    tbl = MagicMock()
    tbl.data = data
    tbl.export_to_markdown.return_value = "| a |\n|---|\n| 1 |"
    tbl.caption_text.return_value = ""
    return tbl


def test_extract_table_artifacts_stamps_document_id():
    from src.ingest.support.docling import _extract_table_artifacts

    t1 = _fake_docling_table([["A", "B"], ["1", "2"]])
    t2 = _fake_docling_table([["X"], ["y"]])
    doc = SimpleNamespace(tables=[t1, t2])

    arts = _extract_table_artifacts(doc, document_id="datasheet_v1")
    assert len(arts) == 2
    assert all(a.document_id == "datasheet_v1" for a in arts)


def test_extract_table_artifacts_defaults_document_id_empty():
    from src.ingest.support.docling import _extract_table_artifacts

    t1 = _fake_docling_table([["A"], ["1"]])
    arts = _extract_table_artifacts(SimpleNamespace(tables=[t1]))
    assert arts and arts[0].document_id == ""


# ---------------------------------------------------------------------------
# Chunk metadata carries document_id
# ---------------------------------------------------------------------------

def test_table_chunks_carry_document_id_in_metadata():
    """Drive the table-chunk emitter and confirm document_id is on every chunk.

    The emitter lives inside ``_apply_table_chunk_policy`` as a closure; the
    simplest way to reach it without mocking the whole pipeline is to call
    the public policy entrypoint with one synthetic chunk + one TableArtifact
    that has a recognisable signature, so the artifact's chunks get spliced in.
    """
    from src.ingest.support.docling import _apply_adaptive_table_chunking
    from src.ingest.support.parser_base import Chunk

    tbl = TableArtifact(
        table_id="table-1",
        markdown="| Year | Count |\n|---|---|\n| 2024 | 42 |\n| 2025 | 7 |",
        cells=[["Year", "Count"], ["2024", "42"], ["2025", "7"]],
        num_rows=3,
        num_cols=2,
        has_header=True,
        section_path="Results",
        caption="Annual counts",
        self_ref="#/tables/0",
        document_id="my_doc_42",
    )
    # A regular chunk so the policy has something to walk; table chunks get
    # appended at the end when no in-stream table signature matches.
    base = Chunk(
        text="some prose",
        section_path="Intro",
        heading="Intro",
        heading_level=1,
        chunk_index=0,
        extra_metadata={"chunk_type": "text"},
    )

    cfg = SimpleNamespace(
        max_table_rows_for_row_chunks=50,
        max_table_cols_for_row_chunks=20,
        table_summary_max_chars=4000,
    )
    out = _apply_adaptive_table_chunking([base], [tbl], cfg)

    table_chunks = [c for c in out if c.extra_metadata.get("chunk_type", "").startswith("table_")]
    assert table_chunks, "expected at least one table chunk to be emitted"
    for c in table_chunks:
        assert c.extra_metadata.get("document_id") == "my_doc_42"
