# @summary
# Tests that _extract_table_artifacts populates TableArtifact.section_path
# from the table's enclosing heading chain by walking docling_document.iterate_items().
# Covers single-heading, nested, no-heading, whitespace preservation, and an
# end-to-end DoclingParser.parse() -> chunk() pass-through to table chunks'
# heading_path metadata.
# Exports: TestTableSectionPathResolution, TestTableSectionPathEndToEnd
# Deps: src.ingest.support.docling, src.ingest.support.parser_base, pytest, unittest.mock
# @end-summary

"""Unit + e2e tests for table section_path resolution.

The production bug: ``_extract_table_artifacts`` previously hardcoded
``section_path=""`` and never walked the table's ancestor heading chain.
These tests pin the new contract: the heading stack at the time the table
appears in ``docling_document.iterate_items()`` becomes the table's
``section_path`` (joined with ``" > "``).
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Any, Generator, Iterable
from unittest.mock import MagicMock, patch

import pytest

from src.ingest.common.types import IngestionConfig, Runtime
from src.ingest.support import docling as _docling_mod
from src.ingest.support.docling import (
    DoclingParser,
    _extract_table_artifacts,
)


# ---------------------------------------------------------------------------
# Mock builders — produce items whose duck-typed surface matches Docling's
# NodeItem / SectionHeaderItem / TableItem just enough for the walker.
# ---------------------------------------------------------------------------


def _make_heading(text: str, *, level: int = 1, self_ref: str = "") -> Any:
    h = MagicMock()
    h.label = "section_header" if level >= 1 else "title"
    h.text = text
    h.level = level
    h.self_ref = self_ref or f"#/texts/{id(h)}"
    return h


def _make_title(text: str, *, self_ref: str = "") -> Any:
    t = MagicMock()
    t.label = "title"
    t.text = text
    t.self_ref = self_ref or f"#/texts/{id(t)}"
    return t


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


def _make_doc(items: Iterable[Any], tables: list[Any]) -> Any:
    """Build a mock DoclingDocument whose iterate_items() yields the given items.

    Each iterate_items() entry is the (node, level) tuple matching the real
    Docling API.
    """
    doc = MagicMock()
    seq = [(it, 1) for it in items]
    doc.iterate_items = MagicMock(return_value=iter(seq))
    doc.tables = tables
    doc.pictures = []
    doc.export_to_markdown.return_value = ""
    return doc


# ---------------------------------------------------------------------------
# Unit tests for _extract_table_artifacts section_path population
# ---------------------------------------------------------------------------


class TestTableSectionPathResolution:
    """_extract_table_artifacts must derive section_path from the heading chain."""

    def test_table_directly_under_single_h1_gets_that_heading_as_section_path(self):
        """A table preceded by exactly one H1 heading inherits that heading."""
        h1 = _make_heading("Chapter Title", level=1)
        tbl = _make_table_mock(self_ref="#/tables/0")
        doc = _make_doc(items=[h1, tbl], tables=[tbl])

        artifacts = _extract_table_artifacts(doc)

        assert len(artifacts) == 1
        assert artifacts[0].section_path == "Chapter Title"

    def test_table_under_nested_h1_h2_h3_gets_full_breadcrumb(self):
        """Nested headings (H1 > H2 > H3) compose into a ' > '-joined breadcrumb."""
        h1 = _make_heading("Chapter Title", level=1)
        h2 = _make_heading("Section", level=2)
        h3 = _make_heading("Subsection", level=3)
        tbl = _make_table_mock(self_ref="#/tables/0")
        doc = _make_doc(items=[h1, h2, h3, tbl], tables=[tbl])

        artifacts = _extract_table_artifacts(doc)

        assert len(artifacts) == 1
        assert artifacts[0].section_path == "Chapter Title > Section > Subsection"

    def test_table_with_no_enclosing_heading_emits_empty_section_path(self):
        """Tables that appear before any heading must keep section_path=''."""
        tbl = _make_table_mock(self_ref="#/tables/0")
        doc = _make_doc(items=[tbl], tables=[tbl])

        artifacts = _extract_table_artifacts(doc)

        assert len(artifacts) == 1
        assert artifacts[0].section_path == ""

    def test_heading_text_internal_whitespace_is_preserved(self):
        """The walker must not collapse or strip whitespace inside heading text."""
        h1 = _make_heading("Chapter  One  Title", level=1)
        tbl = _make_table_mock(self_ref="#/tables/0")
        doc = _make_doc(items=[h1, tbl], tables=[tbl])

        artifacts = _extract_table_artifacts(doc)
        assert artifacts[0].section_path == "Chapter  One  Title"

    def test_heading_stack_pops_when_same_or_shallower_heading_encountered(self):
        """A new H2 after H2>H3 chain pops the H3 sibling; H1 still leads."""
        h1 = _make_heading("Top", level=1)
        h2a = _make_heading("A", level=2)
        h3 = _make_heading("Deep", level=3)
        h2b = _make_heading("B", level=2)  # pops h3 and h2a
        tbl = _make_table_mock(self_ref="#/tables/0")
        doc = _make_doc(items=[h1, h2a, h3, h2b, tbl], tables=[tbl])

        artifacts = _extract_table_artifacts(doc)
        assert artifacts[0].section_path == "Top > B"

    def test_title_item_seeds_the_breadcrumb_chain(self):
        """A TitleItem (label='title') is treated as the root level of the chain."""
        title = _make_title("Document Title")
        h1 = _make_heading("Chapter", level=1)
        tbl = _make_table_mock(self_ref="#/tables/0")
        doc = _make_doc(items=[title, h1, tbl], tables=[tbl])

        artifacts = _extract_table_artifacts(doc)
        assert artifacts[0].section_path == "Document Title > Chapter"

    def test_multiple_tables_get_section_path_for_their_own_position(self):
        """Each table inherits the heading stack as it stood at its own position."""
        h1 = _make_heading("Intro", level=1)
        tbl_a = _make_table_mock(self_ref="#/tables/0")
        h2 = _make_heading("Details", level=2)
        tbl_b = _make_table_mock(self_ref="#/tables/1")
        doc = _make_doc(items=[h1, tbl_a, h2, tbl_b], tables=[tbl_a, tbl_b])

        artifacts = _extract_table_artifacts(doc)
        assert len(artifacts) == 2
        assert artifacts[0].section_path == "Intro"
        assert artifacts[1].section_path == "Intro > Details"

    def test_missing_iterate_items_falls_back_to_empty_section_path(self):
        """Documents lacking iterate_items API must not raise; section_path=''."""
        tbl = _make_table_mock(self_ref="#/tables/0")
        doc = MagicMock(spec=["tables", "pictures", "export_to_markdown"])
        doc.tables = [tbl]
        doc.pictures = []

        artifacts = _extract_table_artifacts(doc)
        assert len(artifacts) == 1
        assert artifacts[0].section_path == ""


# ---------------------------------------------------------------------------
# End-to-end: DoclingParser.parse() -> chunk() carries heading_path through
# to table_summary and table_row chunks.
# ---------------------------------------------------------------------------


@contextmanager
def _docling_sys_modules(mock_doc) -> Generator[MagicMock, None, None]:
    """Inject docling.* mocks so parse_with_docling's lazy imports resolve."""
    mock_result = MagicMock()
    mock_result.document = mock_doc

    mock_converter = MagicMock()
    mock_converter.convert.return_value = mock_result

    MockDocumentConverter = MagicMock(return_value=mock_converter)

    mock_dc_module = MagicMock()
    mock_dc_module.DocumentConverter = MockDocumentConverter
    mock_dc_module.PdfFormatOption = MagicMock()

    pipeline_options_module = MagicMock()
    pipeline_options_module.PdfPipelineOptions = MagicMock(return_value=MagicMock())
    pipeline_options_module.PictureDescriptionVlmEngineOptions = MagicMock()
    pipeline_options_module.RapidOcrOptions = MagicMock()
    pipeline_options_module.TableFormerMode = MagicMock(ACCURATE="ACCURATE", FAST="FAST")

    base_models_module = MagicMock()
    base_models_module.InputFormat = MagicMock(PDF="PDF")

    injected = {
        "docling": MagicMock(),
        "docling.document_converter": mock_dc_module,
        "docling.datamodel": MagicMock(),
        "docling.datamodel.pipeline_options": pipeline_options_module,
        "docling.datamodel.base_models": base_models_module,
    }
    original = {k: sys.modules.get(k) for k in injected}
    sys.modules.update(injected)
    try:
        yield MockDocumentConverter
    finally:
        for k, v in original.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


class TestTableSectionPathEndToEnd:
    """Parse -> chunk surfaces heading_path on table_summary chunks."""

    def test_parse_then_chunk_emits_heading_path_on_table_summary(self, tmp_path):
        """DoclingParser.parse() -> chunk() propagates section_path to the
        first table_summary chunk as a non-empty heading_path list."""
        h1 = _make_heading("Chapter Title", level=1)
        h2 = _make_heading("Section", level=2)
        cells = [
            ["Field", "Offset"],
            ["CTRL", "0x00"],
            ["STAT", "0x04"],
        ]
        tbl = _make_table_mock(self_ref="#/tables/0", cells=cells, caption="Reg")
        doc = _make_doc(items=[h1, h2, tbl], tables=[tbl])
        # parse_with_docling reads these on the produced document
        doc.export_to_markdown.return_value = "# Chapter Title\n\n## Section\n\nSome text.\n"
        doc.model_dump_json = MagicMock(return_value='{"body": {"children": []}}')

        config = IngestionConfig(
            enable_docling_parser=True,
            enable_adaptive_table_chunking=True,
        )

        source = tmp_path / "spec.pdf"
        source.write_bytes(b"%PDF-1.4 fake")

        # Build raw HybridChunker output (used by chunk() not by table chunks,
        # but chunk() always runs the hybrid path before appending table chunks).
        raw_chunk = MagicMock()
        raw_chunk.text = "Some text."
        raw_chunk.meta = MagicMock()
        raw_chunk.meta.headings = ["Chapter Title", "Section"]
        raw_chunk.meta.doc_items = []
        mock_chunker = MagicMock()
        mock_chunker.chunk = MagicMock(return_value=[raw_chunk])
        mock_hc_cls = MagicMock(return_value=mock_chunker)

        with _docling_sys_modules(doc), patch.dict(
            "sys.modules",
            {"docling_core.transforms.chunker": MagicMock(HybridChunker=mock_hc_cls)},
        ), patch.object(_docling_mod, "_get_or_build_tokenizer", return_value=MagicMock()):
            parser = DoclingParser()
            parse_result = parser.parse(source, config)

            # Confirm the unit-level invariant on TableArtifact first.
            assert len(parse_result.tables) == 1
            assert parse_result.tables[0].section_path == "Chapter Title > Section"

            chunks = parser.chunk(parse_result)

        table_summaries = [
            c
            for c in chunks
            if getattr(c, "extra_metadata", {}).get("chunk_type") == "table_summary"
        ]
        assert table_summaries, "expected at least one table_summary chunk"
        summary = table_summaries[0]
        assert summary.heading_path == ["Chapter Title", "Section"]
        assert summary.section_path == "Chapter Title > Section"
