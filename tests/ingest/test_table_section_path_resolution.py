# @summary
# Tests that _extract_table_artifacts populates TableArtifact.section_path
# from the table's enclosing heading chain by walking docling_document.iterate_items(),
# and that DoclingParser.parse() -> chunk() stamps that heading/section_path onto
# the emitted table_row block chunks. Also exercises the token-budget row-block
# chunking contract of _apply_adaptive_table_chunking directly: greedy lossless
# row packing, header/separator/breadcrumb/caption restated per block, atomic
# over-budget rows, header-only tables, ragged rows, and per-chunk metadata.
# Exports: TestTableSectionPathResolution, TestTableSectionPathEndToEnd, TestTableRowBlockChunking
# Deps: src.ingest.support.docling, src.ingest.support.parser_base, pytest, unittest.mock
# @end-summary

"""Unit + e2e tests for table section_path resolution and row-block chunking.

The production bug: ``_extract_table_artifacts`` previously hardcoded
``section_path=""`` and never walked the table's ancestor heading chain.
The section-path tests below pin that contract: the heading stack at the time
the table appears in ``docling_document.iterate_items()`` becomes the table's
``section_path`` (joined with ``" > "``).

The chunking tests pin the *current* table-chunking contract (after the
token-budget refactor): ``_apply_adaptive_table_chunking`` emits ONLY
``chunk_type="table_row"`` blocks — there is no longer any ``table_summary``
chunk. Each table is split into one or more row blocks that each restate the
breadcrumb + caption + markdown header + separator and greedily pack as many
whole body rows as fit under the single token budget
(``hybrid_chunker_max_tokens``). The split is lossless: every body row appears
in exactly one block, a single over-budget row gets its own atomic block, and a
header-only table emits exactly one chunk.
"""

from __future__ import annotations

import sys
import types
from contextlib import contextmanager
from typing import Any, Generator, Iterable
from unittest.mock import MagicMock, patch

import pytest

from src.ingest.common.types import IngestionConfig, Runtime
from src.ingest.support import docling as _docling_mod
from src.ingest.support.docling import (
    DoclingParser,
    _apply_adaptive_table_chunking,
    _extract_table_artifacts,
)
from src.ingest.support.parser_base import TableArtifact


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
# to the emitted table_row block chunks.
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
    """Parse -> chunk surfaces heading_path on the emitted table_row block chunks."""

    def test_parse_then_chunk_emits_heading_path_on_table_row_block(self, tmp_path):
        """DoclingParser.parse() -> chunk() propagates section_path through to the
        table_row block chunk(s) as a non-empty heading_path list — and emits NO
        table_summary chunk (that chunk_type was removed in the token-budget
        refactor)."""
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

        # The refactor removed the table_summary chunk_type entirely — nothing
        # may emit one anymore.
        chunk_types = {
            getattr(c, "extra_metadata", {}).get("chunk_type") for c in chunks
        }
        assert "table_summary" not in chunk_types

        table_rows = [
            c
            for c in chunks
            if getattr(c, "extra_metadata", {}).get("chunk_type") == "table_row"
        ]
        assert table_rows, "expected at least one table_row chunk"
        # The heading/section breadcrumb resolved on the TableArtifact must be
        # stamped onto the emitted row block chunk(s).
        for row_chunk in table_rows:
            assert row_chunk.heading_path == ["Chapter Title", "Section"]
            assert row_chunk.section_path == "Chapter Title > Section"
            # And the breadcrumb is prepended into the embedded text itself
            # (table_embed_prepend_section_path defaults True).
            assert row_chunk.text.startswith("Chapter Title\nSection\n")

        # The small table fits one block; its body rows are all present.
        joined = "\n".join(c.text for c in table_rows)
        assert "| CTRL | 0x00 |" in joined
        assert "| STAT | 0x04 |" in joined


# ---------------------------------------------------------------------------
# Token-budget row-block chunking contract for _apply_adaptive_table_chunking.
#
# These tests replace the deleted summary-emission / summary-truncation /
# row+col-gate / group-size tests: that behavior no longer exists. The new
# contract is a single token budget (hybrid_chunker_max_tokens) that greedily
# packs whole body rows into one or more lossless table_row blocks, restating
# the breadcrumb + caption + header + separator in each block.
# ---------------------------------------------------------------------------


def _make_table_artifact(
    *,
    cells: list[list[str]],
    section_path: str = "",
    caption: str = "",
    self_ref: str = "#/tables/0",
    table_id: str = "table-1",
    document_id: str = "doc-1",
    caption_label: str = "",
) -> TableArtifact:
    """Build a TableArtifact directly (the row-block packer reads .cells,
    .section_path, .caption, .num_rows/.num_cols/.has_header, .self_ref,
    .table_id, .markdown — not a Docling object)."""
    num_rows = len(cells)
    num_cols = max((len(r) for r in cells), default=0)
    md_lines = ["| " + " | ".join(cells[0]) + " |", "| " + " | ".join("---" for _ in cells[0]) + " |"]
    md_lines += ["| " + " | ".join(r) + " |" for r in cells[1:]]
    return TableArtifact(
        table_id=table_id,
        markdown="\n".join(md_lines),
        cells=cells,
        num_rows=num_rows,
        num_cols=num_cols,
        has_header=True,
        section_path=section_path,
        caption=caption,
        caption_label=caption_label,
        self_ref=self_ref,
        document_id=document_id,
    )


def _cfg(**overrides: Any) -> Any:
    """Lightweight cfg stub. NOTE: SimpleNamespace silently ignores unknown
    attrs, so the chunker only reads the two surviving knobs plus the token
    budget — there are no row/col gates, group-size, summary-char-cap, or
    summary-include-body fields anymore (those IngestionConfig fields were
    deleted)."""
    base = dict(
        hybrid_chunker_max_tokens=1024,
        table_embed_prepend_section_path=True,
        enable_adaptive_table_chunking=True,
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


def _row_chunks(out: list) -> list:
    return [c for c in out if c.extra_metadata.get("chunk_type") == "table_row"]


class TestTableRowBlockChunking:
    """_apply_adaptive_table_chunking emits only lossless table_row blocks."""

    def test_only_table_row_chunks_are_emitted_never_a_summary(self):
        """The chunker must emit chunk_type='table_row' exclusively — the
        'table_summary' chunk_type was removed entirely."""
        tbl = _make_table_artifact(
            cells=[["A", "B"], ["1", "2"], ["3", "4"]],
            section_path="Top",
        )
        out = _apply_adaptive_table_chunking([], [tbl], _cfg())

        types_seen = {c.extra_metadata.get("chunk_type") for c in out}
        assert types_seen == {"table_row"}
        assert "table_summary" not in types_seen

    def test_small_table_fits_one_block_with_all_metadata(self):
        """A table whose rows fit the budget yields exactly one block carrying
        the full per-chunk metadata contract."""
        cells = [["Field", "Offset"], ["CTRL", "0x00"], ["STAT", "0x04"]]
        tbl = _make_table_artifact(
            cells=cells,
            section_path="Chapter > Section",
            caption="Reg",
            self_ref="#/tables/7",
            document_id="spec.pdf",
        )
        out = _row_chunks(_apply_adaptive_table_chunking([], [tbl], _cfg()))

        assert len(out) == 1
        c = out[0]
        m = c.extra_metadata
        assert m["chunk_type"] == "table_row"
        assert m["table_id"] == "table-1"
        # group_id = self_ref when present.
        assert m["table_group_id"] == "#/tables/7"
        assert m["document_id"] == "spec.pdf"
        assert m["table_row_index"] == 0
        assert m["table_row_block_index"] == 0
        assert m["table_row_block_start"] == 0
        assert m["table_row_block_count"] == 2  # two body rows
        assert m["table_block_total"] == 1
        assert m["table_num_rows"] == 3
        assert m["table_num_cols"] == 2
        assert m["table_has_header"] is True
        assert m["table_caption"] == "Reg"
        # table_markdown stashed on the first (only) block.
        assert m["table_markdown"] == tbl.markdown
        # breadcrumb + caption + header + separator + both body rows.
        assert c.text == (
            "Chapter\nSection\nReg\n"
            "| Field | Offset |\n| --- | --- |\n"
            "| CTRL | 0x00 |\n| STAT | 0x04 |"
        )

    def test_group_id_falls_back_to_table_id_when_self_ref_empty(self):
        """table_group_id = self_ref or table_id."""
        tbl = _make_table_artifact(
            cells=[["A", "B"], ["1", "2"]],
            self_ref="",
            table_id="table-9",
        )
        out = _row_chunks(_apply_adaptive_table_chunking([], [tbl], _cfg()))
        assert out[0].extra_metadata["table_group_id"] == "table-9"

    def test_large_table_splits_into_multiple_token_bounded_blocks(self):
        """A table far larger than the budget splits into multiple blocks, each
        within the token budget, and each restating the header/separator."""
        body = [[f"REG{i}", f"0x{i:04X}", f"desc number {i} of register"] for i in range(40)]
        cells = [["Name", "Addr", "Desc"]] + body
        tbl = _make_table_artifact(cells=cells, section_path="Registers")
        budget = 40
        out = _row_chunks(
            _apply_adaptive_table_chunking([], [tbl], _cfg(hybrid_chunker_max_tokens=budget))
        )

        assert len(out) > 1, "large table must split into multiple blocks"
        total = out[0].extra_metadata["table_block_total"]
        assert total == len(out)
        for idx, c in enumerate(out):
            m = c.extra_metadata
            assert m["table_row_block_index"] == idx
            # Header + separator restated in EVERY block.
            assert "| Name | Addr | Desc |" in c.text
            assert "| --- | --- | --- |" in c.text
            # Breadcrumb restated in every block.
            assert c.text.startswith("Registers\n")

    def test_every_body_row_covered_exactly_once_in_order(self):
        """LOSSLESS: concatenating the blocks' body rows in order reproduces the
        original body rows, once each."""
        body = [[f"R{i:03d}", f"v{i}"] for i in range(31)]
        cells = [["Key", "Val"]] + body
        tbl = _make_table_artifact(cells=cells, section_path="S")
        out = _row_chunks(
            _apply_adaptive_table_chunking([], [tbl], _cfg(hybrid_chunker_max_tokens=30))
        )

        expected_rows = ["| " + " | ".join(r) + " |" for r in body]
        recovered: list[str] = []
        running_start = 0
        for c in out:
            m = c.extra_metadata
            # block_start aligns with how many rows preceded this block.
            assert m["table_row_block_start"] == running_start
            assert m["table_row_index"] == m["table_row_block_start"]
            block_rows = [
                ln for ln in c.text.splitlines() if ln.startswith("| R")
            ]
            assert len(block_rows) == m["table_row_block_count"]
            recovered.extend(block_rows)
            running_start += m["table_row_block_count"]

        # Order-preserving, complete, no duplicates, no drops.
        assert recovered == expected_rows
        # Sum of block counts equals total body rows.
        assert sum(c.extra_metadata["table_row_block_count"] for c in out) == len(body)

    def test_table_markdown_only_on_first_block(self):
        """table_markdown metadata is stashed ONLY on block_index == 0."""
        body = [[f"R{i}", f"{i}"] for i in range(20)]
        cells = [["K", "V"]] + body
        tbl = _make_table_artifact(cells=cells, section_path="S")
        out = _row_chunks(
            _apply_adaptive_table_chunking([], [tbl], _cfg(hybrid_chunker_max_tokens=30))
        )

        assert len(out) > 1
        assert "table_markdown" in out[0].extra_metadata
        assert all("table_markdown" not in c.extra_metadata for c in out[1:])

    def test_oversized_row_becomes_its_own_atomic_block(self):
        """A single row whose markdown alone exceeds the budget is never split
        or dropped — it occupies its own block."""
        huge = "X" * 600
        cells = [
            ["A", "B"],
            ["small", "row"],
            [huge, "tail"],
            ["after", "row2"],
        ]
        tbl = _make_table_artifact(cells=cells)
        out = _row_chunks(
            _apply_adaptive_table_chunking([], [tbl], _cfg(hybrid_chunker_max_tokens=30))
        )

        # The huge row must appear in exactly one block, alone.
        owning = [c for c in out if huge in c.text]
        assert len(owning) == 1
        assert owning[0].extra_metadata["table_row_block_count"] == 1
        # And it is still present (never dropped) and lossless overall.
        all_body = [
            ln
            for c in out
            for ln in c.text.splitlines()
            if ln.startswith("| ") and not ln.startswith("| A | B |") and "---" not in ln
        ]
        assert any(huge in ln for ln in all_body)
        assert sum(c.extra_metadata["table_row_block_count"] for c in out) == 3

    def test_header_only_table_emits_exactly_one_chunk(self):
        """A table with zero body rows emits exactly ONE chunk carrying just the
        header + separator (count 0, total 1)."""
        cells = [["H1", "H2"]]
        tbl = _make_table_artifact(cells=cells, section_path="Sec")
        out = _row_chunks(_apply_adaptive_table_chunking([], [tbl], _cfg()))

        assert len(out) == 1
        m = out[0].extra_metadata
        assert m["table_row_block_count"] == 0
        assert m["table_block_total"] == 1
        assert out[0].text == "Sec\n| H1 | H2 |\n| --- | --- |"

    def test_breadcrumb_omitted_when_prepend_disabled(self):
        """When table_embed_prepend_section_path is False, no breadcrumb is
        prepended even if a section_path exists."""
        cells = [["A", "B"], ["1", "2"]]
        tbl = _make_table_artifact(cells=cells, section_path="Top > Sub")
        out = _row_chunks(
            _apply_adaptive_table_chunking(
                [], [tbl], _cfg(table_embed_prepend_section_path=False)
            )
        )
        assert len(out) == 1
        assert out[0].text.startswith("| A | B |")
        assert "Top" not in out[0].text

    def test_breadcrumb_omitted_when_no_section_path(self):
        """No section_path -> no breadcrumb prefix, even with prepend enabled."""
        cells = [["A", "B"], ["1", "2"]]
        tbl = _make_table_artifact(cells=cells, section_path="")
        out = _row_chunks(_apply_adaptive_table_chunking([], [tbl], _cfg()))
        assert out[0].text.startswith("| A | B |")

    def test_ragged_row_keeps_all_cells_no_zip_to_header_width(self):
        """A body row wider than the header keeps ALL its cells (no truncation
        to header width)."""
        cells = [["A", "B"], ["x", "y", "z"]]
        tbl = _make_table_artifact(cells=cells)
        out = _row_chunks(_apply_adaptive_table_chunking([], [tbl], _cfg()))
        assert len(out) == 1
        assert "| x | y | z |" in out[0].text

    def test_no_table_caption_metadata_when_caption_absent(self):
        """table_caption is stamped only when the artifact has a caption."""
        cells = [["A", "B"], ["1", "2"]]
        tbl = _make_table_artifact(cells=cells, caption="")
        out = _row_chunks(_apply_adaptive_table_chunking([], [tbl], _cfg()))
        assert "table_caption" not in out[0].extra_metadata

    def test_constructing_ingestion_config_with_deleted_table_fields_raises(self):
        """The summary/gate/group-size knobs were removed from IngestionConfig;
        passing them must now raise TypeError (guards against silent re-add)."""
        for deleted in (
            "max_table_rows_for_row_chunks",
            "max_table_cols_for_row_chunks",
            "table_row_chunk_group_size",
            "table_summary_max_chars",
            "table_summary_include_body",
        ):
            with pytest.raises(TypeError):
                IngestionConfig(**{deleted: 1})
