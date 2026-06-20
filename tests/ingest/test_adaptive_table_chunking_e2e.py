# @summary
# End-to-end integration test for adaptive table chunking (Workstream C).
# Drives DoclingParser.parse() → DoclingParser.chunk() → embedding chunking_node
# with Docling library fully mocked via sys.modules injection. Asserts the
# table-aware contract: summary-per-table, row-chunks for small/uniform tables,
# table_id/page_no/heading_path propagation, chunk_index contiguity, and
# disable-flag pass-through.
# Exports: TestAdaptiveTableChunkingE2E
# Deps: src.ingest.support.docling, src.ingest.embedding.nodes.chunking,
#       src.ingest.common.types, unittest.mock, pytest
# @end-summary

"""Integration test: parse → chunk → embedding-node for adaptive table chunking.

No real Docling models, no network, no real PDF. Docling submodules are mocked
through sys.modules so the lazy imports in parse_with_docling() and
DoclingParser.chunk() resolve to fakes. The mock DoclingDocument exposes a
single inline table with grid cells matching the shape consumed by
_table_to_cells / _detect_header_row in src/ingest/support/docling.py.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import pytest

from src.ingest.common.types import IngestionConfig, Runtime
from src.ingest.support import docling as _docling_mod
from src.ingest.support.docling import DoclingParser


# ---------------------------------------------------------------------------
# Mock builders
# ---------------------------------------------------------------------------


def _make_grid_cell(text: str, *, column_header: bool = False, row_header: bool = False):
    cell = MagicMock()
    cell.text = text
    cell.column_header = column_header
    cell.row_header = row_header
    return cell


def _make_table_item(
    cells: list[list[str]],
    *,
    caption: str = "Register Map",
    page_no: int = 7,
) -> Any:
    """Build a mock Docling TableItem whose .data.grid mirrors _table_to_cells shape."""
    grid = []
    for r_idx, row in enumerate(cells):
        grid_row = [
            _make_grid_cell(c, column_header=(r_idx == 0)) for c in row
        ]
        grid.append(grid_row)

    tbl = MagicMock()
    tbl.data = MagicMock()
    tbl.data.grid = grid
    tbl.export_to_markdown = MagicMock(return_value=_render_markdown(cells))
    tbl.caption_text = MagicMock(return_value=caption)
    tbl.label = "table"
    tbl.self_ref = "#/tables/0"
    # WHY: page_ref derivation reads tbl.prov[*].page_no — provide a single prov entry
    prov_entry = MagicMock()
    prov_entry.page_no = page_no
    tbl.prov = [prov_entry]
    return tbl


def _render_markdown(cells: list[list[str]]) -> str:
    width = max(len(r) for r in cells)
    norm = [r + [""] * (width - len(r)) for r in cells]
    header = "| " + " | ".join(norm[0]) + " |"
    sep = "| " + " | ".join(["---"] * width) + " |"
    body = "\n".join("| " + " | ".join(r) + " |" for r in norm[1:])
    return "\n".join([header, sep, body]).strip()


def _make_docling_document(table_cells: list[list[str]], *, caption: str = "Register Map", page_no: int = 7):
    """Construct a mock DoclingDocument with one inline TableItem."""
    table_md = _render_markdown(table_cells)
    body_md = (
        "# Spec\n\n"
        "Some prose about the register map.\n\n"
        + table_md
        + "\n\nClosing prose paragraph.\n"
    )

    doc = MagicMock()
    doc.export_to_markdown.return_value = body_md
    doc.pictures = []
    doc.model_dump_json.return_value = '{"body": {"children": []}}'
    table_item = _make_table_item(table_cells, caption=caption, page_no=page_no)
    doc.tables = [table_item]

    # WHY: section_path resolution walks iterate_items() and snapshots the
    # heading stack at each table. Emitting one H1 before the table makes
    # TableArtifact.section_path == "Spec", matching the body markdown above.
    heading = MagicMock()
    heading.label = "section_header"
    heading.level = 1
    heading.text = "Spec"
    heading.self_ref = "#/texts/0"
    doc.iterate_items = MagicMock(return_value=iter([(heading, 1), (table_item, 1)]))
    return doc, table_md


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


def _make_raw_chunk(text: str, headings: list[str] | None = None) -> MagicMock:
    chunk = MagicMock()
    chunk.text = text
    chunk.meta = MagicMock()
    chunk.meta.headings = list(headings or [])
    chunk.meta.doc_items = []
    return chunk


@contextmanager
def _e2e_mocked_docling(mock_doc, raw_chunks):
    """Context manager that holds all docling.* mocks active for parse + chunk."""
    mock_chunker = MagicMock()
    mock_chunker.chunk = MagicMock(return_value=raw_chunks)
    mock_hc_cls = MagicMock(return_value=mock_chunker)

    with _docling_sys_modules(mock_doc), patch.dict(
        "sys.modules",
        {"docling_core.transforms.chunker": MagicMock(HybridChunker=mock_hc_cls)},
    ), patch.object(_docling_mod, "_get_or_build_tokenizer", return_value=MagicMock()):
        yield


def _run_e2e(
    *,
    table_cells: list[list[str]],
    raw_chunks_factory,
    config: IngestionConfig,
    caption: str = "Register Map",
    page_no: int = 7,
    tmp_path,
):
    """End-to-end: parse → chunk via DoclingParser → embedding chunking_node.

    Returns (processed_chunks, table_md, parse_result, parser_chunks).
    """
    mock_doc, table_md = _make_docling_document(
        table_cells, caption=caption, page_no=page_no
    )
    raw_chunks = raw_chunks_factory(table_md)

    source = tmp_path / "spec.pdf"
    source.write_bytes(b"%PDF-1.4 fake")

    with _e2e_mocked_docling(mock_doc, raw_chunks):
        parser = DoclingParser()
        parse_result = parser.parse(source, config)
        # Drive embedding node — it will invoke parser.chunk(parse_result)
        # under the same mock context so HybridChunker resolves to our mock.
        processed = _run_through_embedding_node(parser, parse_result, config)

    return processed, table_md, parse_result


def _run_through_embedding_node(parser, parse_result, config: IngestionConfig) -> list:
    """Feed parser+parse_result through chunking_node and return ProcessedChunks."""
    from src.ingest.embedding.nodes.chunking import chunking_node

    runtime = Runtime(
        config=config,
        embedder=MagicMock(),
        weaviate_client=MagicMock(),
    )
    state = {
        "raw_text": parse_result.markdown,
        "cleaned_text": parse_result.markdown,
        "source_name": "spec.pdf",
        "source_key": "local_fs:spec",
        "source_uri": "file:///tmp/spec.pdf",
        "source_id": "spec:1",
        "connector": "local_fs",
        "source_version": "1",
        "errors": [],
        "processing_log": [],
        "runtime": runtime,
        "parse_result": parse_result,
        "parser_instance": parser,
    }
    update = chunking_node(state)
    assert "chunks" in update, f"chunking_node failed: {update}"
    return update["chunks"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


SMALL_TABLE_CELLS = [
    ["Field", "Offset", "Description"],
    ["CTRL", "0x00", "Control register"],
    ["STAT", "0x04", "Status register"],
    ["DATA", "0x08", "Data port"],
    ["IRQ", "0x0C", "Interrupt mask"],
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAdaptiveTableChunkingE2E:
    """Parse → chunk → embedding-node integration for adaptive table chunking."""

    def test_small_table_full_pipeline_emits_summary_and_row_chunks(self, tmp_path):
        """Small uniform table yields one table_summary + N table_row chunks
        with table_id, page_no, heading_path, total_chunks, and contiguous
        chunk_index propagated end-to-end through chunking_node."""
        cells = SMALL_TABLE_CELLS
        body_rows = len(cells) - 1
        headers = cells[0]

        def factory(table_md: str) -> list:
            return [
                _make_raw_chunk("# Spec\n\nSome prose about the register map.", headings=["Spec"]),
                _make_raw_chunk(table_md, headings=["Spec"]),
                _make_raw_chunk("Closing prose paragraph.", headings=["Spec"]),
            ]

        # Pin group size to 1 so this end-to-end test still exercises the
        # per-row metadata-propagation path (the row-grouping behaviour itself is
        # covered by the unit tests in test_docling_adaptive_table_chunking.py).
        config = IngestionConfig(
            enable_docling_parser=True,
            enable_adaptive_table_chunking=True,
            table_row_chunk_group_size=1,
        )
        processed, table_md, parse_result = _run_e2e(
            table_cells=cells,
            raw_chunks_factory=factory,
            config=config,
            tmp_path=tmp_path,
        )

        tables = parse_result.tables
        assert len(tables) == 1
        tbl = tables[0]
        assert tbl.has_header is True
        assert tbl.page_ref is not None and tbl.page_ref.page_no == 7
        types = [c.metadata.get("chunk_type") for c in processed]

        # Exactly one summary chunk per table
        assert types.count("table_summary") == 1
        # Row chunks: one per body row
        assert types.count("table_row") == body_rows

        summary = next(c for c in processed if c.metadata.get("chunk_type") == "table_summary")
        # The heading breadcrumb is prepended to the embedded text (parity with
        # prose contextualize), so the structured summary follows the breadcrumb.
        assert "Table: Register Map" in summary.text, summary.text
        # All header names present, separated by " | "
        assert f"Columns: {' | '.join(headers)}" in summary.text
        assert f"Rows: {body_rows}" in summary.text

        # Table id consistency + contiguous row indices
        table_id = summary.metadata.get("table_id")
        assert table_id == tbl.table_id
        row_chunks = [c for c in processed if c.metadata.get("chunk_type") == "table_row"]
        row_indices = sorted(c.metadata["table_row_index"] for c in row_chunks)
        assert row_indices == list(range(body_rows))
        for rc in row_chunks:
            assert rc.metadata["table_id"] == table_id

        # page_no propagates to BOTH summary and row chunks
        assert summary.metadata.get("page_no") == 7
        for rc in row_chunks:
            assert rc.metadata.get("page_no") == 7

        # heading_path: _extract_table_artifacts walks iterate_items() and
        # snapshots the enclosing heading stack. The mock doc emits one H1
        # ("Spec") before the table, so section_path == "Spec" and every
        # downstream table chunk must carry heading_path=["Spec"].
        assert tbl.section_path == "Spec"
        for c in [summary] + row_chunks:
            assert c.metadata.get("heading_path") == ["Spec"], (
                f"heading_path should be ['Spec'] when section_path={tbl.section_path!r}"
            )

        # chunk_index contiguous 0..M-1 with total_chunks == M on every chunk
        M = len(processed)
        indices = [c.metadata["chunk_index"] for c in processed]
        assert indices == list(range(M))
        for c in processed:
            assert c.metadata["total_chunks"] == M

        # Original table-dominant raw chunk dropped
        assert not any(c.text == table_md for c in processed)
        # Surrounding prose chunks preserved
        prose_texts = [c.text for c in processed]
        assert any("Some prose about the register map" in t for t in prose_texts)
        assert any("Closing prose paragraph" in t for t in prose_texts)

    def test_disabled_flag_preserves_raw_hybridchunker_output(self, tmp_path):
        """enable_adaptive_table_chunking=False → no table_summary/table_row chunks;
        original raw chunks pass through unchanged in count and text."""
        cells = SMALL_TABLE_CELLS

        raw_texts: list[str] = []

        def factory(table_md: str) -> list:
            texts = [
                "# Spec\n\nSome prose about the register map.",
                table_md,
                "Closing prose paragraph.",
            ]
            raw_texts.extend(texts)
            return [_make_raw_chunk(t, headings=["Spec"]) for t in texts]

        config = IngestionConfig(
            enable_docling_parser=True,
            enable_adaptive_table_chunking=False,
            # Isolate the table-flag behaviour from the native min-body coalesce
            # floor: these raw chunks are deliberately sub-floor and share one
            # heading, so the default coalesce would merge them and change the
            # count this test asserts. 0 disables coalescing.
            native_min_chunk_chars=0,
        )
        processed, table_md, parse_result = _run_e2e(
            table_cells=cells,
            raw_chunks_factory=factory,
            config=config,
            tmp_path=tmp_path,
        )
        types = [c.metadata.get("chunk_type") for c in processed]
        assert "table_summary" not in types
        assert "table_row" not in types

        assert len(processed) == len(raw_texts)
        # The table chunk text is preserved (though chunking_node may tag it as
        # chunk_type=table via _match_table_artifact — that's a separate path
        # and acceptable; we only assert the texts and the absence of the
        # adaptive table_summary/table_row types).
        out_texts = [c.text for c in processed]
        for original in raw_texts:
            assert original in out_texts

    def test_large_table_emits_only_summary_and_drops_raw_table_chunk(self, tmp_path):
        """Tables with body rows > max_table_rows_for_row_chunks emit only the
        summary chunk; the table-dominant raw chunk is still dropped."""
        header = ["Field", "Value"]
        body = [[f"F{i}", f"V{i}"] for i in range(40)]
        cells = [header] + body

        def factory(table_md: str) -> list:
            return [
                _make_raw_chunk("Intro paragraph.", headings=["Spec"]),
                _make_raw_chunk(table_md, headings=["Spec"]),
                _make_raw_chunk("Trailing paragraph.", headings=["Spec"]),
            ]

        config = IngestionConfig(
            enable_docling_parser=True,
            enable_adaptive_table_chunking=True,
            # explicit default to make intent obvious
            max_table_rows_for_row_chunks=32,
        )
        processed, table_md, parse_result = _run_e2e(
            table_cells=cells,
            raw_chunks_factory=factory,
            config=config,
            tmp_path=tmp_path,
        )
        types = [c.metadata.get("chunk_type") for c in processed]
        assert types.count("table_summary") == 1
        assert types.count("table_row") == 0
        # Raw table-dominant chunk dropped
        assert not any(c.text == table_md for c in processed)
        # Surrounding prose preserved
        out_texts = [c.text for c in processed]
        assert "Intro paragraph." in out_texts
        assert "Trailing paragraph." in out_texts
