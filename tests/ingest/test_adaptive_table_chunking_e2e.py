# @summary
# End-to-end integration test for adaptive table chunking (Workstream C).
# Drives DoclingParser.parse() → DoclingParser.chunk() → embedding chunking_node
# with Docling library fully mocked via sys.modules injection. Asserts the
# token-budget row-block contract: every table becomes one or more
# chunk_type="table_row" blocks (NO table_summary), the raw table-dominant chunk
# is dropped, the markdown header + separator are restated in every block, row
# coverage is lossless (every body row appears exactly once across blocks in
# order), large tables split into MULTIPLE token-bounded blocks, and
# table_id/page_no/heading_path/block metadata propagate end-to-end.
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

The token budget that drives row-block packing is the SAME knob the prose
chunker uses: ``IngestionConfig.hybrid_chunker_max_tokens``. The tokenizer is
mocked with a deterministic word-count fake (``_make_word_count_tokenizer``) so
``_make_token_counter`` returns a length-proportional count — that lets these
tests force a large table to split into multiple bounded blocks by lowering the
budget, with no dependence on a real HF tokenizer.
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


def _make_word_count_tokenizer() -> MagicMock:
    """Return a fake docling tokenizer whose ``.tokenizer.encode`` yields one
    token per whitespace-delimited word.

    ``_make_token_counter`` reads ``tok.tokenizer`` and calls
    ``hf.encode(text, add_special_tokens=False)``, counting ``len(...)``. A bare
    ``MagicMock`` makes ``len()`` return 0 for every text (so all rows would pack
    into a single block regardless of budget), which defeats the multi-block
    test. A word-count encoder gives a deterministic, length-proportional token
    count so a small ``hybrid_chunker_max_tokens`` forces a large table to split
    into several bounded blocks — exercising the real packing path.
    """
    class _WordEncoder:
        def encode(self, text, add_special_tokens=False):  # noqa: D401, ANN001
            return (text or "").split()

    tok = MagicMock()
    tok.tokenizer = _WordEncoder()
    return tok


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
    """Context manager that holds all docling.* mocks active for parse + chunk.

    The HybridChunker tokenizer resolves to a deterministic word-count fake so
    table row-block packing is bounded by ``hybrid_chunker_max_tokens`` in a
    reproducible way (see ``_make_word_count_tokenizer``).
    """
    mock_chunker = MagicMock()
    mock_chunker.chunk = MagicMock(return_value=raw_chunks)
    mock_hc_cls = MagicMock(return_value=mock_chunker)

    with _docling_sys_modules(mock_doc), patch.dict(
        "sys.modules",
        {"docling_core.transforms.chunker": MagicMock(HybridChunker=mock_hc_cls)},
    ), patch.object(
        _docling_mod, "_get_or_build_tokenizer", return_value=_make_word_count_tokenizer()
    ):
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

    Returns (processed_chunks, table_md, parse_result).
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
# Row-block markdown helpers (mirror docling._make_table_chunks emission)
# ---------------------------------------------------------------------------


def _expected_header_md(headers: list[str]) -> str:
    return "| " + " | ".join("" if h is None else str(h) for h in headers) + " |"


def _expected_sep_md(headers: list[str]) -> str:
    return "| " + " | ".join("---" for _ in headers) + " |"


def _expected_row_md(row: list[str]) -> str:
    return "| " + " | ".join("" if c is None else str(c) for c in row) + " |"


def _body_rows_from_chunk_text(text: str, header_md: str) -> list[str]:
    """Extract the body data rows from a table_row chunk's embedded text.

    The block restates breadcrumb + caption + header + separator, then N body
    rows. We treat a ``| ... |`` line as a body row when it is NOT the restated
    column-header line (``header_md``) and NOT the ``--- | ---`` separator.
    Callers compare these against the source table to assert lossless, in-order
    coverage.
    """
    header_norm = header_md.strip()
    rows: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not (stripped.startswith("|") and stripped.endswith("|")):
            continue
        if stripped == header_norm:
            continue  # restated column-header line
        # Separator row: cells are all dashes.
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if cells and all(set(c) <= {"-"} and c for c in cells):
            continue
        rows.append(stripped)
    return rows


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

    def test_small_table_full_pipeline_emits_row_blocks_no_summary(self, tmp_path):
        """Small table → table_row block(s) that REPLACE the raw table-dominant
        chunk. No table_summary chunk type exists. Every body row is covered
        exactly once across the block(s), the markdown header + separator are
        restated in every block, and table_id/page_no/heading_path/block metadata
        propagate end-to-end through chunking_node.

        Replaces the old "summary + per-row chunks (group_size=1)" test: there is
        no summary path and no group_size knob anymore — the contract is now
        token-budget row blocks. With a generous budget the small table packs
        into a single lossless block.
        """
        cells = SMALL_TABLE_CELLS
        body_rows = cells[1:]
        n_body = len(body_rows)
        headers = cells[0]
        header_md = _expected_header_md(headers)
        sep_md = _expected_sep_md(headers)

        def factory(table_md: str) -> list:
            return [
                _make_raw_chunk("# Spec\n\nSome prose about the register map.", headings=["Spec"]),
                _make_raw_chunk(table_md, headings=["Spec"]),
                _make_raw_chunk("Closing prose paragraph.", headings=["Spec"]),
            ]

        # Generous budget → the whole small table fits in one block. The token
        # budget is the only knob; there is no row/col gate or group_size.
        config = IngestionConfig(
            enable_docling_parser=True,
            enable_adaptive_table_chunking=True,
            hybrid_chunker_max_tokens=1024,
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
        # The summary chunk_type is GONE — nothing emits it.
        assert "table_summary" not in types
        # At least one row-block chunk replaces the raw table chunk.
        row_chunks = [c for c in processed if c.metadata.get("chunk_type") == "table_row"]
        assert len(row_chunks) >= 1
        # Generous budget packs the small table into exactly one block.
        assert len(row_chunks) == 1
        block = row_chunks[0]

        # Header + separator restated in the block; breadcrumb + caption present.
        assert header_md in block.text
        assert sep_md in block.text
        assert "Register Map" in block.text  # caption restated
        assert "Spec" in block.text  # breadcrumb restated

        # Lossless coverage: every source body row appears exactly once.
        covered = _body_rows_from_chunk_text(block.text, header_md)
        expected_rows = [_expected_row_md(r) for r in body_rows]
        assert covered == expected_rows, (covered, expected_rows)

        # Block metadata.
        m = block.metadata
        assert m["table_id"] == tbl.table_id
        assert m["table_group_id"] == (tbl.self_ref or tbl.table_id) == "#/tables/0"
        assert m["table_row_index"] == 0
        assert m["table_row_block_index"] == 0
        assert m["table_row_block_start"] == 0
        assert m["table_row_block_count"] == n_body
        assert m["table_block_total"] == 1
        assert m["table_num_rows"] == tbl.num_rows
        assert m["table_num_cols"] == tbl.num_cols
        assert m["table_has_header"] is True
        assert m["table_caption"] == "Register Map"
        # Full markdown stashed only on the first block.
        assert "table_markdown" in m and m["table_markdown"]

        # page_no propagates to the row block.
        assert m.get("page_no") == 7

        # heading_path: _extract_table_artifacts walks iterate_items() and
        # snapshots the enclosing heading stack. The mock doc emits one H1
        # ("Spec") before the table, so section_path == "Spec" and every
        # downstream table chunk must carry heading_path=["Spec"].
        assert tbl.section_path == "Spec"
        for c in row_chunks:
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

    def test_large_table_splits_into_multiple_lossless_row_blocks(self, tmp_path):
        """Large table → MULTIPLE token-bounded table_row blocks. The raw
        table-dominant chunk is still dropped, NO summary chunk is emitted, every
        body row is covered exactly once (concatenating blocks' rows in order
        reproduces all body rows), and the header + separator are restated in
        every block.

        Replaces the old "large table emits ONLY a summary, no row chunks" test:
        there is no summary path and no row-count gate anymore. A large table is
        split into several bounded blocks instead of being summarized away.
        """
        header = ["Field", "Value"]
        body = [[f"F{i}", f"V{i}"] for i in range(40)]
        cells = [header] + body
        header_md = _expected_header_md(header)
        sep_md = _expected_sep_md(header)

        def factory(table_md: str) -> list:
            return [
                _make_raw_chunk("Intro paragraph.", headings=["Spec"]),
                _make_raw_chunk(table_md, headings=["Spec"]),
                _make_raw_chunk("Trailing paragraph.", headings=["Spec"]),
            ]

        # The ONLY knob is the shared token budget. A small budget forces the
        # 40-row table to split across many bounded blocks (the word-count fake
        # tokenizer makes packing deterministic: ~6 tokens/row + ~13-token
        # prefix → ~3 rows per 32-token block).
        config = IngestionConfig(
            enable_docling_parser=True,
            enable_adaptive_table_chunking=True,
            hybrid_chunker_max_tokens=32,
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

        types = [c.metadata.get("chunk_type") for c in processed]
        # No summary chunk type exists anymore.
        assert "table_summary" not in types

        row_chunks = [c for c in processed if c.metadata.get("chunk_type") == "table_row"]
        # MULTIPLE token-bounded blocks (not one, not zero).
        assert len(row_chunks) > 1, f"expected multiple row blocks, got {len(row_chunks)}"

        # Order the blocks by their declared block index and assert the metadata
        # is internally consistent.
        row_chunks_sorted = sorted(
            row_chunks, key=lambda c: c.metadata["table_row_block_index"]
        )
        total_blocks = len(row_chunks_sorted)
        block_indices = [c.metadata["table_row_block_index"] for c in row_chunks_sorted]
        assert block_indices == list(range(total_blocks))
        for c in row_chunks_sorted:
            assert c.metadata["table_block_total"] == total_blocks
            assert c.metadata["table_id"] == tbl.table_id
            assert c.metadata["table_group_id"] == (tbl.self_ref or tbl.table_id)
            assert c.metadata.get("page_no") == 7
            assert c.metadata.get("heading_path") == ["Spec"]
            assert c.metadata["table_num_rows"] == tbl.num_rows
            assert c.metadata["table_num_cols"] == tbl.num_cols
            # Header + separator restated in EVERY block.
            assert header_md in c.text
            assert sep_md in c.text

        # table_markdown stashed ONLY on the first block.
        assert "table_markdown" in row_chunks_sorted[0].metadata
        for c in row_chunks_sorted[1:]:
            assert "table_markdown" not in c.metadata

        # table_row_index / table_row_block_start are the index of the first body
        # row in each block and must be contiguous & consistent with block_count.
        running = 0
        for c in row_chunks_sorted:
            assert c.metadata["table_row_index"] == running
            assert c.metadata["table_row_block_start"] == running
            running += c.metadata["table_row_block_count"]
        # All body rows accounted for across the blocks' declared counts.
        assert running == len(body)

        # LOSSLESS: concatenating each block's body rows in block order
        # reproduces every source body row exactly once, in order.
        covered: list[str] = []
        for c in row_chunks_sorted:
            block_rows = _body_rows_from_chunk_text(c.text, header_md)
            # Each block's declared count matches the rows actually in its text.
            assert len(block_rows) == c.metadata["table_row_block_count"]
            covered.extend(block_rows)
        expected_rows = [_expected_row_md(r) for r in body]
        assert covered == expected_rows, (
            f"row coverage not lossless: {len(covered)} covered vs "
            f"{len(expected_rows)} source rows"
        )

        # Raw table-dominant chunk dropped.
        assert not any(c.text == table_md for c in processed)
        # Surrounding prose preserved.
        out_texts = [c.text for c in processed]
        assert "Intro paragraph." in out_texts
        assert "Trailing paragraph." in out_texts

    def test_header_only_table_emits_single_block(self, tmp_path):
        """A header-only table (zero body rows) emits EXACTLY ONE table_row chunk
        carrying just the header + separator (and breadcrumb/caption), with
        block metadata reflecting a single empty-body block.

        Guards the ``if not blocks: blocks = [[]]`` branch of the new packer —
        the old gate/summary world had no analogue for this edge.
        """
        cells = [["Field", "Offset", "Description"]]  # header only, no body rows
        headers = cells[0]
        header_md = _expected_header_md(headers)
        sep_md = _expected_sep_md(headers)

        def factory(table_md: str) -> list:
            return [
                _make_raw_chunk("Intro paragraph.", headings=["Spec"]),
                _make_raw_chunk(table_md, headings=["Spec"]),
                _make_raw_chunk("Trailing paragraph.", headings=["Spec"]),
            ]

        config = IngestionConfig(
            enable_docling_parser=True,
            enable_adaptive_table_chunking=True,
            hybrid_chunker_max_tokens=1024,
        )
        processed, table_md, parse_result = _run_e2e(
            table_cells=cells,
            raw_chunks_factory=factory,
            config=config,
            tmp_path=tmp_path,
        )

        types = [c.metadata.get("chunk_type") for c in processed]
        assert "table_summary" not in types
        row_chunks = [c for c in processed if c.metadata.get("chunk_type") == "table_row"]
        assert len(row_chunks) == 1
        block = row_chunks[0]
        assert header_md in block.text
        assert sep_md in block.text
        # Zero body rows in the block text.
        assert _body_rows_from_chunk_text(block.text, header_md) == []
        m = block.metadata
        assert m["table_block_total"] == 1
        assert m["table_row_block_index"] == 0
        assert m["table_row_block_count"] == 0
        assert m["table_row_index"] == 0

    def test_no_otel_table_metrics_module(self):
        """The src.ingest.support.table_metrics module and its ragweave.ingest.table.*
        OTel counters were removed with the summary/gate path. Importing the module
        must fail.

        Replaces the old deleted-counter test class: that behaviour has no
        new-world analogue beyond asserting the module is gone, so the class was
        collapsed into this single negative-existence check.
        """
        with pytest.raises(ModuleNotFoundError):
            import src.ingest.support.table_metrics  # noqa: F401

    def test_removed_config_fields_raise_type_error(self):
        """The summary/gate/group-size IngestionConfig fields were deleted.
        Constructing IngestionConfig with any of them must raise TypeError so
        stale call sites fail loudly rather than silently no-op.
        """
        for field in (
            "max_table_rows_for_row_chunks",
            "max_table_cols_for_row_chunks",
            "table_row_chunk_group_size",
            "table_summary_max_chars",
            "table_summary_include_body",
        ):
            with pytest.raises(TypeError):
                IngestionConfig(**{"enable_docling_parser": True, field: 1})

    def test_removed_docling_helpers_are_gone(self):
        """The summary/gate helper functions were deleted from the docling module.
        Referencing them must raise AttributeError (no compatibility shim).
        """
        for name in (
            "_classify_table_for_row_chunks",
            "_table_is_small_uniform",
            "_build_table_summary_text",
            "_truncate_table_summary_text",
        ):
            assert not hasattr(_docling_mod, name), f"{name} should have been removed"
