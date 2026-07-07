# @summary
# Tests for the lossless, token-budget table chunking inside DoclingParser.chunk().
# Covers: table_row block emission (no summary chunk_type), header/separator/caption/
# breadcrumb restated in every block, greedy whole-row packing under one token budget,
# large/wide tables splitting into multiple lossless blocks (no truncation), header-only
# tables, headerless + ragged tables keeping every cell, disable flag pass-through,
# prose-mention false positives, chunk_index re-sequencing, group-id join, metadata keys.
# Exports: TestAdaptiveTableChunking
# Deps: src.ingest.support.docling, src.ingest.support.parser_base,
#       src.ingest.common.types, unittest.mock, pytest
# @end-summary

"""Lossless token-budget table-chunking tests for ``DoclingParser.chunk()``.

HybridChunker and the tokenizer loader are stubbed via ``patch.dict`` on
``sys.modules`` and ``patch.object`` on the module-local helper so these tests
run without docling installed. Inputs are constructed as lightweight doubles
of ``DoclingDocument`` + ``TableArtifact`` so the post-processing logic is
exercised in isolation.

New contract (source: ``src/ingest/support/docling.py``
``_apply_adaptive_table_chunking`` / ``_make_table_chunks`` /
``_make_token_counter``):

* Each table emits ONLY ``chunk_type="table_row"`` block chunks — there is no
  ``"table_summary"`` chunk_type anymore.
* ONE knob: the shared token budget (``hybrid_chunker_max_tokens``). There are
  no row/col gates, no summary path, no char cap, no group_size.
* Lossless: every body row appears in exactly one block; concatenating the
  blocks' rows in order reproduces all body rows. Header + separator + caption +
  breadcrumb are RESTATED in every block. A single over-budget row is its own
  atomic block. A header-only table emits exactly one chunk.
"""

from __future__ import annotations

from typing import Any, Callable
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
    token_counter: Callable[[str], int] | None = None,
) -> list[Any]:
    """Drive DoclingParser.chunk() with stubbed HybridChunker + tokenizer.

    By default the tokenizer is a ``MagicMock`` whose ``encode(...)`` yields a
    zero-length result, so ``_make_token_counter`` reports 0 tokens for every
    string and the whole table folds into one block (handy for the "single
    block carries every row" assertions). Pass ``token_counter`` to install a
    deterministic counter (e.g. character-based) and force multi-block packing.
    """
    if config is None:
        config = IngestionConfig()
    mock_chunker = MagicMock()
    mock_chunker.chunk = MagicMock(return_value=raw_chunks)
    mock_hc_cls = MagicMock(return_value=mock_chunker)

    ctxs = [
        patch.dict(
            "sys.modules",
            {
                "docling_core.transforms.chunker": MagicMock(
                    HybridChunker=mock_hc_cls
                ),
            },
        ),
        patch.object(
            _docling_mod, "_get_or_build_tokenizer", return_value=MagicMock()
        ),
    ]
    if token_counter is not None:
        ctxs.append(
            patch.object(
                _docling_mod,
                "_make_token_counter",
                return_value=token_counter,
            )
        )

    from contextlib import ExitStack

    with ExitStack() as stack:
        for ctx in ctxs:
            stack.enter_context(ctx)
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


def _char_token_counter(per: float = 1.0) -> Callable[[str], int]:
    """A deterministic token counter: ``ceil(len(text) / per)`` tokens.

    ``per=1.0`` makes one character ≈ one token, so byte-length math drives the
    greedy packer and block boundaries are exactly predictable in tests.
    """
    import math

    def _count(text: str) -> int:
        return int(math.ceil(len(text or "") / per))

    return _count


def _table_rows(out: list[Any]) -> list[Any]:
    return [c for c in out if c.extra_metadata.get("chunk_type") == "table_row"]


def _adaptive_chunks(out: list[Any]) -> list[Any]:
    return [
        c
        for c in out
        if c.extra_metadata.get("chunk_type") in ("table_summary", "table_row")
    ]


class TestAdaptiveTableChunking:
    """Behavior of DoclingParser.chunk() lossless token-budget table chunking."""

    def test_no_summary_chunk_type_is_ever_emitted(self):
        """The summary path is gone: no chunk carries chunk_type='table_summary'."""
        cells = [["Col A", "Col B"], ["x1", "y1"], ["x2", "y2"], ["x3", "y3"]]
        tbl = _make_table_artifact(cells=cells)
        prose = _make_raw_chunk("Some intro prose.", headings=["Top", "Sub"])
        table_chunk = _make_raw_chunk(tbl.markdown, headings=["Top", "Sub"])

        out = _run_chunk([prose, table_chunk], [tbl])
        types = [c.extra_metadata.get("chunk_type") for c in out]
        assert "table_summary" not in types
        # Only table_row chunks are emitted for the table.
        assert types.count("table_row") >= 1

    def test_small_table_packs_all_rows_into_one_block(self):
        """A small table under the budget → ONE table_row block holding every row;
        original table-dominant chunk dropped (prose + 1 block = 2 chunks)."""
        cells = [["Col A", "Col B"], ["x1", "y1"], ["x2", "y2"], ["x3", "y3"]]
        tbl = _make_table_artifact(cells=cells)
        prose = _make_raw_chunk("Some intro prose.", headings=["Top", "Sub"])
        table_chunk = _make_raw_chunk(tbl.markdown, headings=["Top", "Sub"])

        out = _run_chunk([prose, table_chunk], [tbl])
        rows = _table_rows(out)
        # Zero-token MagicMock counter ⇒ no overflow ⇒ all 3 body rows in 1 block.
        assert len(rows) == 1
        # prose preserved, table-dominant chunk dropped → 2 chunks total.
        assert len(out) == 2
        block = rows[0]
        assert block.extra_metadata.get("table_row_block_index") == 0
        assert block.extra_metadata.get("table_row_block_start") == 0
        assert block.extra_metadata.get("table_row_block_count") == 3
        assert block.extra_metadata.get("table_block_total") == 1
        # Header restated + every body cell present (markdown row form).
        assert "| Col A | Col B |" in block.text
        assert "| --- | --- |" in block.text
        for cell in ("x1", "y1", "x2", "y2", "x3", "y3"):
            assert cell in block.text

    def test_header_and_breadcrumb_and_caption_restated_in_every_block(self):
        """Splitting into multiple blocks restates header+sep+caption+breadcrumb
        in each block (no header-blind continuation block)."""
        cells = [["Col A", "Col B"]] + [[f"x{i}", f"y{i}"] for i in range(6)]
        tbl = _make_table_artifact(cells=cells, caption="Pin map")
        table_chunk = _make_raw_chunk(tbl.markdown)
        # Tight char-budget counter forces several blocks. Each whole row is
        # ~13 chars ("| xN | yN |"); with 1 char≈1 token a ~70-token budget fits
        # the prefix + a couple of rows, guaranteeing >1 block.
        cfg = IngestionConfig(hybrid_chunker_max_tokens=70)
        out = _run_chunk(
            [table_chunk], [tbl], config=cfg, token_counter=_char_token_counter(per=1.0)
        )
        rows = _table_rows(out)
        assert len(rows) >= 2  # genuinely split
        for blk in rows:
            assert "Top\nSub" in blk.text  # breadcrumb restated
            assert "Pin map" in blk.text  # caption restated
            assert "| Col A | Col B |" in blk.text  # header restated
            assert "| --- | --- |" in blk.text  # separator restated

    def test_large_table_splits_into_multiple_lossless_blocks(self):
        """A tall table over the token budget → multiple table_row blocks; every
        body row is covered exactly once and the block ordinals are contiguous."""
        header = ["Col A", "Col B"]
        body = [[f"a{i:02d}", f"b{i:02d}"] for i in range(40)]
        cells = [header] + body
        tbl = _make_table_artifact(cells=cells)
        cfg = IngestionConfig(hybrid_chunker_max_tokens=60)
        table_chunk = _make_raw_chunk(tbl.markdown)
        out = _run_chunk(
            [table_chunk], [tbl], config=cfg, token_counter=_char_token_counter(per=1.0)
        )
        rows = _table_rows(out)
        # No summary path; only row blocks, and definitely more than one.
        assert all(
            c.extra_metadata.get("chunk_type") != "table_summary" for c in out
        )
        assert len(rows) > 1
        # Block ordinals are 0..N-1 contiguous in emission order.
        ordinals = [r.extra_metadata["table_row_block_index"] for r in rows]
        assert ordinals == list(range(len(rows)))
        assert all(
            r.extra_metadata["table_block_total"] == len(rows) for r in rows
        )
        # LOSSLESS: block_start values tile [0..40) with no gaps/overlaps.
        starts = [r.extra_metadata["table_row_block_start"] for r in rows]
        counts = [r.extra_metadata["table_row_block_count"] for r in rows]
        assert starts == sorted(starts)
        cursor = 0
        for s, n in zip(starts, counts):
            assert s == cursor  # contiguous, no gap, no overlap
            cursor += n
        assert cursor == len(body)  # every body row covered exactly once
        # Every body row's markdown appears in exactly one block.
        for i in range(40):
            row_md = f"| a{i:02d} | b{i:02d} |"
            hits = sum(1 for r in rows if row_md in r.text)
            assert hits == 1, f"row {i} appeared in {hits} blocks"

    def test_blocks_pack_multiple_rows_before_overflowing(self):
        """The greedy packer accumulates SEVERAL rows per block until the next
        would overflow — not one-row-per-block. Uses a tiny prefix (no
        breadcrumb, no caption) so the budget fits ~3 rows, exercising the
        accumulate-then-flush transition that 1-row-per-block budgets miss.

        header ``| A | B |`` + sep ``| --- | --- |`` → prefix = 23 chars/tokens;
        each row ``| xN | yN |`` = 11 chars, +1 join = 12 tokens. At budget 60:
        23 + 12*3 = 59 ≤ 60 (3 rows fit), + a 4th (71) overflows → blocks of
        [3, 3, 3, 1] for 10 body rows.
        """
        header = ["A", "B"]
        body = [[f"x{i}", f"y{i}"] for i in range(10)]
        cells = [header] + body
        tbl = _make_table_artifact(cells=cells, caption="", section_path="")
        cfg = IngestionConfig(hybrid_chunker_max_tokens=60)
        table_chunk = _make_raw_chunk(tbl.markdown)
        out = _run_chunk(
            [table_chunk], [tbl], config=cfg, token_counter=_char_token_counter(per=1.0)
        )
        rows = _table_rows(out)
        counts = [r.extra_metadata["table_row_block_count"] for r in rows]

        # THE point of this test: at least one block carries 2+ rows (the multi-
        # row greedy accumulation, not a degenerate one-row-per-block regime).
        assert max(counts) >= 2, f"expected multi-row blocks, got counts={counts}"
        # Exact deterministic packing for this budget/prefix/row geometry.
        assert counts == [3, 3, 3, 1], f"unexpected packing: {counts}"
        # Greedy respected the cap: every packed (non-atomic) block stays within
        # the budget (token count == char length at per=1.0).
        for r in rows:
            n = r.extra_metadata["table_row_block_count"]
            if n >= 2:
                assert len(r.text) <= 60, (
                    f"block of {n} rows exceeded budget: {len(r.text)} tokens"
                )
        # Still lossless: contiguous tiling of [0..10), every row covered once.
        starts = [r.extra_metadata["table_row_block_start"] for r in rows]
        cursor = 0
        for s, n in zip(starts, counts):
            assert s == cursor
            cursor += n
        assert cursor == len(body)
        for i in range(10):
            assert sum(1 for r in rows if f"| x{i} | y{i} |" in r.text) == 1
        # No summary chunk, ever.
        assert all(c.extra_metadata.get("chunk_type") != "table_summary" for c in out)

    def test_no_truncation_full_markdown_preserved_on_first_block(self):
        """A wide+tall table is NOT truncated — it splits into multiple lossless
        blocks, and the full table markdown is stashed on the first block only."""
        headers = [f"col_{i:02d}" for i in range(8)]
        body = [[f"v{r}_{c}" for c in range(8)] for r in range(30)]
        cells = [headers] + body
        tbl = _make_table_artifact(cells=cells, caption="Wide datasheet table")
        cfg = IngestionConfig(hybrid_chunker_max_tokens=120)
        table_chunk = _make_raw_chunk(tbl.markdown)
        out = _run_chunk(
            [table_chunk], [tbl], config=cfg, token_counter=_char_token_counter(per=1.0)
        )
        rows = _table_rows(out)
        assert len(rows) > 1
        # No truncation marker anywhere in any block's embedded text.
        for r in rows:
            assert "[truncated" not in r.text
        # All 30 body rows present across the blocks, none dropped.
        for r_idx in range(30):
            row_md = "| " + " | ".join(f"v{r_idx}_{c}" for c in range(8)) + " |"
            assert sum(1 for r in rows if row_md in r.text) == 1
        # table_markdown stashed on block 0 only, and equals the FULL markdown
        # (never truncated).
        first = next(
            r for r in rows if r.extra_metadata["table_row_block_index"] == 0
        )
        assert first.extra_metadata["table_markdown"] == tbl.markdown
        assert all(
            "table_markdown" not in r.extra_metadata
            for r in rows
            if r.extra_metadata["table_row_block_index"] != 0
        )

    def test_oversized_single_row_becomes_its_own_atomic_block(self):
        """A row whose markdown alone exceeds the budget is emitted in its OWN
        block (never split, never dropped) and short rows still pack together."""
        header = ["A", "B"]
        big_cell = "X" * 400  # this single row alone blows past the budget
        cells = [
            header,
            ["s1", "t1"],
            [big_cell, "huge"],
            ["s2", "t2"],
        ]
        tbl = _make_table_artifact(cells=cells, caption="")
        cfg = IngestionConfig(hybrid_chunker_max_tokens=80)
        table_chunk = _make_raw_chunk(tbl.markdown)
        out = _run_chunk(
            [table_chunk], [tbl], config=cfg, token_counter=_char_token_counter(per=1.0)
        )
        rows = _table_rows(out)
        # The oversized row is preserved whole in exactly one block.
        big_hits = [r for r in rows if big_cell in r.text]
        assert len(big_hits) == 1
        # That block contains ONLY the big row (atomic).
        assert big_hits[0].extra_metadata["table_row_block_count"] == 1
        # Lossless: all 3 body rows covered exactly once across blocks.
        counts = sum(r.extra_metadata["table_row_block_count"] for r in rows)
        assert counts == 3
        for cell in ("s1", "t1", "s2", "t2"):
            assert sum(1 for r in rows if cell in r.text) == 1

    def test_header_only_table_emits_exactly_one_block(self):
        """A header-only table (zero body rows) emits exactly ONE table_row chunk
        carrying just header + separator, with an empty body."""
        cells = [["Col A", "Col B"]]
        tbl = _make_table_artifact(cells=cells, caption="Headers only")
        table_chunk = _make_raw_chunk(tbl.markdown)
        out = _run_chunk([table_chunk], [tbl])
        rows = _table_rows(out)
        assert len(rows) == 1
        block = rows[0]
        assert block.extra_metadata["table_row_block_count"] == 0
        assert block.extra_metadata["table_block_total"] == 1
        assert "| Col A | Col B |" in block.text
        assert "| --- | --- |" in block.text

    def test_headerless_table_emits_all_rows_losslessly(self):
        """A table without a real header still emits row blocks covering every
        row — no gate drops it. (cells[0] is restated as the markdown header.)"""
        cells = [["a", "b"], ["c", "d"], ["e", "f"]]
        tbl = _make_table_artifact(cells=cells, has_header=False, caption="")
        cfg = IngestionConfig(hybrid_chunker_max_tokens=40)
        table_chunk = _make_raw_chunk(tbl.markdown)
        out = _run_chunk(
            [table_chunk], [tbl], config=cfg, token_counter=_char_token_counter(per=1.0)
        )
        rows = _table_rows(out)
        assert all(
            c.extra_metadata.get("chunk_type") != "table_summary" for c in out
        )
        assert len(rows) >= 1
        # has_header=False is faithfully recorded.
        assert all(r.extra_metadata["table_has_header"] is False for r in rows)
        # Body rows (cells[1:]) are all present exactly once across blocks.
        for cell_row in (["c", "d"], ["e", "f"]):
            row_md = "| " + " | ".join(cell_row) + " |"
            assert sum(1 for r in rows if row_md in r.text) == 1

    def test_ragged_rows_keep_all_cells_no_zip_to_header_width(self):
        """A ragged-row table keeps ALL cells of each body row (a wider row is
        NOT zipped down to header width) and emits every row losslessly."""
        cells = [["H1", "H2"], ["a", "b", "EXTRA"], ["d"]]
        tbl = _make_table_artifact(cells=cells)
        table_chunk = _make_raw_chunk(tbl.markdown)
        out = _run_chunk([table_chunk], [tbl])
        rows = _table_rows(out)
        assert all(
            c.extra_metadata.get("chunk_type") != "table_summary" for c in out
        )
        assert len(rows) >= 1
        joined = "\n".join(r.text for r in rows)
        # The extra cell on the wide row is preserved (3 cells, not 2).
        assert "| a | b | EXTRA |" in joined
        # The short row keeps its single cell.
        assert "| d |" in joined
        # Lossless count: 2 body rows total across all blocks.
        total = sum(r.extra_metadata["table_row_block_count"] for r in rows)
        assert total == 2

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

    def test_table_group_id_joins_all_blocks_of_one_table(self):
        """All row blocks from one table share table_group_id; distinct tables
        get distinct ids (= self_ref when present)."""
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
        for c in _adaptive_chunks(out):
            assert c.extra_metadata.get("table_group_id")

    def test_table_group_id_falls_back_to_table_id_when_self_ref_missing(self):
        """When parser has no self_ref, group_id falls back to table_id (still stable within doc)."""
        cells = [["H1", "H2"], ["a", "b"]]
        tbl = _make_table_artifact(cells=cells)
        # default TableArtifact.self_ref == ""
        assert tbl.self_ref == ""
        table_chunk = _make_raw_chunk(tbl.markdown)
        out = _run_chunk([table_chunk], [tbl])
        for c in _adaptive_chunks(out):
            assert c.extra_metadata.get("table_group_id") == "table-1"

    def test_concatenating_blocks_reproduces_every_body_row_in_order(self):
        """Lossless ordering invariant: walking blocks by block_index and
        flattening their rows reproduces the full body row sequence in order."""
        header = ["Col A", "Col B"]
        body = [[f"r{i:02d}a", f"r{i:02d}b"] for i in range(25)]
        cells = [header] + body
        tbl = _make_table_artifact(cells=cells, caption="")
        cfg = IngestionConfig(hybrid_chunker_max_tokens=55)
        table_chunk = _make_raw_chunk(tbl.markdown)
        out = _run_chunk(
            [table_chunk], [tbl], config=cfg, token_counter=_char_token_counter(per=1.0)
        )
        rows = sorted(
            _table_rows(out), key=lambda r: r.extra_metadata["table_row_block_index"]
        )
        # Reconstruct body-row markdown from each block in order, then check the
        # flattened sequence equals the original body rows in order.
        reconstructed: list[str] = []
        for blk in rows:
            for line in blk.text.splitlines():
                # body rows are "| rNNa | rNNb |"; skip header/sep/caption/breadcrumb
                if line.startswith("| r") and line.endswith(" |"):
                    reconstructed.append(line)
        expected = ["| " + " | ".join(r) + " |" for r in body]
        assert reconstructed == expected

    def test_metadata_keys_propagate_to_row_chunks(self):
        """The new metadata contract: table_id, group/document ids, row/block
        indices, table dims, page_ref, heading_path all present on row chunks."""
        cells = [["H1", "H2"], ["a", "b"], ["c", "d"]]
        tbl = _make_table_artifact(
            cells=cells, section_path="Alpha > Beta", page_no=7, caption="Cap"
        )
        tbl.self_ref = "#/tables/3"
        tbl.document_id = "doc-99"
        tbl.caption_label = "Table 5"
        table_chunk = _make_raw_chunk(tbl.markdown)
        out = _run_chunk([table_chunk], [tbl])
        rows = _table_rows(out)
        assert len(rows) == 1  # zero-token counter folds the 2 body rows together
        block = rows[0]
        m = block.extra_metadata
        assert m["chunk_type"] == "table_row"
        assert m["table_id"] == "table-1"
        assert m["table_group_id"] == "#/tables/3"
        assert m["document_id"] == "doc-99"
        assert m["caption_label"] == "Table 5"
        assert m["table_caption"] == "Cap"
        assert m["table_row_index"] == 0
        assert m["table_row_block_index"] == 0
        assert m["table_row_block_start"] == 0
        assert m["table_row_block_count"] == 2
        assert m["table_block_total"] == 1
        assert m["table_num_rows"] == tbl.num_rows
        assert m["table_num_cols"] == tbl.num_cols
        assert m["table_has_header"] is True
        assert m["table_markdown"] == tbl.markdown  # block 0 carries full markdown
        # Chunk-level provenance fields.
        assert block.heading_path == ["Alpha", "Beta"]
        assert block.section_path == "Alpha > Beta"
        assert block.page_ref is not None
        assert block.page_ref.page_no == 7

    def test_table_caption_metadata_absent_when_no_caption(self):
        """table_caption is set ONLY when the table has a caption."""
        cells = [["H1", "H2"], ["a", "b"]]
        tbl = _make_table_artifact(cells=cells, caption="")
        table_chunk = _make_raw_chunk(tbl.markdown)
        out = _run_chunk([table_chunk], [tbl])
        for r in _table_rows(out):
            assert "table_caption" not in r.extra_metadata
