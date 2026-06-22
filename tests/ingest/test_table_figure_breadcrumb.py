# @summary
# Tests for table/figure embedded-text breadcrumb prepend (parity with prose
# contextualize) and the token-budget "row block" table chunker.
# Covers (table side): breadcrumb prepend per row-block + disable flag, NO
# table_summary chunk_type, lossless cell/row coverage, header restated in every
# block, large table -> multiple token-bounded blocks, atomic oversized row,
# header-only table, ragged-row cell retention, and per-block metadata.
# Covers (figure side): figure-chunk breadcrumb prepend + disable flag.
# @end-summary

"""Tests for RAG_INGESTION_TABLE_EMBED_PREPEND_SECTION_PATH behaviour and the
token-budget row-block table chunker.

Table chunking emits ONLY ``chunk_type="table_row"`` chunks. Each table becomes
one or more "row block" chunks: every block restates breadcrumb + caption +
markdown header + separator, then packs as many whole body rows as fit under the
token budget (``hybrid_chunker_max_tokens`` -> RAG_INGESTION_HYBRID_CHUNKER_MAX_TOKENS).
Coverage is lossless — every body row appears in exactly one block. There is no
``table_summary`` chunk_type, no row/col gate, no summary truncation, no
group-size knob.

Reuses the stubbed-HybridChunker harness from test_docling_adaptive_table_chunking.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.ingest.common.types import IngestionConfig
from src.ingest.common.shared import chunk_body_text
from src.ingest.support import docling as _docling_mod
from src.ingest.support.docling import _figure_artifacts_to_chunks
from tests.ingest.test_docling_adaptive_table_chunking import (
    _make_table_artifact,
    _make_raw_chunk,
    _run_chunk,
)


def _rows(out):
    """Return the emitted table_row block chunks, in order."""
    return [c for c in out if c.extra_metadata.get("chunk_type") == "table_row"]


def _chunk_types(out):
    return [c.extra_metadata.get("chunk_type") for c in out]


def _char_token_counter(_cfg, _max_tokens):
    """A deterministic char-based token counter (~4 chars/token).

    The stubbed harness patches ``_get_or_build_tokenizer`` with a MagicMock,
    whose ``encode`` returns a MagicMock that ``len()``-es to 0 — so the real
    counter treats every row as 0 tokens and never splits. Patching
    ``_make_token_counter`` with this lets a small ``hybrid_chunker_max_tokens``
    actually force multi-block packing so we can assert the split is lossless.
    """
    return lambda t: max(1, len(t or "") // 4)


def _body_row_lines(text: str) -> list[str]:
    """Markdown body rows in a block's text (header/separator/breadcrumb stripped)."""
    lines = text.splitlines()
    out: list[str] = []
    seen_sep = False
    for ln in lines:
        s = ln.strip()
        if not s.startswith("|"):
            continue
        if set(s) <= {"|", "-", " "}:  # the "| --- | --- |" separator row
            seen_sep = True
            continue
        if seen_sep:
            out.append(ln)
    return out


# ---------------------------------------------------------------------------
# Breadcrumb prepend (parity with prose contextualize)
# ---------------------------------------------------------------------------

def test_row_block_prepends_breadcrumb_by_default():
    """Each table_row block restates the heading breadcrumb at its head."""
    cells = [["Field", "Value"], ["AWVALID", "1"], ["AWREADY", "0"]]
    tbl = _make_table_artifact(cells=cells, section_path="Ch5 > 5.2 Signals", caption="Sig")
    out = _run_chunk([_make_raw_chunk(tbl.markdown)], [tbl])
    rows = _rows(out)
    assert rows, "expected at least one row-block chunk"
    for r in rows:
        # Breadcrumb is "\n".join(heading_path) + "\n" (byte-compatible with the
        # prose contextualize() prefix the body-aware gate strips).
        assert r.text.startswith("Ch5\n5.2 Signals\n")
        body = chunk_body_text(r.text, r.heading_path)
        # The breadcrumb is gone after stripping; the caption begins the body.
        assert body.startswith("Sig")
        assert not body.startswith("Ch5")


def test_row_block_breadcrumb_disabled_by_flag():
    """table_embed_prepend_section_path=False omits the breadcrumb from blocks."""
    cells = [["Field", "Value"], ["AWVALID", "1"], ["AWREADY", "0"]]
    tbl = _make_table_artifact(cells=cells, section_path="Ch5 > 5.2 Signals", caption="Sig")
    cfg = IngestionConfig(table_embed_prepend_section_path=False)
    out = _run_chunk([_make_raw_chunk(tbl.markdown)], [tbl], config=cfg)
    rows = _rows(out)
    assert rows
    for r in rows:
        assert not r.text.startswith("Ch5")
        # With no breadcrumb the caption is the first line, then header/separator.
        assert r.text.startswith("Sig\n| Field | Value |\n| --- | --- |")


def test_caption_present_in_every_block():
    """The caption is restated at the head of every block (after the breadcrumb)."""
    cells = [["Field", "Value"]] + [[f"R{i}", str(i)] for i in range(30)]
    tbl = _make_table_artifact(cells=cells, section_path="Ch7 > 7.1 Map", caption="Registers")
    with patch.object(_docling_mod, "_make_token_counter", _char_token_counter):
        cfg = IngestionConfig(hybrid_chunker_max_tokens=60)
        out = _run_chunk([_make_raw_chunk(tbl.markdown)], [tbl], config=cfg)
    rows = _rows(out)
    assert len(rows) > 1, "budget should have forced more than one block"
    for r in rows:
        assert "Registers" in r.text


# ---------------------------------------------------------------------------
# NO table_summary chunk_type (summary emission was removed)
# ---------------------------------------------------------------------------

def test_no_table_summary_chunk_type_emitted():
    """The chunker never emits a table_summary chunk — only table_row blocks.

    Replaces the old summary-emission / summary-body-fold tests: there is no
    summary chunk to fold cell values into anymore.
    """
    cells = [["Field", "Value"]] + [[f"REG{i}", f"0x{i:02X}"] for i in range(40)]
    tbl = _make_table_artifact(cells=cells, section_path="Ch7 > 7.1 Map", caption="Registers")
    out = _run_chunk([_make_raw_chunk(tbl.markdown)], [tbl])
    assert "table_summary" not in _chunk_types(out)
    assert _rows(out), "a table must still emit at least one row-block chunk"


# ---------------------------------------------------------------------------
# Lossless coverage (replaces summary-only cell-value fold)
# ---------------------------------------------------------------------------

def test_all_cell_values_present_lossless_single_block():
    """Every body cell value survives in the row blocks (lossless, no fold)."""
    cells = [["Field", "Value"]] + [[f"REG{i}", f"0x{i:02X}"] for i in range(40)]
    tbl = _make_table_artifact(cells=cells, section_path="Ch7 > 7.1 Map", caption="Registers")
    out = _run_chunk([_make_raw_chunk(tbl.markdown)], [tbl])
    rows = _rows(out)
    joined = "\n".join(r.text for r in rows)
    # Spot-check specific cells that the old summary-only path would have folded.
    assert "REG37" in joined
    assert "0x25" in joined  # 37 == 0x25
    # Exhaustive: every body cell value appears somewhere in the blocks.
    for i in range(40):
        assert f"REG{i}" in joined
        assert f"0x{i:02X}" in joined


def test_large_table_splits_into_multiple_lossless_blocks():
    """A large table packs into multiple token-bounded blocks; every body row is
    covered in exactly one block (concatenation reproduces all rows).

    Replaces the old "large table -> summary only" / row-col-gate tests: large
    tables now fan out into lossless row blocks rather than collapsing to a
    summary.
    """
    body = [[f"REG{i:03d}", f"0x{i:04X}", f"desc-{i}"] for i in range(80)]
    cells = [["Reg", "Addr", "Desc"]] + body
    tbl = _make_table_artifact(cells=cells, section_path="Ch9 > 9.1 Regs", caption="Big")
    with patch.object(_docling_mod, "_make_token_counter", _char_token_counter):
        cfg = IngestionConfig(hybrid_chunker_max_tokens=64)
        out = _run_chunk([_make_raw_chunk(tbl.markdown)], [tbl], config=cfg)
    rows = _rows(out)
    assert len(rows) > 1, "small budget should force several blocks"

    # Blocks are ordered 0..N-1 and all carry the same block-total.
    block_total = rows[0].extra_metadata["table_block_total"]
    assert block_total == len(rows)
    assert [r.extra_metadata["table_row_block_index"] for r in rows] == list(range(len(rows)))

    # Per-block row counts sum to the body row count (every row covered once).
    counts = [r.extra_metadata["table_row_block_count"] for r in rows]
    assert sum(counts) == len(body)

    # table_row_block_start is contiguous and matches the running row offset.
    expected_start = 0
    for r, n in zip(rows, counts):
        assert r.extra_metadata["table_row_block_start"] == expected_start
        assert r.extra_metadata["table_row_index"] == expected_start
        expected_start += n

    # Concatenating the body-row markdown lines reproduces ALL body rows exactly.
    all_lines: list[str] = []
    for r in rows:
        all_lines.extend(_body_row_lines(r.text))
    assert len(all_lines) == len(body)
    for i in range(80):
        assert any(f"REG{i:03d}" in ln for ln in all_lines)


def test_header_restated_in_every_block():
    """The markdown header + separator are restated at the top of every block."""
    body = [[f"k{i}", f"v{i}"] for i in range(60)]
    cells = [["Key", "Val"]] + body
    tbl = _make_table_artifact(cells=cells, section_path="Ch2", caption="")
    with patch.object(_docling_mod, "_make_token_counter", _char_token_counter):
        cfg = IngestionConfig(hybrid_chunker_max_tokens=48)
        out = _run_chunk([_make_raw_chunk(tbl.markdown)], [tbl], config=cfg)
    rows = _rows(out)
    assert len(rows) > 1
    for r in rows:
        assert "| Key | Val |" in r.text
        assert "| --- | --- |" in r.text


def test_oversized_single_row_becomes_its_own_atomic_block():
    """A row whose markdown alone exceeds the budget gets its own block — never
    split, never dropped, and never merged with neighbours."""
    cells = [
        ["Field", "Value"],
        ["small", "x"],
        ["big", "Z" * 500],
        ["small2", "y"],
    ]
    tbl = _make_table_artifact(cells=cells, section_path="Ch1", caption="")
    with patch.object(_docling_mod, "_make_token_counter", _char_token_counter):
        cfg = IngestionConfig(hybrid_chunker_max_tokens=40)
        out = _run_chunk([_make_raw_chunk(tbl.markdown)], [tbl], config=cfg)
    rows = _rows(out)
    counts = [r.extra_metadata["table_row_block_count"] for r in rows]
    assert sum(counts) == 3  # all three body rows present
    # The oversized row appears exactly once across all blocks (never split).
    joined = "\n".join(r.text for r in rows)
    assert joined.count("Z" * 500) == 1
    # The block carrying the oversized row holds it alone.
    oversized_blocks = [r for r in rows if ("Z" * 500) in r.text]
    assert len(oversized_blocks) == 1
    assert oversized_blocks[0].extra_metadata["table_row_block_count"] == 1


def test_header_only_table_emits_exactly_one_chunk():
    """A table with zero body rows emits exactly one chunk carrying just the
    breadcrumb/caption/header/separator (no body rows)."""
    cells = [["ColA", "ColB", "ColC"]]
    tbl = _make_table_artifact(cells=cells, section_path="Ch2", caption="HdrOnly")
    out = _run_chunk([_make_raw_chunk(tbl.markdown)], [tbl])
    rows = _rows(out)
    assert len(rows) == 1
    only = rows[0]
    assert only.extra_metadata["table_row_block_count"] == 0
    assert only.extra_metadata["table_block_total"] == 1
    assert only.text == "Ch2\nHdrOnly\n| ColA | ColB | ColC |\n| --- | --- | --- |"
    # No stray body-row lines.
    assert _body_row_lines(only.text) == []


def test_ragged_row_keeps_all_cells():
    """A body row with MORE cells than the header keeps every cell (no zip to
    header width). Such a table is still emitted as row blocks (no summary)."""
    cells = [["H1", "H2"], ["a", "b"], ["c", "d", "EXTRA_CELL"]]
    tbl = _make_table_artifact(cells=cells, section_path="Ch3", caption="Rag")
    out = _run_chunk([_make_raw_chunk(tbl.markdown)], [tbl])
    rows = _rows(out)
    assert rows
    assert "table_summary" not in _chunk_types(out)
    joined = "\n".join(r.text for r in rows)
    assert "EXTRA_CELL" in joined
    assert "| c | d | EXTRA_CELL |" in joined


def test_table_markdown_only_on_first_block():
    """The full markdown is stashed on metadata of block 0 only (not repeated)."""
    body = [[f"k{i}", f"v{i}"] for i in range(50)]
    cells = [["Key", "Val"]] + body
    tbl = _make_table_artifact(cells=cells, section_path="Ch4", caption="MD")
    with patch.object(_docling_mod, "_make_token_counter", _char_token_counter):
        cfg = IngestionConfig(hybrid_chunker_max_tokens=48)
        out = _run_chunk([_make_raw_chunk(tbl.markdown)], [tbl], config=cfg)
    rows = _rows(out)
    assert len(rows) > 1
    assert rows[0].extra_metadata.get("table_markdown") == tbl.markdown
    for r in rows[1:]:
        assert "table_markdown" not in r.extra_metadata


def test_row_block_metadata_propagates():
    """Each row block carries the table provenance + geometry metadata."""
    cells = [["H1", "H2"], ["a", "b"], ["c", "d"]]
    tbl = _make_table_artifact(
        cells=cells, section_path="Alpha > Beta", page_no=7, caption="Demo Caption"
    )
    out = _run_chunk([_make_raw_chunk(tbl.markdown)], [tbl])
    rows = _rows(out)
    assert rows
    for r in rows:
        m = r.extra_metadata
        assert m["chunk_type"] == "table_row"
        assert m["table_id"] == "table-1"
        # self_ref is "" on the default artifact, so group_id falls back to table_id.
        assert m["table_group_id"] == "table-1"
        assert m["table_num_rows"] == 3
        assert m["table_num_cols"] == 2
        assert m["table_has_header"] is True
        assert m["table_caption"] == "Demo Caption"
        assert r.heading_path == ["Alpha", "Beta"]
        assert r.section_path == "Alpha > Beta"
        assert r.page_ref is not None
        assert r.page_ref.page_no == 7


# ---------------------------------------------------------------------------
# Figure chunks (unchanged behaviour — kept)
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
