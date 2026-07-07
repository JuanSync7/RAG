"""Property-based tests for table-aware chunking heuristics.

Validates the table-side behaviour of ``src/ingest/support/docling.py``:

* ``_is_table_dominant(chunk_text, table_md, signature)`` — decides whether
  a chunk should be dropped because it duplicates a table that will be
  re-emitted as its own structured chunk(s). (UNCHANGED by the refactor.)
* ``_apply_adaptive_table_chunking(chunks, tables, cfg)`` — the NEW
  token-budget row-block chunker. Each table becomes one or more
  ``chunk_type="table_row"`` chunks. There is exactly ONE knob (the token
  budget ``hybrid_chunker_max_tokens``); there are NO row/col gates, NO
  summary path, NO char cap, and NO ``group_size``. The contract is:

    - every body row appears in exactly ONE emitted block (lossless coverage),
    - the header row + markdown separator (and caption/breadcrumb) are RESTATED
      in every block,
    - raising the row count never drops rows and (under a fixed budget) never
      decreases the block count,
    - a single row whose markdown alone exceeds the budget becomes its OWN
      atomic block (never split, never dropped),
    - a header-only table emits EXACTLY ONE chunk carrying just header+separator,
    - NO emitted chunk has ``chunk_type == "table_summary"`` — that type no
      longer exists.

These tests rely on Hypothesis to surface adversarial inputs that hand-rolled
example tests would miss (e.g. signatures that happen to also be markdown
table separators, ragged grids, exact 60% boundary chunks, row counts that
straddle a packing boundary).

REFACTOR NOTE — what was replaced and why:

    The old P5/P6/P7/P8/P9 tests asserted gate properties of the now-deleted
    ``_table_is_small_uniform`` helper (row-count gate, col-count gate, ragged
    rejection, single-column rejection, "acceptance implies preconditions").
    That helper and its gate semantics no longer exist: a table is NEVER
    rejected — it is always chunked losslessly under the token budget. Those
    five tests have been replaced by property tests of the new lossless
    row-block contract below (P5–P10). There is no summary path, no
    ``table_metrics`` OTel-counter module, and no summary-truncation helper
    anymore, so any test about those was removed outright (no new-world
    analogue) rather than rewritten.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.ingest.support.docling import (
    _apply_adaptive_table_chunking,
    _cells_to_markdown,
    _is_table_dominant,
    _table_signature,
)
from src.ingest.support.parser_base import TableArtifact

# WHY: Hypothesis strings can contain '|' or '\n' which corrupt markdown rows
# and shift the "signature" line; restrict cell alphabet to keep tests focused
# on the heuristic logic, not markdown-escaping quirks (the heuristic itself
# does not escape either).
CELL_ALPHABET = st.text(
    alphabet=st.characters(
        whitelist_categories=("Lu", "Ll", "Nd"),
        whitelist_characters=" _-",
    ),
    min_size=1,
    max_size=8,
)


@st.composite
def st_cells(
    draw,
    min_rows: int = 2,
    max_rows: int = 10,
    min_cols: int = 2,
    max_cols: int = 6,
) -> list[list[str]]:
    """Generate a rectangular row-major cell grid (all rows equal length)."""
    n_cols = draw(st.integers(min_value=min_cols, max_value=max_cols))
    n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    return [
        [draw(CELL_ALPHABET) for _ in range(n_cols)] for _ in range(n_rows)
    ]


@st.composite
def st_ragged_cells(draw) -> list[list[str]]:
    """Generate cells where at least one body row has a different length."""
    n_cols = draw(st.integers(min_value=2, max_value=5))
    n_rows = draw(st.integers(min_value=2, max_value=6))
    rows = [
        [draw(CELL_ALPHABET) for _ in range(n_cols)] for _ in range(n_rows)
    ]
    # Pick a row (not the header) and add or drop a cell.
    victim_idx = draw(st.integers(min_value=1, max_value=n_rows - 1))
    if draw(st.booleans()) and len(rows[victim_idx]) > 1:
        rows[victim_idx] = rows[victim_idx][:-1]
    else:
        rows[victim_idx] = rows[victim_idx] + [draw(CELL_ALPHABET)]
    return rows


def _make_table(
    cells: list[list[str]],
    *,
    has_header: bool = True,
    table_id: str = "table-1",
    section_path: str = "",
    caption: str = "",
    self_ref: str = "",
) -> TableArtifact:
    """Construct a TableArtifact from a cell grid, rendering markdown."""
    num_rows = len(cells)
    num_cols = max((len(r) for r in cells), default=0)
    md = _cells_to_markdown(cells) if cells else ""
    return TableArtifact(
        table_id=table_id,
        markdown=md,
        cells=cells,
        num_rows=num_rows,
        num_cols=num_cols,
        has_header=has_header,
        section_path=section_path,
        caption=caption,
        self_ref=self_ref,
        document_id="doc-1",
    )


@st.composite
def st_table_artifact(draw) -> TableArtifact:
    """Generate a well-formed TableArtifact with rendered markdown."""
    cells = draw(st_cells())
    has_header = draw(st.booleans())
    section_path = draw(st.sampled_from(["", "Ch1", "Ch1 > Sec2", "A > B > C"]))
    caption = draw(st.sampled_from(["", "Table 1", "Pin assignments"]))
    return _make_table(
        cells, has_header=has_header, section_path=section_path, caption=caption
    )


_HYP = settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


# --------------------------------------------------------------------------- #
# Helpers for the new row-block contract
# --------------------------------------------------------------------------- #


def _cfg(
    max_tokens: int = 1024,
    *,
    prepend: bool = True,
) -> SimpleNamespace:
    """Config stand-in for ``_apply_adaptive_table_chunking``.

    NOTE: a ``SimpleNamespace`` silently accepts unknown attributes, so we are
    careful to set ONLY the fields the new chunker reads — the single token
    budget (``hybrid_chunker_max_tokens``) and ``table_embed_prepend_section_path``.
    The deleted gate/summary/group-size fields are intentionally absent.
    """
    return SimpleNamespace(
        hybrid_chunker_max_tokens=max_tokens,
        table_embed_prepend_section_path=prepend,
        enable_adaptive_table_chunking=True,
    )


def _body_row_md(row: list[str]) -> str:
    """Render one body row exactly as the chunker does (lossless, no zip)."""
    return "| " + " | ".join("" if c is None else str(c) for c in row) + " |"


def _rows_in_block(text: str, header_md: str, sep_md: str) -> list[str]:
    """Extract the body-row markdown lines from a block's text.

    A block is ``[breadcrumb] [caption] header sep row...`` joined by newlines.
    Body rows are the pipe-rows that follow the separator line.
    """
    lines = text.split("\n")
    # Find the separator row; body rows are everything after it.
    sep_idx = None
    for i, ln in enumerate(lines):
        if ln == sep_md:
            sep_idx = i
            break
    assert sep_idx is not None, f"separator row missing from block:\n{text}"
    # The line immediately before the separator must be the header row.
    assert lines[sep_idx - 1] == header_md, f"header row not directly above separator:\n{text}"
    return lines[sep_idx + 1 :]


def _chunk_type(chunk) -> str:
    return chunk.extra_metadata.get("chunk_type", "")


# --------------------------------------------------------------------------- #
# _is_table_dominant properties  (UNCHANGED by the refactor — kept verbatim)
# --------------------------------------------------------------------------- #


@_HYP
@given(tbl=st_table_artifact(), pad_extra=st.integers(min_value=1, max_value=2000))
def test_p1_length_gate_rejects_padded_chunks(tbl: TableArtifact, pad_extra: int) -> None:
    """P1: When chunk = markdown + padding so md/chunk_len < 0.60, the heuristic
    must return False (the 60% length gate fails even with the signature present).
    """
    md = tbl.markdown
    sig = _table_signature(md)
    if not md or not sig:
        return  # vacuous: no signature implies trivially False
    # Choose pad so that len(md) / (len(md) + pad) < 0.6
    # i.e. pad > len(md) * (1/0.6 - 1) = len(md) * 2/3.
    min_pad = int(len(md) * (1.0 / 0.6 - 1.0)) + 1
    pad = min_pad + pad_extra
    chunk = md + ("X" * pad)
    assert _is_table_dominant(chunk, md, sig) is False


@_HYP
@given(tbl=st_table_artifact())
def test_p2_pure_markdown_is_always_table_dominant(tbl: TableArtifact) -> None:
    """P2: When the chunk text *is* the table markdown, the heuristic must
    return True (signature trivially matches; ratio == 1.0)."""
    md = tbl.markdown
    sig = _table_signature(md)
    if not md or not sig:
        return  # vacuous
    assert _is_table_dominant(md, md, sig) is True


@_HYP
@given(
    tbl=st_table_artifact(),
    noise=st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
        min_size=10,
        max_size=500,
    ),
)
def test_p3_signature_match_is_prerequisite(tbl: TableArtifact, noise: str) -> None:
    """P3: If chunk_text does not contain the table signature, the heuristic
    returns False regardless of length ratio."""
    md = tbl.markdown
    sig = _table_signature(md)
    if not md or not sig or sig in noise:
        return  # filter cases where noise accidentally embeds the signature
    assert _is_table_dominant(noise, md, sig) is False


@_HYP
@given(chunk_text=st.text(min_size=0, max_size=200))
def test_p4_empty_markdown_is_never_table_dominant(chunk_text: str) -> None:
    """P4: A table with empty markdown (and therefore empty signature) is
    never table-dominant; the signature-prerequisite branch returns False."""
    assert _is_table_dominant(chunk_text, "", "") is False


# --------------------------------------------------------------------------- #
# NEW row-block contract properties (replaces the deleted gate properties)
# --------------------------------------------------------------------------- #


@_HYP
@given(cells=st_cells(min_rows=2, max_rows=10), budget=st.integers(min_value=8, max_value=4096))
def test_p5_every_body_row_covered_exactly_once(
    cells: list[list[str]], budget: int
) -> None:
    """P5 (lossless coverage): every body row appears in EXACTLY one emitted
    block, and concatenating the blocks' rows in order reproduces all body rows
    in their original order. No row is dropped, duplicated, or reordered —
    regardless of token budget.

    Replaces old P5 (row-count gate): tables are never rejected; instead the
    invariant is that no body row is ever lost.
    """
    tbl = _make_table(cells, has_header=True, section_path="Ch1 > Sec")
    cfg = _cfg(max_tokens=budget)
    out = _apply_adaptive_table_chunking([], [tbl], cfg)
    assert out, "a non-empty table must emit at least one chunk"

    expected_rows = [_body_row_md(r) for r in cells[1:]]
    header_md = "| " + " | ".join(str(h) for h in cells[0]) + " |"
    sep_md = "| " + " | ".join("---" for _ in cells[0]) + " |"

    recovered: list[str] = []
    for ch in out:
        recovered.extend(_rows_in_block(ch.text, header_md, sep_md))

    # Lossless: exact same multiset AND order.
    assert recovered == expected_rows
    # Exactly-once: count matches body-row count (catches duplication).
    assert len(recovered) == len(expected_rows)


@_HYP
@given(cells=st_cells(min_rows=2, max_rows=10), budget=st.integers(min_value=8, max_value=4096))
def test_p6_header_and_separator_restated_in_every_block(
    cells: list[list[str]], budget: int
) -> None:
    """P6 (header restated): the markdown header row AND its separator row
    appear in EVERY emitted block, so no block is ever header-blind.

    Replaces old P6 (col-count gate): wide tables are not rejected; the new
    guarantee is that the header is always carried with the rows.
    """
    tbl = _make_table(cells, has_header=True, section_path="Ch1")
    cfg = _cfg(max_tokens=budget)
    out = _apply_adaptive_table_chunking([], [tbl], cfg)
    assert out

    header_md = "| " + " | ".join(str(h) for h in cells[0]) + " |"
    sep_md = "| " + " | ".join("---" for _ in cells[0]) + " |"
    for ch in out:
        lines = ch.text.split("\n")
        assert header_md in lines, f"header missing from a block:\n{ch.text}"
        assert sep_md in lines, f"separator missing from a block:\n{ch.text}"
        # header is immediately above the separator
        assert lines.index(header_md) + 1 == lines.index(sep_md)


@_HYP
@given(extra_rows=st.integers(min_value=1, max_value=40))
def test_p7_more_rows_never_drops_rows_and_never_fewer_blocks(
    extra_rows: int,
) -> None:
    """P7 (monotonic): under a FIXED token budget, adding body rows never drops
    rows and never decreases the block count. Coverage stays lossless and the
    chunker only ever needs more (or equal) blocks to hold more rows.

    Replaces old P9 ("acceptance implies preconditions"): there is no
    accept/reject decision anymore, so the meaningful monotonic property is
    that growth in rows is absorbed by growth (not loss) of blocks.
    """
    budget = 24  # small, fixed: forces multiple blocks so the comparison bites.

    def blocks_and_rows(n_body: int) -> tuple[int, int]:
        cells = [["H1", "H2"]] + [[f"r{i}a", f"r{i}b"] for i in range(n_body)]
        tbl = _make_table(cells, has_header=True, section_path="Ch1 > Sec")
        out = _apply_adaptive_table_chunking([], [tbl], _cfg(max_tokens=budget))
        total_rows = sum(c.extra_metadata["table_row_block_count"] for c in out)
        return len(out), total_rows

    base_n = 5
    grown_n = base_n + extra_rows
    base_blocks, base_rows = blocks_and_rows(base_n)
    grown_blocks, grown_rows = blocks_and_rows(grown_n)

    # Lossless on both sizes (rows captured == body rows).
    assert base_rows == base_n
    assert grown_rows == grown_n
    # More rows -> never fewer blocks; never fewer captured rows.
    assert grown_blocks >= base_blocks
    assert grown_rows > base_rows


@_HYP
@given(
    n_before=st.integers(min_value=0, max_value=3),
    n_after=st.integers(min_value=0, max_value=3),
    oversize_chars=st.integers(min_value=2000, max_value=8000),
)
def test_p8_oversized_single_row_is_its_own_atomic_block(
    n_before: int, n_after: int, oversize_chars: int
) -> None:
    """P8 (atomic oversize): a single body row whose markdown alone exceeds the
    token budget becomes its OWN block (block_count == 1) and is never split or
    dropped. Surrounding normal rows are still covered losslessly.

    Replaces old P8 (single-column rejection): degenerate shapes are no longer
    rejected; the relevant edge case now is the un-splittable giant row.
    """
    budget = 20
    big = "X" * oversize_chars
    cells = [["H1", "H2"]]
    body: list[list[str]] = []
    for i in range(n_before):
        body.append([f"b{i}a", f"b{i}b"])
    body.append([big, big])
    big_row_index = len(body) - 1
    for i in range(n_after):
        body.append([f"a{i}a", f"a{i}b"])
    cells.extend(body)

    tbl = _make_table(cells, has_header=True)
    out = _apply_adaptive_table_chunking([], [tbl], _cfg(max_tokens=budget))

    # Lossless coverage still holds.
    total_rows = sum(c.extra_metadata["table_row_block_count"] for c in out)
    assert total_rows == len(body)

    # The block whose first row is the oversized row holds exactly one row.
    big_block = next(
        c
        for c in out
        if c.extra_metadata["table_row_block_start"] == big_row_index
    )
    assert big_block.extra_metadata["table_row_block_count"] == 1
    # The giant cell content survives intact (never truncated/split).
    assert big in big_block.text


def test_p9_header_only_table_emits_exactly_one_chunk() -> None:
    """P9 (header-only): a table with zero body rows emits EXACTLY ONE chunk
    carrying just the header + separator (block_count == 0, total == 1).

    Replaces old P7 (ragged rejection) in role as a structural edge-case test:
    the empty-body case is now first-class output rather than a rejection.
    """
    tbl = _make_table([["A", "B", "C"]], has_header=True, section_path="Ch1")
    out = _apply_adaptive_table_chunking([], [tbl], _cfg(max_tokens=1024))
    assert len(out) == 1
    meta = out[0].extra_metadata
    assert meta["chunk_type"] == "table_row"
    assert meta["table_row_block_count"] == 0
    assert meta["table_block_total"] == 1
    header_md = "| A | B | C |"
    sep_md = "| --- | --- | --- |"
    lines = out[0].text.split("\n")
    assert header_md in lines
    assert sep_md in lines
    # No body rows beyond header+separator.
    assert _rows_in_block(out[0].text, header_md, sep_md) == []


@_HYP
@given(cells=st_ragged_cells(), budget=st.integers(min_value=8, max_value=4096))
def test_p10_ragged_rows_keep_all_cells_and_stay_lossless(
    cells: list[list[str]], budget: int
) -> None:
    """P10 (ragged lossless): ragged tables are chunked, not rejected. Every
    body row is emitted with ALL of its cells (no zip against header width — a
    wider row keeps its extra cells; a narrower row is not padded), and total
    coverage is still lossless.

    Replaces old P7 (ragged rejection): the new chunker never drops a row for
    being ragged; instead it must preserve every cell verbatim.
    """
    tbl = _make_table(cells, has_header=True)
    out = _apply_adaptive_table_chunking([], [tbl], _cfg(max_tokens=budget))
    assert out

    expected_rows = [_body_row_md(r) for r in cells[1:]]
    header_md = _body_row_md(cells[0])
    sep_md = "| " + " | ".join("---" for _ in cells[0]) + " |"

    recovered: list[str] = []
    for ch in out:
        recovered.extend(_rows_in_block(ch.text, header_md, sep_md))
    assert recovered == expected_rows


@_HYP
@given(cells=st_cells(min_rows=1, max_rows=10), budget=st.integers(min_value=8, max_value=4096))
def test_p11_no_chunk_is_a_table_summary(
    cells: list[list[str]], budget: int
) -> None:
    """P11 (no summary type): the chunker emits ONLY ``chunk_type="table_row"``.
    The ``table_summary`` chunk type no longer exists — nothing may emit one,
    for any table shape or budget.

    Replaces the deleted summary-emission / summary-truncation tests: there is
    no summary path to assert about, so the invariant is simply that the type
    never appears.
    """
    tbl = _make_table(cells, has_header=True, section_path="Ch1 > Sec", caption="Cap")
    out = _apply_adaptive_table_chunking([], [tbl], _cfg(max_tokens=budget))
    assert out
    types = {_chunk_type(c) for c in out}
    assert types == {"table_row"}
    assert "table_summary" not in types


@_HYP
@given(cells=st_cells(min_rows=2, max_rows=8), budget=st.integers(min_value=8, max_value=4096))
def test_p12_metadata_contract_per_block(
    cells: list[list[str]], budget: int
) -> None:
    """P12 (metadata): per-block metadata matches the documented contract:

    * ``table_row_index == table_row_block_start == index of the block's first
      body row``, and these advance contiguously across blocks;
    * ``table_row_block_index`` is a 0-based ordinal 0..total-1;
    * ``table_block_total`` equals the number of emitted chunks;
    * ``table_markdown`` is present ONLY on the first block;
    * ``table_group_id`` == ``self_ref`` (or ``table_id`` when no self_ref);
    * row/col/header stats and ids are stamped on every block.
    """
    self_ref = "#/tables/7"
    tbl = _make_table(
        cells,
        has_header=True,
        table_id="tbl-x",
        section_path="Ch1 > Sec",
        caption="My caption",
        self_ref=self_ref,
    )
    out = _apply_adaptive_table_chunking([], [tbl], _cfg(max_tokens=budget))
    total = len(out)
    assert total >= 1

    n_body = len(cells) - 1
    running_start = 0
    for idx, ch in enumerate(out):
        m = ch.extra_metadata
        assert m["chunk_type"] == "table_row"
        assert m["table_id"] == "tbl-x"
        assert m["table_group_id"] == self_ref  # self_ref wins over table_id
        assert m["document_id"] == "doc-1"
        assert m["table_block_total"] == total
        assert m["table_row_block_index"] == idx
        assert m["table_row_index"] == running_start
        assert m["table_row_block_start"] == running_start
        assert m["table_num_rows"] == tbl.num_rows
        assert m["table_num_cols"] == tbl.num_cols
        assert m["table_has_header"] is True
        # table_markdown lives only on the first block.
        if idx == 0:
            assert m.get("table_markdown")
        else:
            assert "table_markdown" not in m
        running_start += m["table_row_block_count"]

    # Contiguous, complete coverage of all body rows.
    assert running_start == n_body


@_HYP
@given(cells=st_cells(min_rows=2, max_rows=6))
def test_p13_breadcrumb_present_only_when_enabled(cells: list[list[str]]) -> None:
    """P13 (breadcrumb toggle): the heading breadcrumb is prepended to a block's
    text ONLY when ``table_embed_prepend_section_path`` is True and a heading
    path exists. With the toggle off, the block starts at the header row.
    """
    tbl = _make_table(cells, has_header=True, section_path="Alpha > Beta")

    on = _apply_adaptive_table_chunking([], [tbl], _cfg(prepend=True))[0]
    off = _apply_adaptive_table_chunking([], [tbl], _cfg(prepend=False))[0]

    assert on.text.startswith("Alpha\nBeta\n")
    # With prepend off, no heading lines lead the block (it opens at the header
    # markdown row "| ... |").
    assert not off.text.startswith("Alpha")
    assert off.text.split("\n")[0].startswith("|")


# --------------------------------------------------------------------------- #
# Sanity check: TableArtifact construction works as expected (catches generator drift)
# --------------------------------------------------------------------------- #


def test_make_table_round_trips_signature() -> None:
    """Generator sanity: a hand-built table renders a non-empty signature that
    appears as a substring of its own markdown."""
    cells = [["A", "B"], ["1", "2"]]
    tbl = _make_table(cells, has_header=True)
    sig = _table_signature(tbl.markdown)
    assert sig and sig in tbl.markdown


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
