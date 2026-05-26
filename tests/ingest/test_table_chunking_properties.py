"""Property-based tests for table-aware chunking heuristics.

Validates the two private heuristics in ``src/ingest/support/docling.py``:

* ``_is_table_dominant(chunk_text, table_md, signature)`` — decides whether
  a chunk should be dropped because it duplicates a table that will be
  re-emitted as its own structured chunk(s).
* ``_table_is_small_uniform(tbl, max_rows, max_cols)`` — decides whether a
  table is "well-shaped" enough to split into per-row chunks.

These tests rely on Hypothesis to surface adversarial inputs that hand-rolled
example tests would miss (e.g. signatures that happen to also be markdown
table separators, ragged grids, exact 60% boundary chunks).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.ingest.support.docling import (
    _cells_to_markdown,
    _is_table_dominant,
    _table_is_small_uniform,
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
    )


@st.composite
def st_table_artifact(draw) -> TableArtifact:
    """Generate a well-formed TableArtifact with rendered markdown."""
    cells = draw(st_cells())
    has_header = draw(st.booleans())
    return _make_table(cells, has_header=has_header)


_HYP = settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


# --------------------------------------------------------------------------- #
# _is_table_dominant properties
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
# _table_is_small_uniform properties
# --------------------------------------------------------------------------- #


def _cfg(max_rows: int = 32, max_cols: int = 12) -> SimpleNamespace:
    return SimpleNamespace(
        max_table_rows_for_row_chunks=max_rows,
        max_table_cols_for_row_chunks=max_cols,
    )


@_HYP
@given(
    body_extra=st.integers(min_value=1, max_value=20),
    n_cols=st.integers(min_value=2, max_value=12),
)
def test_p5_above_threshold_rows_rejected(body_extra: int, n_cols: int) -> None:
    """P5: When body_rows > max_table_rows_for_row_chunks, the heuristic
    returns False even when every other condition is satisfied."""
    cfg = _cfg(max_rows=4)
    body_rows = cfg.max_table_rows_for_row_chunks + body_extra
    cells = [["h"] * n_cols] + [["x"] * n_cols for _ in range(body_rows)]
    tbl = _make_table(cells, has_header=True)
    assert (
        _table_is_small_uniform(
            tbl, cfg.max_table_rows_for_row_chunks, cfg.max_table_cols_for_row_chunks
        )
        is False
    )


@_HYP
@given(
    cols_extra=st.integers(min_value=1, max_value=10),
    n_body_rows=st.integers(min_value=1, max_value=8),
)
def test_p6_above_threshold_cols_rejected(cols_extra: int, n_body_rows: int) -> None:
    """P6: When num_cols > max_table_cols_for_row_chunks, returns False."""
    cfg = _cfg(max_cols=3)
    n_cols = cfg.max_table_cols_for_row_chunks + cols_extra
    cells = [["h"] * n_cols] + [["x"] * n_cols for _ in range(n_body_rows)]
    tbl = _make_table(cells, has_header=True)
    assert (
        _table_is_small_uniform(
            tbl, cfg.max_table_rows_for_row_chunks, cfg.max_table_cols_for_row_chunks
        )
        is False
    )


@_HYP
@given(cells=st_ragged_cells())
def test_p7_ragged_rows_rejected(cells: list[list[str]]) -> None:
    """P7: Ragged grids (some body row has a different length from the
    header) must be rejected — typed-row chunking requires aligned columns."""
    tbl = _make_table(cells, has_header=True)
    assert _table_is_small_uniform(tbl, 32, 12) is False


@_HYP
@given(n_body_rows=st.integers(min_value=1, max_value=10))
def test_p8_single_column_rejected(n_body_rows: int) -> None:
    """P8: num_cols < 2 (typically a single-column table) is rejected; per-row
    chunks of a 1-column table degenerate to plain text."""
    cells = [["h"]] + [["x"]] * n_body_rows
    tbl = _make_table(cells, has_header=True)
    assert _table_is_small_uniform(tbl, 32, 12) is False


@_HYP
@given(tbl=st_table_artifact())
def test_p9_acceptance_implies_all_preconditions(tbl: TableArtifact) -> None:
    """P9: If the heuristic accepts a table, then ALL preconditions hold:
    has_header is True; 1 <= body_rows <= max_rows; 2 <= num_cols <= max_cols;
    every body row has the same length as the header."""
    max_rows, max_cols = 32, 12
    if not _table_is_small_uniform(tbl, max_rows, max_cols):
        return  # only assert the forward implication
    assert tbl.has_header is True
    assert tbl.cells  # non-empty
    body = tbl.cells[1:]
    assert 1 <= len(body) <= max_rows
    assert 2 <= tbl.num_cols <= max_cols
    header_len = len(tbl.cells[0])
    assert all(len(r) == header_len for r in body)


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
