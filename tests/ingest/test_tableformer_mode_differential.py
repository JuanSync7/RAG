# @summary
# Differential floor test: runs the synthetic register-map PDF through the
# real Docling pipeline under both TableFormer modes ("fast" = v1, "accurate"
# = v2) and asserts each mode meets an *absolute* quality floor for row
# extraction under the token-budget table-chunking contract. Locks the
# regression boundary if Docling ever flips its default TableFormer mode.
# Skips cleanly when models are unavailable. A companion synthetic test
# (no real Docling) exercises the multi-block split path that the small
# real table cannot reach.
# Exports: test_tableformer_mode_floor, test_large_table_splits_into_lossless_blocks
# Deps: reportlab, docling (real), src.ingest.support.docling, pytest
# @end-summary

"""Differential floor test across TableFormer modes (fast vs accurate).

Sibling to ``test_real_docling_table_smoke.py`` (which exercises only the
ACCURATE/v2 path). This test parametrizes over the ``tableformer_mode`` config
knob and asserts a *floor* — not a strict ordering — for each mode against the
same synthetic register-map fixture. The asymmetric thresholds reflect
empirical observation: v1/fast can occasionally drop or merge a row on tightly
gridded synthetic tables, while v2/accurate consistently recovers all 5 body
rows. The point is to pin behavior so Docling default flips don't silently
degrade downstream chunking; this test does NOT claim accurate > fast.

Table-chunking contract under test (``_apply_adaptive_table_chunking`` /
``_make_table_chunks`` in ``src.ingest.support.docling``):

* A table is emitted ONLY as ``chunk_type="table_row"`` chunks — there is no
  ``"table_summary"`` chunk_type anymore. This test asserts that none is
  emitted.
* Each chunk is a "row block": breadcrumb + caption (if any) + markdown header
  row + separator row + N whole body rows, greedily packed under ONE token
  budget (``hybrid_chunker_max_tokens`` or the package default). The header and
  separator are RESTATED in every block.
* The packing is LOSSLESS: every recovered body row appears in exactly one
  block; concatenating the blocks' rows in order reproduces all body rows.

Because the synthetic register map is small (5 body rows), all rows fit under
the default ~1024-token budget into a SINGLE block — so the floor here is
"≥1 row-block chunk, losslessly covering ≥N offsets", not "one chunk per row".
The multi-block split path is covered by the synthetic companion test below,
which drives a wide table under a deliberately tiny token budget.

Gated behind ``@pytest.mark.slow`` and ``@pytest.mark.integration`` and
self-skips when Docling models can't be loaded.
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Synthetic PDF builder (duplicated from test_real_docling_table_smoke.py on
# purpose — these two tests share a fixture but pragmatically inlining is
# clearer than coupling them via an external conftest fixture for now).
# ---------------------------------------------------------------------------

TABLE_ROWS: list[list[str]] = [
    ["Offset", "Name", "Reset", "Description"],
    ["0x00", "CTRL", "0x00000000", "Control register"],
    ["0x04", "STATUS", "0x00000001", "Status register"],
    ["0x08", "MASK", "0x000000FF", "Interrupt mask register"],
    ["0x0C", "DATA", "0x00000000", "Data register"],
    ["0x10", "CFG", "0x0000A5A5", "Configuration register"],
]
BODY_OFFSETS = [r[0] for r in TABLE_ROWS[1:]]  # 0x00 .. 0x10


def _synthesize_register_pdf(path: Path) -> None:
    """Render a deterministic register-map PDF with reportlab."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        str(path),
        pagesize=letter,
        title="Register Map",
        author="ragweave-test",
    )
    story = [
        Paragraph("Register Map", styles["Heading1"]),
        Spacer(1, 8),
        Paragraph("GPIO Control Registers", styles["Heading2"]),
        Spacer(1, 8),
        Paragraph(
            "The following table enumerates the GPIO control register map "
            "for the peripheral block.",
            styles["BodyText"],
        ),
        Spacer(1, 12),
        Table(
            TABLE_ROWS,
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ]
            ),
        ),
        Spacer(1, 12),
        Paragraph(
            "All registers reset to the values listed above on power-on reset.",
            styles["BodyText"],
        ),
    ]
    doc.build(story)


# ---------------------------------------------------------------------------
# Empirical floors (justification for the asymmetry).
#
# Observed against this synthetic 4-col x 6-row register map on the worktree
# Docling pinning:
#   - "accurate" (TF v2): recovers all 5/5 body rows with all 5 offsets.
#   - "fast"     (TF v1): recovers >= 4/5 body rows; occasionally merges/drops
#     one row on synthetic grids with thin borders. We DO NOT assert
#     accurate > fast (that would be flaky); we assert each mode meets its
#     own absolute floor so future Docling/TableFormer changes that *worsen*
#     either mode are caught.
#
# Under the token-budget chunker the 5 body rows fit in ONE block, so the
# per-mode row-chunk floor is just "≥1 row block emitted" (the table-row path
# ran). Row RECOVERY is asserted via the offset-coverage floor instead, which
# is the property that actually distinguishes the two TableFormer modes.
# ---------------------------------------------------------------------------

MODE_FLOORS: dict[str, dict[str, int]] = {
    # min_table_rows: TableArtifact.num_rows lower bound (header + body)
    # min_offsets:    distinct body-row offsets (0x00 .. 0x10) found across row chunks
    "fast": {"min_table_rows": 5, "min_offsets": 4},
    "accurate": {"min_table_rows": 5, "min_offsets": 5},
}


def _row_chunks(chunks: list) -> list:
    """All ``chunk_type='table_row'`` chunks in document order."""
    return [
        c
        for c in chunks
        if (getattr(c, "extra_metadata", {}) or {}).get("chunk_type") == "table_row"
    ]


def _md_body_rows(text: str) -> list[str]:
    """Extract the markdown body-row lines (``| ... |``) from a row-block chunk,
    excluding the markdown header row and the ``| --- | ... |`` separator.

    A row-block chunk's text is::

        <breadcrumb lines>
        <caption>
        | h1 | h2 | ...      <- header row (restated every block)
        | --- | --- | ...    <- separator row (restated every block)
        | c1 | c2 | ...      <- body rows
        ...

    We treat the separator line as the marker; everything after it that still
    looks like a ``|``-delimited markdown row is a body row.
    """
    lines = (text or "").splitlines()
    sep_idx = None
    for i, ln in enumerate(lines):
        cells = [c.strip() for c in ln.split("|")]
        # Separator row: every interior cell is a run of dashes.
        interior = [c for c in cells if c != ""]
        if interior and all(set(c) == {"-"} for c in interior):
            sep_idx = i
            break
    if sep_idx is None:
        return []
    body: list[str] = []
    for ln in lines[sep_idx + 1 :]:
        if ln.lstrip().startswith("|"):
            body.append(ln.strip())
    return body


def _first_cell(md_row: str) -> str:
    """First (Offset) cell of a ``| c1 | c2 | ... |`` markdown body row.

    Splitting on ``|`` yields ``["", " c1 ", " c2 ", ..., ""]``; index 1 is the
    first cell. Used for EXACT offset matching — a loose substring check would
    spuriously match ``0x00`` inside a reset value like ``0x00000000``.
    """
    parts = md_row.split("|")
    return parts[1].strip() if len(parts) > 1 else md_row.strip()


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("tableformer_mode", ["fast", "accurate"])
def test_tableformer_mode_floor(tmp_path: Path, tableformer_mode: str) -> None:
    """Each TableFormer mode must meet its own absolute row-recovery floor under
    the token-budget table-chunking contract.

    Locks the regression boundary independently for ``fast`` (TF v1) and
    ``accurate`` (TF v2) so a Docling default flip or model regression on
    either branch surfaces here. Asserts the new chunking contract:
    table_row-only emission (NO table_summary), header restated per block, and
    lossless body-row coverage.
    """
    from src.ingest.common.types import IngestionConfig
    from src.ingest.support.docling import DoclingParser

    pdf_path = tmp_path / f"register_map_{tableformer_mode}.pdf"
    _synthesize_register_pdf(pdf_path)
    assert pdf_path.exists() and pdf_path.stat().st_size > 0

    config = IngestionConfig()
    config.tableformer_mode = tableformer_mode
    try:
        config.docling_auto_download = True
    except Exception:
        pass

    try:
        DoclingParser.ensure_ready(config)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Docling models unavailable: {exc!r}")

    parser = DoclingParser()
    try:
        parse_result = parser.parse(pdf_path, config)
        chunks = parser.chunk(parse_result)
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(
            f"Real Docling parse/chunk raised under mode={tableformer_mode!r}: {exc!r}"
        ) from exc

    floors = MODE_FLOORS[tableformer_mode]

    # ---- TableArtifact shape floor ------------------------------------------
    tables = list(getattr(parse_result, "tables", []) or [])
    assert tables, (
        f"[mode={tableformer_mode}] expected >=1 TableArtifact; got 0. "
        f"markdown_head={(parse_result.markdown or '')[:300]!r}"
    )
    tbl = tables[0]
    num_rows = getattr(tbl, "num_rows", 0) or 0
    num_cols = getattr(tbl, "num_cols", 0) or 0
    assert num_rows >= floors["min_table_rows"], (
        f"[mode={tableformer_mode}] TableArtifact.num_rows={num_rows} "
        f"(floor={floors['min_table_rows']})"
    )
    assert num_cols >= 4, (
        f"[mode={tableformer_mode}] TableArtifact.num_cols={num_cols} (floor=4)"
    )

    chunk_types = [
        (getattr(c, "extra_metadata", {}) or {}).get("chunk_type") for c in chunks
    ]

    # ---- NO table_summary chunk_type may be emitted -------------------------
    # The summary path was removed; the chunker emits ONLY table_row blocks.
    assert "table_summary" not in chunk_types, (
        f"[mode={tableformer_mode}] removed chunk_type='table_summary' was emitted; "
        f"chunk_types={chunk_types}"
    )

    # ---- Row-block emission floor -------------------------------------------
    # The table-row path must run (≥1 block). The 5 body rows fit under the
    # default token budget into a single block, so we do NOT assert one chunk
    # per row — that was the old per-row contract. Row RECOVERY is asserted via
    # the offset-coverage floor below.
    row_chunks = _row_chunks(chunks)
    assert len(row_chunks) >= 1, (
        f"[mode={tableformer_mode}] no chunk_type='table_row' emitted; "
        f"chunk_types={chunk_types}"
    )

    # ---- Header is restated in EVERY block ----------------------------------
    expected_headers = ["offset", "name", "reset", "description"]
    for bi, c in enumerate(row_chunks):
        ct_lower = (c.text or "").lower()
        missing = [h for h in expected_headers if h not in ct_lower]
        assert not missing, (
            f"[mode={tableformer_mode}] row block {bi} missing restated header "
            f"columns {missing}; text[:300]={c.text[:300]!r}"
        )

    # ---- Lossless body-row coverage -----------------------------------------
    # Every recovered body row appears in exactly one block (no row dropped, no
    # row duplicated across blocks). We key on the EXACT first (Offset) cell —
    # a loose substring match would spuriously fire because ``0x00`` is a prefix
    # of reset values like ``0x00000000``.
    all_body_rows: list[str] = []
    for c in row_chunks:
        all_body_rows.extend(_md_body_rows(c.text))
    first_cells = [_first_cell(r).lower() for r in all_body_rows]
    for off in BODY_OFFSETS:
        occurrences = sum(1 for fc in first_cells if fc == off.lower())
        assert occurrences <= 1, (
            f"[mode={tableformer_mode}] offset {off!r} appears as the first cell "
            f"of {occurrences} row blocks (lossless coverage requires each row "
            f"in exactly one block); body_rows={all_body_rows!r}"
        )

    # ---- Offset coverage floor (mode-differential row recovery) -------------
    matched_offsets = [o for o in BODY_OFFSETS if o.lower() in first_cells]
    assert len(matched_offsets) >= floors["min_offsets"], (
        f"[mode={tableformer_mode}] matched_offsets={matched_offsets} "
        f"(floor={floors['min_offsets']} of {BODY_OFFSETS}); "
        f"body_rows_sample={all_body_rows[:6]!r}"
    )

    # ---- Block-metadata invariants ------------------------------------------
    # Block ordinals are contiguous 0..N-1, table_block_total is consistent,
    # the group id is stable across blocks, and table_markdown is stamped ONLY
    # on the first block.
    total = (row_chunks[0].extra_metadata or {}).get("table_block_total")
    assert total == len(row_chunks), (
        f"[mode={tableformer_mode}] table_block_total={total} != "
        f"emitted blocks={len(row_chunks)}"
    )
    group_ids = {
        (c.extra_metadata or {}).get("table_group_id") for c in row_chunks
    }
    assert len(group_ids) == 1 and next(iter(group_ids)), (
        f"[mode={tableformer_mode}] inconsistent/empty table_group_id across "
        f"blocks: {group_ids}"
    )
    block_indices = [
        (c.extra_metadata or {}).get("table_row_block_index") for c in row_chunks
    ]
    assert block_indices == list(range(len(row_chunks))), (
        f"[mode={tableformer_mode}] table_row_block_index not contiguous 0..N-1: "
        f"{block_indices}"
    )
    md_blocks = [
        bi
        for bi, c in enumerate(row_chunks)
        if "table_markdown" in (c.extra_metadata or {})
    ]
    assert md_blocks == [0], (
        f"[mode={tableformer_mode}] table_markdown must be stamped only on block 0; "
        f"found on blocks {md_blocks}"
    )


# ---------------------------------------------------------------------------
# Synthetic multi-block split test (NO real Docling).
#
# The real register-map table is too small to exercise the greedy row-packing
# split path: its 5 body rows fit in one block under the default token budget.
# This test drives ``_apply_adaptive_table_chunking`` directly with a wide
# table and a deliberately tiny token budget so the packer MUST emit multiple
# blocks, then asserts the new contract's hard properties:
#
#   * multiple token-bounded blocks are emitted (split happened),
#   * coverage is lossless (every body row in exactly one block, in order),
#   * the header + separator are restated in every block,
#   * NO table_summary chunk_type is ever emitted,
#   * block ordinals are contiguous and table_markdown is on block 0 only.
#
# This replaces the old per-row "min_row_chunks" floor, which described the
# removed one-chunk-per-row behaviour. It is real-Docling-independent so it
# runs even when TableFormer models are unavailable.
# ---------------------------------------------------------------------------


def _make_table_artifact(cells: list[list[str]], **overrides):
    from src.ingest.support.parser_base import TableArtifact

    kwargs = dict(
        table_id="table-1",
        markdown="\n".join("| " + " | ".join(r) + " |" for r in cells),
        cells=cells,
        num_rows=len(cells),
        num_cols=max((len(r) for r in cells), default=0),
        has_header=True,
        section_path="Register Map > GPIO Control Registers",
        caption="Table 1: GPIO register map",
        caption_label="Table 1",
        self_ref="#/tables/0",
        document_id="reg_map",
    )
    kwargs.update(overrides)
    return TableArtifact(**kwargs)


def test_large_table_splits_into_lossless_blocks() -> None:
    """A table whose rows exceed the token budget splits into multiple lossless
    row blocks, header restated per block, with NO table_summary chunk_type.

    Drives ``_apply_adaptive_table_chunking`` directly (no real Docling) with a
    tiny ``hybrid_chunker_max_tokens`` so the greedy packer is forced to emit
    several blocks — the path the small real register table cannot reach.
    """
    from src.ingest.support.docling import _apply_adaptive_table_chunking

    header = ["Offset", "Name", "Reset", "Description"]
    body = [
        [f"0x{i * 4:02X}", f"REG{i}", f"0x{i:08X}", f"Register number {i} long-ish text"]
        for i in range(12)
    ]
    cells = [header] + body
    tbl = _make_table_artifact(cells)

    # SimpleNamespace cfg: only the surviving knobs. A tiny token budget forces
    # multi-row-per-block packing to overflow into several blocks. (SimpleNamespace
    # silently ignores unknown attrs, so the DELETED fields are simply absent —
    # never set them here: that would be misleading and they no longer exist on
    # the real IngestionConfig.)
    cfg = types.SimpleNamespace(
        table_embed_prepend_section_path=True,
        enable_adaptive_table_chunking=True,
        hybrid_chunker_max_tokens=24,  # tiny → forces a split
    )

    out = _apply_adaptive_table_chunking([], [tbl], cfg)
    row_chunks = _row_chunks(out)

    # ---- Only table_row chunks; never a table_summary -----------------------
    chunk_types = {
        (getattr(c, "extra_metadata", {}) or {}).get("chunk_type") for c in out
    }
    assert chunk_types == {"table_row"}, (
        f"expected only table_row chunks; got chunk_types={chunk_types}"
    )

    # ---- Split actually happened --------------------------------------------
    assert len(row_chunks) >= 2, (
        f"tiny token budget should force >=2 blocks; got {len(row_chunks)}"
    )

    # ---- Header + separator restated in every block -------------------------
    header_md = "| " + " | ".join(header) + " |"
    sep_md = "| " + " | ".join("---" for _ in header) + " |"
    for bi, c in enumerate(row_chunks):
        assert header_md in c.text, (
            f"block {bi} missing restated header row; text={c.text!r}"
        )
        assert sep_md in c.text, (
            f"block {bi} missing restated separator row; text={c.text!r}"
        )
        # Breadcrumb + caption are restated too.
        assert "GPIO Control Registers" in c.text, (
            f"block {bi} missing restated breadcrumb; text={c.text!r}"
        )
        assert "Table 1: GPIO register map" in c.text, (
            f"block {bi} missing restated caption; text={c.text!r}"
        )

    # ---- Lossless coverage: every body row in exactly one block, in order ---
    recovered: list[str] = []
    for c in row_chunks:
        recovered.extend(_md_body_rows(c.text))
    expected = ["| " + " | ".join(r) + " |" for r in body]
    assert recovered == expected, (
        "row blocks must losslessly reproduce every body row in order; "
        f"recovered={recovered!r}\nexpected={expected!r}"
    )

    # ---- Block-metadata invariants ------------------------------------------
    total = (row_chunks[0].extra_metadata or {}).get("table_block_total")
    assert total == len(row_chunks)
    assert [
        (c.extra_metadata or {}).get("table_row_block_index") for c in row_chunks
    ] == list(range(len(row_chunks)))
    # table_row_block_start / table_row_index advance by each block's row count.
    cursor = 0
    for c in row_chunks:
        meta = c.extra_metadata or {}
        assert meta.get("table_row_block_start") == cursor
        assert meta.get("table_row_index") == cursor
        cursor += int(meta.get("table_row_block_count") or 0)
    assert cursor == len(body), (
        f"summed table_row_block_count={cursor} != body rows={len(body)}"
    )
    # table_markdown stamped only on the first block.
    md_blocks = [
        bi
        for bi, c in enumerate(row_chunks)
        if "table_markdown" in (c.extra_metadata or {})
    ]
    assert md_blocks == [0], (
        f"table_markdown must be on block 0 only; found on blocks {md_blocks}"
    )
    # Group id stable + derived from self_ref.
    group_ids = {(c.extra_metadata or {}).get("table_group_id") for c in row_chunks}
    assert group_ids == {"#/tables/0"}, f"unexpected table_group_id(s): {group_ids}"


def test_header_only_table_emits_single_block() -> None:
    """A header-only table (zero body rows) emits EXACTLY ONE table_row chunk
    carrying just the header + separator (no body rows, no table_summary)."""
    from src.ingest.support.docling import _apply_adaptive_table_chunking

    header = ["Offset", "Name", "Reset", "Description"]
    tbl = _make_table_artifact([header], num_rows=1)

    cfg = types.SimpleNamespace(
        table_embed_prepend_section_path=True,
        enable_adaptive_table_chunking=True,
        hybrid_chunker_max_tokens=1024,
    )

    out = _apply_adaptive_table_chunking([], [tbl], cfg)
    row_chunks = _row_chunks(out)

    assert len(out) == 1 and len(row_chunks) == 1, (
        f"header-only table must emit exactly one table_row chunk; got {out!r}"
    )
    c = row_chunks[0]
    assert (c.extra_metadata or {}).get("chunk_type") == "table_row"
    assert _md_body_rows(c.text) == [], (
        f"header-only block must carry zero body rows; text={c.text!r}"
    )
    header_md = "| " + " | ".join(header) + " |"
    assert header_md in c.text
    assert (c.extra_metadata or {}).get("table_block_total") == 1
    assert (c.extra_metadata or {}).get("table_row_block_count") == 0
