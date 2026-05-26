# @summary
# Differential floor test: runs the synthetic register-map PDF through the
# real Docling pipeline under both TableFormer modes ("fast" = v1, "accurate"
# = v2) and asserts each mode meets an *absolute* quality floor for row
# extraction. Locks the regression boundary if Docling ever flips its default
# TableFormer mode. Skips cleanly when models are unavailable.
# Exports: test_tableformer_mode_floor
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

Gated behind ``@pytest.mark.slow`` and ``@pytest.mark.integration`` and
self-skips when Docling models can't be loaded.
"""

from __future__ import annotations

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
# ---------------------------------------------------------------------------

MODE_FLOORS: dict[str, dict[str, int]] = {
    # min_table_rows: TableArtifact.num_rows lower bound (header + body)
    # min_row_chunks: number of emitted chunk_type='table_row' chunks
    # min_offsets:    distinct body-row offsets (0x00 .. 0x10) found across row chunks
    "fast": {"min_table_rows": 5, "min_row_chunks": 4, "min_offsets": 4},
    "accurate": {"min_table_rows": 5, "min_row_chunks": 5, "min_offsets": 5},
}


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("tableformer_mode", ["fast", "accurate"])
def test_tableformer_mode_floor(tmp_path: Path, tableformer_mode: str) -> None:
    """Each TableFormer mode must meet its own absolute row-recovery floor.

    Locks the regression boundary independently for ``fast`` (TF v1) and
    ``accurate`` (TF v2) so a Docling default flip or model regression on
    either branch surfaces here.
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

    # ---- Row-chunk count floor ----------------------------------------------
    row_chunks = [
        c
        for c in chunks
        if (getattr(c, "extra_metadata", {}) or {}).get("chunk_type") == "table_row"
    ]
    assert len(row_chunks) >= floors["min_row_chunks"], (
        f"[mode={tableformer_mode}] row_chunks={len(row_chunks)} "
        f"(floor={floors['min_row_chunks']}); "
        f"chunk_types={[(getattr(c, 'extra_metadata', {}) or {}).get('chunk_type') for c in chunks]}"
    )

    # ---- Offset coverage floor ----------------------------------------------
    joined_row_text = "\n".join((c.text or "") for c in row_chunks).lower()
    matched_offsets = [o for o in BODY_OFFSETS if o.lower() in joined_row_text]
    assert len(matched_offsets) >= floors["min_offsets"], (
        f"[mode={tableformer_mode}] matched_offsets={matched_offsets} "
        f"(floor={floors['min_offsets']} of {BODY_OFFSETS}); "
        f"row_text_sample={joined_row_text[:400]!r}"
    )
