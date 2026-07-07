# @summary
# Real-Docling smoke test: parses a synthesized register-map PDF with the
# actual Docling pipeline (TF v2 ACCURATE + OCR) and asserts the new adaptive
# table chunking emits the expected table_summary + row chunks. Skips
# gracefully when Docling models are unavailable.
# Exports: test_real_docling_table_smoke
# Deps: reportlab, docling (real), src.ingest.support.docling, pytest
# @end-summary

"""Real-Docling smoke test for adaptive table chunking + TF v2 + OCR.

Distinct from ``test_adaptive_table_chunking_e2e.py`` (which mocks Docling at
``sys.modules``): this test boots the *real* Docling pipeline so model loading,
TF v2 invocation, OCR, and the adaptive table chunker are exercised end-to-end
on a synthesized PDF.

The test is gated behind ``@pytest.mark.slow`` and ``@pytest.mark.integration``
and self-skips when models can't be loaded.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Deterministic PDF builder
# ---------------------------------------------------------------------------

TABLE_ROWS: list[list[str]] = [
    ["Offset", "Name", "Reset", "Description"],
    ["0x00", "CTRL", "0x00000000", "Control register"],
    ["0x04", "STATUS", "0x00000001", "Status register"],
    ["0x08", "MASK", "0x000000FF", "Interrupt mask register"],
    ["0x0C", "DATA", "0x00000000", "Data register"],
    ["0x10", "CFG", "0x0000A5A5", "Configuration register"],
]


def _synthesize_register_pdf(path: Path) -> None:
    """Render a deterministic register-map PDF with reportlab.

    Layout (one page is fine):
      - "Register Map" heading
      - "GPIO Control Registers" subheading
      - One sentence of prose
      - 4-column / 6-row (1 header + 5 body) table
      - One sentence of prose
    """
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
# Diagnostics helpers
# ---------------------------------------------------------------------------


def _dump_diagnostics(
    parse_result_markdown: str,
    chunks: list,
    tables: list,
) -> Path:
    """Write parse markdown + chunk metadata JSON to a temp dir for debugging."""
    out_dir = Path(tempfile.mkdtemp(prefix="real_docling_smoke_"))
    (out_dir / "parsed.md").write_text(parse_result_markdown or "", encoding="utf-8")

    chunk_blob = []
    for c in chunks:
        chunk_blob.append(
            {
                "text": getattr(c, "text", ""),
                "section_path": getattr(c, "section_path", ""),
                "heading": getattr(c, "heading", ""),
                "heading_path": list(getattr(c, "heading_path", []) or []),
                "heading_level": getattr(c, "heading_level", 0),
                "chunk_index": getattr(c, "chunk_index", -1),
                "extra_metadata": getattr(c, "extra_metadata", {}) or {},
                "page_ref": repr(getattr(c, "page_ref", None)),
            }
        )
    (out_dir / "chunks.json").write_text(
        json.dumps(chunk_blob, indent=2, default=str), encoding="utf-8"
    )

    table_blob = []
    for t in tables:
        table_blob.append(
            {
                "num_rows": getattr(t, "num_rows", None),
                "num_cols": getattr(t, "num_cols", None),
                "caption": getattr(t, "caption", None),
                "has_header": getattr(t, "has_header", None),
            }
        )
    (out_dir / "tables.json").write_text(
        json.dumps(table_blob, indent=2, default=str), encoding="utf-8"
    )
    return out_dir


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.integration
def test_real_docling_table_smoke(tmp_path: Path) -> None:
    """Real Docling parse+chunk over a synthesized register-map PDF.

    Asserts adaptive table chunking emits a table_summary chunk plus row
    chunks that reference real cell content, and that the TableArtifact's
    shape matches the rendered table. Skips when models are unavailable.
    """
    # Deferred imports: keep the test importable even if optional deps shift.
    from src.ingest.common.types import IngestionConfig
    from src.ingest.support.docling import DoclingParser

    pdf_path = tmp_path / "register_map.pdf"
    _synthesize_register_pdf(pdf_path)
    assert pdf_path.exists() and pdf_path.stat().st_size > 0

    config = IngestionConfig()
    # Defaults already enable TF v2 ACCURATE, OCR, and adaptive table chunking.
    # Force-allow model auto-download on this host so first-run smoke is honest.
    try:
        config.docling_auto_download = True  # dataclass field
    except Exception:
        pass

    # Skip cleanly if models can't be made ready (no network / restricted CI).
    try:
        DoclingParser.ensure_ready(config)
    except Exception as exc:  # noqa: BLE001 — any failure → skip
        pytest.skip(f"Docling models unavailable: {exc!r}")

    parser = DoclingParser()
    try:
        parse_result = parser.parse(pdf_path, config)
        chunks = parser.chunk(parse_result)
    except Exception as exc:  # noqa: BLE001
        # Surface a debug dir even on hard parse failure (no chunks yet).
        diag = _dump_diagnostics(
            getattr(parse_result, "markdown", "") if "parse_result" in locals() else "",
            chunks if "chunks" in locals() else [],
            list(getattr(parse_result, "tables", []) or [])
            if "parse_result" in locals()
            else [],
        )
        print(f"\n[diagnostics] {diag}")
        raise AssertionError(f"Real Docling parse/chunk raised: {exc!r}") from exc

    tables = list(getattr(parse_result, "tables", []) or [])

    # Best-effort diagnostics path — only printed on failure below.
    diag_dir = _dump_diagnostics(parse_result.markdown, chunks, tables)

    def _fail(msg: str) -> None:
        print(f"\n[diagnostics] {diag_dir}")
        raise AssertionError(msg)

    # ---- TableArtifact shape -------------------------------------------------
    if not tables:
        _fail("Expected at least one TableArtifact in parse_result.tables; got 0.")
    tbl = tables[0]
    if not (getattr(tbl, "num_rows", 0) >= 5):
        _fail(
            f"TableArtifact.num_rows={getattr(tbl, 'num_rows', None)} "
            f"(want >= 5); caption={getattr(tbl, 'caption', None)!r}"
        )
    if not (getattr(tbl, "num_cols", 0) >= 4):
        _fail(
            f"TableArtifact.num_cols={getattr(tbl, 'num_cols', None)} "
            f"(want >= 4); caption={getattr(tbl, 'caption', None)!r}"
        )

    # ---- table_summary chunk presence + header content -----------------------
    summary_chunks = [
        c for c in chunks
        if (getattr(c, "extra_metadata", {}) or {}).get("chunk_type") == "table_summary"
    ]
    if not summary_chunks:
        _fail(
            "No chunk with chunk_type='table_summary' was emitted. "
            f"chunk_types observed: "
            f"{sorted({(getattr(c, 'extra_metadata', {}) or {}).get('chunk_type') for c in chunks})}"
        )
    summary = summary_chunks[0]
    summary_text_lower = (summary.text or "").lower()
    expected_headers = ["offset", "name", "reset", "description"]
    missing_headers = [h for h in expected_headers if h not in summary_text_lower]
    if missing_headers:
        _fail(
            f"table_summary text missing headers {missing_headers}. "
            f"summary.text[:300]={summary.text[:300]!r}"
        )

    # ---- row chunks ----------------------------------------------------------
    row_chunks = [
        c for c in chunks
        if (getattr(c, "extra_metadata", {}) or {}).get("chunk_type") == "table_row"
    ]
    # Allow ≥80% of 5 body rows = 4 rows (TF v2 may miss/merge rows).
    if len(row_chunks) < 4:
        _fail(
            f"Expected >= 4 table_row chunks (>=80% of 5 body rows); "
            f"got {len(row_chunks)}. "
            f"chunk_types: "
            f"{[(getattr(c, 'extra_metadata', {}) or {}).get('chunk_type') for c in chunks]}"
        )

    # Each row chunk's text should contain its Offset value, loose substring.
    expected_offsets = [r[0] for r in TABLE_ROWS[1:]]  # 0x00 .. 0x10
    joined_row_text = "\n".join((c.text or "") for c in row_chunks).lower()
    matched_offsets = [o for o in expected_offsets if o.lower() in joined_row_text]
    if len(matched_offsets) < 4:
        _fail(
            f"Row chunks only captured offsets {matched_offsets}; "
            f"expected >= 4 of {expected_offsets}. "
            f"row_text_sample={joined_row_text[:400]!r}"
        )

    # ---- heading_path / section_path fix validation --------------------------
    heading_path_hits = []
    for c in chunks:
        hp = list(getattr(c, "heading_path", []) or [])
        if not hp:
            continue
        joined = " > ".join(hp).lower()
        if "register map" in joined or "gpio control registers" in joined:
            heading_path_hits.append(hp)
    # NOTE: The section_path propagation fix may still be pending on a parallel
    # branch. We assert here but tolerate empty results with a clear failure so
    # the operator can choose to xfail it temporarily.
    if not heading_path_hits:
        _fail(
            "No chunk had heading_path containing 'Register Map' or "
            "'GPIO Control Registers'. The section_path propagation fix may "
            "not have landed on this branch. "
            f"heading_paths observed: "
            f"{[list(getattr(c, 'heading_path', []) or []) for c in chunks]}"
        )

    # If we got here, everything passed — diagnostics are noise. Print only on
    # failure paths above (each _fail prints before raising).
