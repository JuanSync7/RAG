# @summary
# Real-Docling datasheet-shape smoke test: synthesizes a multi-page,
# multi-table PDF resembling a small ASIC datasheet (register map,
# electrical characteristics, nested bit-field details) and exercises the
# full table-aware chunking chain end-to-end -- TF v2 extraction, heading
# replay across pages, adaptive chunker stamping, ``table_group_id``
# stability, and ``table_markdown`` stash. Asserts contract floors only
# (no exact counts). Skips cleanly when Docling models are unavailable.
# Exports: test_real_docling_datasheet_smoke
# Deps: reportlab, docling (real), src.ingest.support.docling, pytest
# @end-summary

"""Real-Docling smoke test over a multi-page, multi-table datasheet shape.

Existing real-pipeline tests cover a single-page single-table register map.
Real ASIC datasheets are multi-page, multi-table, and frequently include
dense electrical-characteristics grids alongside register tables and
nested bit-field sub-sections. This test locks down the end-to-end
contract for the full table-aware chunking story over that shape:

1. Multiple ``TableArtifact`` entries with non-empty ``markdown``,
   ``section_path``, and ``self_ref``.
2. Heading provenance preserved across pages -- including a nested
   sub-heading for the bit-field table.
3. Adaptive chunker emits ``table_summary`` + ``table_row`` chunks
   keyed by a stable ``table_group_id``; row indices are contiguous.
4. ``table_markdown`` is stashed on each summary chunk.
5. Determinism: re-parsing the same PDF with a fresh ``DoclingParser``
   yields the same set of ``table_group_id`` values (important for
   idempotent re-ingest).

Gated behind ``slow`` + ``integration`` markers and self-skips when
models can't be loaded.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Synthetic datasheet content (ASCII-only to avoid font-fallback flakiness)
# ---------------------------------------------------------------------------

REGISTER_MAP_ROWS: list[list[str]] = [
    ["Address", "Name", "Width", "Reset", "Description"],
    ["0x00", "CTRL", "8", "0x00", "Control register"],
    ["0x01", "STATUS", "8", "0x01", "Status register"],
    ["0x02", "MASK", "8", "0xFF", "Interrupt mask"],
    ["0x04", "DATA", "16", "0x0000", "Data sample register"],
    ["0x08", "CFG", "16", "0xA5A5", "Configuration register"],
    ["0x0C", "GAIN", "8", "0x10", "Programmable gain"],
    ["0x10", "OFFSET", "16", "0x0000", "Offset trim register"],
    ["0x14", "ID", "8", "0xAD", "Device identification"],
]

ELEC_CHAR_ROWS: list[list[str]] = [
    ["Parameter", "Symbol", "Min", "Typ", "Max", "Unit"],
    ["Supply voltage", "VDD", "1.7", "1.8", "1.9", "V"],
    ["Analog supply", "AVDD", "3.0", "3.3", "3.6", "V"],
    ["Reference voltage", "VREF", "1.0", "1.2", "1.5", "V"],
    ["Input range", "VIN", "-1.0", "0.0", "1.0", "V"],
    ["Sample rate", "FS", "0.1", "1.0", "10.0", "MSPS"],
    ["Resolution", "N", "12", "12", "12", "bits"],
    ["INL", "INL", "-1.0", "0.2", "1.0", "LSB"],
    ["DNL", "DNL", "-1.0", "0.1", "1.0", "LSB"],
    ["Power consumption", "PD", "5.0", "8.0", "12.0", "mW"],
    ["Operating temperature", "TA", "-40", "25", "85", "C"],
]

BIT_FIELD_ROWS: list[list[str]] = [
    ["Bit", "Name", "Access", "Description"],
    ["7", "EN", "RW", "Module enable"],
    ["6", "MODE", "RW", "Operating mode"],
    ["5:4", "GAIN", "RW", "Gain select"],
    ["3:1", "RSVD", "RO", "Reserved"],
    ["0", "START", "WO", "Start conversion"],
]


def _synthesize_datasheet_pdf(path: Path) -> None:
    """Render a 4-page synthetic ASIC-datasheet-shaped PDF with reportlab.

    Layout:
      Page 1 -- Title + Overview prose.
      Page 2 -- "Register Map" heading + register-map table.
      Page 3 -- "Electrical Characteristics" heading + parameter table.
      Page 4 -- "Bit Field Details" + nested "CTRL Register Bits"
                sub-heading + bit-field table.
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import (
        PageBreak,
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
        title="Synthetic ADC Datasheet",
        author="ragweave-test",
    )

    def _styled_table(rows: list[list[str]]) -> Table:
        return Table(
            rows,
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ]
            ),
        )

    story = [
        # Page 1: cover + overview
        Paragraph("Synthetic ADC Datasheet", styles["Title"]),
        Spacer(1, 12),
        Paragraph("Overview", styles["Heading1"]),
        Spacer(1, 6),
        Paragraph(
            "This datasheet describes a synthetic 12-bit successive "
            "approximation analog-to-digital converter used solely for "
            "ingest pipeline regression testing.",
            styles["BodyText"],
        ),
        Spacer(1, 6),
        Paragraph(
            "The device exposes a small register file, a representative "
            "set of electrical characteristics, and a single control "
            "register whose bit fields are documented below.",
            styles["BodyText"],
        ),
        Spacer(1, 6),
        Paragraph(
            "All values are illustrative and have no physical meaning.",
            styles["BodyText"],
        ),
        PageBreak(),

        # Page 2: register map
        Paragraph("Register Map", styles["Heading1"]),
        Spacer(1, 8),
        Paragraph(
            "The register map below lists the byte-addressable control "
            "and status registers exposed by the device.",
            styles["BodyText"],
        ),
        Spacer(1, 10),
        _styled_table(REGISTER_MAP_ROWS),
        Spacer(1, 10),
        Paragraph(
            "All registers reset to the listed values on power-on reset.",
            styles["BodyText"],
        ),
        PageBreak(),

        # Page 3: electrical characteristics
        Paragraph("Electrical Characteristics", styles["Heading1"]),
        Spacer(1, 8),
        Paragraph(
            "Specifications apply over the full operating temperature "
            "range unless otherwise noted.",
            styles["BodyText"],
        ),
        Spacer(1, 10),
        _styled_table(ELEC_CHAR_ROWS),
        Spacer(1, 10),
        Paragraph(
            "Typical values are characterized at nominal supply and 25 C.",
            styles["BodyText"],
        ),
        PageBreak(),

        # Page 4: nested heading + bit-field details
        Paragraph("Bit Field Details", styles["Heading1"]),
        Spacer(1, 8),
        Paragraph(
            "The following sub-section documents the individual bit "
            "fields of the control register.",
            styles["BodyText"],
        ),
        Spacer(1, 8),
        Paragraph("CTRL Register Bits", styles["Heading2"]),
        Spacer(1, 8),
        _styled_table(BIT_FIELD_ROWS),
        Spacer(1, 10),
        Paragraph(
            "Reserved fields must be written as zero for forward "
            "compatibility.",
            styles["BodyText"],
        ),
    ]
    doc.build(story)


# ---------------------------------------------------------------------------
# Diagnostics helpers
# ---------------------------------------------------------------------------


def _dump_diagnostics(parse_result_markdown: str, chunks: list, tables: list) -> Path:
    """Write parse markdown + chunk metadata + table summaries for debugging."""
    out_dir = Path(tempfile.mkdtemp(prefix="real_docling_datasheet_"))
    (out_dir / "parsed.md").write_text(parse_result_markdown or "", encoding="utf-8")

    chunk_blob = []
    for c in chunks:
        chunk_blob.append(
            {
                "text": (getattr(c, "text", "") or "")[:400],
                "section_path": getattr(c, "section_path", ""),
                "heading": getattr(c, "heading", ""),
                "heading_path": list(getattr(c, "heading_path", []) or []),
                "extra_metadata": getattr(c, "extra_metadata", {}) or {},
            }
        )
    (out_dir / "chunks.json").write_text(
        json.dumps(chunk_blob, indent=2, default=str), encoding="utf-8"
    )

    table_blob = []
    for t in tables:
        table_blob.append(
            {
                "table_id": getattr(t, "table_id", None),
                "self_ref": getattr(t, "self_ref", None),
                "section_path": getattr(t, "section_path", None),
                "num_rows": getattr(t, "num_rows", None),
                "num_cols": getattr(t, "num_cols", None),
                "caption": getattr(t, "caption", None),
                "markdown_head": (getattr(t, "markdown", "") or "")[:200],
            }
        )
    (out_dir / "tables.json").write_text(
        json.dumps(table_blob, indent=2, default=str), encoding="utf-8"
    )
    return out_dir


def _build_config():
    from src.ingest.common.types import IngestionConfig

    config = IngestionConfig()
    try:
        config.docling_auto_download = True
    except Exception:  # noqa: BLE001
        pass
    return config


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.integration
def test_real_docling_datasheet_smoke(tmp_path: Path) -> None:
    """End-to-end real-Docling parse + chunk over a multi-page datasheet.

    Asserts the full table-aware chunking contract: multiple
    ``TableArtifact`` entries with heading provenance (including a
    nested sub-heading), adaptive chunker emits ``table_summary`` +
    ``table_row`` chunks with stable ``table_group_id``s and contiguous
    row indices, ``table_markdown`` is stashed on the summary, and
    re-parsing yields the same set of group ids.
    """
    from src.ingest.support.docling import DoclingParser

    pdf_path = tmp_path / "synthetic_adc_datasheet.pdf"
    _synthesize_datasheet_pdf(pdf_path)
    assert pdf_path.exists() and pdf_path.stat().st_size > 0

    config = _build_config()

    # Skip cleanly if models can't be made ready (no network / restricted CI).
    try:
        DoclingParser.ensure_ready(config)
    except Exception as exc:  # noqa: BLE001 -- any failure -> skip
        pytest.skip(f"Docling models unavailable: {exc!r}")

    parser = DoclingParser()
    try:
        parse_result = parser.parse(pdf_path, config)
        chunks = parser.chunk(parse_result)
    except Exception as exc:  # noqa: BLE001
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
    diag_dir = _dump_diagnostics(parse_result.markdown, chunks, tables)

    def _fail(msg: str) -> None:
        print(f"\n[diagnostics] {diag_dir}")
        raise AssertionError(msg)

    # ---- (1) TableArtifact floor: >= 3 tables ------------------------------
    if len(tables) < 3:
        _fail(
            f"Expected >= 3 TableArtifact entries; got {len(tables)}. "
            f"sections: {[getattr(t, 'section_path', '') for t in tables]!r}"
        )

    # ---- (2) Each artifact has non-empty markdown / section_path / self_ref
    for i, tbl in enumerate(tables):
        md = getattr(tbl, "markdown", "") or ""
        sp = getattr(tbl, "section_path", "") or ""
        sr = getattr(tbl, "self_ref", "") or ""
        if not md.strip():
            _fail(f"Table[{i}] has empty markdown; section_path={sp!r}")
        if not sp.strip():
            _fail(f"Table[{i}] has empty section_path; self_ref={sr!r}")
        if not sr.strip():
            _fail(f"Table[{i}] has empty self_ref; section_path={sp!r}")

    # ---- (3) Heading provenance --------------------------------------------
    section_paths_lower = [
        (getattr(t, "section_path", "") or "").lower() for t in tables
    ]

    has_register_map = any("register map" in sp for sp in section_paths_lower)
    has_elec_char = any("electrical characteristics" in sp for sp in section_paths_lower)
    if not (has_register_map or has_elec_char):
        _fail(
            "No table had section_path containing 'Register Map' or "
            "'Electrical Characteristics'. "
            f"section_paths={section_paths_lower!r}"
        )

    # Nested heading for the bit-field table: section_path now reflects the
    # full ancestor chain, so we assert both the outer "Bit Field Details"
    # and the inner "CTRL Register Bits" co-occur in the same breadcrumb
    # (joined by " > "). Same-level Docling headings with no intervening
    # body content are now treated as parent/child rather than siblings.
    has_full_chain = any(
        "bit field details" in sp and "ctrl register bits" in sp
        for sp in section_paths_lower
    )
    if not has_full_chain:
        _fail(
            "No table had section_path containing the full nested chain "
            "'Bit Field Details > CTRL Register Bits' -- heading replay "
            f"across pages may be broken. section_paths={section_paths_lower!r}"
        )

    # ---- (4) Adaptive chunker: >= 3 table_summary chunks with distinct gids
    summary_chunks = [
        c for c in chunks
        if (getattr(c, "extra_metadata", {}) or {}).get("chunk_type") == "table_summary"
    ]
    if len(summary_chunks) < 3:
        _fail(
            f"Expected >= 3 table_summary chunks; got {len(summary_chunks)}. "
            f"chunk_types: "
            f"{sorted({(getattr(c, 'extra_metadata', {}) or {}).get('chunk_type') for c in chunks})}"
        )

    summary_group_ids = [
        (c.extra_metadata or {}).get("table_group_id", "") for c in summary_chunks
    ]
    if any(not gid for gid in summary_group_ids):
        _fail(
            f"Some table_summary chunks have empty table_group_id: "
            f"{summary_group_ids!r}"
        )
    if len(set(summary_group_ids)) != len(summary_group_ids):
        _fail(
            f"table_group_id collision across summary chunks: "
            f"{summary_group_ids!r}"
        )

    # ---- (5) Row chunks contiguous and share their summary's group id ------
    row_chunks = [
        c for c in chunks
        if (getattr(c, "extra_metadata", {}) or {}).get("chunk_type") == "table_row"
    ]
    if not row_chunks:
        _fail(
            "Expected at least one table_row chunk for the small uniform "
            "tables in this datasheet; got 0."
        )

    # Group rows by table_group_id and validate per-group contiguity.
    rows_by_gid: dict[str, list[int]] = {}
    for c in row_chunks:
        meta = c.extra_metadata or {}
        gid = meta.get("table_group_id", "")
        idx = meta.get("table_row_index", -1)
        if not gid:
            _fail(f"Row chunk missing table_group_id: meta={meta!r}")
        rows_by_gid.setdefault(gid, []).append(int(idx))

    # At least one group must have contiguous 0..N-1 row indices, and that
    # group's id must match one of the summary chunks.
    contiguous_found = False
    for gid, idxs in rows_by_gid.items():
        if gid not in summary_group_ids:
            _fail(
                f"Row chunks for gid={gid!r} have no matching table_summary "
                f"chunk. summary gids={summary_group_ids!r}"
            )
        sorted_idxs = sorted(idxs)
        if sorted_idxs == list(range(len(sorted_idxs))) and len(sorted_idxs) >= 4:
            contiguous_found = True
    if not contiguous_found:
        _fail(
            "No table_group_id had contiguous 0..N-1 row indices with N>=4. "
            f"rows_by_gid={rows_by_gid!r}"
        )

    # ---- (6) table_markdown present and non-empty on each summary ---------
    for c in summary_chunks:
        md = (c.extra_metadata or {}).get("table_markdown", "")
        if not md or not md.strip():
            _fail(
                "table_summary chunk has empty table_markdown; "
                f"gid={(c.extra_metadata or {}).get('table_group_id')!r}"
            )

    # ---- (7) Determinism: re-parse with a fresh parser, same group ids ----
    parser2 = DoclingParser()
    try:
        parse_result2 = parser2.parse(pdf_path, config)
        chunks2 = parser2.chunk(parse_result2)
    except Exception as exc:  # noqa: BLE001
        _fail(f"Re-parse for determinism check raised: {exc!r}")

    summary_gids_2 = {
        (c.extra_metadata or {}).get("table_group_id", "")
        for c in chunks2
        if (getattr(c, "extra_metadata", {}) or {}).get("chunk_type") == "table_summary"
    }
    summary_gids_1 = set(summary_group_ids)
    if summary_gids_1 != summary_gids_2:
        _fail(
            "table_group_id sets diverged across two parses -- not "
            "deterministic for idempotent re-ingest. "
            f"first={sorted(summary_gids_1)!r} second={sorted(summary_gids_2)!r}"
        )
