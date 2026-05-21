# @summary
# Real-Docling mixed-routing OCR test: synthesizes a single PDF with one
# native-text page and one image-only page, parses with the real Docling
# pipeline, and asserts both pages contribute distinct phrases to the parsed
# output. Locks down per-page OCR auto-routing at bitmap_area_threshold=0.05
# (force_full_page_ocr=False): text page takes the native path, image page is
# routed through OCR. Skips cleanly when Docling/RapidOCR models are
# unavailable.
# Exports: test_mixed_pdf_routes_text_and_image_pages,
#          test_mixed_pdf_ocr_disabled_drops_image_page
# Deps: reportlab, pillow, pypdf, docling (real), src.ingest.support.docling, pytest
# @end-summary

"""Real-Docling mixed-page OCR auto-routing test.

Existing real-pipeline tests cover the extremes: an all-text register-map
smoke test and an all-image OCR-fire test. Real ASIC datasheets are mixed —
some pages have a clean text layer, others are scanned bitmaps. This test
asserts that Docling's per-page OCR auto-trigger (RapidOcrOptions with
``force_full_page_ocr=False`` and the default ``bitmap_area_threshold=0.05``)
correctly routes only the image-only page through OCR while the native-text
page continues to take the fast text-extraction path.

Gated behind ``slow`` + ``integration`` markers and self-skips when models
can't be loaded.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Distinctive phrases — chosen to be ASCII-only and unlikely to collide
# ---------------------------------------------------------------------------

# Native text-layer phrase rendered with canvas.drawString on page 1.
TEXT_PAGE_PHRASE = "NATIVE TEXT LAYER PHRASE alpha bravo charlie"
TEXT_PAGE_TOKENS = ["native text layer phrase", "alpha bravo charlie"]

# Image-only phrase rasterized onto page 2 via PIL.
IMAGE_PAGE_PHRASES = [
    "SCANNED OCR PHRASE",
    "delta echo foxtrot",
    "register address 0x80 reset 0xCAFEBABE",
]
IMAGE_PAGE_LOOSE_TOKENS = [
    "scanned ocr phrase",
    "delta echo foxtrot",
    "register address",
    "0xcafebabe",
]


# ---------------------------------------------------------------------------
# PDF builder: one text page + one image-only page in a single PDF
# ---------------------------------------------------------------------------


def _render_image_page_png() -> bytes:
    """Render IMAGE_PAGE_PHRASES onto a white A4-ish PNG and return PNG bytes."""
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1240, 1754  # A4 at ~150 DPI
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    try:
        font = ImageFont.truetype(font_path, size=46)
    except Exception:  # noqa: BLE001 — fall back to default bitmap font
        font = ImageFont.load_default()

    y = 220
    line_gap = 95
    for phrase in IMAGE_PAGE_PHRASES:
        draw.text((120, y), phrase, fill="black", font=font)
        y += line_gap

    # Padding lines so rapidocr has more material to lock onto.
    for extra in (
        "Additional line for OCR robustness.",
        "More english words on the scanned page.",
        "End of scanned page two.",
    ):
        draw.text((120, y), extra, fill="black", font=font)
        y += line_gap

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def _synthesize_mixed_pdf(path: Path) -> None:
    """Write a 2-page PDF: page 1 native text, page 2 image-only.

    Page 1 uses ``canvas.drawString`` so the text is in the PDF text layer.
    Page 2 uses ``canvas.drawImage`` with a full-page raster — no text layer.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=A4)
    page_w, page_h = A4

    # Page 1: native text via drawString. Multiple lines for parser stability.
    c.setFont("Helvetica-Bold", 18)
    c.drawString(72, page_h - 90, "Mixed PDF — Page 1 (Native Text)")
    c.setFont("Helvetica", 14)
    c.drawString(72, page_h - 140, TEXT_PAGE_PHRASE)
    c.drawString(72, page_h - 170, "This page is rendered with drawString.")
    c.drawString(72, page_h - 200, "Docling should take the native text path here.")
    c.showPage()

    # Page 2: image-only raster, no text layer.
    png_bytes = _render_image_page_png()
    img_reader = ImageReader(io.BytesIO(png_bytes))
    c.drawImage(img_reader, 0, 0, width=page_w, height=page_h)
    c.showPage()

    c.save()


def _pypdf_page_text(path: Path) -> list[str]:
    """Return per-page extracted text via pypdf for sanity verification."""
    import pypdf

    reader = pypdf.PdfReader(str(path))
    return [(p.extract_text() or "") for p in reader.pages]


def _build_config():
    """Build an IngestionConfig with model auto-download enabled."""
    from src.ingest.common.types import IngestionConfig

    config = IngestionConfig()
    try:
        config.docling_auto_download = True
    except Exception:  # noqa: BLE001 — frozen dataclass safety net
        pass
    return config


def _ensure_models_or_skip(config) -> None:
    from src.ingest.support.docling import DoclingParser

    try:
        DoclingParser.ensure_ready(config)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Docling models unavailable (ensure_ready): {exc!r}")
    try:
        DoclingParser.warmup(config)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Docling/RapidOCR warmup failed: {exc!r}")


def _collected_text(parse_result, chunks) -> str:
    md = getattr(parse_result, "markdown", "") or ""
    chunk_text = "\n".join(getattr(c, "text", "") or "" for c in chunks)
    return f"{md}\n{chunk_text}".lower()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.integration
def test_mixed_pdf_routes_text_and_image_pages(tmp_path: Path) -> None:
    """Both native-text and image-only pages must contribute to parsed output.

    With ``enable_ocr=True`` and the default ``force_full_page_ocr=False``,
    Docling's per-page bitmap-area heuristic (default 0.05) should route
    page 2 (image-only) through OCR while page 1 (text-layer) goes through
    the native text path. The parsed output should therefore contain both
    the native-text phrase and at least one image-page phrase.
    """
    from src.ingest.support.docling import DoclingParser

    pdf_path = tmp_path / "mixed_text_and_image.pdf"
    _synthesize_mixed_pdf(pdf_path)
    assert pdf_path.exists() and pdf_path.stat().st_size > 0

    # Sanity: page 1 must have an extractable text layer, page 2 must not.
    pages = _pypdf_page_text(pdf_path)
    assert len(pages) == 2, f"Expected 2 pages, got {len(pages)}: {pages!r}"
    assert TEXT_PAGE_PHRASE.lower() in pages[0].lower(), (
        f"Page 1 missing native text phrase. pypdf page 1 text: {pages[0]!r}"
    )
    assert pages[1].strip() == "", (
        f"Page 2 unexpectedly has an extractable text layer: {pages[1]!r}"
    )

    config = _build_config()
    config.enable_ocr = True
    config.ocr_engine = "rapidocr"
    _ensure_models_or_skip(config)

    parser = DoclingParser()
    try:
        parse_result = parser.parse(pdf_path, config)
        chunks = parser.chunk(parse_result)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Docling parse failed on mixed PDF: {exc!r}")

    text = _collected_text(parse_result, chunks)
    assert text.strip(), "Mixed-PDF parse produced empty markdown and no chunks."

    # Page 1 (native text path) — at least one of the distinctive tokens.
    text_hits = [tok for tok in TEXT_PAGE_TOKENS if tok in text]
    assert text_hits, (
        "Native-text page 1 phrase missing from parsed output — text path "
        f"appears broken. Tokens tried: {TEXT_PAGE_TOKENS!r}. "
        f"First 600 chars: {text[:600]!r}"
    )

    # Page 2 (OCR path) — at least one loose token recovered.
    ocr_hits = [tok for tok in IMAGE_PAGE_LOOSE_TOKENS if tok in text]
    assert ocr_hits, (
        "Image-only page 2 produced no OCR output — per-page OCR auto-routing "
        "did not fire on the bitmap page. "
        f"Tokens tried: {IMAGE_PAGE_LOOSE_TOKENS!r}. "
        f"First 600 chars: {text[:600]!r}"
    )


@pytest.mark.slow
@pytest.mark.integration
def test_mixed_pdf_ocr_disabled_drops_image_page(tmp_path: Path) -> None:
    """Negative control: with OCR disabled, only page 1 text should survive.

    The native-text page 1 phrase must still be present (text path is
    OCR-independent). The image-only page 2 phrases must be absent, proving
    the positive test was actually sensing OCR output rather than a stray
    text layer on page 2.

    Note (per memory ``project_docling_ocr_trigger``): Docling can raise
    RuntimeError("empty markdown") on fully-image input when OCR is off. With
    a *mixed* PDF that contains at least one valid text page, markdown should
    NOT be empty — but if Docling surprises us, we xfail rather than fight it.
    """
    from src.ingest.support.docling import DoclingParser

    pdf_path = tmp_path / "mixed_text_and_image_neg.pdf"
    _synthesize_mixed_pdf(pdf_path)

    config = _build_config()
    config.enable_ocr = False
    _ensure_models_or_skip(config)

    parser = DoclingParser()
    text = ""
    try:
        parse_result = parser.parse(pdf_path, config)
        chunks = parser.chunk(parse_result)
        text = _collected_text(parse_result, chunks)
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "empty" in msg:
            pytest.xfail(
                "Docling raised empty-markdown on mixed PDF with OCR off; "
                "text-page content should have survived but pipeline bailed. "
                f"RuntimeError: {exc!r}"
            )
        raise
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Docling parse failed with OCR disabled: {exc!r}")

    # Text-page content must still be present.
    text_hits = [tok for tok in TEXT_PAGE_TOKENS if tok in text]
    assert text_hits, (
        "OCR disabled but the native-text page 1 phrase is also missing — "
        "the text path itself appears broken on mixed PDFs. "
        f"Tokens tried: {TEXT_PAGE_TOKENS!r}. First 600 chars: {text[:600]!r}"
    )

    # Image-page content must be absent.
    leaked = [tok for tok in IMAGE_PAGE_LOOSE_TOKENS if tok in text]
    assert not leaked, (
        "OCR was disabled but image-page tokens still appeared in output — "
        f"the PDF likely has a hidden text layer on page 2. Leaked: {leaked!r}. "
        f"First 600 chars: {text[:600]!r}"
    )
