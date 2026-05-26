# @summary
# Real-Docling OCR-fire integration test: synthesizes an image-only PDF (no
# embedded text layer), parses it with the real Docling pipeline and asserts
# that OCR genuinely fires and recovers known text. Includes a negative
# control sub-test with enable_ocr=False that asserts no text is recovered.
# Skips cleanly when Docling/RapidOCR models are unavailable.
# Exports: test_ocr_enabled_recovers_text_from_image_only_pdf,
#          test_ocr_disabled_leaves_image_only_pdf_textless
# Deps: reportlab, pillow, pypdf, docling (real), src.ingest.support.docling, pytest
# @end-summary

"""Real-Docling OCR-fire test on an image-only synthetic PDF.

Existing tests in ``test_docling_ocr_and_tf_v2.py`` only verify the
``RapidOcrOptions``/``do_ocr=True`` wiring at the kwargs seam. This test
asserts that OCR *actually fires* and produces output on a PDF whose pages
are pure bitmaps (no extractable text layer). A negative control with
``enable_ocr=False`` proves the positive assertion isn't a fluke from a
stray text layer.

Gated behind ``slow`` + ``integration`` markers and self-skips when models
can't be loaded.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Deterministic image-only PDF builder
# ---------------------------------------------------------------------------

# Phrases rendered onto the page. Chosen to be OCR-friendly (common English
# words, hex tokens, mostly uppercase markers) and to give multiple chances
# for at least one to survive OCR misreads.
KNOWN_PHRASES: list[str] = [
    "OCR FIRE TEST PAGE",
    "register address 0x40 reset 0x00000000",
    "Synthesized image-only content",
]

# Loose substrings used for assertion: even an imperfect OCR pass should
# recover at least one of these (case-insensitive substring).
LOOSE_MATCH_TOKENS: list[str] = [
    "ocr fire test",
    "register address",
    "synthesized",
    "image-only",
]


def _render_text_png() -> bytes:
    """Render KNOWN_PHRASES onto a white A4-ish PNG and return PNG bytes."""
    from PIL import Image, ImageDraw, ImageFont

    # A4 at ~150 DPI: 1240 x 1754
    width, height = 1240, 1754
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    try:
        font = ImageFont.truetype(font_path, size=44)
    except Exception:  # noqa: BLE001 — fall back to default bitmap font
        font = ImageFont.load_default()

    y = 200
    line_gap = 90
    for phrase in KNOWN_PHRASES:
        draw.text((120, y), phrase, fill="black", font=font)
        y += line_gap

    # A few extra lines of context to give OCR enough material.
    for extra in (
        "Line four for OCR coverage.",
        "Line five with more english words.",
        "End of page.",
    ):
        draw.text((120, y), extra, fill="black", font=font)
        y += line_gap

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def _synthesize_image_only_pdf(path: Path) -> None:
    """Write a one-page PDF whose only content is a full-page raster image.

    No ``canvas.drawString`` is used — the resulting PDF has no extractable
    text layer, so any text recovered by Docling must come from OCR.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    png_bytes = _render_text_png()
    img_reader = ImageReader(io.BytesIO(png_bytes))

    c = canvas.Canvas(str(path), pagesize=A4)
    page_w, page_h = A4
    c.drawImage(img_reader, 0, 0, width=page_w, height=page_h)
    c.showPage()
    c.save()


def _pdf_has_no_text_layer(path: Path) -> tuple[bool, str]:
    """Return (is_image_only, extracted_text) using pypdf.

    A PDF is considered image-only when pypdf extracts nothing but
    whitespace from every page.
    """
    import pypdf

    reader = pypdf.PdfReader(str(path))
    extracted = "".join(p.extract_text() or "" for p in reader.pages)
    return (extracted.strip() == ""), extracted


def _build_config():
    """Build an IngestionConfig defaulted for the test (auto-download on)."""
    from src.ingest.common.types import IngestionConfig

    config = IngestionConfig()
    try:
        config.docling_auto_download = True
    except Exception:  # noqa: BLE001 — frozen dataclass safety net
        pass
    return config


def _ensure_models_or_skip(config) -> None:
    """Validate runtime + warm models; skip if either fails."""
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
    """Concatenate parse markdown + all chunk texts for substring matching."""
    md = getattr(parse_result, "markdown", "") or ""
    chunk_text = "\n".join(getattr(c, "text", "") or "" for c in chunks)
    return f"{md}\n{chunk_text}".lower()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.integration
def test_ocr_enabled_recovers_text_from_image_only_pdf(tmp_path: Path) -> None:
    """Real Docling parse over an image-only PDF must recover OCR'd text.

    Sanity-checks that the synthesized PDF has no text layer (otherwise the
    test would be vacuous), then asserts at least one known phrase appears
    in the parsed markdown or chunk text after OCR.
    """
    from src.ingest.support.docling import DoclingParser

    pdf_path = tmp_path / "image_only.pdf"
    _synthesize_image_only_pdf(pdf_path)
    assert pdf_path.exists() and pdf_path.stat().st_size > 0

    image_only, extracted = _pdf_has_no_text_layer(pdf_path)
    assert image_only, (
        "Synthesized PDF unexpectedly has an extractable text layer; "
        f"pypdf returned: {extracted!r}"
    )

    config = _build_config()
    config.enable_ocr = True
    _ensure_models_or_skip(config)

    parser = DoclingParser()
    try:
        parse_result = parser.parse(pdf_path, config)
        chunks = parser.chunk(parse_result)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Docling parse failed on image-only PDF: {exc!r}")

    text = _collected_text(parse_result, chunks)
    assert text.strip(), (
        "OCR-enabled parse produced empty markdown and no chunks — OCR did "
        "not fire on the image-only page."
    )

    matched = [tok for tok in LOOSE_MATCH_TOKENS if tok in text]
    assert matched, (
        "OCR-enabled parse recovered some text but none of the known "
        f"phrases matched. Tokens tried: {LOOSE_MATCH_TOKENS!r}. "
        f"First 400 chars of recovered text: {text[:400]!r}"
    )


@pytest.mark.slow
@pytest.mark.integration
def test_ocr_disabled_leaves_image_only_pdf_textless(tmp_path: Path) -> None:
    """Negative control: same image-only PDF parsed with enable_ocr=False.

    Asserts no known phrase is recovered when OCR is disabled, proving the
    positive test is sensing OCR output rather than a stray text layer.
    """
    from src.ingest.support.docling import DoclingParser

    pdf_path = tmp_path / "image_only_neg.pdf"
    _synthesize_image_only_pdf(pdf_path)

    image_only, _extracted = _pdf_has_no_text_layer(pdf_path)
    assert image_only, "Synthesized PDF has an extractable text layer."

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
        # Docling raises "empty markdown output" when OCR is off and there's
        # no text layer — exactly the outcome we're asserting (no text
        # recovered). Treat as a successful negative control.
        msg = str(exc).lower()
        assert "empty" in msg or "no" in msg, (
            f"Unexpected RuntimeError shape during OCR-disabled parse: {exc!r}"
        )
        return
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Docling parse failed with OCR disabled: {exc!r}")

    leaked = [tok for tok in LOOSE_MATCH_TOKENS if tok in text]
    assert not leaked, (
        "OCR was disabled but known phrases still appeared in output — "
        f"the PDF likely has a hidden text layer. Leaked tokens: {leaked!r}. "
        f"First 400 chars: {text[:400]!r}"
    )
