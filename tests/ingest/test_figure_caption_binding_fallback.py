"""FIG-4 triage fix: when Docling fails to bind a caption TextItem to a
PictureItem (``pic.captions`` empty), ``_extract_figure_artifacts`` should
fall back to the nearest forward TextItem on the same page whose text
matches the canonical ``Figure N`` / ``Figure N-N`` / ``Figure N.N`` regex.

Background: triage of the 2026-05-24 ESP32-S3 soak showed that 4 of 5
unresolved prose-figure refs (Figure 2-1, 4-2, 7-1, 7-2) had their caption
text correctly emitted by Docling as a TextItem on the right page, but
``pic.captions`` was empty on the matching PictureItem, so the resolver
filter (on ``caption_label`` equality) could not match. This fallback
closes that gap without touching Docling itself.

Triage report: ``docs/soak/figure_artifacts_esp32_2026-05-24_triage.md``.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from src.ingest.support.docling import _extract_figure_artifacts


def _label(value: str) -> SimpleNamespace:
    return SimpleNamespace(value=value)


def _picture(self_ref: str, page_no: int) -> MagicMock:
    """A PictureItem with NO bound captions — the failure mode we observed."""
    pic = MagicMock()
    pic.label = _label("picture")
    pic.self_ref = self_ref
    pic.captions = []  # the bug: caption not bound
    pic.caption_text.return_value = ""
    pic.prov = [SimpleNamespace(page_no=page_no, bbox=None)]
    pic.image = None
    return pic


def _textitem(text: str, page_no: int, self_ref: str) -> SimpleNamespace:
    return SimpleNamespace(
        label=_label("text"),
        text=text,
        self_ref=self_ref,
        prov=[SimpleNamespace(page_no=page_no, bbox=None)],
    )


def _doc(pictures, walk_items):
    doc = SimpleNamespace()
    doc.pictures = pictures
    doc.tables = []
    doc.iterate_items = lambda: iter([(it, 0) for it in walk_items])
    return doc


def test_caption_binding_fallback_forward_textitem_same_page():
    """ESP32-S3 page-77 pattern: PictureItem with no captions, followed
    immediately by a "Figure 7-1. ..." TextItem on the SAME page. The
    extractor must adopt the TextItem text as the caption."""
    pic = _picture("#/pictures/77", page_no=77)
    cap = _textitem("Figure 7-1. QFN56 (7x7 mm) Package", page_no=77,
                    self_ref="#/texts/cap-77")

    doc = _doc(pictures=[pic], walk_items=[pic, cap])
    figs = _extract_figure_artifacts(doc, document_id="esp32-s3")

    assert len(figs) == 1
    assert figs[0].caption_label == "Figure 7-1"
    assert figs[0].caption.startswith("Figure 7-1")


def test_caption_binding_fallback_does_not_cross_page():
    """An unbound PictureItem on page N must NOT adopt a Figure-caption
    TextItem that lives on page N+1 — that would invent provenance."""
    pic = _picture("#/pictures/0", page_no=42)
    cap_next_page = _textitem(
        "Figure 7-1. Package", page_no=43, self_ref="#/texts/cap-43"
    )

    doc = _doc(pictures=[pic], walk_items=[pic, cap_next_page])
    figs = _extract_figure_artifacts(doc, document_id="d")

    assert len(figs) == 1
    assert figs[0].caption == ""
    assert figs[0].caption_label == ""


def test_caption_binding_fallback_ignores_non_figure_textitem():
    """Forward TextItems that aren't figure captions must not be adopted."""
    pic = _picture("#/pictures/0", page_no=10)
    body = _textitem("This paragraph happens to follow the picture.",
                     page_no=10, self_ref="#/texts/body")

    doc = _doc(pictures=[pic], walk_items=[pic, body])
    figs = _extract_figure_artifacts(doc, document_id="d")

    assert figs[0].caption == ""
    assert figs[0].caption_label == ""


def test_caption_binding_fallback_skipped_when_pic_captions_present():
    """If Docling DID bind a caption, we trust it — fallback must not fire."""
    pic = MagicMock()
    pic.label = _label("picture")
    pic.self_ref = "#/pictures/0"
    pic.captions = [SimpleNamespace(cref="#/pictures/0/cap")]
    pic.caption_text.return_value = "Figure 1. Real caption"
    pic.prov = [SimpleNamespace(page_no=5, bbox=None)]
    pic.image = None

    distractor = _textitem("Figure 99. Wrong caption", page_no=5,
                           self_ref="#/texts/x")

    doc = _doc(pictures=[pic], walk_items=[pic, distractor])
    figs = _extract_figure_artifacts(doc, document_id="d")

    assert figs[0].caption == "Figure 1. Real caption"
    assert figs[0].caption_label == "Figure 1"
