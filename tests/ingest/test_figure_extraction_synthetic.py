"""Synthetic-Docling tests for ``_extract_figure_artifacts``.

These tests use duck-typed namespaces in place of real DoclingDocument /
PictureItem objects. They cover the contract — list shape, caption parsing,
section_path replay, page_no, image_uri, and defensive fallbacks. The real
extraction-quality signal lives in ``test_real_docling_figures.py``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.ingest.support.docling import _extract_figure_artifacts
from src.ingest.support.parser_base import FigureArtifact


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _label(value: str) -> SimpleNamespace:
    """Mimic Docling's DocItemLabel enum (has ``.value``)."""
    return SimpleNamespace(value=value)


def _heading(text: str, level: int = 1, self_ref: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        label=_label("section_header"),
        text=text,
        level=level,
        self_ref=self_ref or f"#/headings/{text}",
    )


def _picture(
    self_ref: str,
    caption: str = "",
    page_no: int | None = None,
    image_uri: str | None = None,
) -> MagicMock:
    pic = MagicMock()
    pic.label = _label("picture")
    pic.self_ref = self_ref
    # captions is truthy iff a caption is supplied so the defensive
    # ``if getattr(pic, "captions", None)`` gate fires correctly.
    if caption:
        pic.captions = [SimpleNamespace(cref=f"{self_ref}/cap")]
        pic.caption_text.return_value = caption
    else:
        pic.captions = []
        pic.caption_text.return_value = ""
    if page_no is not None:
        pic.prov = [SimpleNamespace(page_no=page_no, bbox=None)]
    else:
        pic.prov = []
    if image_uri is not None:
        pic.image = SimpleNamespace(uri=image_uri)
    else:
        pic.image = None
    return pic


def _doc(pictures, walk_items=None):
    """Build a fake DoclingDocument whose ``iterate_items()`` replays the
    supplied flat sibling order, mirroring real Docling's body topology."""
    doc = SimpleNamespace()
    doc.pictures = pictures
    doc.tables = []
    if walk_items is None:
        walk_items = pictures
    doc.iterate_items = lambda: iter([(it, 0) for it in walk_items])
    return doc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_extract_figure_artifacts_none_doc_returns_empty():
    assert _extract_figure_artifacts(None) == []


def test_extract_figure_artifacts_no_pictures_attr():
    doc = SimpleNamespace()
    assert _extract_figure_artifacts(doc) == []


def test_extract_figure_artifacts_basic_shape():
    pic = _picture("#/pictures/0", caption="Figure 1: clock tree", page_no=12)
    doc = _doc([pic])
    figs = _extract_figure_artifacts(doc, document_id="esp32-s3")

    assert len(figs) == 1
    fig = figs[0]
    assert isinstance(fig, FigureArtifact)
    assert fig.document_id == "esp32-s3"
    assert fig.caption == "Figure 1: clock tree"
    assert fig.caption_label == "Figure 1"
    assert fig.page_no == 12
    assert fig.self_ref == "#/pictures/0"
    assert fig.image_uri == ""


def test_extract_figure_artifacts_caption_label_variants():
    captions = [
        ("Figure 4-1: SoC block diagram", "Figure 4-1"),
        ("Fig. 2 — pin layout", "Figure 2"),
        ("Block diagram without prefix", ""),
        ("", ""),
    ]
    pics = [
        _picture(f"#/pictures/{i}", caption=cap)
        for i, (cap, _) in enumerate(captions)
    ]
    doc = _doc(pics)
    figs = _extract_figure_artifacts(doc)
    assert [f.caption_label for f in figs] == [exp for _, exp in captions]


def test_extract_figure_artifacts_section_path_replay():
    """Heading-stack replay: pictures interleaved with section_headers
    should pick up the active heading breadcrumb at their position."""
    h1 = _heading("Chapter 4", level=1, self_ref="#/headings/ch4")
    h2 = _heading("Functional Description", level=2, self_ref="#/headings/fd")
    pic_a = _picture("#/pictures/0", caption="Figure 4-1: SoC")
    h3 = _heading("Power", level=2, self_ref="#/headings/pow")
    pic_b = _picture("#/pictures/1", caption="Figure 4-2: rails")

    doc = _doc(
        pictures=[pic_a, pic_b],
        walk_items=[h1, h2, pic_a, h3, pic_b],
    )
    figs = _extract_figure_artifacts(doc, document_id="d")
    by_ref = {f.self_ref: f for f in figs}

    assert by_ref["#/pictures/0"].section_path == "Chapter 4 > Functional Description"
    assert by_ref["#/pictures/1"].section_path == "Chapter 4 > Power"


def test_extract_figure_artifacts_image_uri_passthrough():
    pic = _picture(
        "#/pictures/0",
        caption="Figure 3",
        image_uri="data:image/png;base64,AAA",
    )
    doc = _doc([pic])
    figs = _extract_figure_artifacts(doc)
    assert figs[0].image_uri == "data:image/png;base64,AAA"


def test_extract_figure_artifacts_tolerates_caption_text_raising():
    pic = MagicMock()
    pic.label = _label("picture")
    pic.self_ref = "#/pictures/0"
    pic.captions = [object()]  # truthy → caption_text path is taken
    pic.caption_text.side_effect = RuntimeError("caption resolve blew up")
    pic.prov = []
    pic.image = None
    doc = _doc([pic])

    figs = _extract_figure_artifacts(doc)
    assert len(figs) == 1
    assert figs[0].caption == ""
    assert figs[0].caption_label == ""


@pytest.mark.parametrize("bad_page_no", ["not-a-number", None])
def test_extract_figure_artifacts_invalid_page_no_falls_back_to_zero(bad_page_no):
    pic = _picture("#/pictures/0", caption="Figure 1")
    pic.prov = [SimpleNamespace(page_no=bad_page_no, bbox=None)]
    doc = _doc([pic])
    figs = _extract_figure_artifacts(doc)
    assert figs[0].page_no == 0
