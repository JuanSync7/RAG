"""Tests for src/ingest/support/vision.py — pure logic + mock-based."""

from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.ingest.support.vision import (
    VisionDescription,
    VisionImageCandidate,
    _decode_data_url,
    _describe_image,
    _extract_image_candidates,
    _resolve_file_image_bytes,
    ensure_vision_ready,
    generate_vision_notes,
)

# ---------------------------------------------------------------------------
# Shared config mock
# ---------------------------------------------------------------------------

_CFG = type(
    "Config",
    (),
    {
        "enable_vision_processing": True,
        "vision_temperature": 0.1,
        "vision_max_tokens": 500,
        "vision_timeout_seconds": 30,
        "vision_max_figures": 5,
        "vision_max_image_bytes": 10_000_000,
    },
)()

_CFG_DISABLED = type(
    "Config",
    (),
    {
        "enable_vision_processing": False,
        "vision_temperature": 0.1,
        "vision_max_tokens": 500,
        "vision_timeout_seconds": 30,
        "vision_max_figures": 5,
        "vision_max_image_bytes": 10_000_000,
    },
)()


# ---------------------------------------------------------------------------
# VisionDescription.as_note()
# ---------------------------------------------------------------------------


def test_as_note_with_visible_text_and_tags():
    desc = VisionDescription(
        figure_label="Figure 1",
        source_ref="data-url",
        caption="A chart",
        visible_text="Revenue Q4",
        tags=["chart", "bar", "revenue"],
    )
    note = desc.as_note()
    assert "Figure 1: A chart" in note
    assert "text=Revenue Q4" in note
    assert "tags=chart, bar, revenue" in note
    assert " | " in note


def test_as_note_with_visible_text_only():
    desc = VisionDescription(
        figure_label="Figure 2",
        source_ref="data-url",
        caption="Photo",
        visible_text="Hello world",
        tags=[],
    )
    note = desc.as_note()
    assert "text=Hello world" in note
    assert "tags=" not in note


def test_as_note_with_tags_only():
    desc = VisionDescription(
        figure_label="Figure 3",
        source_ref="data-url",
        caption="Diagram",
        visible_text="",
        tags=["network", "topology"],
    )
    note = desc.as_note()
    assert "text=" not in note
    assert "tags=network, topology" in note


def test_as_note_empty():
    desc = VisionDescription(
        figure_label="Figure 4",
        source_ref="data-url",
        caption="Plain",
        visible_text="",
        tags=[],
    )
    note = desc.as_note()
    assert note == "Figure 4: Plain"


def test_as_note_truncates_at_8_tags():
    desc = VisionDescription(
        figure_label="Figure 5",
        source_ref="data-url",
        caption="X",
        visible_text="",
        tags=[f"tag{i}" for i in range(12)],
    )
    note = desc.as_note()
    # Only the first 8 tags
    assert "tag8" not in note
    assert "tag7" in note


# ---------------------------------------------------------------------------
# _decode_data_url()
# ---------------------------------------------------------------------------


def _make_data_url(data: bytes, mime: str = "image/png") -> str:
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def test_decode_data_url_valid():
    raw = b"\x89PNG\r\nfakedata"
    src = _make_data_url(raw, "image/png")
    result = _decode_data_url(src)
    assert result is not None
    img_bytes, mime = result
    assert img_bytes == raw
    assert mime == "image/png"


def test_decode_data_url_invalid_base64():
    # Include characters that are truly illegal in base64 (e.g. non-ASCII bytes encoded as escape)
    # Python's b64decode with validate=False is lenient; the source code uses binascii.Error catch.
    # Trigger it with a string that has only invalid padding (odd length with no valid decode).
    # The safest approach: use a string that produces binascii.Error: embed a non-ASCII char.
    src = "data:image/png;base64,\xff\xfe"  # non-ASCII → binascii.Error
    result = _decode_data_url(src)
    assert result is None


def test_decode_data_url_non_data_url():
    result = _decode_data_url("https://example.com/image.png")
    assert result is None


def test_decode_data_url_empty_string():
    result = _decode_data_url("")
    assert result is None


def test_decode_data_url_missing_mime():
    raw = b"pixels"
    b64 = base64.b64encode(raw).decode("ascii")
    src = f"data:;base64,{b64}"
    result = _decode_data_url(src)
    assert result is not None
    _, mime = result
    assert mime == "image/png"  # falls back to image/png


# ---------------------------------------------------------------------------
# _resolve_file_image_bytes()
# ---------------------------------------------------------------------------


def test_resolve_file_image_bytes_file_uri(tmp_path):
    img_file = tmp_path / "img.png"
    img_file.write_bytes(b"\x89PNG")
    src = f"file://{img_file}"
    result = _resolve_file_image_bytes(src, source_path=tmp_path / "doc.md")
    assert result is not None
    data, mime, resolved = result
    assert data == b"\x89PNG"
    assert "png" in mime


def test_resolve_file_image_bytes_relative_path(tmp_path):
    img_file = tmp_path / "assets" / "photo.jpg"
    img_file.parent.mkdir(parents=True)
    img_file.write_bytes(b"JFIF")
    source_path = tmp_path / "docs" / "doc.md"
    source_path.parent.mkdir(parents=True)
    src = "../assets/photo.jpg"
    result = _resolve_file_image_bytes(src, source_path=source_path)
    assert result is not None
    data, _, _ = result
    assert data == b"JFIF"


def test_resolve_file_image_bytes_http_returns_none(tmp_path):
    result = _resolve_file_image_bytes("http://example.com/img.png", source_path=tmp_path / "doc.md")
    assert result is None


def test_resolve_file_image_bytes_https_returns_none(tmp_path):
    result = _resolve_file_image_bytes("https://example.com/img.png", source_path=tmp_path / "doc.md")
    assert result is None


def test_resolve_file_image_bytes_empty_returns_none(tmp_path):
    result = _resolve_file_image_bytes("", source_path=tmp_path / "doc.md")
    assert result is None


def test_resolve_file_image_bytes_nonexistent_returns_none(tmp_path):
    result = _resolve_file_image_bytes("nonexistent.png", source_path=tmp_path / "doc.md")
    assert result is None


# ---------------------------------------------------------------------------
# _extract_image_candidates() — mock-based
# ---------------------------------------------------------------------------


def _write_small_png(path: Path) -> None:
    path.write_bytes(b"\x89PNG\r\nfakeimage")


def test_mock_extract_image_candidates_data_url(tmp_path):
    raw = b"\x89PNG\r\nfakeimage"
    data_url = _make_data_url(raw, "image/png")
    md = f"![Alt text]({data_url})"
    candidates = _extract_image_candidates(
        md,
        source_path=tmp_path / "doc.md",
        max_figures=5,
        max_image_bytes=10_000_000,
    )
    assert len(candidates) == 1
    assert candidates[0].source_ref == "data-url"
    assert candidates[0].alt_text == "Alt text"
    assert candidates[0].mime_type == "image/png"


def test_mock_extract_image_candidates_file_image(tmp_path):
    img = tmp_path / "img.png"
    _write_small_png(img)
    md = f"![Figure](img.png)"
    candidates = _extract_image_candidates(
        md,
        source_path=tmp_path / "doc.md",
        max_figures=5,
        max_image_bytes=10_000_000,
    )
    assert len(candidates) == 1
    assert str(img) in candidates[0].source_ref


def test_mock_extract_image_candidates_max_figures_limit(tmp_path):
    imgs = []
    for i in range(4):
        img = tmp_path / f"img{i}.png"
        _write_small_png(img)
        imgs.append(img)

    lines = "\n".join(f"![fig{i}](img{i}.png)" for i in range(4))
    candidates = _extract_image_candidates(
        lines,
        source_path=tmp_path / "doc.md",
        max_figures=2,
        max_image_bytes=10_000_000,
    )
    assert len(candidates) == 2


def test_mock_extract_image_candidates_max_image_bytes_limit(tmp_path):
    img = tmp_path / "big.png"
    img.write_bytes(b"X" * 200)  # 200 bytes
    md = "![big](big.png)"
    candidates = _extract_image_candidates(
        md,
        source_path=tmp_path / "doc.md",
        max_figures=5,
        max_image_bytes=100,  # only allow 100 bytes
    )
    assert len(candidates) == 0


def test_mock_extract_image_candidates_http_skipped(tmp_path):
    md = "![external](https://example.com/img.png)"
    candidates = _extract_image_candidates(
        md,
        source_path=tmp_path / "doc.md",
        max_figures=5,
        max_image_bytes=10_000_000,
    )
    assert len(candidates) == 0


# ---------------------------------------------------------------------------
# _describe_image() — mock-based
# ---------------------------------------------------------------------------


def _make_candidate(figure_label: str = "Figure 1") -> VisionImageCandidate:
    return VisionImageCandidate(
        figure_label=figure_label,
        alt_text="alt",
        source_ref="data-url",
        image_b64=base64.b64encode(b"fake").decode("ascii"),
        mime_type="image/png",
    )


def test_mock_describe_image_success():
    mock_provider = MagicMock()
    mock_response = MagicMock()
    mock_response.content = '{"caption": "A bar chart", "visible_text": "Q4", "tags": ["chart", "bar"]}'
    mock_provider.vision_completion.return_value = mock_response

    with patch("src.ingest.support.vision.get_llm_provider", return_value=mock_provider):
        candidate = _make_candidate()
        result = _describe_image(candidate, _CFG)

    assert result is not None
    assert result.caption == "A bar chart"
    assert result.visible_text == "Q4"
    assert "chart" in result.tags


def test_mock_describe_image_exception_returns_none():
    mock_provider = MagicMock()
    mock_provider.vision_completion.side_effect = RuntimeError("VLM not available")

    with patch("src.ingest.support.vision.get_llm_provider", return_value=mock_provider):
        candidate = _make_candidate()
        result = _describe_image(candidate, _CFG)

    assert result is None


def test_mock_describe_image_empty_caption_uses_default():
    mock_provider = MagicMock()
    mock_response = MagicMock()
    mock_response.content = '{"caption": "", "visible_text": "", "tags": []}'
    mock_provider.vision_completion.return_value = mock_response

    with patch("src.ingest.support.vision.get_llm_provider", return_value=mock_provider):
        candidate = _make_candidate()
        result = _describe_image(candidate, _CFG)

    assert result is not None
    assert result.caption == "No caption generated."


# ---------------------------------------------------------------------------
# ensure_vision_ready() — mock-based
# ---------------------------------------------------------------------------


def test_mock_ensure_vision_ready_disabled_early_return():
    """Vision disabled: no provider call, no exception."""
    with patch("src.ingest.support.vision.get_llm_provider") as mock_get:
        ensure_vision_ready(_CFG_DISABLED)
        mock_get.assert_not_called()


def test_mock_ensure_vision_ready_enabled_available():
    mock_provider = MagicMock()
    mock_provider.is_available.return_value = True

    with patch("src.ingest.support.vision.get_llm_provider", return_value=mock_provider):
        ensure_vision_ready(_CFG)  # should not raise

    mock_provider.is_available.assert_called_once_with(model_alias="vision")


def test_mock_ensure_vision_ready_enabled_not_available():
    mock_provider = MagicMock()
    mock_provider.is_available.return_value = False

    with patch("src.ingest.support.vision.get_llm_provider", return_value=mock_provider):
        with pytest.raises(RuntimeError, match="not reachable"):
            ensure_vision_ready(_CFG)


# ---------------------------------------------------------------------------
# generate_vision_notes() — mock-based
# ---------------------------------------------------------------------------


def test_mock_generate_vision_notes_no_images(tmp_path):
    md = "# Title\n\nNo images here."
    with patch("src.ingest.support.vision.get_llm_provider"):
        notes, count = generate_vision_notes(md, source_path=tmp_path / "doc.md", config=_CFG)
    assert notes == []
    assert count == 0


def test_mock_generate_vision_notes_with_images(tmp_path):
    img = tmp_path / "chart.png"
    _write_small_png(img)
    md = "![Chart](chart.png)"

    mock_provider = MagicMock()
    mock_response = MagicMock()
    mock_response.content = '{"caption": "Revenue chart", "visible_text": "Q4", "tags": ["revenue"]}'
    mock_provider.vision_completion.return_value = mock_response

    with patch("src.ingest.support.vision.get_llm_provider", return_value=mock_provider):
        notes, count = generate_vision_notes(md, source_path=tmp_path / "doc.md", config=_CFG)

    assert count == 1
    assert len(notes) == 1
    assert "Revenue chart" in notes[0]


def test_mock_generate_vision_notes_failed_description_skipped(tmp_path):
    img = tmp_path / "chart.png"
    _write_small_png(img)
    md = "![Chart](chart.png)"

    mock_provider = MagicMock()
    mock_provider.vision_completion.side_effect = RuntimeError("VLM unavailable")

    with patch("src.ingest.support.vision.get_llm_provider", return_value=mock_provider):
        notes, count = generate_vision_notes(md, source_path=tmp_path / "doc.md", config=_CFG)

    assert notes == []
    assert count == 0
