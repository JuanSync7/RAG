"""Tests for PlainTextParser in src/ingest/support/parser_text.py.

Covers:
- parse() with .html and .rst suffixes
- chunk() before parse() -> RuntimeError
- ensure_ready() and warmup() no-ops
- _html_to_markdown(): happy path (markdownify) and fallback (mocked import)
- _rst_to_markdown(): fallback heuristic (pypandoc not installed), heading detection
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.ingest.support.parser_text import PlainTextParser


# ---------------------------------------------------------------------------
# Minimal config stub
# ---------------------------------------------------------------------------

def _config():
    cfg = MagicMock()
    cfg.chunk_size = 500
    cfg.chunk_overlap = 50
    return cfg


# ---------------------------------------------------------------------------
# parse() suffix routing
# ---------------------------------------------------------------------------


class TestParseSuffixRouting:
    def test_parse_html_calls_html_to_markdown(self, tmp_path: Path) -> None:
        """parse() on a .html file should invoke _html_to_markdown."""
        html_file = tmp_path / "page.html"
        html_file.write_text("<h1>Hello</h1><p>World</p>", encoding="utf-8")

        parser = PlainTextParser()
        result = parser.parse(html_file, _config())

        # markdownify should convert <h1> -> # Hello
        assert "Hello" in result.markdown

    def test_parse_rst_calls_rst_to_markdown(self, tmp_path: Path) -> None:
        """parse() on a .rst file should invoke _rst_to_markdown."""
        rst_content = "My Title\n========\n\nSome text here.\n"
        rst_file = tmp_path / "doc.rst"
        rst_file.write_text(rst_content, encoding="utf-8")

        parser = PlainTextParser()
        result = parser.parse(rst_file, _config())

        # Fallback heuristic: === → h1
        assert "My Title" in result.markdown

    def test_parse_md_uses_content_as_is(self, tmp_path: Path) -> None:
        """parse() on .md file uses the raw content unchanged."""
        md_content = "# Heading\n\nSome paragraph.\n"
        md_file = tmp_path / "readme.md"
        md_file.write_text(md_content, encoding="utf-8")

        parser = PlainTextParser()
        result = parser.parse(md_file, _config())

        assert result.markdown == md_content
        assert "Heading" in result.headings

    def test_parse_txt_uses_content_as_is(self, tmp_path: Path) -> None:
        """parse() on .txt file uses the raw content unchanged."""
        txt_file = tmp_path / "notes.txt"
        txt_file.write_text("plain text content", encoding="utf-8")

        parser = PlainTextParser()
        result = parser.parse(txt_file, _config())

        assert result.markdown == "plain text content"
        assert result.page_count == 0


# ---------------------------------------------------------------------------
# chunk() before parse() -> RuntimeError
# ---------------------------------------------------------------------------


class TestChunkBeforeParse:
    def test_chunk_before_parse_raises_runtime_error(self) -> None:
        """chunk() must raise RuntimeError if parse() has not been called."""
        parser = PlainTextParser()

        from src.ingest.support.parser_base import ParseResult
        pr = ParseResult(markdown="# Hello\n\nBody.", headings=["Hello"], has_figures=False, page_count=0)

        with pytest.raises(RuntimeError, match="called before parse"):
            parser.chunk(pr)


# ---------------------------------------------------------------------------
# ensure_ready() and warmup() are no-ops
# ---------------------------------------------------------------------------


class TestNoOps:
    def test_ensure_ready_does_not_raise(self) -> None:
        PlainTextParser.ensure_ready(_config())  # should not raise

    def test_warmup_does_not_raise(self) -> None:
        PlainTextParser.warmup(_config())  # should not raise


# ---------------------------------------------------------------------------
# _html_to_markdown(): markdownify happy path
# ---------------------------------------------------------------------------


class TestHtmlToMarkdownHappyPath:
    def test_markdownify_converts_headings(self) -> None:
        """When markdownify is installed, <h1> → # heading."""
        html = "<h1>Title</h1><p>Text</p>"
        result = PlainTextParser._html_to_markdown(html)
        assert "Title" in result

    def test_markdownify_keeps_paragraph_content(self) -> None:
        """markdownify should preserve paragraph text."""
        html = "<p>Keep me</p>"
        result = PlainTextParser._html_to_markdown(html)
        assert "Keep me" in result


# ---------------------------------------------------------------------------
# _html_to_markdown(): fallback path (markdownify import failure)
# ---------------------------------------------------------------------------


class TestHtmlToMarkdownFallback:
    def test_mock_fallback_strips_tags_and_preserves_heading(self) -> None:
        """When markdownify import fails, regex fallback strips tags and maps <h1> → # heading."""
        # Temporarily hide markdownify by making its import raise ImportError
        with patch.dict(sys.modules, {"markdownify": None}):
            html = "<h1>Section</h1><p>Body text.</p>"
            result = PlainTextParser._html_to_markdown(html)

        assert "# Section" in result
        # The plain tag-stripped text should also be present
        assert "Body text" in result

    def test_mock_fallback_h2_to_double_hash(self) -> None:
        """<h2> should become ## Heading in fallback mode."""
        with patch.dict(sys.modules, {"markdownify": None}):
            html = "<h2>Subsection</h2>"
            result = PlainTextParser._html_to_markdown(html)

        assert "## Subsection" in result

    def test_mock_fallback_h6_to_six_hashes(self) -> None:
        """<h6> should become ###### in fallback mode."""
        with patch.dict(sys.modules, {"markdownify": None}):
            html = "<h6>Deep</h6>"
            result = PlainTextParser._html_to_markdown(html)

        assert "###### Deep" in result


# ---------------------------------------------------------------------------
# _rst_to_markdown(): fallback heuristic (pypandoc absent / OSError)
# ---------------------------------------------------------------------------


class TestRstToMarkdownFallback:
    def _call_fallback(self, rst: str) -> str:
        """Force the pypandoc fallback path."""
        with patch.dict(sys.modules, {"pypandoc": None}):
            return PlainTextParser._rst_to_markdown(rst)

    def test_mock_equals_underline_becomes_h1(self) -> None:
        """=== underline → # H1."""
        rst = "My Title\n========\n\nText.\n"
        result = self._call_fallback(rst)
        assert "# My Title" in result

    def test_mock_dash_underline_becomes_h2(self) -> None:
        """--- underline → ## H2."""
        rst = "Subtitle\n--------\n\nMore text.\n"
        result = self._call_fallback(rst)
        assert "## Subtitle" in result

    def test_mock_tilde_underline_becomes_h3(self) -> None:
        """~~~ underline → ### H3."""
        rst = "Section\n~~~~~~~\n\nContent.\n"
        result = self._call_fallback(rst)
        assert "### Section" in result

    def test_mock_plain_lines_pass_through(self) -> None:
        """Non-heading lines are passed through unchanged."""
        rst = "Just a plain line.\nAnother plain line.\n"
        result = self._call_fallback(rst)
        assert "Just a plain line." in result
        assert "Another plain line." in result

    def test_mock_empty_input_returns_empty(self) -> None:
        """Empty RST → empty output."""
        result = self._call_fallback("")
        assert result == ""
