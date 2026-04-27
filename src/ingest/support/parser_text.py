# @summary
# Plain text parser for .md, .txt, .rst, .html/.htm files.
# Exports: PlainTextParser
# Deps: pathlib, re, src.ingest.support.parser_base
# @end-summary

"""Plain text parser implementation.

Handles markdown, plain text, reStructuredText, and HTML files with minimal
processing. Heading-aware markdown chunking via the shared chunk_with_markdown()
utility.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from src.ingest.support.parser_base import (
    Chunk,
    DocumentParser,
    ParseResult,
    chunk_with_markdown,
)

logger = logging.getLogger(__name__)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_IMAGE_RE = re.compile(r"!\[.*?\]\(.*?\)")


class PlainTextParser:
    """Plain text parser implementing DocumentParser protocol. FR-3280.

    Handles .md, .txt, .rst, .html/.htm with minimal processing.
    No external model or service is invoked. FR-3281 AC 3.
    """

    def __init__(self) -> None:
        self._config: Any = None
        self._suffix: str = ""
        self._source_path: Path | None = None
        self._source_key: str | None = None
        self._doc_store_client: Any = None

    def configure_storage(self, *, source_key: str, doc_store_client: Any) -> None:
        """Optional: pass identity + doc-store client so figure uploads (when
        ``config.store_figures_in_db=True``) get a stable key. No-op when
        either argument is None.
        """
        self._source_key = source_key
        self._doc_store_client = doc_store_client

    def parse(self, file_path: Path, config: Any) -> ParseResult:
        """Read file, normalise to markdown, extract headings. FR-3281.

        - .md, .txt: content used as-is.
        - .html/.htm: convert to markdown (strip tags, preserve headings).
        - .rst: convert to markdown or treat as plain text with heading heuristic.

        Args:
            file_path: Path to the source file.
            config: IngestionConfig instance.

        Returns:
            ParseResult with headings extracted and page_count=0.
        """
        self._config = config
        content = file_path.read_text(encoding="utf-8", errors="replace")
        suffix = file_path.suffix.lower()
        self._suffix = suffix
        self._source_path = file_path

        if suffix in (".html", ".htm"):
            content = self._html_to_markdown(content)
        elif suffix == ".rst":
            content = self._rst_to_markdown(content)
        # .md, .txt: use content as-is

        headings = self._extract_headings(content)
        has_figures = bool(_IMAGE_RE.search(content))

        return ParseResult(
            markdown=content,
            headings=headings,
            has_figures=has_figures,
            page_count=0,
        )

    def chunk(self, parse_result: ParseResult) -> list[Chunk]:
        """Chunk using a structure-aware path for markdown, char-splitter elsewhere.

        For .md / .markdown / .html / .rst (already converted to markdown in parse()),
        try Docling's HybridChunker — it preserves lists, tables, and requirement
        blocks as atomic units, which the LangChain splitter cannot. Fall back to
        chunk_with_markdown() (MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter)
        if Docling is unavailable, the markdown is empty, or HybridChunker raises.

        For .txt the source is unstructured, so the legacy char-splitter is fine.

        Toggle: config.use_docling_chunker_for_markdown (default True).

        Raises:
            RuntimeError: If called before parse().
        """
        if self._config is None:
            raise RuntimeError(
                "PlainTextParser.chunk() called before parse(). "
                "Call parse() first to populate config."
            )

        markdown_suffixes = {".md", ".markdown", ".html", ".htm", ".rst"}
        use_docling_chunker = getattr(
            self._config, "use_docling_chunker_for_markdown", True
        )
        if (
            use_docling_chunker
            and self._suffix in markdown_suffixes
            and parse_result.markdown
        ):
            try:
                from src.ingest.support.docling import chunk_markdown_via_docling

                max_tokens = getattr(
                    self._config, "hybrid_chunker_max_tokens", 512
                )
                return chunk_markdown_via_docling(
                    parse_result.markdown,
                    max_tokens=max_tokens,
                    source_path=self._source_path,
                    config=self._config,
                    source_key=self._source_key,
                    doc_store_client=self._doc_store_client,
                )
            except Exception as exc:
                logger.warning(
                    "HybridChunker failed for suffix=%s, falling back to "
                    "char-splitter: %s",
                    self._suffix, exc,
                )

        return chunk_with_markdown(parse_result, self._config)

    @classmethod
    def ensure_ready(cls, config: Any) -> None:
        """No-op. Plain text parsing has no external dependencies. FR-3204."""
        pass

    @classmethod
    def warmup(cls, config: Any) -> None:
        """No-op. No expensive assets to pre-load. FR-3207."""
        pass

    @staticmethod
    def _extract_headings(text: str) -> list[str]:
        """Extract markdown heading text in document order.

        Reuses the same logic as docling.py's _extract_headings_from_markdown().
        """
        headings: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                heading = stripped.lstrip("#").strip()
                if heading:
                    headings.append(heading)
        return headings

    @staticmethod
    def _html_to_markdown(html: str) -> str:
        """Convert HTML to markdown. FR-3281 item 2.

        Uses markdownify if available, otherwise falls back to a minimal
        regex-based tag stripper that preserves heading structure.
        """
        try:
            from markdownify import markdownify as md
            return md(html, heading_style="ATX", strip=["script", "style"])
        except ImportError:
            logger.warning(
                "markdownify not installed; falling back to basic HTML stripping. "
                "Install with: uv add markdownify"
            )
            import re as _re
            text = html
            for level in range(1, 7):
                text = _re.sub(
                    rf"<h{level}[^>]*>(.*?)</h{level}>",
                    lambda m, lv=level: "#" * lv + " " + m.group(1),
                    text,
                    flags=_re.IGNORECASE | _re.DOTALL,
                )
            text = _re.sub(r"<[^>]+>", "", text)
            return text.strip()

    @staticmethod
    def _rst_to_markdown(rst: str) -> str:
        """Convert reStructuredText to markdown. FR-3281 item 3.

        Uses pypandoc if available, otherwise applies a heading heuristic
        that detects RST underline patterns (===, ---, ~~~).
        """
        try:
            import pypandoc
            return pypandoc.convert_text(rst, "md", format="rst")
        except (ImportError, OSError):
            logger.warning(
                "pypandoc not installed; falling back to RST heading heuristic. "
                "Install with: uv add pypandoc"
            )
            import re as _re
            lines = rst.splitlines()
            result: list[str] = []
            i = 0
            while i < len(lines):
                if (
                    i + 1 < len(lines)
                    and lines[i].strip()
                    and _re.match(r"^[=\-~]{3,}$", lines[i + 1].strip())
                ):
                    char = lines[i + 1].strip()[0]
                    level = {"=": 1, "-": 2, "~": 3}.get(char, 2)
                    result.append("#" * level + " " + lines[i].strip())
                    i += 2
                else:
                    result.append(lines[i])
                    i += 1
            return "\n".join(result)
