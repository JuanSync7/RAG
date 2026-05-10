# @summary
# Abstract parser protocol and unified data contracts for the Document Parsing Abstraction.
# Exports: DocumentParser, ParseResult, Chunk, chunk_with_markdown, validate_extra_metadata
# Deps: dataclasses, pathlib, typing, src.ingest.support.markdown
# @end-summary

"""Abstract parser protocol and unified data contracts.

Encapsulation rule: parser-internal types (DoclingDocument, tree-sitter Tree,
DeepDoc internal objects) MUST NOT appear in ParseResult or Chunk. Parser
implementations retain internal state on the instance between parse() and
chunk() calls — the pipeline never inspects or serialises this state.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass
class ParseResult:
    """Unified output of parser.parse(). FR-3201.

    Contains parser-agnostic document metadata. No parser-internal types
    (DoclingDocument, tree-sitter Tree) may appear here. All fields are
    JSON-serialisable by design.

    Attributes:
        markdown: Document content as markdown text.
        headings: Heading text in document order.
        has_figures: Whether the parser detected figures or images.
        page_count: Total pages in source. 0 for code/text files.
    """

    markdown: str
    headings: list[str]
    has_figures: bool
    page_count: int


@dataclass
class Chunk:
    """Unified output element of parser.chunk(). FR-3202.

    extra_metadata values MUST be JSON-serialisable. Code parser chunks
    carry language, function_name, class_name, docstring, imports,
    decorators keys. Document parser chunks typically leave extra_metadata
    empty.

    Attributes:
        text: Chunk content (raw source code for code parsers, markdown
            segments for document/text parsers).
        section_path: Hierarchical breadcrumb string
            (e.g., "Chapter 1 > Background > Prior Work"). Empty string if
            no section hierarchy.
        heading: Nearest heading for this chunk. Empty string if none.
        heading_level: Depth of nearest heading (1=top-level, 0=none).
        chunk_index: Zero-based index within the document.
        extra_metadata: Parser-specific metadata. Must be JSON-serialisable.
    """

    text: str
    section_path: str
    heading: str
    heading_level: int
    chunk_index: int
    extra_metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class DocumentParser(Protocol):
    """Abstract parser interface. FR-3200.

    Implementations MUST encapsulate parser-internal types. Pipeline nodes
    access parsers ONLY through this protocol. Per-document instance
    lifecycle is RECOMMENDED (FR-3206): create a new parser instance per
    document, call parse() then chunk() sequentially.
    """

    def parse(self, file_path: Path, config: Any) -> ParseResult:
        """Parse a source file into a ParseResult.

        Internal state (e.g., DoclingDocument, AST) MAY be retained for use
        by chunk() but MUST NOT appear in ParseResult.

        Args:
            file_path: Path to the source file.
            config: IngestionConfig instance.

        Returns:
            ParseResult with parser-agnostic metadata.
        """
        ...

    def chunk(self, parse_result: ParseResult) -> list[Chunk]:
        """Produce chunks from a previously parsed document.

        Uses parser-internal state retained from parse(). Calling chunk()
        before parse() MUST raise RuntimeError.

        Args:
            parse_result: The ParseResult from a prior parse() call.

        Returns:
            List of Chunk objects with populated metadata.
        """
        ...

    @classmethod
    def ensure_ready(cls, config: Any) -> None:
        """Validate runtime dependencies. Called at pipeline startup. FR-3204.

        Raises:
            RuntimeError: If parser dependencies are missing or configuration
                is invalid. The error message MUST identify the specific
                missing dependency and provide installation instructions.
        """
        ...

    @classmethod
    def warmup(cls, config: Any) -> None:
        """Download/compile expensive assets. FR-3207. Optional.

        Called during deployment or container startup, separate from
        ensure_ready(). Implementations without expensive init should no-op.
        """
        ...


def validate_extra_metadata(meta: dict[str, Any]) -> None:
    """Validate that all extra_metadata values are JSON-serialisable.

    Called by parser implementations in debug/test mode. Raises ValueError
    if any value cannot be serialised.

    Args:
        meta: The extra_metadata dict to validate.

    Raises:
        ValueError: If any value is not JSON-serialisable.
    """
    try:
        json.dumps(meta, default=str)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"extra_metadata contains non-JSON-serialisable value: {exc}"
        ) from exc


def chunk_with_markdown(parse_result: ParseResult, config: Any) -> list[Chunk]:
    """Shared markdown chunker. FR-3321.

    Wraps the existing chunk_markdown() from src.ingest.support.markdown and
    maps output to Chunk dataclass. Used by PlainTextParser.chunk() natively
    and by any parser when chunker="markdown" override is active.

    Args:
        parse_result: ParseResult whose markdown field is chunked.
        config: IngestionConfig with chunk_size, chunk_overlap settings.

    Returns:
        List of Chunk objects with section_path, heading, heading_level
        populated from markdown heading hierarchy.
    """
    # Lazy import to avoid circular imports at module load time.
    from src.ingest.support.markdown import _build_section_metadata, chunk_markdown

    raw_chunks = chunk_markdown(
        parse_result.markdown,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        embedder=None,  # Semantic splitting handled at a higher level if needed
    )

    chunks: list[Chunk] = []
    for idx, raw in enumerate(raw_chunks):
        section_meta = _build_section_metadata(raw.get("header_metadata", {}))
        chunks.append(
            Chunk(
                text=raw["text"],
                section_path=section_meta["section_path"],
                heading=section_meta["heading"],
                heading_level=section_meta["heading_level"],
                chunk_index=idx,
                extra_metadata={},
            )
        )
    return chunks
