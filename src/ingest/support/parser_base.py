# @summary
# Abstract parser protocol and unified data contracts for the Document Parsing Abstraction.
# Exports: DocumentParser, ParseResult, Chunk, TableArtifact, FigureArtifact, PageRef, chunk_with_markdown, validate_extra_metadata
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
class PageRef:
    """Reference to a single source page. FR-3212.

    Carried by chunks and table artifacts when the parser exposes per-element
    page provenance (Docling does; regex/text fallback does not). Consumers
    use this for citation rendering and per-page retrieval filtering.

    Attributes:
        page_no: 1-based page number within the source document.
        page_label: Display label (e.g., "iv", "12a"). Empty when unknown.
        bbox: Optional bounding box ``(x0, y0, x1, y1)`` in PDF coordinates;
            ``None`` when not available.
    """

    page_no: int
    page_label: str = ""
    bbox: tuple[float, float, float, float] | None = None


@dataclass
class TableArtifact:
    """Structured table extracted from a parsed document. FR-3211.

    A first-class artifact independent of the chunk stream. Surfaces the
    table's markdown rendering for embedding/display alongside a structured
    cell grid for typed retrieval (e.g., column lookups, programmatic joins).

    Tables are emitted by DoclingParser via DoclingDocument.tables. Parsers
    that do not produce structured tables (text/code/markdown) leave
    ``ParseResult.tables`` empty.

    Attributes:
        table_id: Stable per-document identifier (e.g., "table-1").
        markdown: GitHub-flavored markdown rendering of the table; safe to
            embed directly into a chunk.
        cells: Row-major 2D cell grid as plain strings. First row is typically
            the header row when ``has_header`` is True.
        num_rows: Total row count (including header if present).
        num_cols: Column count (max across rows).
        has_header: True when the parser identified a header row at index 0.
        section_path: Hierarchical breadcrumb of headings containing this
            table. Empty when no enclosing heading exists.
        caption: Caption text if the parser detected one; empty otherwise.
        caption_label: Normalised caption label extracted from ``caption``
            (e.g., ``"Table 5-2"``); empty when no prefix could be parsed.
            Surfaced into chunk metadata for xref resolution.
        self_ref: Parser-internal stable reference (e.g., Docling
            ``TableItem.self_ref``) used to join sibling chunks emitted from
            the same source table. Empty string when the parser does not
            expose one.
        document_id: Stable pointer to the parent document (e.g., the source
            file's stem). Empty string when the parser/caller did not stamp
            one — backward compatible with callers that pre-date the field.
    """

    table_id: str
    markdown: str
    cells: list[list[str]]
    num_rows: int
    num_cols: int
    has_header: bool = False
    section_path: str = ""
    caption: str = ""
    caption_label: str = ""
    page_ref: PageRef | None = None
    self_ref: str = ""
    document_id: str = ""


@dataclass
class FigureArtifact:
    """Structured figure/picture extracted from a parsed document.

    Sibling to :class:`TableArtifact`, surfaced for xref expansion ("see Figure
    4-1") and citation/UI rendering. Lightweight by design — image bytes are
    not embedded; ``image_uri`` carries a best-effort pointer to where the
    image lives (data URI, file URI, or empty when Docling did not surface
    one).

    Emitted by :class:`DoclingParser` from ``DoclingDocument.pictures``.
    Parsers without figure support leave ``ParseResult.figures`` empty.

    Attributes:
        document_id: Stable pointer to the parent document (typically the
            source file's stem). Empty when not stamped by the caller.
        caption: Caption text extracted from the figure's caption refs;
            empty when no caption was detected.
        caption_label: Normalised caption label (e.g. ``"Figure 4-1"``)
            parsed from ``caption``. Empty when no prefix matched.
        section_path: Hierarchical breadcrumb of enclosing headings,
            joined with ``" > "``. Empty when no enclosing heading exists.
        page_no: 1-based page number from the figure's provenance.
            ``0`` when unknown.
        self_ref: Parser-internal stable reference (e.g. Docling
            ``PictureItem.self_ref``). Empty when the parser does not
            expose one.
        image_uri: Best-effort URI for the figure image (data URI, file
            URI, or http(s)). Empty when Docling did not attach an image
            ref to this picture.
    """

    document_id: str = ""
    caption: str = ""
    caption_label: str = ""
    section_path: str = ""
    page_no: int = 0
    self_ref: str = ""
    image_uri: str = ""


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
        tables: Structured table artifacts extracted by the parser. Empty
            for parsers without structured-table support. FR-3211.
        figures: Structured figure artifacts extracted by the parser. Empty
            for parsers without figure support.
    """

    markdown: str
    headings: list[str]
    has_figures: bool
    page_count: int
    tables: list[TableArtifact] = field(default_factory=list)
    figures: list[FigureArtifact] = field(default_factory=list)


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
    heading_path: list[str] = field(default_factory=list)
    page_ref: PageRef | None = None


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
        section_path = section_meta["section_path"]
        heading_path = [h for h in section_path.split(" > ") if h] if section_path else []
        chunks.append(
            Chunk(
                text=raw["text"],
                section_path=section_path,
                heading=section_meta["heading"],
                heading_level=section_meta["heading_level"],
                chunk_index=idx,
                extra_metadata={},
                heading_path=heading_path,
                page_ref=None,
            )
        )
    return chunks
