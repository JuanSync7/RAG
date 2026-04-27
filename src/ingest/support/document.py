# @summary
# Document text preprocessing helpers for ingestion (cleaning, metadata, chunking).
# Exports: DocumentMetadata, strip_boilerplate, normalize_unicode, clean_whitespace,
#          strip_section_markers, strip_trailing_short_lines, clean_text,
#          extract_metadata, metadata_to_dict, chunk_text, process_document
# Deps: re, unicodedata, dataclasses, typing, langchain_text_splitters, config.settings,
#       src.ingest.common.schemas
# @end-summary
"""Document text preprocessing helpers for ingestion.

This module provides a deterministic, multi-stage preprocessing pipeline for
raw text extracted from documents. It focuses on robustness against real-world
artifacts (banners, signatures, boilerplate), then produces cleaned chunks with
attached metadata for downstream embedding and storage.
"""

import re
import unicodedata
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import CHUNK_SIZE, CHUNK_OVERLAP, DEFAULT_TENANT_ID
from src.ingest.common import ProcessedChunk


@dataclass
class DocumentMetadata:
    """Metadata extracted from a document.

    Attributes:
        source: Source identifier (e.g., filename).
        title: Optional title extracted from a header block.
        author: Optional author/owner extracted from a header block.
        date: Optional date string extracted from a header block.
        tags: Optional list of tags extracted from a header block.
    """
    source: str = "unknown"
    title: str | None = None
    author: str | None = None
    date: str | None = None
    tags: list[str] | None = None


# --- Stage 1: Header/Footer/Boilerplate Removal ---

# Patterns for common boilerplate blocks to strip entirely
_BOILERPLATE_PATTERNS: list[re.Pattern[str]] = [
    # Delimited header blocks: lines of ====... or ----... with content between
    # them (e.g., banner blocks). Only matches consecutive non-blank lines
    # sandwiched between delimiter lines.
    re.compile(
        r"^[=]{3,}\s*\n(?:[^\n]*\n)*?[=]{3,}\s*$",
        re.MULTILINE,
    ),
    # Metadata key-value header lines (Title:, Author:, Date:, etc.)
    # that appear before the main content
    re.compile(
        r"^(?:Title|Author|Date|Department|Tags|Classification|Document ID)"
        r"\s*:.*$",
        re.MULTILINE | re.IGNORECASE,
    ),
    # Page footers: "Page X of Y | ... | ..."
    re.compile(r"^Page \d+.*$", re.MULTILINE),
    # Generated/modified timestamps
    re.compile(r"^Generated:.*$", re.MULTILINE),
    re.compile(r"^Last Modified:.*$", re.MULTILINE),
    # Copyright lines
    re.compile(r"^©.*$", re.MULTILINE),
    # Email headers (Subject/From/To/Date/MIME/Content-Type block)
    re.compile(
        r"^\s*(?:Subject|From|To|Date|MIME-Version|Content-Type):.*?"
        r"(?=\n\s*\n|\n\s*-{3,})",
        re.DOTALL | re.MULTILINE,
    ),
    # Email greetings and sign-offs
    re.compile(r"^\s*Hi everyone,?\s*$", re.MULTILINE),
    re.compile(r"^\s*(?:Best|Regards|Cheers|Thanks),?\s*$", re.MULTILINE),
    # Email signature blocks ("-- \n...")
    re.compile(r"\n\s*--\s*\n.*$", re.DOTALL),
    # Confidentiality disclaimers
    re.compile(
        r"This email and any attachments are confidential.*$",
        re.DOTALL | re.IGNORECASE,
    ),
    # Table of contents blocks
    re.compile(r"\[TOC\]\s*\n(?:\s*\d+\..*\n)+", re.MULTILINE),
    # Draft/version markers
    re.compile(r"^.*(?:DRAFT|Do Not Distribute).*$", re.MULTILINE | re.IGNORECASE),
    # Document version lines
    re.compile(r"^Document version.*$", re.MULTILINE | re.IGNORECASE),
    # "For more info / contact" lines
    re.compile(r"^For (?:more information|related articles),?\s*(?:contact|visit):.*$", re.MULTILINE),
    # Reference sections: "[1] Author..." style citations
    re.compile(r"^\[\d+\]\s+.*$", re.MULTILINE),
    # Internal wiki/URL-only lines
    re.compile(r"^\s*(?:Internal wiki|See also):?\s*https?://\S+\s*$", re.MULTILINE),
    # "Prepared by" / "Reviewed by" lines
    re.compile(r"^\s*(?:Prepared|Reviewed) by\s*:.*$", re.MULTILINE | re.IGNORECASE),
    # "Last updated" standalone lines
    re.compile(r"^\s*Last updated\s*:.*$", re.MULTILINE | re.IGNORECASE),
    # Separator-only lines (--- or ===)
    re.compile(r"^\s*[-=]{3,}\s*$", re.MULTILINE),
    # NOTE/TODO internal markers
    re.compile(r"^(?:NOTE|TODO|FIXME|HACK):.*$", re.MULTILINE),
    # "Following up on..." transitional email filler
    re.compile(r"^.*Following up on.*(?:write-up|knowledge base).*$", re.MULTILINE),
    # "Let me know if..." closing filler
    re.compile(r"^.*Let me know if you have questions.*$", re.MULTILINE),
    # Sign-off name + title lines (after "Best,")
    re.compile(r"^\s*(?:Principal|Senior|Lead|Staff)\s+\w+.*$", re.MULTILINE),
    # "We'll be doing..." meeting/event references
    re.compile(r"^.*(?:deep-dive|tech talk|meeting|session).*(?:Friday|Monday|next week).*$", re.MULTILINE | re.IGNORECASE),
]


def strip_boilerplate(text: str) -> str:
    """Remove headers, footers, email boilerplate, and other non-content noise.

    Args:
        text: Input text to clean.

    Returns:
        Text with boilerplate patterns removed.
    """
    for pattern in _BOILERPLATE_PATTERNS:
        text = pattern.sub("", text)
    return text


# --- Stage 2: Text Cleaning ---

# Smart quotes and typographic characters to normalize
_UNICODE_REPLACEMENTS = {
    "\u2018": "'",   # left single quote
    "\u2019": "'",   # right single quote
    "\u201c": '"',   # left double quote
    "\u201d": '"',   # right double quote
    "\u2013": "-",   # en dash
    "\u2014": "--",  # em dash
    "\u2026": "...", # ellipsis
    "\u00a0": " ",   # non-breaking space
}

# Pre-built translation table for single-character replacements (O(n) one-pass).
# Multi-character replacements (em dash → "--", ellipsis → "...") are handled
# separately via str.replace since str.maketrans only supports 1-to-1 mappings.
_UNICODE_SINGLE = {k: v for k, v in _UNICODE_REPLACEMENTS.items() if len(v) == 1}
_UNICODE_MULTI = {k: v for k, v in _UNICODE_REPLACEMENTS.items() if len(v) > 1}
_UNICODE_TRANS = str.maketrans(_UNICODE_SINGLE)


def normalize_unicode(text: str) -> str:
    """Normalize typographic unicode characters to simpler equivalents.

    Single-character replacements are done in one pass via ``str.translate``
    (O(n)); multi-character replacements (em dash, ellipsis) use ``str.replace``.

    Args:
        text: Input text.

    Returns:
        Text with common typographic characters replaced and NFC-normalized.
    """
    text = text.translate(_UNICODE_TRANS)
    for char, replacement in _UNICODE_MULTI.items():
        text = text.replace(char, replacement)
    # Normalize remaining unicode to NFC form
    text = unicodedata.normalize("NFC", text)
    return text


def clean_whitespace(text: str) -> str:
    """Collapse excessive whitespace while preserving paragraph structure.

    Args:
        text: Input text.

    Returns:
        Cleaned text with normalized whitespace and paragraph breaks.
    """
    # Replace tabs with spaces
    text = text.replace("\t", " ")
    # Collapse multiple spaces within lines to single space
    text = re.sub(r"[^\S\n]+", " ", text)
    # Collapse 3+ consecutive newlines to double newline (paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip trailing whitespace per line
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()


def strip_section_markers(text: str) -> str:
    """Normalize markdown/wiki section headers to plain text.

    Args:
        text: Input text.

    Returns:
        Text with header markers removed while preserving heading text.
    """
    # Markdown headers: ## Heading -> Heading
    text = re.sub(r"^\s*#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Wiki-style headers: == Heading == -> Heading
    text = re.sub(r"^\s*={2,}\s*(.*?)\s*={2,}\s*$", r"\1", text, flags=re.MULTILINE)
    # Numbered section headers: "1. INTRODUCTION" or "2.1 Supervised Learning"
    # Matches "N." or "N.N" prefix followed by ALL CAPS text
    text = re.sub(
        r"^\s*(\d+(?:\.\d+)*)\.?\s+([A-Z][A-Z &/]+)$",
        lambda m: f"{m.group(2).title()}",
        text,
        flags=re.MULTILINE,
    )
    return text


def strip_trailing_short_lines(text: str, max_words: int = 4) -> str:
    """Remove very short trailing lines (likely signature/name remnants).

    Args:
        text: Input text.
        max_words: Maximum words allowed for a line to be considered "short".

    Returns:
        Text with short trailing lines removed (best effort).
    """
    lines = text.rstrip().split("\n")
    while lines and len(lines[-1].split()) <= max_words and not lines[-1].strip() == "":
        # Don't strip if it looks like a real sentence ending
        last = lines[-1].strip()
        if last.endswith(".") or last.endswith("?") or last.endswith("!"):
            break
        lines.pop()
    return "\n".join(lines)


def clean_text(text: str) -> str:
    """Run the full text cleaning pipeline.

    Args:
        text: Raw text.

    Returns:
        Cleaned text.
    """
    text = strip_boilerplate(text)
    text = normalize_unicode(text)
    text = clean_whitespace(text)
    text = strip_section_markers(text)
    text = strip_trailing_short_lines(text)
    text = clean_whitespace(text)  # final pass after marker removal
    return text


# --- Stage 3: Metadata Extraction ---

# Key-value patterns commonly found in document headers
_METADATA_KV_PATTERN = re.compile(
    r"^\s*(?P<key>Title|Author|Date|Department|Tags|Subject|From|Prepared by|Reviewed by|Last updated)"
    r"\s*:\s*(?P<value>.+)$",
    re.MULTILINE | re.IGNORECASE,
)


def extract_metadata(raw_text: str, source: str) -> DocumentMetadata:
    """Extract structured metadata from the raw document text (before cleaning).

    Args:
        raw_text: Raw document text (including headers/boilerplate).
        source: Source identifier (e.g., filename) used for defaults.

    Returns:
        Extracted `DocumentMetadata`.
    """
    metadata = DocumentMetadata(source=source)

    for match in _METADATA_KV_PATTERN.finditer(raw_text):
        key = match.group("key").lower().strip()
        value = match.group("value").strip()

        if key in ("title", "subject"):
            metadata.title = value
        elif key in ("author", "prepared by"):
            metadata.author = value
        elif key in ("date", "last updated"):
            metadata.date = value
        elif key == "tags":
            metadata.tags = [t.strip() for t in value.split(",")]

    return metadata


def metadata_to_dict(meta: DocumentMetadata) -> dict:
    """Convert `DocumentMetadata` to a flat dict for storage.

    Args:
        meta: Metadata object.

    Returns:
        Flat dictionary suitable for attaching to chunk metadata.
    """
    d = {"source": meta.source, "tenant_id": DEFAULT_TENANT_ID}
    if meta.title:
        d["title"] = meta.title
    if meta.author:
        d["author"] = meta.author
    if meta.date:
        d["date"] = meta.date
    if meta.tags:
        d["tags"] = ", ".join(meta.tags)
    return d


# --- Stage 4: Chunking ---

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks using recursive character splitting.

    Args:
        text: Cleaned text to split.
        chunk_size: Target maximum chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks in characters.

    Returns:
        List of chunk strings.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


# --- Full Pipeline ---

def process_document(raw_text: str, source: str = "unknown") -> list[ProcessedChunk]:
    """Full document processing pipeline.

    Stages:
        1. Extract metadata from raw text (before cleaning)
        2. Clean text (boilerplate removal, unicode normalization, whitespace)
        3. Chunk cleaned text
        4. Attach metadata to each chunk

    Args:
        raw_text: The raw document text with all its artifacts.
        source: Source identifier (e.g., filename).

    Returns:
        List of ProcessedChunk objects ready for embedding.
    """
    # Stage 1: Extract metadata from raw text (before we strip headers)
    doc_metadata = extract_metadata(raw_text, source)
    base_metadata = metadata_to_dict(doc_metadata)

    # Stage 2: Clean text
    cleaned = clean_text(raw_text)

    if not cleaned:
        return []

    # Stage 3: Chunk text
    chunks = chunk_text(cleaned)

    # Stage 4: Build processed chunks with metadata
    processed = []
    for i, chunk in enumerate(chunks):
        metadata = {**base_metadata, "chunk_index": i, "total_chunks": len(chunks)}
        processed.append(ProcessedChunk(text=chunk, metadata=metadata))

    return processed
