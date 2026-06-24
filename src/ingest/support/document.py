# @summary
# Document text preprocessing helpers for ingestion (cleaning, metadata, chunking).
# Exports: DocumentMetadata, strip_boilerplate, normalize_unicode, clean_whitespace,
#          strip_trailing_short_lines, extract_metadata, metadata_to_dict
# Deps: re, unicodedata, dataclasses, config.settings
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

from config.settings import DEFAULT_TENANT_ID, RAG_INGEST_STRIP_TRAILING_SHORT_LINES_MAX_WORDS


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


def strip_trailing_short_lines(text: str, max_words: int = RAG_INGEST_STRIP_TRAILING_SHORT_LINES_MAX_WORDS) -> str:
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
