# @summary
# Markdown-preserving document processor. Main exports: process_document_markdown, chunk_markdown, clean_document.
# Also exports _embeddings_are_unusable — a defensive guard at the semantic
# chunker boundary that detects pathological embedder outputs (all-None or
# all-NaN rows produced by misconfigured embedding backends) and triggers a
# safe fallback to plain sentences instead of letting np.dot crash the
# entire chunking stage with a NoneType TypeError.
# Deps: re, numpy, langchain_text_splitters, src.ingest.support.document, src.ingest.common.schemas, config.settings
# @end-summary
"""
Markdown-preserving document processor.

Instead of stripping section markers, normalizes all heading formats
(wiki ==, numbered ALL-CAPS, markdown ##) to standard markdown, then
chunks using MarkdownHeaderTextSplitter to preserve document structure.
"""

import logging
import re

import numpy as np
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from config.settings import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MIN_CHUNK_CHARS,
    SEMANTIC_CHUNKING_ENABLED,
    SEMANTIC_SIMILARITY_THRESHOLD,
)
from src.ingest.common import ProcessedChunk
from src.ingest.support.document import (
    extract_metadata,
    metadata_to_dict,
    strip_boilerplate,
    normalize_unicode,
    clean_whitespace,
    strip_trailing_short_lines,
)

logger = logging.getLogger(__name__)


# Module-level pre-compiled regex patterns
_WIKI_HEADING_RE = re.compile(
    r"^\s*(\={2,})\s*(.*?)\s*\={2,}\s*$",
    re.MULTILINE,
)
# NOTE: the title group must NOT include a literal space inside its character
# class — a space matchable by both the class AND the inter-word ``\s+`` makes
# the pattern ambiguous and triggers catastrophic backtracking on long
# non-matching lines (a 1.2MB ARM CHI spec hung heading normalization for
# minutes). Word separation is handled solely by the ``(?:\s+...)*`` group.
_NUMBERED_HEADING_RE = re.compile(
    r"^\s*(\d+(?:\.\d+)*)\.?\s+([A-Z][A-Za-z&/()\-]*(?:\s+[A-Za-z][A-Za-z&/()\-]*)*)$",
    re.MULTILINE,
)
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+|\n\n+')

# Headers the markdown splitter will split on
_HEADERS_TO_SPLIT = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]


def normalize_headings_to_markdown(text: str) -> str:
    """Convert wiki-style and numbered headings to markdown format.

    - ``== Heading ==`` → ``## Heading`` (level from ``=`` count)
    - ``1. INTRODUCTION`` → ``## Introduction``
    - ``2.1 Supervised Learning`` → ``### Supervised Learning``
    - Existing ``## Heading`` markdown → unchanged
    """
    # Wiki == Heading == -> ## Heading
    def _wiki_to_md(m):
        level = len(m.group(1))
        heading = m.group(2).strip()
        return "#" * min(level, 6) + " " + heading

    text = _WIKI_HEADING_RE.sub(_wiki_to_md, text)

    # Numbered ALL-CAPS sections: "1. INTRODUCTION" -> "## Introduction"
    # "2.1 Supervised Learning" -> "### Supervised Learning"
    # Depth: dots in numbering + 2 (so "1." = h2, "2.1" = h3)
    def _numbered_to_md(m):
        numbering = m.group(1)
        depth = numbering.count(".") + 2
        heading_text = m.group(2).strip()
        if heading_text == heading_text.upper():
            heading_text = heading_text.title()
        return "#" * min(depth, 6) + " " + heading_text

    text = _NUMBERED_HEADING_RE.sub(_numbered_to_md, text)

    return text


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using a regex heuristic.

    Args:
        text: Input text.

    Returns:
        List of non-empty sentence-like segments.
    """
    raw = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in raw if s.strip()]


def _embeddings_are_unusable(embeddings: object) -> bool:
    """Return True when an embedder's output cannot be used for cosine similarity.

    Specifically guards the ``_semantic_split`` cosine path against:

    * ``None`` (embedder returned nothing usable).
    * empty sequences / arrays.
    * any row whose values are all ``None`` or all ``NaN`` — these arise when
      an upstream embedding service serializes NaN/Inf as JSON ``null`` (e.g.
      a misconfigured TEI bge-m3 ONNX backend) and the JSON decoder turns
      them back into Python ``None``. ``np.dot`` on such a row raises
      ``TypeError: unsupported operand type(s) for *: 'NoneType' and 'NoneType'``
      which we want to avoid letting bubble up and crash the chunking stage.

    The check is intentionally cheap (samples first/last row only) and never
    raises — when in doubt, declare the batch unusable.
    """
    if embeddings is None:
        return True
    try:
        n = len(embeddings)
    except TypeError:
        return True
    if n == 0:
        return True
    # Inspect the first row — that's enough to catch the all-null upstream
    # failure mode. We deliberately do not iterate every row: a healthy
    # embedder cannot return some-good / some-None mixes from one batch
    # call, and per-row inspection would be O(N×D) on the hot path.
    sample = embeddings[0]
    if sample is None:
        return True
    try:
        arr = np.asarray(sample, dtype=float)
    except (TypeError, ValueError):
        # ``np.asarray`` on a list-of-None coerces to dtype=object and yields
        # ``None`` entries — convert via float to surface them as NaN. If
        # that fails entirely, treat as unusable.
        try:
            arr = np.asarray([float("nan") if x is None else x for x in sample], dtype=float)
        except Exception:
            return True
    if arr.size == 0:
        return True
    if not np.all(np.isfinite(arr)):
        return True
    return False


def _coalesce_segments(
    segments: list[str],
    min_chars: int = MIN_CHUNK_CHARS,
    max_chars: int = CHUNK_SIZE,
) -> list[str]:
    """Merge adjacent segments so no emitted segment is smaller than
    ``min_chars`` unless it cannot be merged without exceeding ``max_chars``.

    Order-preserving. This is the floor that prevents semantic/sentence
    splitting from emitting a flood of tiny fragments: a sub-``min_chars``
    buffer keeps absorbing following segments until it reaches ``min_chars`` or
    the next segment would push it past ``max_chars``. A trailing sub-min
    buffer is folded back into the previous segment when there is room.
    """
    if min_chars <= 0:
        return [s for s in segments if s.strip()]
    out: list[str] = []
    buf = ""
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        if not buf:
            buf = seg
        elif len(buf) < min_chars and len(buf) + 1 + len(seg) <= max_chars:
            buf = f"{buf} {seg}"
        else:
            out.append(buf)
            buf = seg
    if buf:
        if out and len(buf) < min_chars and len(out[-1]) + 1 + len(buf) <= max_chars:
            out[-1] = f"{out[-1]} {buf}"
        else:
            out.append(buf)
    return out


def _semantic_split(
    text: str,
    embedder,
    threshold: float = SEMANTIC_SIMILARITY_THRESHOLD,
    min_chars: int = MIN_CHUNK_CHARS,
    max_chars: int = CHUNK_SIZE,
) -> list[str]:
    """Split text into semantically coherent chunks.

    Embeds each sentence, computes cosine similarity between consecutive
    sentences, and splits where similarity drops below threshold. A final
    coalescing pass (``_coalesce_segments``) merges undersized adjacent
    segments so heterogeneous, table-dense content cannot fragment into a
    flood of single-sentence chunks.
    """
    if embedder is None:
        return [text]

    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return [text]

    # Batch encode all sentences (returns L2-normalized numpy array)
    try:
        embeddings = embedder.encode_sentences(sentences)
    except Exception:
        logger.debug("Semantic sentence embedding failed; falling back to plain sentences", exc_info=True)
        return _coalesce_segments(sentences, min_chars, max_chars)

    # Defensive: an embedder may return an array of ``None`` (or NaN-only) rows
    # when the upstream service is misconfigured (e.g. TEI bge-m3 ONNX yielding
    # all-null vectors). ``np.dot(None, None)`` then raises
    # ``TypeError: unsupported operand type(s) for *: 'NoneType' and 'NoneType'``
    # which masks the real failure and crashes the whole chunking stage. Detect
    # the pathological shape and fall back to plain sentences — the rest of the
    # pipeline still runs (the embedding-storage node will surface the bad
    # vectors as its own error, where it belongs).
    if _embeddings_are_unusable(embeddings):
        logger.warning(
            "Semantic chunking embedder returned unusable vectors "
            "(None/NaN rows or shape mismatch); falling back to plain sentences."
        )
        return _coalesce_segments(sentences, min_chars, max_chars)

    # Cosine similarity between consecutive sentences (dot product on normalized vectors)
    try:
        similarities = np.array([
            np.dot(embeddings[i], embeddings[i + 1])
            for i in range(len(embeddings) - 1)
        ])
    except TypeError:
        # Belt-and-braces: if a stray None slipped past the guard above,
        # do not crash the chunking node.
        logger.warning(
            "Semantic chunking similarity computation failed on None-typed "
            "embedding rows; falling back to plain sentences.",
            exc_info=True,
        )
        return _coalesce_segments(sentences, min_chars, max_chars)

    # Split at points where similarity drops below threshold
    chunks = []
    current_group = [sentences[0]]

    for i, sim in enumerate(similarities):
        if sim < threshold:
            chunks.append(" ".join(current_group))
            current_group = [sentences[i + 1]]
        else:
            current_group.append(sentences[i + 1])

    if current_group:
        chunks.append(" ".join(current_group))

    return _coalesce_segments(chunks, min_chars, max_chars)


def chunk_markdown(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    embedder=None,
) -> list[dict]:
    """Split text using markdown headers, then semantic or character splitting.

    If embedder is provided and SEMANTIC_CHUNKING_ENABLED, uses semantic
    similarity to split oversized sections. Falls back to character splitting
    for any chunks still exceeding chunk_size.

    Args:
        text: Markdown text to split.
        chunk_size: Target maximum chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks in characters.
        embedder: Optional embedder with an ``encode_sentences`` method.

    Returns:
        List of dictionaries with ``text`` and ``header_metadata`` keys.
    """
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_HEADERS_TO_SPLIT,
        strip_headers=False,
    )
    md_splits = md_splitter.split_text(text)

    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    use_semantic = SEMANTIC_CHUNKING_ENABLED and embedder is not None

    final_chunks = []
    for doc in md_splits:
        if len(doc.page_content) <= chunk_size:
            final_chunks.append({
                "text": doc.page_content,
                "header_metadata": doc.metadata,
            })
        elif use_semantic:
            # Semantic split first, then character split if still too large
            semantic_chunks = _semantic_split(doc.page_content, embedder)
            for sc in semantic_chunks:
                if len(sc) <= chunk_size:
                    final_chunks.append({
                        "text": sc,
                        "header_metadata": doc.metadata,
                    })
                else:
                    sub_texts = char_splitter.split_text(sc)
                    for sub in sub_texts:
                        final_chunks.append({
                            "text": sub,
                            "header_metadata": doc.metadata,
                        })
        else:
            sub_texts = char_splitter.split_text(doc.page_content)
            for sub in sub_texts:
                final_chunks.append({
                    "text": sub,
                    "header_metadata": doc.metadata,
                })

    return final_chunks


def _build_section_metadata(header_meta: dict) -> dict:
    """Convert header metadata to flat section fields.

    Args:
        header_meta: Metadata payload produced by `MarkdownHeaderTextSplitter`.

    Returns:
        Dictionary containing ``section_path``, ``heading``, and ``heading_level``.
    """
    # LangChain versions vary between "h1"..."h4" and "Header 1"..."Header 4".
    levels = [
        ("h1", "Header 1"),
        ("h2", "Header 2"),
        ("h3", "Header 3"),
        ("h4", "Header 4"),
    ]
    path_parts = []
    for compact_key, long_key in levels:
        value = header_meta.get(compact_key)
        if value is None:
            value = header_meta.get(long_key)
        if value:
            path_parts.append(value)
    heading = path_parts[-1] if path_parts else ""
    heading_level = len(path_parts)
    return {
        "section_path": " > ".join(path_parts),
        "heading": heading,
        "heading_level": heading_level,
    }


def clean_document(raw_text: str) -> str:
    """Run the full cleaning pipeline on raw text (no chunking).

    Stages:
        1. Strip boilerplate (headers, footers, signatures)
        2. Normalize unicode and whitespace
        3. Normalize headings to markdown format

    Returns:
        Cleaned document text ready for chunking, or empty string.
    """
    cleaned = strip_boilerplate(raw_text)
    cleaned = normalize_unicode(cleaned)
    cleaned = clean_whitespace(cleaned)
    cleaned = normalize_headings_to_markdown(cleaned)
    cleaned = strip_trailing_short_lines(cleaned)
    cleaned = clean_whitespace(cleaned)
    return cleaned


def process_document_markdown(
    raw_text: str, source: str = "unknown", embedder=None
) -> list[ProcessedChunk]:
    """Full markdown-preserving document processing pipeline.

    Stages:
        1. Extract metadata from raw text (before cleaning)
        2. Clean document (boilerplate, unicode, headings)
        3. Chunk using markdown-aware splitter
        4. Attach metadata to each chunk

    Args:
        raw_text: The raw document text.
        source: Source identifier (e.g., filename).
        embedder: Optional embedder used for semantic splitting.

    Returns:
        List of ProcessedChunk objects ready for embedding.
    """
    # Stage 1: Extract metadata from raw text
    doc_metadata = extract_metadata(raw_text, source)
    base_metadata = metadata_to_dict(doc_metadata)

    # Stage 2: Clean document
    cleaned = clean_document(raw_text)

    if not cleaned:
        return []

    # Stage 3: Chunk using markdown-aware splitter (with optional semantic splitting)
    chunks = chunk_markdown(cleaned, embedder=embedder)

    # Stage 4: Build processed chunks with metadata
    processed = []
    for i, chunk in enumerate(chunks):
        section_meta = _build_section_metadata(chunk["header_metadata"])
        metadata = {
            **base_metadata,
            **section_meta,
            "chunk_index": i,
            "total_chunks": len(chunks),
        }
        processed.append(ProcessedChunk(text=chunk["text"], metadata=metadata))

    return processed
