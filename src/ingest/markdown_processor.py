# @summary
# Markdown-preserving document processor. Main exports: process_document_markdown, chunk_markdown, clean_document. Dependencies: re, numpy, langchain_text_splitters, extract_metadata, metadata_to_dict, strip_boilerplate, normalize_unicode, clean_whitespace, strip_trailing_short_lines, _split_sentences, _semantic_split, _build_section_metadata.
# @end-summary
"""
Markdown-preserving document processor.

Instead of stripping section markers, normalizes all heading formats
(wiki ==, numbered ALL-CAPS, markdown ##) to standard markdown, then
chunks using MarkdownHeaderTextSplitter to preserve document structure.
"""

import re
from typing import List, Optional

import numpy as np
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from config.settings import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SEMANTIC_CHUNKING_ENABLED,
    SEMANTIC_SIMILARITY_THRESHOLD,
)
from src.ingest.document_processor import (
    ProcessedChunk,
    extract_metadata,
    metadata_to_dict,
    strip_boilerplate,
    normalize_unicode,
    clean_whitespace,
    strip_trailing_short_lines,
)


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

    text = re.sub(
        r"^\s*(\={2,})\s*(.*?)\s*\={2,}\s*$",
        _wiki_to_md,
        text,
        flags=re.MULTILINE,
    )

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

    text = re.sub(
        r"^\s*(\d+(?:\.\d+)*)\.?\s+([A-Z][A-Za-z &/()\-]+(?:\s+[A-Za-z][A-Za-z &/()\-]*)*)$",
        _numbered_to_md,
        text,
        flags=re.MULTILINE,
    )

    return text


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using regex."""
    raw = re.split(r'(?<=[.!?])\s+|\n\n+', text)
    return [s.strip() for s in raw if s.strip()]


def _semantic_split(
    text: str,
    embedder,
    threshold: float = SEMANTIC_SIMILARITY_THRESHOLD,
) -> List[str]:
    """Split text into semantically coherent chunks.

    Embeds each sentence, computes cosine similarity between consecutive
    sentences, and splits where similarity drops below threshold.
    """
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return [text]

    # Batch encode all sentences (returns L2-normalized numpy array)
    embeddings = embedder.encode_sentences(sentences)

    # Cosine similarity between consecutive sentences (dot product on normalized vectors)
    similarities = np.array([
        np.dot(embeddings[i], embeddings[i + 1])
        for i in range(len(embeddings) - 1)
    ])

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

    return chunks


def chunk_markdown(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    embedder=None,
) -> List[dict]:
    """Split text using markdown headers, then semantic or character splitting.

    If embedder is provided and SEMANTIC_CHUNKING_ENABLED, uses semantic
    similarity to split oversized sections. Falls back to character splitting
    for any chunks still exceeding chunk_size.

    Returns list of dicts with 'text' and 'header_metadata' keys.
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
    """Convert header metadata from MarkdownHeaderTextSplitter to flat fields."""
    path_parts = [header_meta[k] for k in ("h1", "h2", "h3", "h4") if k in header_meta]
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
) -> List[ProcessedChunk]:
    """Full markdown-preserving document processing pipeline.

    Stages:
        1. Extract metadata from raw text (before cleaning)
        2. Clean document (boilerplate, unicode, headings)
        3. Chunk using markdown-aware splitter
        4. Attach metadata to each chunk

    Args:
        raw_text: The raw document text.
        source: Source identifier (e.g., filename).

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
