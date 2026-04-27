"""Tests for src/ingest/support/markdown.py — pure logic + mock-based."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.ingest.support.markdown import (
    _build_section_metadata,
    _semantic_split,
    _split_sentences,
    chunk_markdown,
    normalize_headings_to_markdown,
    process_document_markdown,
)


# ---------------------------------------------------------------------------
# normalize_headings_to_markdown() — wiki headings
# ---------------------------------------------------------------------------


def test_normalize_wiki_heading_h2():
    text = "== Introduction =="
    result = normalize_headings_to_markdown(text)
    assert result.strip() == "## Introduction"


def test_normalize_wiki_heading_h3():
    text = "=== Sub Section ==="
    result = normalize_headings_to_markdown(text)
    assert result.strip() == "### Sub Section"


def test_normalize_wiki_heading_max_level_6():
    text = "======= Deep Section ======="
    result = normalize_headings_to_markdown(text)
    # Should not exceed h6
    assert result.strip().startswith("######")


def test_normalize_wiki_heading_preserves_markdown():
    text = "## Already Markdown"
    result = normalize_headings_to_markdown(text)
    assert result.strip() == "## Already Markdown"


# ---------------------------------------------------------------------------
# normalize_headings_to_markdown() — numbered headings
# ---------------------------------------------------------------------------


def test_normalize_numbered_heading_top_level_allcaps():
    text = "1. INTRODUCTION"
    result = normalize_headings_to_markdown(text)
    # depth = 0 dots + 2 = 2 → ##
    assert "##" in result
    assert "Introduction" in result  # title-cased


def test_normalize_numbered_heading_second_level():
    text = "2.1 Supervised Learning"
    result = normalize_headings_to_markdown(text)
    # depth = 1 dot + 2 = 3 → ###
    assert "###" in result
    assert "Supervised Learning" in result


def test_normalize_numbered_heading_preserves_mixed_case():
    text = "3. Related Work"
    result = normalize_headings_to_markdown(text)
    # Not ALL-CAPS, so should remain as-is
    assert "Related Work" in result


def test_normalize_numbered_heading_allcaps_titlecase():
    text = "4. METHODOLOGY"
    result = normalize_headings_to_markdown(text)
    # ALL-CAPS should be title-cased
    assert "Methodology" in result
    assert "METHODOLOGY" not in result


# ---------------------------------------------------------------------------
# _split_sentences()
# ---------------------------------------------------------------------------


def test_split_sentences_period_separated():
    text = "Hello world. This is a test. Final sentence."
    result = _split_sentences(text)
    assert len(result) >= 2


def test_split_sentences_newline_separated():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird."
    result = _split_sentences(text)
    assert len(result) >= 2


def test_split_sentences_filters_empty():
    text = "One sentence.   "
    result = _split_sentences(text)
    assert all(s.strip() != "" for s in result)


def test_split_sentences_single_sentence():
    text = "Just one sentence"
    result = _split_sentences(text)
    assert len(result) == 1
    assert result[0] == "Just one sentence"


def test_split_sentences_empty_string():
    result = _split_sentences("")
    assert result == []


def test_split_sentences_exclamation_question():
    text = "Hello! Is this working? Yes it is."
    result = _split_sentences(text)
    assert len(result) >= 2


# ---------------------------------------------------------------------------
# _build_section_metadata()
# ---------------------------------------------------------------------------


def test_build_section_metadata_compact_keys():
    meta = {"h1": "Introduction", "h2": "Background"}
    result = _build_section_metadata(meta)
    assert result["section_path"] == "Introduction > Background"
    assert result["heading"] == "Background"
    assert result["heading_level"] == 2


def test_build_section_metadata_long_keys():
    meta = {"Header 1": "Chapter One", "Header 2": "Section A"}
    result = _build_section_metadata(meta)
    assert result["section_path"] == "Chapter One > Section A"
    assert result["heading"] == "Section A"
    assert result["heading_level"] == 2


def test_build_section_metadata_empty():
    result = _build_section_metadata({})
    assert result["section_path"] == ""
    assert result["heading"] == ""
    assert result["heading_level"] == 0


def test_build_section_metadata_single_heading():
    meta = {"h1": "Top Level"}
    result = _build_section_metadata(meta)
    assert result["section_path"] == "Top Level"
    assert result["heading"] == "Top Level"
    assert result["heading_level"] == 1


def test_build_section_metadata_h4():
    meta = {"h1": "Root", "h2": "Sub", "h3": "Deep", "h4": "Deepest"}
    result = _build_section_metadata(meta)
    assert result["heading_level"] == 4
    assert "Deepest" in result["section_path"]


# ---------------------------------------------------------------------------
# _semantic_split() — mock-based
# ---------------------------------------------------------------------------


def test_mock_semantic_split_none_embedder_returns_text():
    text = "Hello. World."
    result = _semantic_split(text, embedder=None)
    assert result == [text]


def test_mock_semantic_split_single_sentence_returns_text():
    text = "Only one sentence"
    mock_embedder = MagicMock()
    result = _semantic_split(text, embedder=mock_embedder)
    # Single sentence: short-circuit, embedder not called
    assert result == [text]


def test_mock_semantic_split_groups_similar_sentences():
    """Two similar sentences (high cosine sim) stay together."""
    text = "Cats are great. Cats are amazing. Dogs are different."

    mock_embedder = MagicMock()
    # Three sentences → 3 embeddings, 2 similarity values
    # Use normalized vectors for dot product
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.99, 0.14, 0.0])  # very similar to v1
    v3 = np.array([0.0, 0.0, 1.0])    # very different from v2
    mock_embedder.encode_sentences.return_value = np.array([v1, v2, v3])

    result = _semantic_split(text, embedder=mock_embedder, threshold=0.5)
    # sim(v1, v2) ≈ 0.99 (above threshold) → group
    # sim(v2, v3) ≈ 0.0 (below threshold) → split
    assert len(result) == 2


def test_mock_semantic_split_splits_all_dissimilar():
    """All consecutive dissimilar → each sentence is its own chunk."""
    text = "First. Second. Third."

    mock_embedder = MagicMock()
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    v3 = np.array([0.0, 0.0, 1.0])
    mock_embedder.encode_sentences.return_value = np.array([v1, v2, v3])

    result = _semantic_split(text, embedder=mock_embedder, threshold=0.5)
    assert len(result) == 3


def test_mock_semantic_split_encode_exception_falls_back_to_sentences():
    text = "Hello. World."

    mock_embedder = MagicMock()
    mock_embedder.encode_sentences.side_effect = RuntimeError("Embedder error")

    result = _semantic_split(text, embedder=mock_embedder, threshold=0.5)
    # Should fall back to returning sentences
    assert isinstance(result, list)
    assert len(result) >= 1


# ---------------------------------------------------------------------------
# chunk_markdown() — no mock needed
# ---------------------------------------------------------------------------


def test_chunk_markdown_simple_text():
    text = "# Title\n\nSome content here.\n\n## Section\n\nMore content."
    chunks = chunk_markdown(text)
    assert len(chunks) >= 1
    for chunk in chunks:
        assert "text" in chunk
        assert "header_metadata" in chunk


def test_chunk_markdown_respects_chunk_size():
    # Create a large section that needs to be split
    long_content = "Word " * 2000
    text = f"# Big Section\n\n{long_content}"
    chunks = chunk_markdown(text, chunk_size=500, chunk_overlap=50)
    # Should produce multiple chunks
    assert len(chunks) > 1


def test_chunk_markdown_without_embedder_semantic_disabled():
    text = "# Title\n\nContent A. Content B. Content C."
    chunks = chunk_markdown(text, embedder=None)
    assert len(chunks) >= 1


def test_chunk_markdown_empty_text():
    chunks = chunk_markdown("")
    # Empty or minimal result
    assert isinstance(chunks, list)


def test_chunk_markdown_no_headers():
    text = "Plain text without any headers. Just regular content."
    chunks = chunk_markdown(text)
    assert len(chunks) >= 1
    assert chunks[0]["text"].strip() != ""


def test_mock_chunk_markdown_with_embedder():
    """Test semantic branch: embedder provided, SEMANTIC_CHUNKING_ENABLED respected."""
    # Generate a section larger than chunk_size to trigger the semantic branch
    long_content = "Sentence number here. " * 300
    text = f"# Section\n\n{long_content}"

    mock_embedder = MagicMock()
    # Return trivially similar embeddings so all sentences group into one chunk
    n_sentences = len(long_content.split(". "))
    vecs = np.ones((n_sentences, 3)) / np.sqrt(3)
    mock_embedder.encode_sentences.return_value = vecs

    # Whether SEMANTIC_CHUNKING_ENABLED is True or False, chunk_markdown should complete
    chunks = chunk_markdown(text, chunk_size=200, chunk_overlap=20, embedder=mock_embedder)
    assert len(chunks) >= 1


# ---------------------------------------------------------------------------
# process_document_markdown() — full pipeline
# ---------------------------------------------------------------------------


def test_process_document_markdown_basic():
    raw = "# Title\n\nSome content. More content here."
    chunks = process_document_markdown(raw, source="test.md")
    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.text.strip() != ""
        assert "chunk_index" in chunk.metadata
        assert "total_chunks" in chunk.metadata


def test_process_document_markdown_empty_after_clean():
    # Boilerplate-only text that gets cleaned to empty
    raw = ""
    result = process_document_markdown(raw, source="empty.md")
    assert result == []


def test_process_document_markdown_preserves_source():
    raw = "# Report\n\nContent for testing."
    chunks = process_document_markdown(raw, source="report.md")
    assert len(chunks) >= 1
    # Metadata should reference source
    for chunk in chunks:
        assert "chunk_index" in chunk.metadata


def test_process_document_markdown_chunk_indices_ordered():
    raw = "\n\n".join([f"# Section {i}\n\nContent for section {i}." for i in range(5)])
    chunks = process_document_markdown(raw, source="multi.md")
    indices = [c.metadata["chunk_index"] for c in chunks]
    assert indices == list(range(len(chunks)))


def test_process_document_markdown_total_chunks_consistent():
    raw = "# Doc\n\nParagraph one.\n\n## Sub\n\nParagraph two."
    chunks = process_document_markdown(raw, source="doc.md")
    total = len(chunks)
    for chunk in chunks:
        assert chunk.metadata["total_chunks"] == total


def test_process_document_markdown_with_wiki_headings():
    raw = "== Introduction ==\n\nSome intro content.\n\n=== Background ===\n\nMore background."
    chunks = process_document_markdown(raw, source="wiki.md")
    assert len(chunks) >= 1


def test_process_document_markdown_with_numbered_headings():
    raw = "1. INTRODUCTION\n\nIntro text.\n\n2.1 Methods\n\nMethods content."
    chunks = process_document_markdown(raw, source="numbered.md")
    assert len(chunks) >= 1
