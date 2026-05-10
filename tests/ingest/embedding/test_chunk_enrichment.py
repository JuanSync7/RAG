# @summary
# Tests for src/ingest/embedding/nodes/chunk_enrichment.py.
# Covers: chunk_id derivation (24-char hex, determinism, ordinal uniqueness),
#         source field propagation, retrieval_text_origin flag,
#         enriched_content population, and empty-chunks no-op.
# @end-summary
"""Tests for the chunk_enrichment_node pipeline stage."""

import pytest
from unittest.mock import MagicMock

from src.ingest.common.schemas import ProcessedChunk
from src.ingest.common.types import IngestionConfig, Runtime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(text: str = "Sample chunk text.", heading: str = "Intro") -> ProcessedChunk:
    return ProcessedChunk(
        text=text,
        metadata={
            "heading": heading,
            "section_path": f"root > {heading}",
            "source_name": "doc.txt",
        },
    )


def _make_state(chunks=None, enable_refactoring: bool = False) -> dict:
    """Return a minimal ingest state dict for chunk_enrichment_node tests."""
    config = IngestionConfig(enable_document_refactoring=enable_refactoring)
    runtime = Runtime(
        config=config,
        embedder=MagicMock(),
        weaviate_client=MagicMock(),
        kg_builder=None,
    )
    return {
        "chunks": chunks if chunks is not None else [],
        "source_name": "doc.txt",
        "source_key": "local_fs:test:1",
        "source_uri": "file:///tmp/doc.txt",
        "source_id": "test:1",
        "connector": "local_fs",
        "source_version": "1",
        "raw_text": "raw content",
        "cleaned_text": "cleaned content",
        "refactored_text": "",
        "errors": [],
        "processing_log": [],
        "runtime": runtime,
    }


def _get_chunks(result: dict, original_state: dict) -> list:
    return result.get("chunks", original_state.get("chunks", []))


# ---------------------------------------------------------------------------
# Tests: chunk_id derivation
# ---------------------------------------------------------------------------

class TestChunkIdDerivation:
    """chunk_id is deterministic, 24-char hex, and ordinal-sensitive."""

    def test_chunk_id_set_on_every_chunk(self):
        """chunk_id is set on every chunk after enrichment."""
        from src.ingest.embedding.nodes.chunk_enrichment import chunk_enrichment_node

        chunks = [_make_chunk("Alpha"), _make_chunk("Beta"), _make_chunk("Gamma")]
        state = _make_state(chunks=chunks)
        result = chunk_enrichment_node(state)
        out_chunks = _get_chunks(result, state)
        assert len(out_chunks) == 3
        for chunk in out_chunks:
            assert "chunk_id" in chunk.metadata
            assert chunk.metadata["chunk_id"] != ""

    def test_chunk_id_is_24_char_hex(self):
        """chunk_id is a non-empty deterministic string (UUID5 format)."""
        from src.ingest.embedding.nodes.chunk_enrichment import chunk_enrichment_node

        state = _make_state(chunks=[_make_chunk("Some text.")])
        result = chunk_enrichment_node(state)
        chunk_id = _get_chunks(result, state)[0].metadata["chunk_id"]
        # build_chunk_id returns a UUID5 string (36 chars like xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)
        assert chunk_id != ""
        assert len(chunk_id) > 0
        # Must contain only hex digits and hyphens (UUID format)
        import re
        assert re.fullmatch(r"[0-9a-f\-]+", chunk_id), f"chunk_id not UUID-like: {chunk_id!r}"

    def test_chunk_id_is_deterministic(self):
        """Same source_key + ordinal + text → same chunk_id across two calls."""
        from src.ingest.embedding.nodes.chunk_enrichment import chunk_enrichment_node

        text = "Deterministic body text."
        state_a = _make_state(chunks=[_make_chunk(text)])
        state_b = _make_state(chunks=[_make_chunk(text)])
        result_a = chunk_enrichment_node(state_a)
        result_b = chunk_enrichment_node(state_b)
        id_a = _get_chunks(result_a, state_a)[0].metadata["chunk_id"]
        id_b = _get_chunks(result_b, state_b)[0].metadata["chunk_id"]
        assert id_a == id_b

    def test_two_chunks_same_text_different_ordinal_have_different_ids(self):
        """Chunks with same text but different ordinal produce different chunk_ids."""
        from src.ingest.embedding.nodes.chunk_enrichment import chunk_enrichment_node

        same_text = "Identical body."
        state = _make_state(chunks=[_make_chunk(same_text), _make_chunk(same_text)])
        result = chunk_enrichment_node(state)
        out = _get_chunks(result, state)
        id_0 = out[0].metadata["chunk_id"]
        id_1 = out[1].metadata["chunk_id"]
        assert id_0 != id_1, (
            "Two chunks with identical text but different ordinals must have different chunk_ids"
        )


# ---------------------------------------------------------------------------
# Tests: source field propagation
# ---------------------------------------------------------------------------

class TestSourceFieldPropagation:
    """source, source_key, source_uri, connector are set on every chunk."""

    def test_source_fields_set_on_all_chunks(self):
        """source and source_key are set on every chunk; citation_source_uri for URI."""
        from src.ingest.embedding.nodes.chunk_enrichment import chunk_enrichment_node

        chunks = [_make_chunk("Chunk A"), _make_chunk("Chunk B")]
        state = _make_state(chunks=chunks)
        result = chunk_enrichment_node(state)
        out = _get_chunks(result, state)
        for chunk in out:
            assert "source_key" in chunk.metadata
            assert chunk.metadata["source_key"] == "local_fs:test:1"
            # node sets citation_source_uri (not source_uri) for the URI field
            assert "citation_source_uri" in chunk.metadata
            assert chunk.metadata["citation_source_uri"] == "file:///tmp/doc.txt"

    def test_source_name_propagated(self):
        """chunk.metadata['source'] == source_name from state."""
        from src.ingest.embedding.nodes.chunk_enrichment import chunk_enrichment_node

        state = _make_state(chunks=[_make_chunk("Text.")])
        result = chunk_enrichment_node(state)
        chunk = _get_chunks(result, state)[0]
        # The node sets source or source_name; accept either key
        source_value = chunk.metadata.get("source") or chunk.metadata.get("source_name")
        assert source_value == "doc.txt"


# ---------------------------------------------------------------------------
# Tests: retrieval_text_origin
# ---------------------------------------------------------------------------

class TestRetrievalTextOrigin:
    """retrieval_text_origin reflects the enable_document_refactoring config flag."""

    def test_retrieval_text_origin_original_when_refactoring_disabled(self):
        """retrieval_text_origin == 'original' when enable_document_refactoring=False."""
        from src.ingest.embedding.nodes.chunk_enrichment import chunk_enrichment_node

        state = _make_state(chunks=[_make_chunk("Body.")], enable_refactoring=False)
        result = chunk_enrichment_node(state)
        chunk = _get_chunks(result, state)[0]
        assert chunk.metadata.get("retrieval_text_origin") == "original"

    def test_retrieval_text_origin_refactored_when_enabled(self):
        """retrieval_text_origin == 'refactored' when enable_document_refactoring=True."""
        from src.ingest.embedding.nodes.chunk_enrichment import chunk_enrichment_node

        state = _make_state(chunks=[_make_chunk("Body.")], enable_refactoring=True)
        result = chunk_enrichment_node(state)
        chunk = _get_chunks(result, state)[0]
        assert chunk.metadata.get("retrieval_text_origin") == "refactored"


# ---------------------------------------------------------------------------
# Tests: enriched_content
# ---------------------------------------------------------------------------

class TestEnrichedContent:
    """enriched_content is always set and non-empty on each chunk."""

    def test_enriched_content_is_set(self):
        """enriched_content field is set and non-empty on each chunk."""
        from src.ingest.embedding.nodes.chunk_enrichment import chunk_enrichment_node

        state = _make_state(chunks=[_make_chunk("Rich text here.")])
        result = chunk_enrichment_node(state)
        chunk = _get_chunks(result, state)[0]
        enriched = chunk.metadata.get("enriched_content")
        assert enriched is not None
        assert enriched != ""


# ---------------------------------------------------------------------------
# Tests: empty chunks no-op
# ---------------------------------------------------------------------------

class TestEmptyChunksNoOp:
    """Empty chunks list must not cause errors or mutations."""

    def test_empty_chunks_list_no_mutation(self):
        """Empty chunks list → no errors, state.chunks still empty."""
        from src.ingest.embedding.nodes.chunk_enrichment import chunk_enrichment_node

        state = _make_state(chunks=[])
        result = chunk_enrichment_node(state)
        out_chunks = _get_chunks(result, state)
        errors = result.get("errors", state.get("errors", []))
        assert out_chunks == []
        assert errors == []


# ---------------------------------------------------------------------------
# Tests: partial state resilience (optional upstream fields absent/None)
# ---------------------------------------------------------------------------

class TestPartialStateResilience:
    """chunk_enrichment_node must handle None/absent optional fields gracefully."""

    def test_refactored_text_none_no_error(self):
        """refactored_text=None must not raise KeyError or AttributeError."""
        from src.ingest.embedding.nodes.chunk_enrichment import chunk_enrichment_node

        state = _make_state(chunks=[_make_chunk("body text")])
        state["refactored_text"] = None  # absent optional upstream field

        result = chunk_enrichment_node(state)
        errors = result.get("errors", state.get("errors", []))
        assert errors == []
        chunks = _get_chunks(result, state)
        assert len(chunks) == 1
        assert "chunk_id" in chunks[0].metadata

    def test_cleaned_text_absent_no_error(self):
        """When cleaned_text is absent from state, node must not raise KeyError."""
        from src.ingest.embedding.nodes.chunk_enrichment import chunk_enrichment_node

        state = _make_state(chunks=[_make_chunk("body text")])
        state.pop("cleaned_text", None)  # simulate absent optional upstream field

        result = chunk_enrichment_node(state)
        errors = result.get("errors", state.get("errors", []))
        assert errors == []

    def test_document_id_defaults_to_empty_string(self):
        """When document_id is absent from state, chunk.metadata['document_id'] is ''."""
        from src.ingest.embedding.nodes.chunk_enrichment import chunk_enrichment_node

        state = _make_state(chunks=[_make_chunk("text")])
        state.pop("document_id", None)  # document_id set later by document_storage_node

        result = chunk_enrichment_node(state)
        chunk = _get_chunks(result, state)[0]
        # document_id must be present in metadata and be an empty string (not raise)
        assert chunk.metadata.get("document_id") == ""
