# @summary
# Tests for src/ingest/embedding/nodes/chunking.py.
# Covers: text source selection (refactored vs cleaned), heading normalization,
#         semantic vs standard chunking dispatch, ProcessedChunk metadata fields,
#         and exception handling (empty chunks + error appended).
# @end-summary
"""Tests for the chunking_node pipeline stage."""

import pytest
from unittest.mock import MagicMock, patch

from src.ingest.common.schemas import ProcessedChunk
from src.ingest.common.types import IngestionConfig, Runtime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(
    cleaned: str = "# H1\nText",
    refactored: str = "",
    semantic: bool = False,
    embedder=None,
) -> dict:
    """Return a minimal ingest state dict for chunking_node tests."""
    config = IngestionConfig(
        semantic_chunking=semantic,
        chunk_size=500,
        chunk_overlap=50,
    )
    runtime = Runtime(
        config=config,
        embedder=embedder or MagicMock(),
        weaviate_client=MagicMock(),
        kg_builder=None,
    )
    return {
        "cleaned_text": cleaned,
        "refactored_text": refactored,
        "raw_text": "raw content",
        "source_name": "test.md",
        "source_key": "local_fs:1",
        "source_uri": "file:///tmp/test.md",
        "source_id": "test:1",
        "connector": "local_fs",
        "source_version": "1",
        "errors": [],
        "processing_log": [],
        "runtime": runtime,
    }


def _fake_chunk_dicts(n: int = 2) -> list[dict]:
    """Return a list of fake chunk dicts as chunk_markdown might return them.

    chunk_markdown returns dicts with ``text`` and ``header_metadata`` keys.
    ``header_metadata`` uses LangChain MarkdownHeaderTextSplitter format:
    keys like "h1", "h2", ... which _build_section_metadata converts to
    ``heading`` and ``section_path``.
    """
    return [
        {
            "text": f"Chunk {i} content",
            "header_metadata": {"h1": f"Section {i}"},
        }
        for i in range(n)
    ]


# Patch targets for this module
_CHUNK_MARKDOWN = "src.ingest.embedding.nodes.chunking.chunk_markdown"
_NORMALIZE_HEADINGS = "src.ingest.embedding.nodes.chunking.normalize_headings_to_markdown"
_EXTRACT_METADATA = "src.ingest.embedding.nodes.chunking.extract_metadata"
_METADATA_TO_DICT = "src.ingest.embedding.nodes.chunking.metadata_to_dict"


# ---------------------------------------------------------------------------
# Tests: text source selection
# ---------------------------------------------------------------------------

class TestTextSourceSelection:
    """chunking_node picks refactored_text when non-empty, else cleaned_text."""

    def test_chunking_uses_refactored_text_when_present(self):
        """chunk_markdown called with refactored_text when non-empty."""
        from src.ingest.embedding.nodes.chunking import chunking_node

        refactored = "# Refactored\nRefactored body."
        state = _make_state(cleaned="# Original\nOriginal body.", refactored=refactored)

        with (
            patch(_EXTRACT_METADATA, return_value=MagicMock()),
            patch(_METADATA_TO_DICT, return_value={}),
            patch(_NORMALIZE_HEADINGS, side_effect=lambda t: t) as mock_norm,
            patch(_CHUNK_MARKDOWN, return_value=_fake_chunk_dicts()) as mock_chunk,
        ):
            chunking_node(state)

        # normalize_headings_to_markdown should have received the refactored text
        call_arg = mock_norm.call_args[0][0]
        assert call_arg == refactored

    def test_chunking_falls_back_to_cleaned_text(self):
        """chunk_markdown called with cleaned_text when refactored_text is empty."""
        from src.ingest.embedding.nodes.chunking import chunking_node

        cleaned = "# Cleaned\nCleaned body."
        state = _make_state(cleaned=cleaned, refactored="")

        with (
            patch(_EXTRACT_METADATA, return_value=MagicMock()),
            patch(_METADATA_TO_DICT, return_value={}),
            patch(_NORMALIZE_HEADINGS, side_effect=lambda t: t) as mock_norm,
            patch(_CHUNK_MARKDOWN, return_value=_fake_chunk_dicts()),
        ):
            chunking_node(state)

        call_arg = mock_norm.call_args[0][0]
        assert call_arg == cleaned


# ---------------------------------------------------------------------------
# Tests: heading normalization
# ---------------------------------------------------------------------------

class TestHeadingNormalization:
    """normalize_headings_to_markdown is always called before chunk_markdown."""

    def test_heading_metadata_from_chunk_dict(self):
        """chunk heading metadata extracted from chunk_markdown result dict.

        chunk_markdown returns dicts with "header_metadata" → {"h1": "My Heading"}.
        chunking_node calls _build_section_metadata(header_metadata) which sets
        the "heading" field in the ProcessedChunk metadata.
        """
        from src.ingest.embedding.nodes.chunking import chunking_node

        fake_chunks = [
            {"text": "Body text", "header_metadata": {"h1": "My Heading"}}
        ]
        state = _make_state()

        with (
            patch(_EXTRACT_METADATA, return_value=MagicMock()),
            patch(_METADATA_TO_DICT, return_value={}),
            patch(_NORMALIZE_HEADINGS, side_effect=lambda t: t),
            patch(_CHUNK_MARKDOWN, return_value=fake_chunks),
        ):
            result = chunking_node(state)

        chunks = result.get("chunks", state.get("chunks", []))
        assert len(chunks) == 1
        assert chunks[0].metadata.get("heading") == "My Heading"


# ---------------------------------------------------------------------------
# Tests: ProcessedChunk metadata population
# ---------------------------------------------------------------------------

class TestChunkMetadata:
    """Returned ProcessedChunk objects must contain required metadata fields."""

    def test_chunking_produces_processed_chunks_with_metadata(self):
        """Each chunk has heading, section_path, source, source_key in metadata."""
        from src.ingest.embedding.nodes.chunking import chunking_node

        fake_chunks = [
            {
                "text": "Paragraph text.",
                "header_metadata": {"h1": "Intro"},
            }
        ]
        state = _make_state()

        with (
            patch(_EXTRACT_METADATA, return_value=MagicMock()),
            patch(_METADATA_TO_DICT, return_value={}),
            patch(_NORMALIZE_HEADINGS, side_effect=lambda t: t),
            patch(_CHUNK_MARKDOWN, return_value=fake_chunks),
        ):
            result = chunking_node(state)

        chunks = result.get("chunks", state.get("chunks", []))
        assert chunks, "Expected at least one chunk"
        meta = chunks[0].metadata
        assert "heading" in meta
        assert "section_path" in meta
        # chunking_node sets "source" (not "source_name") from state["source_name"]
        assert "source" in meta
        assert meta["source"] == "test.md"


# ---------------------------------------------------------------------------
# Tests: semantic vs standard mode dispatch
# ---------------------------------------------------------------------------

class TestChunkingMode:
    """semantic_chunking flag controls whether embedder is forwarded to chunk_markdown."""

    def test_chunking_semantic_mode_passes_embedder(self):
        """semantic_chunking=True passes embedder to chunk_markdown."""
        from src.ingest.embedding.nodes.chunking import chunking_node

        mock_embedder = MagicMock(name="embedder")
        state = _make_state(semantic=True, embedder=mock_embedder)

        with (
            patch(_EXTRACT_METADATA, return_value=MagicMock()),
            patch(_METADATA_TO_DICT, return_value={}),
            patch(_NORMALIZE_HEADINGS, side_effect=lambda t: t),
            patch(_CHUNK_MARKDOWN, return_value=_fake_chunk_dicts()) as mock_chunk,
        ):
            chunking_node(state)

        # The embedder should appear somewhere in the call args
        call_args = mock_chunk.call_args
        passed_as_positional = mock_embedder in (call_args.args or ())
        passed_as_keyword = mock_embedder in (call_args.kwargs or {}).values()
        assert passed_as_positional or passed_as_keyword, (
            "Expected runtime.embedder to be forwarded to chunk_markdown in semantic mode"
        )

    def test_chunking_standard_mode_no_embedder(self):
        """semantic_chunking=False calls chunk_markdown without embedder."""
        from src.ingest.embedding.nodes.chunking import chunking_node

        mock_embedder = MagicMock(name="embedder")
        state = _make_state(semantic=False, embedder=mock_embedder)

        with (
            patch(_EXTRACT_METADATA, return_value=MagicMock()),
            patch(_METADATA_TO_DICT, return_value={}),
            patch(_NORMALIZE_HEADINGS, side_effect=lambda t: t),
            patch(_CHUNK_MARKDOWN, return_value=_fake_chunk_dicts()) as mock_chunk,
        ):
            chunking_node(state)

        call_args = mock_chunk.call_args
        passed_as_positional = mock_embedder in (call_args.args or ())
        passed_as_keyword = mock_embedder in (call_args.kwargs or {}).values()
        assert not (passed_as_positional or passed_as_keyword), (
            "Embedder should NOT be forwarded to chunk_markdown in standard mode"
        )


# ---------------------------------------------------------------------------
# Tests: zero chunks
# ---------------------------------------------------------------------------

class TestZeroChunks:
    """chunk_markdown returning [] must be handled gracefully."""

    def test_chunking_returns_empty_list_on_zero_chunks(self):
        """chunk_markdown returns [] → state.chunks is []."""
        from src.ingest.embedding.nodes.chunking import chunking_node

        state = _make_state()

        with (
            patch(_NORMALIZE_HEADINGS, side_effect=lambda t: t),
            patch(_CHUNK_MARKDOWN, return_value=[]),
        ):
            result = chunking_node(state)

        chunks = result.get("chunks", state.get("chunks", []))
        assert chunks == []


# ---------------------------------------------------------------------------
# Tests: exception handling
# ---------------------------------------------------------------------------

class TestChunkingExceptionHandling:
    """Exceptions from chunk_markdown must be caught; chunks=[] and error appended."""

    def test_chunking_exception_returns_empty_chunks(self):
        """chunk_markdown raises → chunks=[], error appended."""
        from src.ingest.embedding.nodes.chunking import chunking_node

        state = _make_state()

        with (
            patch(_NORMALIZE_HEADINGS, side_effect=lambda t: t),
            patch(_CHUNK_MARKDOWN, side_effect=RuntimeError("chunker exploded")),
        ):
            result = chunking_node(state)

        chunks = result.get("chunks", state.get("chunks", []))
        errors = result.get("errors", state.get("errors", []))
        assert chunks == []
        assert len(errors) > 0


# ---------------------------------------------------------------------------
# Tests: partial state resilience (optional upstream fields absent/None)
# ---------------------------------------------------------------------------

class TestPartialStateResilience:
    """Chunking node must handle None/absent optional upstream fields gracefully."""

    def test_refactored_text_none_uses_cleaned_text(self):
        """When refactored_text is None (not just empty), cleaned_text is used."""
        from src.ingest.embedding.nodes.chunking import chunking_node

        state = _make_state(cleaned="# Cleaned\nCleaned body.", refactored="")
        # Replace with explicit None to simulate absent optional field.
        state["refactored_text"] = None

        with (
            patch(_EXTRACT_METADATA, return_value=MagicMock()),
            patch(_METADATA_TO_DICT, return_value={}),
            patch(_NORMALIZE_HEADINGS, side_effect=lambda t: t) as mock_norm,
            patch(_CHUNK_MARKDOWN, return_value=_fake_chunk_dicts()),
        ):
            result = chunking_node(state)

        # Node must not raise; must produce chunks.
        chunks = result.get("chunks", state.get("chunks", []))
        assert chunks is not None
        # normalize_headings must be called with cleaned_text (not None).
        call_arg = mock_norm.call_args[0][0]
        assert call_arg is not None
        assert call_arg == "# Cleaned\nCleaned body."

    def test_docling_document_none_uses_markdown_path(self):
        """When docling_document is None the markdown fallback is used without error."""
        from src.ingest.embedding.nodes.chunking import chunking_node

        state = _make_state()
        state["docling_document"] = None

        with (
            patch(_EXTRACT_METADATA, return_value=MagicMock()),
            patch(_METADATA_TO_DICT, return_value={}),
            patch(_NORMALIZE_HEADINGS, side_effect=lambda t: t),
            patch(_CHUNK_MARKDOWN, return_value=_fake_chunk_dicts()),
        ):
            result = chunking_node(state)

        errors = result.get("errors", state.get("errors", []))
        assert errors == []
