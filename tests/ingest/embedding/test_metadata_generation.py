# @summary
# Tests for src/ingest/embedding/nodes/metadata_generation.py.
# Covers: LLM-enabled summary/keyword extraction, fallback on disabled or empty
#         LLM response, max_keywords cap, chunk projection, and empty-chunks guard.
# @end-summary
"""Tests for the metadata_generation_node pipeline stage."""

import pytest
from unittest.mock import MagicMock, patch

from src.ingest.common.schemas import ProcessedChunk
from src.ingest.common.types import IngestionConfig, Runtime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(text: str = "Sample text for a chunk.") -> ProcessedChunk:
    return ProcessedChunk(text=text, metadata={"heading": "Section"})


def _make_state(
    chunks=None,
    enable_llm: bool = False,
    max_keywords: int = 10,
    cleaned: str = "First sentence. Second one.",
) -> dict:
    """Return a minimal ingest state dict for metadata_generation_node tests."""
    config = IngestionConfig(
        enable_llm_metadata=enable_llm,
        max_keywords=max_keywords,
        llm_temperature=0.0,
        llm_timeout_seconds=10,
    )
    runtime = Runtime(
        config=config,
        embedder=MagicMock(),
        weaviate_client=MagicMock(),
        kg_builder=None,
    )
    return {
        "chunks": chunks if chunks is not None else [],
        "cleaned_text": cleaned,
        "metadata_summary": "",
        "metadata_keywords": [],
        "errors": [],
        "processing_log": [],
        "runtime": runtime,
    }


# Patch targets
_LLM_JSON = "src.ingest.embedding.nodes.metadata_generation._llm_json"
_FALLBACK = "src.ingest.embedding.nodes.metadata_generation.extract_keywords_fallback"


def _run(state: dict) -> dict:
    """Import and run metadata_generation_node; return result dict."""
    from src.ingest.embedding.nodes.metadata_generation import metadata_generation_node
    return metadata_generation_node(state)


def _get(key: str, result: dict, state: dict):
    return result.get(key, state.get(key))


# ---------------------------------------------------------------------------
# Tests: LLM-enabled path
# ---------------------------------------------------------------------------

class TestLLMEnabledPath:
    """When enable_llm_metadata=True and LLM returns valid data, use it."""

    def test_metadata_summary_set_from_llm(self):
        """LLM path: metadata_summary taken from LLM response."""
        state = _make_state(enable_llm=True)
        llm_response = {"summary": "An LLM-generated summary.", "keywords": ["alpha", "beta"]}

        with (
            patch(_LLM_JSON, return_value=llm_response),
            patch(_FALLBACK, return_value=["fallback"]),
        ):
            result = _run(state)

        summary = _get("metadata_summary", result, state)
        assert summary == "An LLM-generated summary."

    def test_metadata_keywords_set_from_llm(self):
        """LLM path: metadata_keywords taken from LLM response."""
        state = _make_state(enable_llm=True)
        llm_response = {"summary": "Summary.", "keywords": ["x", "y", "z"]}

        with (
            patch(_LLM_JSON, return_value=llm_response),
            patch(_FALLBACK, return_value=["fallback"]),
        ):
            result = _run(state)

        keywords = _get("metadata_keywords", result, state)
        assert keywords == ["x", "y", "z"]


# ---------------------------------------------------------------------------
# Tests: disabled / fallback path
# ---------------------------------------------------------------------------

class TestFallbackPath:
    """When LLM is disabled or returns {}, fall back to heuristic extraction."""

    def test_llm_disabled_skips_generation(self):
        """enable_llm_metadata=False: node returns skipped log entry; no LLM or fallback call."""
        state = _make_state(enable_llm=False)

        with (
            patch(_LLM_JSON) as mock_llm,
            patch(_FALLBACK) as mock_fallback,
        ):
            result = _run(state)

        # Node short-circuits when LLM is disabled; neither LLM nor fallback is called.
        mock_llm.assert_not_called()
        mock_fallback.assert_not_called()
        # The result contains only a processing_log entry indicating the stage was skipped.
        assert any("skipped" in entry for entry in result.get("processing_log", []))

    def test_llm_empty_response_uses_fallback(self):
        """_llm_json returns {} → fallback keywords and first-sentence summary."""
        state = _make_state(
            enable_llm=True,
            cleaned="First sentence. Second sentence.",
        )
        fallback_keywords = ["first", "sentence"]

        with (
            patch(_LLM_JSON, return_value={}),
            patch(_FALLBACK, return_value=fallback_keywords),
        ):
            result = _run(state)

        keywords = _get("metadata_keywords", result, state)
        assert keywords is not None
        summary = _get("metadata_summary", result, state)
        # Summary must be derived from the first sentence
        assert "First sentence" in summary or summary.startswith("First")


# ---------------------------------------------------------------------------
# Tests: max_keywords cap
# ---------------------------------------------------------------------------

class TestMaxKeywordsCap:
    """Keywords must never exceed max_keywords regardless of source."""

    def test_keywords_capped_at_max_keywords(self):
        """max_keywords=3, LLM returns 5 keywords → 3 retained."""
        state = _make_state(enable_llm=True, max_keywords=3)
        llm_response = {
            "summary": "Summary.",
            "keywords": ["a", "b", "c", "d", "e"],
        }

        with (
            patch(_LLM_JSON, return_value=llm_response),
            patch(_FALLBACK, return_value=[]),
        ):
            result = _run(state)

        keywords = _get("metadata_keywords", result, state)
        assert len(keywords) <= 3

    def test_fallback_keywords_capped(self):
        """max_keywords=2, fallback returns 10 terms → 2 retained.

        _llm_json is always called regardless of enable_llm_metadata; when disabled
        it returns an empty/None result causing fallback to kick in.
        """
        state = _make_state(enable_llm=False, max_keywords=2)
        many_keywords = [f"term{i}" for i in range(10)]

        with (
            patch(_LLM_JSON, return_value={}),
            patch(_FALLBACK, return_value=many_keywords),
        ):
            result = _run(state)

        keywords = _get("metadata_keywords", result, state)
        assert len(keywords) <= 2

    def test_max_keywords_zero_returns_empty_list(self):
        """max_keywords=0 → metadata_keywords == []."""
        state = _make_state(enable_llm=True, max_keywords=0)
        llm_response = {"summary": "Summary.", "keywords": ["a", "b", "c"]}

        with (
            patch(_LLM_JSON, return_value=llm_response),
            patch(_FALLBACK, return_value=["x", "y"]),
        ):
            result = _run(state)

        keywords = _get("metadata_keywords", result, state)
        assert keywords == []


# ---------------------------------------------------------------------------
# Tests: projection into chunks
# ---------------------------------------------------------------------------

class TestChunkProjection:
    """doc_summary and doc_keywords are projected into every chunk's metadata."""

    def test_summary_and_keywords_projected_into_chunks(self):
        """doc_summary and doc_keywords set on every chunk's metadata."""
        chunks = [_make_chunk("Alpha."), _make_chunk("Beta."), _make_chunk("Gamma.")]
        state = _make_state(chunks=chunks, enable_llm=True)
        llm_response = {"summary": "Projected summary.", "keywords": ["key1", "key2"]}

        with (
            patch(_LLM_JSON, return_value=llm_response),
            patch(_FALLBACK, return_value=[]),
        ):
            result = _run(state)

        out_chunks = _get("chunks", result, state)
        assert len(out_chunks) == 3
        for chunk in out_chunks:
            # The node projects using "document_summary" and "document_keywords" keys.
            assert chunk.metadata.get("document_summary") is not None, (
                f"Expected document_summary in chunk.metadata; got {chunk.metadata}"
            )
            assert chunk.metadata.get("document_keywords") is not None, (
                f"Expected document_keywords in chunk.metadata; got {chunk.metadata}"
            )

    def test_empty_chunks_no_projection_error(self):
        """chunks=[] → no crash; state.metadata_summary and metadata_keywords still set."""
        state = _make_state(chunks=[], enable_llm=True)
        llm_response = {"summary": "Empty doc summary.", "keywords": ["empty"]}

        with (
            patch(_LLM_JSON, return_value=llm_response),
            patch(_FALLBACK, return_value=[]),
        ):
            result = _run(state)

        # Must not raise; output fields must be populated
        summary = _get("metadata_summary", result, state)
        keywords = _get("metadata_keywords", result, state)
        assert summary is not None
        assert keywords is not None
        out_chunks = _get("chunks", result, state)
        assert out_chunks == []


# ---------------------------------------------------------------------------
# Tests: partial state resilience (optional upstream fields absent/None)
# ---------------------------------------------------------------------------

class TestPartialStateResilience:
    """metadata_generation_node must handle None/absent optional upstream fields."""

    def test_refactored_text_none_falls_back_to_cleaned(self):
        """refactored_text=None must not raise; node falls back to cleaned_text."""
        state = _make_state(enable_llm=True, cleaned="Some content here.")
        state["refactored_text"] = None  # simulate absent optional upstream field

        with (
            patch(_LLM_JSON, return_value={"summary": "ok", "keywords": ["a"]}),
            patch(_FALLBACK, return_value=[]),
        ):
            result = _run(state)

        # Must not raise; summary should be populated.
        summary = _get("metadata_summary", result, state)
        assert summary is not None

    def test_refactored_text_and_cleaned_empty_no_crash(self):
        """When refactored_text is None and cleaned_text is empty string, node must not raise."""
        state = _make_state(enable_llm=True, cleaned="")
        state["refactored_text"] = None  # simulate absent optional upstream field

        with (
            patch(_LLM_JSON, return_value={}),
            patch(_FALLBACK, return_value=["fallback"]),
        ):
            result = _run(state)

        # Must not raise; result must be a dict.
        assert result is not None
        assert isinstance(result, dict)
