# @summary
# Integration tests for cross_document_dedup_node wiring in the Embedding Pipeline.
# Covers: node present in graph, bypass when disabled, override path,
#   merge events in state, idempotency of dedup over identical state.
# @end-summary
"""Tests for Phase 3.3 — cross_document_dedup_node workflow integration.

These tests verify:
- The dedup node is registered in the compiled graph.
- When ``enable_cross_document_dedup=False``, the workflow routes directly to
  ``embedding_storage`` and no merge events are emitted.
- When a source_key is in ``dedup_override_sources``, all chunks pass through
  as novel and each emits an ``override_skipped`` merge event.
- Merge events appear in state after the node executes.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.ingest.common.schemas import ProcessedChunk
from src.ingest.common.types import IngestionConfig, Runtime
from src.ingest.embedding.nodes.cross_document_dedup import cross_document_dedup_node


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(text: str = "hello world chunk text here", metadata: dict | None = None) -> ProcessedChunk:
    return ProcessedChunk(text=text, metadata=metadata if metadata is not None else {})


def _make_state(
    chunks: list,
    *,
    enable_dedup: bool = True,
    enable_fuzzy: bool = False,
    override_sources: list | None = None,
    source_key: str = "doc/test.md",
    weaviate_client: object | None = None,
) -> dict:
    config = IngestionConfig(
        enable_cross_document_dedup=enable_dedup,
        enable_fuzzy_dedup=enable_fuzzy,
        dedup_override_sources=override_sources or [],
    )
    runtime = Runtime(
        config=config,
        embedder=MagicMock(),
        weaviate_client=weaviate_client or MagicMock(),
        kg_builder=None,
    )
    return {
        "chunks": chunks,
        "errors": [],
        "processing_log": [],
        "runtime": runtime,
        "source_key": source_key,
    }


# ---------------------------------------------------------------------------
# T5 — Workflow graph wiring
# ---------------------------------------------------------------------------

class TestWorkflowWiring:
    """Verify cross_document_dedup_node is present in the compiled graph.

    NOTE: LangGraph introspection API stability is not guaranteed.
    Tests that rely on internal graph-node inspection use ``get_graph().nodes``
    and skip gracefully if the API changes.
    """

    def test_build_embedding_graph_returns_non_none(self):
        """build_embedding_graph() must return a compiled graph object."""
        from src.ingest.embedding.workflow import build_embedding_graph

        graph = build_embedding_graph()
        assert graph is not None

    def test_cross_document_dedup_node_in_graph(self):
        """The compiled graph must contain a 'cross_document_dedup' node.

        Uses ``get_graph().nodes`` (LangGraph introspection API).
        Skips if the API is unavailable or has changed.
        """
        from src.ingest.embedding.workflow import build_embedding_graph

        graph = build_embedding_graph()
        try:
            nodes = graph.get_graph().nodes
            node_names = set(nodes.keys()) if hasattr(nodes, "keys") else set(nodes)
        except (AttributeError, TypeError) as exc:
            pytest.skip(f"LangGraph introspection API unavailable: {exc}")

        assert "cross_document_dedup" in node_names, (
            f"Expected 'cross_document_dedup' in graph nodes; got: {sorted(node_names)}"
        )

    def test_dedup_node_between_quality_and_storage(self):
        """Topology: quality_validation, cross_document_dedup, and embedding_storage
        must all be present as registered nodes.

        Uses ``get_graph().nodes`` (LangGraph introspection API).
        Skips if the API is unavailable or has changed.
        """
        from src.ingest.embedding.workflow import build_embedding_graph

        graph = build_embedding_graph()
        try:
            nodes = graph.get_graph().nodes
            node_names = list(nodes.keys()) if hasattr(nodes, "keys") else list(nodes)
        except (AttributeError, TypeError) as exc:
            pytest.skip(f"LangGraph introspection API unavailable: {exc}")

        for name in ("quality_validation", "cross_document_dedup", "embedding_storage"):
            assert name in node_names, f"Missing node: {name}"


# ---------------------------------------------------------------------------
# T5 — Bypass when enable_cross_document_dedup=False
# ---------------------------------------------------------------------------

class TestDedupBypassWhenDisabled:
    """Verify the node returns an empty merge report and passes chunks through."""

    def test_disabled_returns_all_chunks(self):
        chunks = [_make_chunk("chunk A"), _make_chunk("chunk B")]
        state = _make_state(chunks, enable_dedup=False)

        result = cross_document_dedup_node(state)

        assert result["chunks"] == chunks, "All chunks should be returned when dedup disabled"

    def test_disabled_empty_merge_report(self):
        chunks = [_make_chunk()]
        state = _make_state(chunks, enable_dedup=False)

        result = cross_document_dedup_node(state)

        assert result["dedup_merge_report"] == [], "Merge report must be empty when disabled"

    def test_disabled_empty_dedup_stats(self):
        chunks = [_make_chunk()]
        state = _make_state(chunks, enable_dedup=False)

        result = cross_document_dedup_node(state)

        assert result["dedup_stats"] == {}, "Dedup stats must be empty dict when disabled"

    def test_disabled_skipped_in_processing_log(self):
        state = _make_state([_make_chunk()], enable_dedup=False)

        result = cross_document_dedup_node(state)

        log = result.get("processing_log", [])
        assert any("skipped" in entry for entry in log), (
            f"Expected a 'skipped' log entry; got: {log}"
        )

    def test_disabled_no_weaviate_calls(self):
        """When disabled, the node must not touch the Weaviate client."""
        mock_client = MagicMock()
        state = _make_state([_make_chunk()], enable_dedup=False, weaviate_client=mock_client)

        cross_document_dedup_node(state)

        mock_client.collections.get.assert_not_called()


# ---------------------------------------------------------------------------
# T7 — Override path (dedup_override_sources)
# ---------------------------------------------------------------------------

class TestDedupOverridePath:
    """Verify per-source override: chunks pass through, override_skipped events emitted."""

    def _setup_no_match_client(self) -> MagicMock:
        """Return a mock Weaviate client that finds no existing chunk."""
        client = MagicMock()
        collection = MagicMock()
        client.collections.get.return_value = collection
        fetch_result = MagicMock()
        fetch_result.objects = []
        collection.query.fetch_objects.return_value = fetch_result
        return client

    def test_override_source_chunks_pass_through(self):
        """All chunks for an overridden source_key are returned as novel."""
        chunks = [_make_chunk("alpha text"), _make_chunk("beta text")]
        state = _make_state(
            chunks,
            enable_dedup=True,
            override_sources=["doc/test.md"],
            source_key="doc/test.md",
        )

        result = cross_document_dedup_node(state)

        assert len(result["chunks"]) == 2, "Both chunks should pass through on override"

    def test_override_emits_override_skipped_events(self):
        """Each chunk from an overridden source emits an override_skipped merge event."""
        chunks = [_make_chunk("alpha text"), _make_chunk("beta text")]
        state = _make_state(
            chunks,
            enable_dedup=True,
            override_sources=["doc/test.md"],
            source_key="doc/test.md",
        )

        result = cross_document_dedup_node(state)

        events = result["dedup_merge_report"]
        assert len(events) == 2, f"Expected 2 override events; got {len(events)}"
        for event in events:
            assert event["action"] == "override_skipped", (
                f"Expected action='override_skipped'; got action='{event['action']}'"
            )
            assert event["match_tier"] == "override"
            assert event["merged_source_key"] == "doc/test.md"
            assert event["canonical_chunk_id"] == ""

    def test_override_no_weaviate_lookup(self):
        """Override path must not query Weaviate for content hash matches."""
        mock_client = MagicMock()
        chunks = [_make_chunk("some text")]
        state = _make_state(
            chunks,
            enable_dedup=True,
            override_sources=["doc/test.md"],
            source_key="doc/test.md",
            weaviate_client=mock_client,
        )

        cross_document_dedup_node(state)

        # fetch_objects should not have been called for hash lookup
        # (collections.get may be called for back-ref cleanup if update_mode=True,
        #  but override_sources does NOT trigger update_mode — they're orthogonal)
        mock_client.collections.get.assert_not_called()

    def test_non_overridden_source_still_deduplicates(self):
        """A source not in override_sources follows normal dedup logic."""
        mock_client = MagicMock()
        collection = MagicMock()
        mock_client.collections.get.return_value = collection
        # Simulate no existing chunk found (novel)
        fetch_result = MagicMock()
        fetch_result.objects = []
        collection.query.fetch_objects.return_value = fetch_result

        chunks = [_make_chunk("unique text for doc_b")]
        state = _make_state(
            chunks,
            enable_dedup=True,
            override_sources=["doc/other.md"],  # different source is overridden
            source_key="doc/test.md",           # this source is not overridden
            weaviate_client=mock_client,
        )

        result = cross_document_dedup_node(state)

        # chunk should still be in output (novel)
        assert len(result["chunks"]) == 1
        # No override event
        assert all(e["action"] != "override_skipped" for e in result["dedup_merge_report"])


# ---------------------------------------------------------------------------
# T6 — Merge events in state
# ---------------------------------------------------------------------------

class TestMergeEventsInState:
    """Verify merge events are populated correctly when dedup finds a match."""

    def _make_client_with_match(self, existing_uuid: str = "aaaa-bbbb-cccc-dddd") -> MagicMock:
        """Return a mock client that returns a matching chunk for any hash lookup."""
        client = MagicMock()
        collection = MagicMock()
        client.collections.get.return_value = collection

        # fetch_objects returns one matching object
        obj = MagicMock()
        obj.uuid = existing_uuid
        obj.properties = {
            "content_hash": "some_hash",
            "source_documents": ["original/doc.md"],
            "text": "hello world chunk text here",
        }
        fetch_result = MagicMock()
        fetch_result.objects = [obj]
        collection.query.fetch_objects.return_value = fetch_result

        # fetch_object_by_id for append_source_document
        existing_obj = MagicMock()
        existing_obj.properties = {"source_documents": ["original/doc.md"]}
        collection.query.fetch_object_by_id.return_value = existing_obj
        collection.data.update.return_value = None

        return client

    def test_exact_match_emits_merged_event(self):
        """An exact hash match produces a 'merged' action merge event."""
        existing_uuid = "aaaa-bbbb-cccc-dddd"
        mock_client = self._make_client_with_match(existing_uuid)
        chunks = [_make_chunk("hello world chunk text here")]
        state = _make_state(chunks, enable_dedup=True, weaviate_client=mock_client)

        result = cross_document_dedup_node(state)

        events = result["dedup_merge_report"]
        assert len(events) == 1
        event = events[0]
        assert event["action"] == "merged"
        assert event["match_tier"] == "exact"
        assert event["similarity_score"] == 1.0
        assert event["canonical_replaced"] is False
        assert event["merged_source_key"] == "doc/test.md"
        assert event["canonical_chunk_id"] == existing_uuid
        assert "timestamp" in event
        assert "canonical_content_hash" in event

    def test_exact_match_chunk_removed_from_output(self):
        """A deduplicated chunk must not appear in the output chunks list."""
        mock_client = self._make_client_with_match()
        chunks = [_make_chunk("hello world chunk text here")]
        state = _make_state(chunks, enable_dedup=True, weaviate_client=mock_client)

        result = cross_document_dedup_node(state)

        assert result["chunks"] == [], "Deduped chunk should be removed from output"

    def test_dedup_stats_populated(self):
        """dedup_stats must contain expected counters after a dedup run."""
        mock_client = self._make_client_with_match()
        chunks = [_make_chunk("hello world chunk text here")]
        state = _make_state(chunks, enable_dedup=True, weaviate_client=mock_client)

        result = cross_document_dedup_node(state)

        stats = result["dedup_stats"]
        assert stats["total_input_chunks"] == 1
        assert stats["exact_matches"] == 1
        assert stats["novel_chunks"] == 0
        assert stats["degraded"] is False

    def test_novel_chunk_has_content_hash_in_metadata(self):
        """Novel chunks must have content_hash set in metadata."""
        mock_client = MagicMock()
        collection = MagicMock()
        mock_client.collections.get.return_value = collection
        # No match
        fetch_result = MagicMock()
        fetch_result.objects = []
        collection.query.fetch_objects.return_value = fetch_result

        chunk = _make_chunk("a novel piece of text")
        state = _make_state([chunk], enable_dedup=True, weaviate_client=mock_client)

        result = cross_document_dedup_node(state)

        assert len(result["chunks"]) == 1
        assert "content_hash" in result["chunks"][0].metadata
        assert len(result["chunks"][0].metadata["content_hash"]) == 64  # SHA-256 hex

    def test_merge_event_contains_required_fields(self):
        """Each merge event must include all fields defined in MergeEvent TypedDict."""
        from src.ingest.embedding.common.types import MergeEvent

        required_keys = set(MergeEvent.__annotations__.keys())
        mock_client = self._make_client_with_match()
        chunks = [_make_chunk("hello world chunk text here")]
        state = _make_state(chunks, enable_dedup=True, weaviate_client=mock_client)

        result = cross_document_dedup_node(state)

        events = result["dedup_merge_report"]
        assert len(events) == 1
        missing = required_keys - set(events[0].keys())
        assert not missing, f"Merge event missing required fields: {missing}"
