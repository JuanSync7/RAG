# @summary
# Tests for src/ingest/embedding/nodes/document_card.py (A3) + workflow wiring.
# Covers: ENABLED path (one card per document_id, embedded card_text, staged
#         records carrying the contract keys + a vector), DISABLED strict no-op
#         (no staged_card_records, embedder untouched), robustness to documents
#         with no usable chunks, and a workflow-wiring assertion that
#         document_card_emission sits between tree_node_synthesis and
#         metadata_generation.
# @end-summary
"""Tests for document_card_emission_node and its placement in the graph."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.ingest.common.schemas import ProcessedChunk
from src.ingest.common.types import IngestionConfig, Runtime
from src.ingest.embedding.nodes.document_card import document_card_emission_node
from src.ingest.embedding.workflow import build_embedding_graph


# ---------------------------------------------------------------------------
# Helpers — mirror tests/ingest/embedding/test_embedding_storage.py idioms.
# ---------------------------------------------------------------------------

def _chunk(document_id: str, heading: str, text: str = "body", **extra) -> ProcessedChunk:
    meta = {
        "document_id": document_id,
        "source": f"{document_id}.md",
        "heading_path": [heading] if heading else [],
        "heading": heading,
        **extra,
    }
    return ProcessedChunk(text=text, metadata=meta)


def _state(chunks, *, build_cards: bool, card_max_headings: int = 60):
    config = IngestionConfig(
        build_document_cards=build_cards,
        card_max_headings=card_max_headings,
    )
    mock_embedder = MagicMock()
    # One vector per card_text in the batched call. The node attaches each
    # returned vector to the corresponding card; a generic side_effect keeps
    # the count flexible regardless of how many documents are present.
    mock_embedder.embed_documents.side_effect = lambda texts: [
        [float(i), float(i) + 0.5] for i, _ in enumerate(texts)
    ]
    runtime = Runtime(
        config=config,
        embedder=mock_embedder,
        weaviate_client=MagicMock(),
    )
    return {
        "chunks": chunks,
        "source_key": "local_fs:doc:1",
        "source_name": "doc.md",
        "errors": [],
        "processing_log": [],
        "runtime": runtime,
    }


_CONTRACT_KEYS = {
    "document_id",
    "title",
    "source",
    "section_headings",
    "card_text",
    "summary",
    "num_chunks",
}


# ---------------------------------------------------------------------------
# ENABLED path
# ---------------------------------------------------------------------------

class TestEnabled:
    def test_two_documents_yield_two_staged_cards(self):
        chunks = [
            _chunk("doc-A", "Overview"),
            _chunk("doc-A", "Details"),
            _chunk("doc-B", "Intro"),
        ]
        state = _state(chunks, build_cards=True)
        result = document_card_emission_node(state)
        cards = result.get("staged_card_records")
        assert cards is not None
        assert len(cards) == 2
        ids = {c["document_id"] for c in cards}
        assert ids == {"doc-A", "doc-B"}

    def test_cards_carry_contract_keys_and_vector(self):
        chunks = [_chunk("doc-A", "Overview"), _chunk("doc-B", "Intro")]
        state = _state(chunks, build_cards=True)
        result = document_card_emission_node(state)
        for card in result["staged_card_records"]:
            assert _CONTRACT_KEYS.issubset(card.keys())
            assert "vector" in card
            assert isinstance(card["vector"], list)
            assert card["vector"]  # non-empty

    def test_embedder_called_with_card_texts(self):
        chunks = [_chunk("doc-A", "Overview"), _chunk("doc-B", "Intro")]
        state = _state(chunks, build_cards=True)
        document_card_emission_node(state)
        embedder = state["runtime"].embedder
        embedder.embed_documents.assert_called_once()
        passed_texts = embedder.embed_documents.call_args[0][0]
        # One card_text per document, each non-empty, batched in one call.
        assert len(passed_texts) == 2
        assert all(isinstance(t, str) and t for t in passed_texts)
        # Headings appear in the embedded card_text.
        joined = "\n".join(passed_texts)
        assert "Overview" in joined and "Intro" in joined

    def test_vector_matches_embedded_card_text_order(self):
        chunks = [_chunk("doc-A", "Overview"), _chunk("doc-B", "Intro")]
        state = _state(chunks, build_cards=True)
        result = document_card_emission_node(state)
        cards = result["staged_card_records"]
        # side_effect assigns [i, i+0.5] to the i-th card_text.
        assert cards[0]["vector"] == [0.0, 0.5]
        assert cards[1]["vector"] == [1.0, 1.5]

    def test_document_order_preserved(self):
        chunks = [
            _chunk("doc-B", "Intro"),
            _chunk("doc-A", "Overview"),
            _chunk("doc-B", "More"),
        ]
        state = _state(chunks, build_cards=True)
        result = document_card_emission_node(state)
        order = [c["document_id"] for c in result["staged_card_records"]]
        # First-seen document order: doc-B before doc-A.
        assert order == ["doc-B", "doc-A"]

    def test_processing_log_records_count(self):
        chunks = [_chunk("doc-A", "Overview"), _chunk("doc-B", "Intro")]
        state = _state(chunks, build_cards=True)
        result = document_card_emission_node(state)
        log = result.get("processing_log", [])
        assert any("document_card" in entry for entry in log)
        assert any("2" in entry for entry in log)

    def test_summary_is_empty_baseline(self):
        """A3 baseline: no LLM summary; card_text excludes summary."""
        chunks = [_chunk("doc-A", "Overview")]
        state = _state(chunks, build_cards=True)
        result = document_card_emission_node(state)
        card = result["staged_card_records"][0]
        assert card["summary"] == ""

    def test_card_max_headings_respected(self):
        chunks = [_chunk("doc-A", f"H{i}") for i in range(10)]
        state = _state(chunks, build_cards=True, card_max_headings=3)
        result = document_card_emission_node(state)
        card = result["staged_card_records"][0]
        assert len(card["section_headings"]) <= 3


# ---------------------------------------------------------------------------
# Robustness: documents with no usable chunks must be skipped, not crash.
# ---------------------------------------------------------------------------

class TestRobustness:
    def test_empty_chunks_no_cards(self):
        state = _state([], build_cards=True)
        result = document_card_emission_node(state)
        assert result.get("staged_card_records", []) == []
        # Embedder must not be called with an empty batch.
        state["runtime"].embedder.embed_documents.assert_not_called()

    def test_chunk_without_document_id_skipped(self):
        # A chunk whose metadata yields no document_id and no usable card_text
        # must be skipped rather than crash; a real doc still produces a card.
        blank = ProcessedChunk(text="", metadata={})
        good = _chunk("doc-A", "Overview")
        state = _state([blank, good], build_cards=True)
        result = document_card_emission_node(state)
        ids = {c["document_id"] for c in result["staged_card_records"]}
        assert "doc-A" in ids
        # The blank/empty group must not produce a spurious card.
        assert "" not in ids


# ---------------------------------------------------------------------------
# DISABLED path — strict no-op (the default).
# ---------------------------------------------------------------------------

class TestDisabled:
    def test_no_staged_card_records_when_disabled(self):
        chunks = [_chunk("doc-A", "Overview"), _chunk("doc-B", "Intro")]
        state = _state(chunks, build_cards=False)
        result = document_card_emission_node(state)
        assert "staged_card_records" not in result

    def test_embedder_not_called_when_disabled(self):
        chunks = [_chunk("doc-A", "Overview")]
        state = _state(chunks, build_cards=False)
        document_card_emission_node(state)
        state["runtime"].embedder.embed_documents.assert_not_called()

    def test_default_config_is_disabled(self):
        # IngestionConfig() default must keep cards OFF.
        assert IngestionConfig().build_document_cards is False


# ---------------------------------------------------------------------------
# Workflow wiring — document_card_emission between tree_node_synthesis and
# metadata_generation.
#
# Two complementary strategies:
#  * node presence via the LangGraph ``get_graph().nodes`` introspection API
#    (repo convention — see tests/ingest/test_dedup_workflow_integration.py;
#    skips gracefully when the API is unavailable, e.g. under the conftest stub).
#  * edge rewiring via a recording ``StateGraph`` proxy. This is stub-independent
#    and pins the exact source->target topology that the stub never records.
# ---------------------------------------------------------------------------

class TestWorkflowWiring:
    def _node_names(self):
        graph = build_embedding_graph()
        try:
            nodes = graph.get_graph().nodes
            return set(nodes.keys()) if hasattr(nodes, "keys") else set(nodes)
        except (AttributeError, TypeError) as exc:
            pytest.skip(f"LangGraph introspection API unavailable: {exc}")

    def test_node_present(self):
        assert "document_card_emission" in self._node_names()

    def test_neighbour_nodes_present(self):
        names = self._node_names()
        for name in ("tree_node_synthesis", "document_card_emission", "metadata_generation"):
            assert name in names, f"missing node: {name}"

    def _recorded_edges(self, monkeypatch):
        """Build the graph through a StateGraph proxy that records add_edge calls.

        Works whether langgraph is the real package or the conftest stub: we
        patch the symbol that ``workflow.py`` looked up at import time.
        """
        import src.ingest.embedding.workflow as wf

        recorded: list[tuple[str, str]] = []
        RealStateGraph = wf.StateGraph

        class _RecordingStateGraph(RealStateGraph):  # type: ignore[misc, valid-type]
            def add_edge(self, source, target, *args, **kwargs):
                recorded.append((source, target))
                return super().add_edge(source, target, *args, **kwargs)

        monkeypatch.setattr(wf, "StateGraph", _RecordingStateGraph)
        wf.build_embedding_graph()
        return set(recorded)

    def test_edge_tree_to_card(self, monkeypatch):
        assert ("tree_node_synthesis", "document_card_emission") in self._recorded_edges(monkeypatch)

    def test_edge_card_to_metadata(self, monkeypatch):
        assert ("document_card_emission", "metadata_generation") in self._recorded_edges(monkeypatch)

    def test_old_direct_edge_removed(self, monkeypatch):
        # The node now sits between them; the direct edge must be gone.
        assert ("tree_node_synthesis", "metadata_generation") not in self._recorded_edges(monkeypatch)

    def test_downstream_edges_intact(self, monkeypatch):
        edges = self._recorded_edges(monkeypatch)
        # The rest of the DAG is unchanged.
        assert ("embedding_storage", "visual_embedding") in edges
        assert ("visual_embedding", "commit") in edges
