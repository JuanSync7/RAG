# @summary
# Tests the enable_kg_phase2b feature flag — when True, the embedding
# LangGraph KG nodes short-circuit and commit_node skips kg_builder.add_chunk
# replay. When False (default), legacy in-pipeline behavior is preserved.
# Exports: (test module)
# Deps: pytest, src.ingest.common.types, src.ingest.embedding.nodes
# @end-summary
"""Step 6b: enable_kg_phase2b feature flag gating tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.ingest.common.schemas import ProcessedChunk
from src.ingest.common.types import IngestionConfig, Runtime
from src.ingest.embedding.nodes.commit_node import commit_node
from src.ingest.embedding.nodes.knowledge_graph_extraction import (
    knowledge_graph_extraction_node,
)
from src.ingest.embedding.nodes.knowledge_graph_storage import (
    knowledge_graph_storage_node,
)

EXTRACTOR_PATH = (
    "src.ingest.embedding.nodes.knowledge_graph_extraction._EXTRACTOR"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(text: str = "Alice knows Bob.") -> ProcessedChunk:
    return ProcessedChunk(text=text, metadata={})


def _runtime(*, kg_builder=None, enable_kg_phase2b: bool = False) -> Runtime:
    config = IngestionConfig(
        enable_knowledge_graph_extraction=True,
        enable_knowledge_graph_storage=True,
        enable_kg_phase2b=enable_kg_phase2b,
    )
    return Runtime(
        config=config,
        embedder=MagicMock(),
        weaviate_client=MagicMock(),
        kg_builder=kg_builder,
    )


def _state(runtime: Runtime, *, chunks=None, staged_kg=None, errors=None):
    return {
        "chunks": chunks or [_chunk()],
        "kg_triples": [],
        "source_key": "doc-1",
        "source_name": "doc-1",
        "errors": errors or [],
        "processing_log": [],
        "runtime": runtime,
        "staging_batch_id": "",
        "document_id": "",
        "staged_minio": None,
        "staged_weaviate_records": [],
        "staged_weaviate_delete_old": False,
        "staged_kg_chunks": staged_kg or [],
    }


# ---------------------------------------------------------------------------
# Default (flag False): existing behavior preserved
# ---------------------------------------------------------------------------


def test_default_flag_false_extraction_runs() -> None:
    runtime = _runtime()
    assert runtime.config.enable_kg_phase2b is False
    state = _state(runtime, chunks=[_chunk()])
    with patch(EXTRACTOR_PATH) as ext:
        ext.extract_entities.return_value = ["Alice", "Bob"]
        ext.extract_relations.return_value = [("Alice", "knows", "Bob")]
        result = knowledge_graph_extraction_node(state)
    assert "kg_triples" in result
    assert len(result["kg_triples"]) == 1


def test_default_flag_false_storage_stages_chunks() -> None:
    runtime = _runtime(kg_builder=MagicMock())
    state = _state(runtime, chunks=[_chunk("hello")])
    result = knowledge_graph_storage_node(state)
    assert len(result.get("staged_kg_chunks", [])) == 1


# ---------------------------------------------------------------------------
# Flag True: short-circuit
# ---------------------------------------------------------------------------


def test_flag_true_extraction_short_circuits() -> None:
    runtime = _runtime(enable_kg_phase2b=True)
    state = _state(runtime, chunks=[_chunk()])
    with patch(EXTRACTOR_PATH) as ext:
        result = knowledge_graph_extraction_node(state)
        ext.extract_entities.assert_not_called()
        ext.extract_relations.assert_not_called()
    assert result.get("kg_triples") in (None, [])
    log = result.get("processing_log", [])
    assert any("knowledge_graph_extraction:skipped" in entry for entry in log)


def test_flag_true_storage_short_circuits() -> None:
    builder = MagicMock()
    runtime = _runtime(kg_builder=builder, enable_kg_phase2b=True)
    state = _state(runtime, chunks=[_chunk("hello")])
    result = knowledge_graph_storage_node(state)
    assert result.get("staged_kg_chunks") == []
    builder.add_chunk.assert_not_called()
    log = result.get("processing_log", [])
    assert any("knowledge_graph_storage:skipped" in entry for entry in log)


def test_flag_true_commit_skips_kg_replay() -> None:
    builder = MagicMock()
    runtime = _runtime(kg_builder=builder, enable_kg_phase2b=True)
    # Even if upstream stages a payload, commit must NOT replay it under flag.
    staged = [("text-a", "doc-1"), ("text-b", "doc-1")]
    state = _state(runtime, staged_kg=staged)
    state["runtime"].db_client = None
    state["runtime"].weaviate_client = None

    commit_node(state)
    builder.add_chunk.assert_not_called()


def test_flag_false_commit_replays_kg() -> None:
    builder = MagicMock()
    runtime = _runtime(kg_builder=builder, enable_kg_phase2b=False)
    staged = [("text-a", "doc-1")]
    state = _state(runtime, staged_kg=staged)
    state["runtime"].db_client = None
    state["runtime"].weaviate_client = None

    commit_node(state)
    builder.add_chunk.assert_called_once_with("text-a", source="doc-1")
