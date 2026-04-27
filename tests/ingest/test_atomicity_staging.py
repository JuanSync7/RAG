# @summary
# TDD tests for Issue #42: stage-only storage nodes + atomic commit_node.
# Covers: nodes stage to state only (no direct DB writes), commit_node flushes
#   all three sinks, rollback on Weaviate failure, rollback on MinIO failure,
#   skip when upstream errors, and graceful no-MinIO path.
# @end-summary
"""TDD tests for the atomicity staging refactor (Issue #42).

Tests are intentionally written *before* implementation so they fail red,
then go green once the implementation is in place.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch, call

import pytest


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_runtime(
    *,
    store_documents: bool = True,
    update_mode: bool = False,
    enable_kg: bool = False,
    db_client=None,
    weaviate_client=None,
    kg_builder=None,
    target_bucket: str = "test-bucket",
    target_collection: str = "TestCol",
):
    """Build a minimal Runtime with controlled config."""
    from src.ingest.common.types import IngestionConfig, Runtime

    embedder = MagicMock()
    embedder.embed_documents = MagicMock(return_value=[[0.1, 0.2]])

    config = IngestionConfig(
        store_documents=store_documents,
        update_mode=update_mode,
        enable_knowledge_graph_storage=enable_kg,
        target_bucket=target_bucket,
        target_collection=target_collection,
    )
    return Runtime(
        config=config,
        embedder=embedder,
        weaviate_client=weaviate_client or MagicMock(),
        kg_builder=kg_builder,
        db_client=db_client or MagicMock(),
    )


def _make_base_state(runtime, **overrides) -> dict:
    """Minimal state dict that satisfies all three storage nodes."""
    from src.ingest.common.schemas import ProcessedChunk

    chunk = ProcessedChunk(
        text="hello world",
        metadata={
            "source": "test.md",
            "source_key": "k1",
            "chunk_index": 0,
        },
    )
    state = {
        "runtime": runtime,
        "source_key": "k1",
        "source_name": "test.md",
        "source_uri": "file:///tmp/test.md",
        "source_id": "test:1",
        "source_version": "1",
        "connector": "local_fs",
        "raw_text": "# Hello",
        "cleaned_text": "# Hello",
        "refactored_text": "",
        "clean_hash": "abc",
        "chunks": [chunk],
        "errors": [],
        "processing_log": [],
        "staging_batch_id": "batch-test-001",
        "source_hash": "deadbeef",
        "trace_id": "",
        "batch_id": "",
        "document_id": "",
        # staging fields start empty
        "staged_minio": None,
        "staged_weaviate_records": [],
        "staged_weaviate_delete_old": False,
        "staged_kg_chunks": [],
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# Test 1: storage nodes stage only — no direct DB writes
# ---------------------------------------------------------------------------


class TestStorageNodesStageOnly:
    """Storage nodes MUST NOT call DB sinks; they only populate staged_* keys."""

    def test_storage_nodes_stage_only_no_db_writes(self, monkeypatch):
        """Run all 3 storage nodes; assert no live DB calls, staged_* populated."""
        from src.ingest.embedding.nodes.document_storage_node import document_storage_node
        from src.ingest.embedding.nodes.embedding_storage import embedding_storage_node
        from src.ingest.embedding.nodes.knowledge_graph_storage import (
            knowledge_graph_storage_node,
        )

        kg_builder = MagicMock()
        runtime = _make_runtime(
            store_documents=True,
            enable_kg=True,
            kg_builder=kg_builder,
        )
        state = _make_base_state(runtime)

        # Patch direct-write helpers at their import sites.
        with (
            patch("src.ingest.embedding.nodes.document_storage_node.put_document") as mock_put,
            patch("src.ingest.embedding.nodes.embedding_storage.add_documents") as mock_add,
        ):
            # Run document storage node
            ds_result = document_storage_node(state)
            state = {**state, **ds_result}

            # Stub embedder to return fake vectors for all chunks
            runtime.embedder.embed_documents.return_value = [[0.1, 0.2]]

            # Run embedding storage node
            es_result = embedding_storage_node(state)
            state = {**state, **es_result}

            # Run KG storage node
            kg_result = knowledge_graph_storage_node(state)
            state = {**state, **kg_result}

        # --- Assertions ---
        # 1) No live writes happened in any storage node
        mock_put.assert_not_called(), "put_document must NOT be called during document_storage_node"
        mock_add.assert_not_called(), "add_documents must NOT be called during embedding_storage_node"
        kg_builder.add_chunk.assert_not_called(), "kg_builder.add_chunk must NOT be called during kg_storage_node"

        # 2) staged_* keys are populated in state
        assert state.get("staged_minio") is not None, "staged_minio must be populated"
        assert isinstance(state.get("staged_minio"), dict)
        assert state["staged_minio"]["document_id"]  # non-empty
        assert state["staged_minio"]["content"]

        assert state.get("staged_weaviate_records"), "staged_weaviate_records must be populated"
        assert len(state["staged_weaviate_records"]) > 0

        assert isinstance(state.get("staged_kg_chunks"), list)
        assert len(state["staged_kg_chunks"]) > 0
        assert state["staged_kg_chunks"][0][0] == "hello world"  # (text, source_name)


# ---------------------------------------------------------------------------
# Test 2: commit_node flushes staged writes
# ---------------------------------------------------------------------------


class TestCommitNodeFlushes:
    """commit_node writes all three sinks exactly once with the staged payloads."""

    def _make_staged_state(self):
        from src.vector_db import DocumentRecord

        runtime = _make_runtime(store_documents=True)
        doc_id = "doc-uuid-1234"
        record = DocumentRecord(
            text="chunk text",
            embedding=[0.1, 0.2],
            metadata={"source": "test.md", "source_key": "k1", "chunk_index": 0},
        )
        state = _make_base_state(
            runtime,
            document_id=doc_id,
            staged_minio={
                "document_id": doc_id,
                "content": "# Hello",
                "metadata": {"source_key": "k1"},
                "bucket": "test-bucket",
            },
            staged_weaviate_records=[record],
            staged_weaviate_delete_old=False,
            staged_kg_chunks=[("chunk text", "test.md")],
        )
        return state

    def test_commit_node_flushes_staged_writes(self, monkeypatch):
        """All three sinks written exactly once; stored_count reflects Weaviate result."""
        from src.ingest.embedding.nodes.commit_node import commit_node

        state = self._make_staged_state()

        with (
            patch("src.ingest.embedding.nodes.commit_node.put_document") as mock_put,
            patch("src.ingest.embedding.nodes.commit_node.add_documents", return_value=1) as mock_add,
        ):
            kg_builder = MagicMock()
            state["runtime"].kg_builder = kg_builder
            state["runtime"].config.enable_knowledge_graph_storage = True

            result = commit_node(state)

        mock_put.assert_called_once()
        mock_add.assert_called_once()
        kg_builder.add_chunk.assert_called_once_with("chunk text", source="test.md")

        assert result["stored_count"] == 1
        assert not result.get("errors"), f"Unexpected errors: {result.get('errors')}"
        assert any("commit:ok" in entry for entry in result.get("processing_log", []))


# ---------------------------------------------------------------------------
# Test 3: rollback on Weaviate failure
# ---------------------------------------------------------------------------


class TestCommitNodeRollbackOnWeaviateFailure:
    """Weaviate failure triggers rollback of Weaviate + MinIO."""

    def test_commit_node_rolls_back_on_weaviate_failure(self, monkeypatch):
        from src.vector_db import DocumentRecord
        from src.ingest.embedding.nodes.commit_node import commit_node

        doc_id = "doc-uuid-wv-fail"
        runtime = _make_runtime(store_documents=True)
        record = DocumentRecord(
            text="chunk",
            embedding=[0.1],
            metadata={"source": "t.md", "source_key": "k1", "chunk_index": 0},
        )
        state = _make_base_state(
            runtime,
            document_id=doc_id,
            staging_batch_id="batch-rollback-001",
            staged_minio={
                "document_id": doc_id,
                "content": "# text",
                "metadata": {"source_key": "k1"},
                "bucket": "test-bucket",
            },
            staged_weaviate_records=[record],
            staged_kg_chunks=[],
        )

        with (
            patch("src.ingest.embedding.nodes.commit_node.put_document") as mock_put,
            patch(
                "src.ingest.embedding.nodes.commit_node.add_documents",
                side_effect=RuntimeError("weaviate down"),
            ),
            patch(
                "src.ingest.embedding.nodes.commit_node.delete_documents_by_staging_batch"
            ) as mock_del_wv,
            patch(
                "src.ingest.embedding.nodes.commit_node.delete_document"
            ) as mock_del_minio,
        ):
            result = commit_node(state)

        # MinIO was committed before Weaviate failure
        mock_put.assert_called_once()
        # Rollback: both stores cleaned up
        mock_del_wv.assert_called_once()
        wv_call_kwargs = mock_del_wv.call_args
        assert "batch-rollback-001" in str(wv_call_kwargs)

        mock_del_minio.assert_called_once()
        minio_call_kwargs = mock_del_minio.call_args
        assert doc_id in str(minio_call_kwargs)

        assert result.get("stored_count", -1) == 0
        errors = result.get("errors", [])
        assert any(e.startswith("commit:") for e in errors), f"No commit: error in {errors}"


# ---------------------------------------------------------------------------
# Test 4: rollback on MinIO failure
# ---------------------------------------------------------------------------


class TestCommitNodeRollbackOnMinIOFailure:
    """MinIO failure: nothing committed yet, so no Weaviate rollback needed."""

    def test_commit_node_rolls_back_on_minio_failure(self, monkeypatch):
        from src.vector_db import DocumentRecord
        from src.ingest.embedding.nodes.commit_node import commit_node

        doc_id = "doc-uuid-minio-fail"
        runtime = _make_runtime(store_documents=True)
        record = DocumentRecord(
            text="chunk",
            embedding=[0.1],
            metadata={"source": "t.md", "source_key": "k1", "chunk_index": 0},
        )
        state = _make_base_state(
            runtime,
            document_id=doc_id,
            staging_batch_id="batch-minio-fail",
            staged_minio={
                "document_id": doc_id,
                "content": "# text",
                "metadata": {"source_key": "k1"},
                "bucket": "test-bucket",
            },
            staged_weaviate_records=[record],
            staged_kg_chunks=[],
        )

        with (
            patch(
                "src.ingest.embedding.nodes.commit_node.put_document",
                side_effect=RuntimeError("minio connection failed"),
            ),
            patch(
                "src.ingest.embedding.nodes.commit_node.add_documents"
            ) as mock_add,
            patch(
                "src.ingest.embedding.nodes.commit_node.delete_documents_by_staging_batch"
            ) as mock_del_wv,
            patch(
                "src.ingest.embedding.nodes.commit_node.delete_document"
            ) as mock_del_minio,
        ):
            result = commit_node(state)

        # MinIO failed → Weaviate was never written → no Weaviate rollback
        mock_add.assert_not_called()
        mock_del_wv.assert_not_called()
        # MinIO also didn't succeed → no MinIO rollback
        mock_del_minio.assert_not_called()

        errors = result.get("errors", [])
        assert any(e.startswith("commit:") for e in errors), f"No commit: error in {errors}"


# ---------------------------------------------------------------------------
# Test 5: commit skipped when upstream errors
# ---------------------------------------------------------------------------


class TestCommitNodeSkipsOnUpstreamErrors:
    """commit_node must be a no-op when upstream state already has errors."""

    def test_commit_node_skipped_when_upstream_errors(self, monkeypatch):
        from src.ingest.embedding.nodes.commit_node import commit_node

        runtime = _make_runtime()
        state = _make_base_state(
            runtime,
            errors=["embedding_storage:some error"],
            staged_minio={"document_id": "x", "content": "y", "metadata": {}, "bucket": "b"},
            staged_weaviate_records=[MagicMock()],
        )

        with (
            patch("src.ingest.embedding.nodes.commit_node.put_document") as mock_put,
            patch("src.ingest.embedding.nodes.commit_node.add_documents") as mock_add,
            patch(
                "src.ingest.embedding.nodes.commit_node.delete_documents_by_staging_batch"
            ) as mock_del_wv,
            patch(
                "src.ingest.embedding.nodes.commit_node.delete_document"
            ) as mock_del_minio,
        ):
            result = commit_node(state)

        mock_put.assert_not_called()
        mock_add.assert_not_called()
        mock_del_wv.assert_not_called()
        mock_del_minio.assert_not_called()

        log = result.get("processing_log", [])
        assert any("commit:skipped:upstream_errors" in entry for entry in log), (
            f"Expected 'commit:skipped:upstream_errors' in log, got: {log}"
        )


# ---------------------------------------------------------------------------
# Test 6: no staged_minio when store_documents=False
# ---------------------------------------------------------------------------


class TestCommitNodeNoMinIOWhenStoreDisabled:
    """commit_node skips MinIO put when staged_minio is None; still flushes Weaviate."""

    def test_commit_node_handles_no_minio_when_store_documents_disabled(self, monkeypatch):
        from src.vector_db import DocumentRecord
        from src.ingest.embedding.nodes.commit_node import commit_node

        runtime = _make_runtime(store_documents=False)
        record = DocumentRecord(
            text="chunk",
            embedding=[0.1],
            metadata={"source": "t.md", "source_key": "k1", "chunk_index": 0},
        )
        state = _make_base_state(
            runtime,
            staged_minio=None,  # store_documents=False → no MinIO staged
            staged_weaviate_records=[record],
            staged_kg_chunks=[],
        )

        with (
            patch("src.ingest.embedding.nodes.commit_node.put_document") as mock_put,
            patch(
                "src.ingest.embedding.nodes.commit_node.add_documents", return_value=1
            ) as mock_add,
        ):
            result = commit_node(state)

        mock_put.assert_not_called()
        mock_add.assert_called_once()
        assert result.get("stored_count") == 1
        assert not result.get("errors")
