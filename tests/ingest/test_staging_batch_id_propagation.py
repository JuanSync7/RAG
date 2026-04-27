"""Step 2: staging_batch_id propagation through the ingest pipeline.

Verifies that:
- ``IngestDocumentWorkflow`` mints a non-empty ``staging_batch_id`` and passes
  the SAME id to both Phase 1 and Phase 2 activities.
- ``embedding_storage_node`` injects ``state["staging_batch_id"]`` into every
  chunk's metadata so it lands as a property in ``add_documents``.
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest


def _import_workflows():
    import src.ingest.temporal.workflows as wf_mod
    return wf_mod


def _import_activity_types():
    from src.ingest.temporal.activities import (
        ActivityArgs, DocProcessingResult, EmbeddingResult, SourceArgs,
    )
    return ActivityArgs, DocProcessingResult, EmbeddingResult, SourceArgs


def _ensure_sandbox(wf_mod):
    import logging
    wf = wf_mod.workflow
    if not hasattr(wf, "logger"):
        wf.logger = logging.getLogger("temporalio.workflow.test")
    if not hasattr(wf, "execute_activity"):
        async def _noop(*a, **kw):
            raise NotImplementedError
        wf.execute_activity = _noop
    if not hasattr(wf, "uuid4"):
        import uuid
        wf.uuid4 = lambda: uuid.uuid4()


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Workflow-level: same staging_batch_id reaches both phases
# ---------------------------------------------------------------------------

class TestWorkflowMintsStagingBatchId:
    def test_same_staging_batch_id_passed_to_both_phases(self):
        wf_mod = _import_workflows()
        _ensure_sandbox(wf_mod)
        ActivityArgs, DocProcessingResult, EmbeddingResult, SourceArgs = _import_activity_types()

        source = SourceArgs(
            source_path="/x", source_name="x", source_uri="file:///x",
            source_key="k", source_id="i", connector="c", source_version="v",
        )
        args = wf_mod.IngestDocumentArgs(source=source, config={})

        seen_ids: list[str] = []

        async def fake_exec(activity_fn, activity_args, **kw):
            seen_ids.append(getattr(activity_args, "staging_batch_id", "<missing>"))
            if len(seen_ids) == 1:
                return DocProcessingResult(
                    errors=[], source_hash="h", clean_hash="c", processing_log=[],
                )
            return EmbeddingResult(
                errors=[], stored_count=1, metadata_summary="",
                metadata_keywords=[], processing_log=[],
            )

        original = wf_mod.workflow.execute_activity
        wf_mod.workflow.execute_activity = fake_exec
        try:
            _run(wf_mod.IngestDocumentWorkflow().run(args))
        finally:
            wf_mod.workflow.execute_activity = original

        assert len(seen_ids) == 2
        assert seen_ids[0] != "<missing>"
        assert seen_ids[0] != ""
        assert seen_ids[0] == seen_ids[1]
        # Looks like a UUID
        assert len(seen_ids[0]) >= 32


# ---------------------------------------------------------------------------
# Node-level: staging_batch_id and source_hash land on chunk metadata
# ---------------------------------------------------------------------------

class TestEmbeddingStorageNodeInjectsAtomicityFields:
    def test_chunk_metadata_carries_staging_batch_id_and_source_hash(self, monkeypatch):
        from src.ingest.embedding.nodes import embedding_storage as es_mod
        from src.ingest.common import IngestionConfig, ProcessedChunk, Runtime

        captured: list[dict] = []

        def fake_add_documents(client, records, collection=None):
            for r in records:
                captured.append(r.metadata)
            return len(records)

        monkeypatch.setattr(es_mod, "add_documents", fake_add_documents)
        # Stub embed step so we don't need a real embedder
        monkeypatch.setattr(
            es_mod, "_embed_batches",
            lambda embedder, batches: ([[0.1] * 4 for batch in batches for _ in batch], [], [True] * len(batches)),
        )

        runtime = Runtime(
            config=IngestionConfig(),
            embedder=MagicMock(),
            weaviate_client=MagicMock(),
            kg_builder=None,
            db_client=None,
        )

        chunk = ProcessedChunk(
            text="hello world",
            metadata={"chunk_index": 0, "source_key": "k", "source": "doc.md"},
        )
        state = {
            "runtime": runtime,
            "source_key": "k",
            "source_name": "doc.md",
            "chunks": [chunk],
            "staging_batch_id": "batch-uuid-xyz",
            "source_hash": "deadbeef",
        }

        es_mod.embedding_storage_node(state)

        assert len(captured) == 1
        meta = captured[0]
        assert meta.get("staging_batch_id") == "batch-uuid-xyz"
        assert meta.get("source_hash") == "deadbeef"
