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
    import uuid
    wf = wf_mod.workflow
    # Replace unconditionally — workflow.logger and workflow.uuid4 exist on the
    # real temporalio module but raise _NotInWorkflowEventLoopError when called
    # outside a Temporal workflow loop.
    wf.logger = logging.getLogger("temporalio.workflow.test")
    wf.uuid4 = uuid.uuid4
    if not hasattr(wf, "execute_activity"):
        async def _noop(*a, **kw):
            raise NotImplementedError
        wf.execute_activity = _noop


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
        """staging_batch_id and source_hash must be in staged_weaviate_records metadata.

        Since Issue #42, embedding_storage_node no longer calls add_documents; it
        stages DocumentRecords in state["staged_weaviate_records"]. The metadata
        fields are verified on the staged records directly.
        """
        from src.ingest.embedding.nodes import embedding_storage as es_mod
        from src.ingest.common import IngestionConfig, ProcessedChunk, Runtime

        # Stub embed step so we don't need a real embedder
        monkeypatch.setattr(
            es_mod, "_embed_batches",
            lambda embedder, batches: ([[0.1] * 4 for batch in batches for _ in batch], [], [True] * len(batches)),
        )

        runtime = Runtime(
            config=IngestionConfig(),
            embedder=MagicMock(),
            weaviate_client=MagicMock(),
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

        result = es_mod.embedding_storage_node(state)

        # add_documents must NOT be called in the staging node.
        staged = result.get("staged_weaviate_records", [])
        assert len(staged) == 1, "Expected exactly 1 staged record"
        meta = staged[0].metadata
        assert meta.get("staging_batch_id") == "batch-uuid-xyz"
        assert meta.get("source_hash") == "deadbeef"
