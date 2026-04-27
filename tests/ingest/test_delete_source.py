"""Tests for DeleteSourceWorkflow and delete_source_activity (Issue #42, PR-B).

Covers:
- Normal deletion of Weaviate chunks and MinIO document.
- Deletion using staging_batch_id filter instead of source_key.
- Idempotent success when stores return zero-match deletes.
- Best-effort: Weaviate failure does not block MinIO cleanup.
- Workflow calls activity with retry policy and schedule_to_close_timeout.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Lazy import helpers
# ---------------------------------------------------------------------------


def _import_activities():
    import src.ingest.temporal.activities as acts
    return acts


def _import_workflows():
    import src.ingest.temporal.workflows as wf_mod
    return wf_mod


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _ensure_workflow_sandbox_attrs(wf_mod):
    """Inject the sandbox-only attributes used inside workflow bodies.

    Replace workflow.logger unconditionally — it exists on the real temporalio
    module but raises _NotInWorkflowEventLoopError when accessed outside a
    Temporal workflow loop. A stdlib logger is a safe stand-in for tests.
    """
    import logging
    wf = wf_mod.workflow
    wf.logger = logging.getLogger("temporalio.workflow.test")
    if not hasattr(wf, "execute_activity"):
        async def _noop(*a, **kw):
            raise NotImplementedError("execute_activity not patched")
        wf.execute_activity = _noop


def _patch_execute_activity(wf_mod, side_effect):
    """Context manager: swap workflow.execute_activity for the duration of a test."""

    class _Patcher:
        def __enter__(self):
            _ensure_workflow_sandbox_attrs(wf_mod)
            self._original = wf_mod.workflow.execute_activity
            wf_mod.workflow.execute_activity = side_effect
            return self

        def __exit__(self, *exc):
            wf_mod.workflow.execute_activity = self._original
            return False

    return _Patcher()


# ---------------------------------------------------------------------------
# Test 1: Normal deletion — both stores succeed
# ---------------------------------------------------------------------------


class TestDeleteSourceActivityRemovesWeaviateAndMinio:
    def test_mock_removes_weaviate_and_minio(self, monkeypatch):
        """Activity returns weaviate_deleted=7, minio_deleted=True, errors=[]."""
        acts = _import_activities()

        fake_wv_client = MagicMock()

        @contextmanager
        def fake_get_client():
            yield fake_wv_client

        delete_by_source_key_calls = []

        def fake_delete_by_source_key(client, source_key, **kw):
            delete_by_source_key_calls.append({"client": client, "source_key": source_key})
            return 7

        delete_document_calls = []

        def fake_delete_document(client, document_id, **kw):
            delete_document_calls.append({"client": client, "document_id": document_id})
            return True

        monkeypatch.setattr("src.ingest.temporal.activities.vector_db.get_client", fake_get_client)
        monkeypatch.setattr(
            "src.ingest.temporal.activities.vector_db.delete_by_source_key",
            fake_delete_by_source_key,
        )
        monkeypatch.setattr("src.ingest.temporal.activities.db.delete_document", fake_delete_document)
        # _get_db_client() singleton — give it a fake client
        fake_db_client = MagicMock()
        monkeypatch.setattr("src.ingest.temporal.activities._db_client", fake_db_client)

        args = acts.DeleteSourceArgs(source_key="k1")
        result = _run(acts.delete_source_activity(args))

        assert result.weaviate_deleted == 7
        assert result.minio_deleted is True
        assert result.errors == []

        # Verify delete_by_source_key called with correct source_key
        assert len(delete_by_source_key_calls) == 1
        assert delete_by_source_key_calls[0]["source_key"] == "k1"

        # Verify delete_document called (document_id derived from source_key)
        assert len(delete_document_calls) == 1
        from src.db import build_document_id
        expected_doc_id = build_document_id("k1")
        assert delete_document_calls[0]["document_id"] == expected_doc_id


# ---------------------------------------------------------------------------
# Test 2: staging_batch_id triggers batch-filter path
# ---------------------------------------------------------------------------


class TestDeleteSourceActivityWithStagingBatchId:
    def test_mock_uses_batch_filter_when_staging_batch_id_set(self, monkeypatch):
        """When staging_batch_id is set, delete_documents_by_staging_batch is called."""
        acts = _import_activities()

        fake_wv_client = MagicMock()

        @contextmanager
        def fake_get_client():
            yield fake_wv_client

        staging_batch_calls = []

        def fake_delete_by_staging_batch(client, staging_batch_id):
            staging_batch_calls.append({"client": client, "batch_id": staging_batch_id})
            return 3

        source_key_calls = []

        def fake_delete_by_source_key(client, source_key, **kw):
            source_key_calls.append(source_key)
            return 99  # should never be called

        import src.vector_db.weaviate.store as wv_store
        monkeypatch.setattr("src.ingest.temporal.activities.vector_db.get_client", fake_get_client)
        monkeypatch.setattr(wv_store, "delete_documents_by_staging_batch", fake_delete_by_staging_batch)
        monkeypatch.setattr(
            "src.ingest.temporal.activities.vector_db.delete_by_source_key",
            fake_delete_by_source_key,
        )
        monkeypatch.setattr(
            "src.ingest.temporal.activities.db.delete_document",
            lambda client, document_id, **kw: True,
        )
        fake_db_client = MagicMock()
        monkeypatch.setattr("src.ingest.temporal.activities._db_client", fake_db_client)

        args = acts.DeleteSourceArgs(source_key="k1", staging_batch_id="batch-xyz")
        result = _run(acts.delete_source_activity(args))

        # Batch-filter path was used
        assert len(staging_batch_calls) == 1
        assert staging_batch_calls[0]["batch_id"] == "batch-xyz"

        # source_key path must NOT have been called
        assert source_key_calls == []

        assert result.weaviate_deleted == 3


# ---------------------------------------------------------------------------
# Test 3: Idempotent on already-deleted source
# ---------------------------------------------------------------------------


class TestDeleteSourceActivityIdempotentOnMissing:
    def test_mock_idempotent_when_nothing_deleted(self, monkeypatch):
        """Zero-match deletes are a success — weaviate_deleted=0, minio_deleted=False, errors=[]."""
        acts = _import_activities()

        @contextmanager
        def fake_get_client():
            yield MagicMock()

        monkeypatch.setattr("src.ingest.temporal.activities.vector_db.get_client", fake_get_client)
        monkeypatch.setattr(
            "src.ingest.temporal.activities.vector_db.delete_by_source_key",
            lambda client, source_key, **kw: 0,
        )
        monkeypatch.setattr(
            "src.ingest.temporal.activities.db.delete_document",
            lambda client, document_id, **kw: False,
        )
        fake_db_client = MagicMock()
        monkeypatch.setattr("src.ingest.temporal.activities._db_client", fake_db_client)

        args = acts.DeleteSourceArgs(source_key="k-gone")
        result = _run(acts.delete_source_activity(args))

        assert result.weaviate_deleted == 0
        assert result.minio_deleted is False
        assert result.errors == []


# ---------------------------------------------------------------------------
# Test 4: Weaviate failure does not block MinIO (best-effort)
# ---------------------------------------------------------------------------


class TestDeleteSourceActivityContinuesOnPartialFailure:
    def test_mock_weaviate_failure_does_not_block_minio(self, monkeypatch):
        """Weaviate exception recorded in errors; MinIO still cleaned up."""
        acts = _import_activities()

        @contextmanager
        def fake_get_client():
            yield MagicMock()

        def raise_weaviate(client, source_key, **kw):
            raise RuntimeError("connection reset")

        minio_calls = []

        def fake_delete_document(client, document_id, **kw):
            minio_calls.append(document_id)
            return True

        monkeypatch.setattr("src.ingest.temporal.activities.vector_db.get_client", fake_get_client)
        monkeypatch.setattr(
            "src.ingest.temporal.activities.vector_db.delete_by_source_key",
            raise_weaviate,
        )
        monkeypatch.setattr(
            "src.ingest.temporal.activities.db.delete_document",
            fake_delete_document,
        )
        fake_db_client = MagicMock()
        monkeypatch.setattr("src.ingest.temporal.activities._db_client", fake_db_client)

        args = acts.DeleteSourceArgs(source_key="k1")
        # Must NOT raise
        result = _run(acts.delete_source_activity(args))

        assert any("weaviate_delete_failed" in e for e in result.errors)
        assert result.minio_deleted is True
        assert len(minio_calls) == 1  # MinIO was still attempted


# ---------------------------------------------------------------------------
# Test 5: Workflow calls activity with retry policy + timeout
# ---------------------------------------------------------------------------


class TestDeleteSourceWorkflowCallsActivityWithRetry:
    def test_mock_workflow_calls_activity_with_retry_and_timeout(self):
        """Workflow passes retry_policy and schedule_to_close_timeout to execute_activity."""
        wf_mod = _import_workflows()
        acts = _import_activities()

        activity_calls = []

        async def fake_exec(activity_fn, activity_args, **kw):
            activity_calls.append({
                "fn": activity_fn,
                "args": activity_args,
                "kw": kw,
            })
            return acts.DeleteSourceResult(
                weaviate_deleted=5,
                minio_deleted=True,
                errors=[],
            )

        wf_args = acts.DeleteSourceArgs(source_key="k1", staging_batch_id="")

        with _patch_execute_activity(wf_mod, fake_exec):
            result = _run(wf_mod.DeleteSourceWorkflow().run(wf_args))

        assert len(activity_calls) == 1

        call = activity_calls[0]
        # Activity function must be delete_source_activity
        assert call["fn"] is acts.delete_source_activity

        # Args forwarded unchanged
        assert call["args"].source_key == "k1"

        # retry_policy must be present
        assert "retry_policy" in call["kw"], "retry_policy not passed to execute_activity"

        # schedule_to_close_timeout must be present
        assert "schedule_to_close_timeout" in call["kw"], (
            "schedule_to_close_timeout not passed to execute_activity"
        )

        # Workflow returns the activity result directly
        assert result.weaviate_deleted == 5
        assert result.minio_deleted is True
        assert result.errors == []
