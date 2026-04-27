"""Idempotent re-ingest: Phase 1 short-circuits when source bytes are unchanged.

The activity computes ``source_hash`` from the raw source bytes early, queries
Weaviate via ``get_source_hash(source_key)``, and skips the rest of the
pipeline when the prior hash matches. Workflow then skips Phase 2.
"""
from __future__ import annotations

import asyncio
import hashlib
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest


def _import_activities():
    import src.ingest.temporal.activities as acts
    return acts


def _import_workflows():
    import src.ingest.temporal.workflows as wf_mod
    return wf_mod


def _make_activity_args(acts, source_path: str, source_key: str = "k1"):
    import dataclasses
    from src.ingest.common import IngestionConfig
    config = IngestionConfig()
    return acts.ActivityArgs(
        source=acts.SourceArgs(
            source_path=source_path,
            source_name="x.md",
            source_uri=f"file://{source_path}",
            source_key=source_key,
            source_id="i",
            connector="local_fs",
            source_version="0",
        ),
        config=dataclasses.asdict(config),
        staging_batch_id="batch-xyz",
    )


def _seed_file(tmp_path, body: bytes = b"hello world") -> str:
    p = tmp_path / "x.md"
    p.write_bytes(body)
    return str(p)


# ---------------------------------------------------------------------------
# Phase-1 activity: short-circuit on hash match
# ---------------------------------------------------------------------------

class TestPhase1Idempotency:
    def test_skipped_when_prior_hash_matches(self, monkeypatch, tmp_path):
        acts = _import_activities()
        path = _seed_file(tmp_path, b"unchanged")
        expected_hash = hashlib.sha256(b"unchanged").hexdigest()

        run_calls = []

        @contextmanager
        def fake_get_client():
            yield MagicMock()

        monkeypatch.setattr(acts.vector_db, "get_client", fake_get_client)
        monkeypatch.setattr(
            "src.ingest.temporal.activities.get_source_hash",
            lambda client, source_key: expected_hash,
        )
        monkeypatch.setattr(
            "src.ingest.temporal.activities.run_document_processing",
            lambda **kw: run_calls.append(kw) or {"errors": [], "source_hash": "", "processing_log": []},
        )

        result = asyncio.run(acts.document_processing_activity(_make_activity_args(acts, path)))

        assert result.skipped is True
        assert result.source_hash == expected_hash
        assert result.errors == []
        assert run_calls == []  # heavy parser never invoked
        assert any("skipped" in entry for entry in result.processing_log)

    def test_runs_full_pipeline_when_hash_differs(self, monkeypatch, tmp_path):
        acts = _import_activities()
        path = _seed_file(tmp_path, b"new content")
        run_calls = []

        @contextmanager
        def fake_get_client():
            yield MagicMock()

        monkeypatch.setattr(acts.vector_db, "get_client", fake_get_client)
        monkeypatch.setattr(
            "src.ingest.temporal.activities.get_source_hash",
            lambda client, source_key: "different-hash",
        )

        def fake_run(**kw):
            run_calls.append(kw)
            return {"errors": [], "source_hash": "", "cleaned_text": "x", "processing_log": []}

        monkeypatch.setattr(
            "src.ingest.temporal.activities.run_document_processing", fake_run
        )

        result = asyncio.run(acts.document_processing_activity(_make_activity_args(acts, path)))

        assert result.skipped is False
        assert len(run_calls) == 1

    def test_runs_full_pipeline_when_no_prior_hash(self, monkeypatch, tmp_path):
        acts = _import_activities()
        path = _seed_file(tmp_path, b"first ingest")
        run_calls = []

        @contextmanager
        def fake_get_client():
            yield MagicMock()

        monkeypatch.setattr(acts.vector_db, "get_client", fake_get_client)
        monkeypatch.setattr(
            "src.ingest.temporal.activities.get_source_hash",
            lambda client, source_key: None,
        )

        def fake_run(**kw):
            run_calls.append(kw)
            return {"errors": [], "source_hash": "", "cleaned_text": "x", "processing_log": []}

        monkeypatch.setattr("src.ingest.temporal.activities.run_document_processing", fake_run)

        result = asyncio.run(acts.document_processing_activity(_make_activity_args(acts, path)))
        assert result.skipped is False
        assert len(run_calls) == 1

    def test_runs_full_pipeline_when_source_path_unreadable(self, monkeypatch, tmp_path):
        """If we can't read raw bytes (e.g. URL/connector), don't short-circuit — fall through."""
        acts = _import_activities()
        run_calls = []

        @contextmanager
        def fake_get_client():
            yield MagicMock()

        monkeypatch.setattr(acts.vector_db, "get_client", fake_get_client)

        def fake_run(**kw):
            run_calls.append(kw)
            return {"errors": [], "source_hash": "", "cleaned_text": "x", "processing_log": []}

        monkeypatch.setattr("src.ingest.temporal.activities.run_document_processing", fake_run)

        # /nonexistent path → bytes read fails; short-circuit must not crash.
        result = asyncio.run(
            acts.document_processing_activity(_make_activity_args(acts, "/nonexistent/path.md"))
        )
        assert result.skipped is False
        assert len(run_calls) == 1


# ---------------------------------------------------------------------------
# Workflow: skip Phase 2 when Phase 1 skipped
# ---------------------------------------------------------------------------

class TestWorkflowSkipsPhase2:
    def test_phase2_not_executed_when_phase1_skipped(self):
        wf_mod = _import_workflows()
        from src.ingest.temporal.activities import (
            DocProcessingResult, EmbeddingResult, SourceArgs,
        )

        # Sandbox attrs (mirror tests/ingest/test_temporal_workflows.py helpers)
        import logging
        import uuid
        wf = wf_mod.workflow
        if not hasattr(wf, "logger"):
            wf.logger = logging.getLogger("test")
        if not hasattr(wf, "uuid4"):
            wf.uuid4 = lambda: uuid.uuid4()
        if not hasattr(wf, "execute_activity"):
            async def _noop(*a, **kw):
                raise NotImplementedError
            wf.execute_activity = _noop

        source = SourceArgs(
            source_path="/x", source_name="x", source_uri="file:///x",
            source_key="k", source_id="i", connector="c", source_version="v",
        )
        args = wf_mod.IngestDocumentArgs(source=source, config={})

        seen = []

        async def fake_exec(activity_fn, activity_args, **kw):
            seen.append(activity_fn.__name__ if hasattr(activity_fn, "__name__") else "phase")
            return DocProcessingResult(
                errors=[], source_hash="h", clean_hash="",
                processing_log=["skipped:unchanged"], skipped=True,
            )

        original = wf.execute_activity
        wf.execute_activity = fake_exec
        try:
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(
                    wf_mod.IngestDocumentWorkflow().run(args)
                )
            finally:
                loop.close()
        finally:
            wf.execute_activity = original

        assert len(seen) == 1, "Phase 2 must not be invoked when Phase 1 is skipped"
        assert result.stored_count == 0
        assert "skipped:unchanged" in result.processing_log
