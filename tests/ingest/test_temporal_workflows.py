"""Tests for src/ingest/temporal/workflows.py.

Uses Temporal's built-in WorkflowEnvironment for testing workflow logic.
Falls back to direct mock patching of the workflow module's activity references.

Covers:
- IngestDocumentWorkflow.run(): Phase 1 → Phase 2 success, Phase 1 error, Phase 2 error
- IngestDirectoryWorkflow.run(): fan-out aggregation, child exceptions, dict conversion
"""

from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch, create_autospec

import pytest


# ---------------------------------------------------------------------------
# Lazy import helpers
# ---------------------------------------------------------------------------

def _import_workflows():
    import src.ingest.temporal.workflows as wf_mod
    return wf_mod


def _import_activity_types():
    from src.ingest.temporal.activities import (
        DocProcessingResult, EmbeddingResult, SourceArgs,
    )
    return DocProcessingResult, EmbeddingResult, SourceArgs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_source_args(key="test-key"):
    _, _, SourceArgs = _import_activity_types()
    return SourceArgs(
        source_path="/tmp/test.pdf", source_name="test.pdf",
        source_uri="file:///tmp/test.pdf", source_key=key,
        source_id="1234:5678", connector="local_fs", source_version="0",
    )


def _make_phase1(errors=None, log=None):
    DocProcessingResult, _, _ = _import_activity_types()
    return DocProcessingResult(
        errors=errors or [], source_hash="abc123", clean_hash="def456",
        processing_log=log or ["doc_processing:ok"],
    )


def _make_phase2(errors=None, stored=3, log=None):
    _, EmbeddingResult, _ = _import_activity_types()
    return EmbeddingResult(
        errors=errors or [], stored_count=stored,
        metadata_summary="summary", metadata_keywords=["kw"],
        processing_log=log or ["embedding:ok"],
    )


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _ensure_workflow_sandbox_attrs(wf_mod):
    """Ensure sandbox-only attributes exist on workflow module for testing.

    With real temporalio installed, ``workflow.logger`` exists but raises
    ``_NotInWorkflowEventLoopError`` when called outside a workflow runtime.
    These tests run workflow code directly (no Temporal harness), so we
    unconditionally replace it with a stdlib logger.
    """
    wf = wf_mod.workflow
    import logging
    wf.logger = logging.getLogger("temporalio.workflow.test")
    if not hasattr(wf, 'execute_activity'):
        async def _noop(*a, **kw):
            raise NotImplementedError("execute_activity not patched")
        wf.execute_activity = _noop
    if not hasattr(wf, 'start_child_workflow'):
        async def _noop2(*a, **kw):
            raise NotImplementedError("start_child_workflow not patched")
        wf.start_child_workflow = _noop2
    if not hasattr(wf, 'info'):
        wf.info = lambda: MagicMock(task_queue="test")


def _patch_execute_activity(wf_mod, side_effect):
    """Monkey-patch workflow.execute_activity for the duration of a with-block."""

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


def _patch_start_child(wf_mod, side_effect):
    """Same monkey-patch approach for start_child_workflow."""

    class _Patcher:
        def __enter__(self):
            _ensure_workflow_sandbox_attrs(wf_mod)
            self._orig_start = wf_mod.workflow.start_child_workflow
            self._orig_info = wf_mod.workflow.info
            wf_mod.workflow.start_child_workflow = side_effect
            mock_info = MagicMock()
            mock_info.task_queue = "test-queue"
            wf_mod.workflow.info = lambda: mock_info
            return self

        def __exit__(self, *exc):
            wf_mod.workflow.start_child_workflow = self._orig_start
            wf_mod.workflow.info = self._orig_info
            return False

    return _Patcher()


# ---------------------------------------------------------------------------
# Tests: IngestDocumentWorkflow
# ---------------------------------------------------------------------------

class TestIngestDocumentWorkflow:

    def test_mock_ingest_doc_wf_success(self):
        """Phase 1 ok → Phase 2 ok → merged logs and stored_count."""
        wf_mod = _import_workflows()
        phase1 = _make_phase1()
        phase2 = _make_phase2(stored=5)
        args = wf_mod.IngestDocumentArgs(source=_make_source_args(), config={})
        calls = {"n": 0}

        async def fake_exec(activity_fn, activity_args, **kw):
            calls["n"] += 1
            return phase1 if calls["n"] == 1 else phase2

        with _patch_execute_activity(wf_mod, fake_exec):
            result = _run(wf_mod.IngestDocumentWorkflow().run(args))

        assert result.source_key == "test-key"
        assert result.errors == []
        assert result.stored_count == 5
        assert "doc_processing:ok" in result.processing_log
        assert "embedding:ok" in result.processing_log

    def test_mock_ingest_doc_wf_phase1_error(self):
        """Phase 1 errors → early return, Phase 2 never called."""
        wf_mod = _import_workflows()
        phase1 = _make_phase1(errors=["parse failed"], log=["doc:error"])
        args = wf_mod.IngestDocumentArgs(source=_make_source_args(), config={})
        exec_calls = []

        async def fake_exec(activity_fn, activity_args, **kw):
            exec_calls.append(1)
            return phase1

        with _patch_execute_activity(wf_mod, fake_exec):
            result = _run(wf_mod.IngestDocumentWorkflow().run(args))

        assert result.errors == ["parse failed"]
        assert result.stored_count == 0
        assert len(exec_calls) == 1  # Phase 2 never called

    def test_mock_ingest_doc_wf_phase2_errors(self):
        """Phase 2 errors are propagated in result."""
        wf_mod = _import_workflows()
        phase1 = _make_phase1()
        phase2 = _make_phase2(errors=["emb fail"], stored=0)
        args = wf_mod.IngestDocumentArgs(source=_make_source_args(), config={})
        calls = {"n": 0}

        async def fake_exec(activity_fn, activity_args, **kw):
            calls["n"] += 1
            return phase1 if calls["n"] == 1 else phase2

        with _patch_execute_activity(wf_mod, fake_exec):
            result = _run(wf_mod.IngestDocumentWorkflow().run(args))

        assert "emb fail" in result.errors
        assert result.stored_count == 0


# ---------------------------------------------------------------------------
# Tests: IngestDirectoryWorkflow
# ---------------------------------------------------------------------------

class TestIngestDirectoryWorkflow:

    def _run_dir(self, args, child_results):
        wf_mod = _import_workflows()
        idx = {"i": 0}

        async def fake_start_child(workflow_cls, child_args, **kw):
            i = idx["i"]
            idx["i"] += 1
            r = child_results[i]
            fut = asyncio.get_event_loop().create_future()
            if isinstance(r, Exception):
                fut.set_exception(r)
            else:
                fut.set_result(r)
            return fut

        with _patch_start_child(wf_mod, fake_start_child):
            return _run(wf_mod.IngestDirectoryWorkflow().run(args))

    def test_mock_ingest_dir_wf_all_success(self):
        wf_mod = _import_workflows()
        c1 = wf_mod.IngestDocumentResult(source_key="k1", errors=[], stored_count=3, processing_log=[])
        c2 = wf_mod.IngestDocumentResult(source_key="k2", errors=[], stored_count=5, processing_log=[])
        args = wf_mod.IngestDirectoryArgs(
            sources=[_make_source_args("k1"), _make_source_args("k2")], config={}
        )
        result = self._run_dir(args, [c1, c2])
        assert result.processed == 2
        assert result.failed == 0
        assert result.stored_chunks == 8

    def test_mock_ingest_dir_wf_child_exception(self):
        wf_mod = _import_workflows()
        c1 = wf_mod.IngestDocumentResult(source_key="k1", errors=[], stored_count=3, processing_log=[])
        args = wf_mod.IngestDirectoryArgs(
            sources=[_make_source_args("k1"), _make_source_args("k2")], config={}
        )
        result = self._run_dir(args, [c1, RuntimeError("boom")])
        assert result.processed == 1
        assert result.failed == 1

    def test_mock_ingest_dir_wf_child_with_errors(self):
        wf_mod = _import_workflows()
        c1 = wf_mod.IngestDocumentResult(source_key="k1", errors=["err"], stored_count=0, processing_log=[])
        c2 = wf_mod.IngestDocumentResult(source_key="k2", errors=[], stored_count=4, processing_log=[])
        args = wf_mod.IngestDirectoryArgs(
            sources=[_make_source_args("k1"), _make_source_args("k2")], config={}
        )
        result = self._run_dir(args, [c1, c2])
        assert result.processed == 1
        assert result.failed == 1
        assert "err" in result.errors

    def test_mock_ingest_dir_wf_empty(self):
        wf_mod = _import_workflows()
        args = wf_mod.IngestDirectoryArgs(sources=[], config={})
        result = self._run_dir(args, [])
        assert result.processed == 0
        assert result.failed == 0

    def test_mock_ingest_dir_wf_dict_sources(self):
        wf_mod = _import_workflows()
        c1 = wf_mod.IngestDocumentResult(source_key="k1", errors=[], stored_count=2, processing_log=[])
        sd = dict(
            source_path="/tmp/t.pdf", source_name="t.pdf",
            source_uri="file:///tmp/t.pdf", source_key="k1",
            source_id="1:2", connector="local_fs", source_version="0",
        )
        args = wf_mod.IngestDirectoryArgs(sources=[sd], config={})
        result = self._run_dir(args, [c1])
        assert result.processed == 1
        assert result.stored_chunks == 2
