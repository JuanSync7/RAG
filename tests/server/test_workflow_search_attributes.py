"""Tests for RAGQueryWorkflow Temporal search-attribute upsert.

Two layers of coverage:

1. Unit-level — exercise the pure helper ``_build_dr_search_attributes`` with
   crafted request/result dicts. Locks the edge-case mapping (DR off, DR on
   with metadata, DR on with missing metadata).

2. Workflow-level — drive ``RAGQueryWorkflow.run`` directly via the same
   monkey-patch pattern used in ``tests/ingest/test_temporal_workflows.py``
   (no Temporal harness needed). Asserts ``workflow.upsert_search_attributes``
   was called with the expected dict.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from server.search_attributes import (
    DR_EARLY_STOPPED,
    DR_ENABLED,
    DR_ITERATIONS,
    DR_LLM_CALLS,
    DR_TOPIC_COUNT,
)
from server.workflows import RAGQueryWorkflow, _build_dr_search_attributes


# ---------------------------------------------------------------------------
# Helpers (mirrors tests/ingest/test_temporal_workflows.py pattern)
# ---------------------------------------------------------------------------


def _ensure_workflow_sandbox_attrs():
    """Replace sandbox-only workflow symbols with stdlib equivalents so we
    can drive ``RAGQueryWorkflow.run`` outside a Temporal worker."""
    import logging
    import server.workflows as wf_mod
    wf = wf_mod.workflow
    wf.logger = logging.getLogger("temporalio.workflow.test")
    if not hasattr(wf, "execute_activity"):
        async def _noop(*a, **kw):  # pragma: no cover
            raise NotImplementedError("execute_activity not patched")
        wf.execute_activity = _noop
    if not hasattr(wf, "upsert_search_attributes"):
        wf.upsert_search_attributes = lambda *a, **kw: None


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Unit tests for _build_dr_search_attributes
# ---------------------------------------------------------------------------


class TestBuildDrSearchAttributes:
    def test_dr_disabled_only_emits_dr_enabled_false(self):
        attrs = _build_dr_search_attributes({"deep_research": False}, {"any": "result"})
        assert attrs == {DR_ENABLED: False}

    def test_dr_enabled_full_metadata(self):
        result = {
            "metadata": {
                "deep_research": {
                    "iteration_count": 4,
                    "llm_call_count": 9,
                    "topic_count": 3,
                    "decomposed": True,
                    "is_unified": False,
                }
            }
        }
        attrs = _build_dr_search_attributes({"deep_research": True}, result)
        assert attrs == {
            DR_ENABLED: True,
            DR_ITERATIONS: 4,
            DR_LLM_CALLS: 9,
            DR_EARLY_STOPPED: False,  # decomposed=True → not early-stopped
            DR_TOPIC_COUNT: 3,
        }

    def test_dr_enabled_early_stop_no_decomposition(self):
        result = {
            "metadata": {
                "deep_research": {
                    "iteration_count": 1,
                    "llm_call_count": 1,
                    "topic_count": 1,
                    "decomposed": False,
                }
            }
        }
        attrs = _build_dr_search_attributes({"deep_research": True}, result)
        assert attrs[DR_EARLY_STOPPED] is True
        assert attrs[DR_TOPIC_COUNT] == 1
        assert attrs[DR_LLM_CALLS] == 1

    def test_dr_enabled_missing_metadata_returns_zero_baseline(self):
        # Sanitizer reject / failure: DR was requested but the chain bailed
        # before the orchestrator produced any metadata.
        attrs = _build_dr_search_attributes({"deep_research": True}, {"results": []})
        assert attrs == {
            DR_ENABLED: True,
            DR_ITERATIONS: 0,
            DR_LLM_CALLS: 0,
            DR_EARLY_STOPPED: True,
            DR_TOPIC_COUNT: 0,
        }

    def test_dr_enabled_metadata_present_but_no_dr_key(self):
        attrs = _build_dr_search_attributes(
            {"deep_research": True}, {"metadata": {"other": "x"}}
        )
        assert attrs[DR_ENABLED] is True
        assert attrs[DR_EARLY_STOPPED] is True
        assert attrs[DR_ITERATIONS] == 0


# ---------------------------------------------------------------------------
# Workflow-level tests
# ---------------------------------------------------------------------------


def _drive_workflow(request: dict, fake_result: dict):
    """Run RAGQueryWorkflow.run with execute_activity + upsert mocked."""
    import server.workflows as wf_mod

    _ensure_workflow_sandbox_attrs()

    captured: dict = {}

    async def _fake_execute_activity(*args, **kwargs):
        return fake_result

    def _fake_upsert(attrs):
        captured["attrs"] = attrs

    orig_exec = wf_mod.workflow.execute_activity
    orig_upsert = wf_mod.workflow.upsert_search_attributes
    wf_mod.workflow.execute_activity = _fake_execute_activity
    wf_mod.workflow.upsert_search_attributes = _fake_upsert
    try:
        wf = RAGQueryWorkflow()
        result = _run(wf.run(request))
    finally:
        wf_mod.workflow.execute_activity = orig_exec
        wf_mod.workflow.upsert_search_attributes = orig_upsert

    return result, captured.get("attrs")


# The in-workflow upsert is currently DISABLED (server/workflows.py: the
# deprecated dict-based ``workflow.upsert_search_attributes`` call is
# commented out pending migration to the typed SearchAttributePair API), so
# the workflow-level assertions cannot observe a call. Re-enable these
# together with that migration; the pure ``_build_dr_search_attributes``
# mapping stays covered by the unit tests above.
_UPSERT_DISABLED_REASON = (
    "workflow.upsert_search_attributes is disabled in RAGQueryWorkflow "
    "(deprecated dict API; see the DISABLED block in server/workflows.py) — "
    "re-enable with the typed SearchAttributePair migration"
)


@pytest.mark.skip(reason=_UPSERT_DISABLED_REASON)
def test_workflow_dr_enabled_multi_topic_upserts_full_attrs():
    fake_result = {
        "results": [],
        "metadata": {
            "deep_research": {
                "iteration_count": 5,
                "llm_call_count": 11,
                "topic_count": 3,
                "decomposed": True,
                "is_unified": False,
            }
        },
    }
    result, attrs = _drive_workflow({"deep_research": True, "query": "q"}, fake_result)
    assert result is fake_result  # workflow returns the activity's dict unchanged
    assert attrs[DR_ENABLED] is True
    assert attrs[DR_ITERATIONS] == 5
    assert attrs[DR_LLM_CALLS] == 11
    assert attrs[DR_TOPIC_COUNT] == 3
    assert attrs[DR_EARLY_STOPPED] is False


@pytest.mark.skip(reason=_UPSERT_DISABLED_REASON)
def test_workflow_dr_enabled_early_stop_marks_early_stopped():
    fake_result = {
        "metadata": {
            "deep_research": {
                "iteration_count": 1,
                "llm_call_count": 1,
                "topic_count": 1,
                "decomposed": False,
            }
        },
    }
    _, attrs = _drive_workflow({"deep_research": True, "query": "q"}, fake_result)
    assert attrs[DR_EARLY_STOPPED] is True
    assert attrs[DR_TOPIC_COUNT] == 1


@pytest.mark.skip(reason=_UPSERT_DISABLED_REASON)
def test_workflow_dr_disabled_only_sets_dr_enabled_false():
    fake_result = {"results": []}
    _, attrs = _drive_workflow({"deep_research": False, "query": "q"}, fake_result)
    assert attrs == {DR_ENABLED: False}
