# @summary
# Offline route tests for server.routes.system: health + root endpoints and the
# build_health_response status matrix / _queue_has_poller poller detection.
# Drives build_health_response directly via asyncio.run with hand-built fake
# Temporal clients so each of temporal_ok/worker_ok/ingest_worker_ok is
# independently controllable.
# Deps: pytest, fastapi.testclient
# @end-summary
"""Tests for the system (health/root) API routes.

The real config has ``trigger_to_queue(TRIGGER_SINGLE) == 'rag-reliability'``
which DIFFERS from ``RAG_QUERY_TASK_QUEUE == 'rag-query'``, so the live ingest
branch is the *separate* ``_queue_has_poller(ingest_queue)`` describe call (the
``ingest_worker_ok = worker_ok`` shortcut is dead in the default config). These
tests exercise the live separate-describe path; one test additionally documents
the shortcut by monkeypatching the queues equal.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Optional

from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.routes import system as system_mod
from server.routes.system import build_health_response, create_system_router
from server.workflows import RAG_QUERY_TASK_QUEUE
from src.ingest.temporal.constants import TRIGGER_SINGLE, trigger_to_queue

LOGGER = logging.getLogger("test.system")

INGEST_QUEUE = trigger_to_queue(TRIGGER_SINGLE)


# ---------------------------------------------------------------------------
# Fake Temporal client building blocks
# ---------------------------------------------------------------------------


class _FakeWorkflowService:
    """Models client.workflow_service for describe_task_queue / get_system_info.

    ``poller_queues`` maps queue-name -> bool (has poller). ``describe_task_queue``
    returns an object with ``.pollers`` truthy iff the queried queue is present.
    """

    def __init__(self, poller_queues: dict[str, bool]):
        self._poller_queues = poller_queues
        self.describe_calls: list[str] = []
        self.system_info_calls = 0

    async def describe_task_queue(self, req, timeout=None):
        name = req.task_queue.name
        self.describe_calls.append(name)
        has = self._poller_queues.get(name, False)
        return SimpleNamespace(pollers=["poller"] if has else [])

    async def get_system_info(self, req, timeout=None):
        self.system_info_calls += 1
        return SimpleNamespace()


class _FakeServiceClient:
    def __init__(self, *, health_result=True, raises=False):
        self._health_result = health_result
        self._raises = raises
        self.check_health_calls = 0

    async def check_health(self, timeout=None):
        self.check_health_calls += 1
        if self._raises:
            raise RuntimeError("boom")
        return self._health_result


class _FakeClient:
    """Fake Temporal client with both the modern check_health path and the
    fallback get_system_info path selectable.
    """

    def __init__(
        self,
        *,
        poller_queues: Optional[dict[str, bool]] = None,
        health_result=True,
        health_raises=False,
        with_service_client=True,
    ):
        self.namespace = "default"
        self.workflow_service = _FakeWorkflowService(poller_queues or {})
        if with_service_client:
            self.service_client = _FakeServiceClient(
                health_result=health_result, raises=health_raises
            )
        else:
            # service_client present but WITHOUT check_health attribute ->
            # getattr(...,"check_health",None) is None -> fallback path.
            self.service_client = SimpleNamespace()


def _run_health(client) -> object:
    return asyncio.run(build_health_response(client, LOGGER))


# ---------------------------------------------------------------------------
# 9. temporal_client is None -> degraded, all flags False
# ---------------------------------------------------------------------------


def test_health_none_client_is_degraded_all_false():
    resp = _run_health(None)
    assert resp.status == "degraded"
    assert resp.temporal_connected is False
    assert resp.worker_available is False
    assert resp.ingest_worker_available is False


# ---------------------------------------------------------------------------
# 10. all healthy
# ---------------------------------------------------------------------------


def test_health_all_healthy():
    client = _FakeClient(
        poller_queues={RAG_QUERY_TASK_QUEUE: True, INGEST_QUEUE: True},
        health_result=True,
    )
    resp = _run_health(client)
    assert resp.status == "healthy"
    assert resp.temporal_connected is True
    assert resp.worker_available is True
    assert resp.ingest_worker_available is True


# ---------------------------------------------------------------------------
# 11. degraded matrix: independently controllable flags
# ---------------------------------------------------------------------------


def test_health_worker_missing_is_degraded():
    # temporal_ok True, query queue has NO poller -> worker_ok False,
    # ingest queue has poller -> ingest_worker_ok True. Status degraded.
    client = _FakeClient(
        poller_queues={RAG_QUERY_TASK_QUEUE: False, INGEST_QUEUE: True},
        health_result=True,
    )
    resp = _run_health(client)
    assert resp.temporal_connected is True
    assert resp.worker_available is False
    assert resp.ingest_worker_available is True
    assert resp.status == "degraded"


def test_health_ingest_worker_missing_is_degraded():
    client = _FakeClient(
        poller_queues={RAG_QUERY_TASK_QUEUE: True, INGEST_QUEUE: False},
        health_result=True,
    )
    resp = _run_health(client)
    assert resp.temporal_connected is True
    assert resp.worker_available is True
    assert resp.ingest_worker_available is False
    assert resp.status == "degraded"


# ---------------------------------------------------------------------------
# 12. check_health raising -> temporal_ok False (exception swallowed) -> degraded
# ---------------------------------------------------------------------------


def test_health_check_health_raises_is_degraded():
    client = _FakeClient(
        poller_queues={RAG_QUERY_TASK_QUEUE: True, INGEST_QUEUE: True},
        health_raises=True,
    )
    resp = _run_health(client)
    assert resp.temporal_connected is False
    # worker checks are skipped when temporal_ok is False.
    assert resp.worker_available is False
    assert resp.ingest_worker_available is False
    assert resp.status == "degraded"
    assert client.workflow_service.describe_calls == []


# ---------------------------------------------------------------------------
# 13. fallback path: no service_client.check_health -> get_system_info -> ok
# ---------------------------------------------------------------------------


def test_health_fallback_get_system_info_sets_temporal_ok():
    client = _FakeClient(
        poller_queues={RAG_QUERY_TASK_QUEUE: True, INGEST_QUEUE: True},
        with_service_client=False,
    )
    resp = _run_health(client)
    assert resp.temporal_connected is True
    assert resp.status == "healthy"
    assert client.workflow_service.system_info_calls == 1


# ---------------------------------------------------------------------------
# 14. _queue_has_poller: WORKFLOW-then-ACTIVITY order; True on first hit; False when neither
# ---------------------------------------------------------------------------


def test_queue_has_poller_true_when_present():
    client = _FakeClient(poller_queues={"q": True})
    out = asyncio.run(system_mod._queue_has_poller(client, "q"))
    assert out is True
    # Returns as soon as the WORKFLOW describe finds pollers -> single call.
    assert client.workflow_service.describe_calls == ["q"]


def test_queue_has_poller_false_when_neither_type_has_pollers():
    client = _FakeClient(poller_queues={"q": False})
    out = asyncio.run(system_mod._queue_has_poller(client, "q"))
    assert out is False
    # Both WORKFLOW and ACTIVITY describes attempted (queried twice).
    assert client.workflow_service.describe_calls == ["q", "q"]


# ---------------------------------------------------------------------------
# 15. ingest-queue branch: live config takes SEPARATE describe; document shortcut
# ---------------------------------------------------------------------------


def test_live_config_takes_separate_ingest_describe():
    # In real config the queues differ, so ingest_worker_ok comes from a second
    # describe of the ingest queue (NOT reused from worker_ok).
    assert INGEST_QUEUE != RAG_QUERY_TASK_QUEUE
    client = _FakeClient(
        poller_queues={RAG_QUERY_TASK_QUEUE: True, INGEST_QUEUE: False},
        health_result=True,
    )
    resp = _run_health(client)
    # worker_ok True but ingest queue separately probed and found empty.
    assert resp.worker_available is True
    assert resp.ingest_worker_available is False
    # The ingest queue name was actually described (separate path proof).
    assert INGEST_QUEUE in client.workflow_service.describe_calls


def test_shortcut_reuses_worker_ok_when_queues_equal(monkeypatch):
    # Force the dead-in-prod shortcut branch: make trigger_to_queue return the
    # query queue so ingest_queue == RAG_QUERY_TASK_QUEUE.
    monkeypatch.setattr(system_mod, "trigger_to_queue", lambda _t: RAG_QUERY_TASK_QUEUE)
    client = _FakeClient(
        poller_queues={RAG_QUERY_TASK_QUEUE: True},  # ingest queue not listed
        health_result=True,
    )
    resp = _run_health(client)
    assert resp.worker_available is True
    # ingest reuses worker_ok WITHOUT a second describe of a distinct queue.
    assert resp.ingest_worker_available is True
    # Only the query queue was described (no ingest-specific describe).
    assert set(client.workflow_service.describe_calls) == {RAG_QUERY_TASK_QUEUE}


# ---------------------------------------------------------------------------
# 16. Endpoints via TestClient
# ---------------------------------------------------------------------------


def _build_client(temporal_client) -> TestClient:
    app = FastAPI()
    app.include_router(
        create_system_router(get_temporal_client=lambda: temporal_client, logger=LOGGER)
    )
    return TestClient(app, raise_server_exceptions=False)


def test_health_endpoint_returns_health_response_shape():
    client = _FakeClient(
        poller_queues={RAG_QUERY_TASK_QUEUE: True, INGEST_QUEUE: True},
        health_result=True,
    )
    resp = _build_client(client).get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {
        "status",
        "temporal_connected",
        "worker_available",
        "ingest_worker_available",
    }
    assert body["status"] == "healthy"
    assert body["temporal_connected"] is True


def test_health_endpoint_none_client_degraded():
    resp = _build_client(None).get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "degraded"


def test_root_endpoint_exact_values():
    resp = _build_client(None).get("/")
    assert resp.status_code == 200
    assert resp.json() == {
        "service": "RAG Query API",
        "docs": "/docs",
        "health": "/health",
        "query_endpoint": "POST /query",
    }
