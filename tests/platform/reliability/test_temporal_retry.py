"""Contract + behavior tests for the Temporal-backed retry provider.

:mod:`src.platform.reliability.temporal_retry` is FAIL-OPEN: when Temporal is
unavailable, :meth:`TemporalRetryProvider.execute` must fall back to invoking
the supplied callable directly. These tests pin that fallback contract, the
payload construction from a :class:`RetryPolicy`, the global operation
registry, and the workflow-dispatch shape of ``_execute_via_temporal`` (mocked
Temporal client). No real Temporal server is contacted: the unavailable path is
forced by monkeypatching ``_execute_via_temporal`` to raise, and the happy path
is exercised against a fake async ``Client``.
"""

from __future__ import annotations

import asyncio
import dataclasses

import pytest

import src.platform.reliability.temporal_retry as tr
from config.settings import TEMPORAL_TARGET_HOST, TEMPORAL_TASK_QUEUE
from src.platform.reliability.temporal_retry import (
    TemporalPayload,
    TemporalRetryProvider,
    register_temporal_operation,
)
from src.platform.schemas import RetryPolicy


@pytest.fixture(autouse=True)
def _clear_registry():
    """Isolate the module-global operation registry around every test."""
    tr._OPERATION_REGISTRY.clear()
    yield
    tr._OPERATION_REGISTRY.clear()


def _raise_unavailable(record=None):
    """Build an async ``_execute_via_temporal`` replacement that always fails.

    Optionally records the payload it was called with into ``record['payload']``
    before raising, so tests can inspect the constructed :class:`TemporalPayload`
    without a live Temporal server.
    """

    async def _fake(self, payload):  # noqa: ANN001
        if record is not None:
            record["payload"] = payload
        raise RuntimeError("no temporal")

    return _fake


# --------------------------------------------------------------------------
# 1. register_temporal_operation
# --------------------------------------------------------------------------


def test_register_temporal_operation_stores_fn_under_name():
    def my_op():
        return "value-1"

    register_temporal_operation("alpha-op", my_op)

    assert tr._OPERATION_REGISTRY["alpha-op"] is my_op


# --------------------------------------------------------------------------
# 2. TemporalPayload contract (frozen + fields)
# --------------------------------------------------------------------------


def test_temporal_payload_is_frozen():
    payload = TemporalPayload(
        operation_name="op",
        idempotency_key="idem",
        max_attempts=3,
        initial_backoff_seconds=0.5,
        max_backoff_seconds=5.0,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        payload.max_attempts = 99  # type: ignore[misc]


def test_temporal_payload_carries_five_fields():
    payload = TemporalPayload(
        operation_name="op-name",
        idempotency_key="idem-key",
        max_attempts=4,
        initial_backoff_seconds=0.75,
        max_backoff_seconds=6.0,
    )
    assert payload.operation_name == "op-name"
    assert payload.idempotency_key == "idem-key"
    assert payload.max_attempts == 4
    assert payload.initial_backoff_seconds == 0.75
    assert payload.max_backoff_seconds == 6.0
    field_names = {f.name for f in dataclasses.fields(TemporalPayload)}
    assert field_names == {
        "operation_name",
        "idempotency_key",
        "max_attempts",
        "initial_backoff_seconds",
        "max_backoff_seconds",
    }


# --------------------------------------------------------------------------
# 3. __init__ sets target_host / task_queue from config
# --------------------------------------------------------------------------


def test_init_sets_target_host_and_task_queue():
    provider = TemporalRetryProvider()
    assert provider.target_host == TEMPORAL_TARGET_HOST
    assert provider.task_queue == TEMPORAL_TASK_QUEUE


# --------------------------------------------------------------------------
# 4. execute fallback returns fn() when Temporal is unavailable
# --------------------------------------------------------------------------


def test_execute_falls_back_to_local_when_temporal_unavailable(monkeypatch):
    monkeypatch.setattr(
        TemporalRetryProvider, "_execute_via_temporal", _raise_unavailable()
    )
    provider = TemporalRetryProvider()

    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        return "fallback-sentinel"

    result = provider.execute("op-fallback", fn)

    assert result == "fallback-sentinel"
    assert calls["n"] == 1
    assert tr._OPERATION_REGISTRY["op-fallback"] is fn


# --------------------------------------------------------------------------
# 5. execute builds payload from a CUSTOM policy
# --------------------------------------------------------------------------


def test_execute_builds_payload_from_custom_policy(monkeypatch):
    record: dict = {}
    monkeypatch.setattr(
        TemporalRetryProvider, "_execute_via_temporal", _raise_unavailable(record)
    )
    provider = TemporalRetryProvider()

    policy = RetryPolicy(
        max_attempts=7,
        initial_backoff_seconds=1.25,
        max_backoff_seconds=9.0,
    )

    provider.execute(
        "op-custom", lambda: "x", policy=policy, idempotency_key="idem-7"
    )

    payload = record["payload"]
    assert payload.max_attempts == 7
    assert payload.initial_backoff_seconds == 1.25
    assert payload.max_backoff_seconds == 9.0
    assert payload.operation_name == "op-custom"
    assert payload.idempotency_key == "idem-7"


# --------------------------------------------------------------------------
# 6. execute defaults policy to RetryPolicy() when None
# --------------------------------------------------------------------------


def test_execute_defaults_policy_when_none(monkeypatch):
    record: dict = {}
    monkeypatch.setattr(
        TemporalRetryProvider, "_execute_via_temporal", _raise_unavailable(record)
    )
    provider = TemporalRetryProvider()

    provider.execute("op-default", lambda: "x", policy=None)

    payload = record["payload"]
    assert payload.max_attempts == 3
    assert payload.initial_backoff_seconds == 0.5
    assert payload.max_backoff_seconds == 5.0


# --------------------------------------------------------------------------
# 7. idempotency_key threaded into payload (None and a real key)
# --------------------------------------------------------------------------


def test_execute_threads_idempotency_key(monkeypatch):
    record: dict = {}
    monkeypatch.setattr(
        TemporalRetryProvider, "_execute_via_temporal", _raise_unavailable(record)
    )
    provider = TemporalRetryProvider()

    provider.execute("op-key", lambda: "x", idempotency_key="real-key-42")
    assert record["payload"].idempotency_key == "real-key-42"

    provider.execute("op-nokey", lambda: "x", idempotency_key=None)
    assert record["payload"].idempotency_key is None


# --------------------------------------------------------------------------
# 8. execute from inside a running event loop still falls back (thread branch)
# --------------------------------------------------------------------------


def test_execute_inside_running_loop_falls_back(monkeypatch):
    monkeypatch.setattr(
        TemporalRetryProvider, "_execute_via_temporal", _raise_unavailable()
    )
    provider = TemporalRetryProvider()

    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        return "loop-sentinel"

    async def driver():
        # Confirm we are genuinely inside a running loop (ThreadPoolExecutor
        # branch), not the asyncio.run branch.
        assert asyncio.get_running_loop().is_running()
        return provider.execute("op-loop", fn)

    result = asyncio.run(driver())

    assert result == "loop-sentinel"
    assert calls["n"] == 1


# --------------------------------------------------------------------------
# 9. _execute_via_temporal happy path (mocked Client)
# --------------------------------------------------------------------------


class _FakeClient:
    def __init__(self, recorder, result):
        self._recorder = recorder
        self._result = result

    @classmethod
    def _make_connector(cls, recorder, result):
        async def connect(target_host):  # noqa: ANN001
            recorder["connect_host"] = target_host
            return cls(recorder, result)

        return connect

    async def execute_workflow(self, workflow, arg, *, id, task_queue):  # noqa: A002
        self._recorder["workflow"] = workflow
        self._recorder["arg"] = arg
        self._recorder["id"] = id
        self._recorder["task_queue"] = task_queue
        return self._result

    async def close(self):
        self._recorder["closed"] = True


def test_execute_via_temporal_happy_path(monkeypatch):
    recorder: dict = {}
    sentinel = "workflow-result-9"

    # The function does `from temporalio.client import Client` at call time,
    # so patch the attribute on the real temporalio.client module.
    import temporalio.client as temporal_client

    class _Client:
        connect = staticmethod(
            _FakeClient._make_connector(recorder, sentinel)
        )

    monkeypatch.setattr(temporal_client, "Client", _Client)

    provider = TemporalRetryProvider()
    payload = TemporalPayload(
        operation_name="op-happy",
        idempotency_key="idem-happy",
        max_attempts=3,
        initial_backoff_seconds=0.5,
        max_backoff_seconds=5.0,
    )

    result = asyncio.run(provider._execute_via_temporal(payload))

    assert result == sentinel
    assert recorder["connect_host"] == provider.target_host
    assert recorder["workflow"] == "RetryWorkflow"
    assert recorder["arg"] == payload.__dict__
    assert recorder["id"] == "rag-retry-op-happy-idem-happy"
    assert recorder["task_queue"] == provider.task_queue
    assert recorder["closed"] is True


# --------------------------------------------------------------------------
# 10. workflow_id fallback when idempotency_key is None
# --------------------------------------------------------------------------


def test_execute_via_temporal_workflow_id_fallback(monkeypatch):
    recorder: dict = {}
    import temporalio.client as temporal_client

    class _Client:
        connect = staticmethod(_FakeClient._make_connector(recorder, "r"))

    monkeypatch.setattr(temporal_client, "Client", _Client)

    provider = TemporalRetryProvider()
    payload = TemporalPayload(
        operation_name="op-noidem",
        idempotency_key=None,
        max_attempts=3,
        initial_backoff_seconds=0.5,
        max_backoff_seconds=5.0,
    )

    asyncio.run(provider._execute_via_temporal(payload))

    assert recorder["id"] == "rag-retry-op-noidem-no-idempotency-key"


# --------------------------------------------------------------------------
# 11. _execute_via_temporal closes client even when execute_workflow raises
# --------------------------------------------------------------------------


def test_execute_via_temporal_closes_client_on_error(monkeypatch):
    recorder: dict = {}

    class _RaisingClient:
        async def execute_workflow(self, *args, **kwargs):
            raise RuntimeError("workflow boom")

        async def close(self):
            recorder["closed"] = True

    async def _connect(target_host):  # noqa: ANN001
        return _RaisingClient()

    import temporalio.client as temporal_client

    class _Client:
        connect = staticmethod(_connect)

    monkeypatch.setattr(temporal_client, "Client", _Client)

    provider = TemporalRetryProvider()
    payload = TemporalPayload(
        operation_name="op-err",
        idempotency_key="k",
        max_attempts=3,
        initial_backoff_seconds=0.5,
        max_backoff_seconds=5.0,
    )

    with pytest.raises(RuntimeError, match="workflow boom"):
        asyncio.run(provider._execute_via_temporal(payload))

    assert recorder["closed"] is True
