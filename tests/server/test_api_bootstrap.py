"""App-level cross-cutting tests for ``server/api.py``.

Covers the bootstrap/cross-cutting logic that the route-level tests do not
exercise directly:

- ``_enforce_rate_limit``: allow path is a no-op; the over-quota path raises
  HTTPException(429) after consulting the limiter with the tenant-scoped key.
- ``_acquire_request_slot`` / ``_release_request_slot``: the inflight semaphore
  is acquired below capacity, rejects with HTTPException(503) at capacity, and
  release decrements / no-ops for an unacquired slot. The at-cap boundary is
  given teeth with a one-below positive partner.
- ``add_request_id_middleware``: a request-id header is generated when absent
  and the client-supplied ``x-request-id`` is propagated unchanged.
- exception handlers: the error-envelope keys, the HTTPException-500 generic
  message scrub, the 422 status, and the unhandled-exception 500 no-leak
  contract (asserted by calling the handler coroutines directly so they are
  covered independently of the route stack).
- ``lifespan``: entering/exiting the context manager performs the documented
  startup (Temporal connect + job sweeper) and shutdown (sweeper cancel +
  client close).

The error-envelope *route* integration is already covered by
``tests/test_api_error_envelope.py``; this module covers the helpers and
middleware that file does not touch and asserts the handler contracts at the
coroutine level.
"""
from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from server import api
from src.platform.security import Principal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _principal() -> Principal:
    return Principal(
        subject="user-1",
        tenant_id="tenant-a",
        roles=["query"],
        auth_type="none",
        project_id="proj-x",
    )


class _StubLimiter:
    """Records the check() call and returns a canned LimitResult-like object."""

    def __init__(self, *, allowed: bool, retry_after_seconds: int = 7):
        self._allowed = allowed
        self._retry_after = retry_after_seconds
        self.calls: list[dict] = []

    def check(self, *, key: str, limit: int, window_seconds: int):
        self.calls.append({"key": key, "limit": limit, "window_seconds": window_seconds})

        class _Result:
            allowed = self._allowed
            retry_after_seconds = self._retry_after

        return _Result()


def _make_request(request_id: str | None = None) -> Request:
    """Build a bare Request, optionally seeding the middleware-set request id.

    The error envelope reads the id from ``request.state.request_id`` (set by
    ``add_request_id_middleware``), not from the inbound header, so the test
    seeds state directly to mirror post-middleware handler invocation.
    """
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/x",
        "headers": [],
        "query_string": b"",
    }
    request = Request(scope)
    if request_id is not None:
        request.state.request_id = request_id
    return request


# ---------------------------------------------------------------------------
# _enforce_rate_limit
# ---------------------------------------------------------------------------


def test_enforce_rate_limit_allows_when_under_quota(monkeypatch):
    """Under quota: returns None and consults the limiter with the scoped key."""
    monkeypatch.setattr(api, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(api, "get_tenant_quota", lambda _tenant: 60)
    limiter = _StubLimiter(allowed=True)
    monkeypatch.setattr(api, "_rate_limiter", limiter)

    result = api._enforce_rate_limit(_principal(), endpoint="/query")

    assert result is None
    assert len(limiter.calls) == 1
    assert limiter.calls[0]["key"] == "tenant-a:proj-x"
    assert limiter.calls[0]["limit"] == 60


def test_enforce_rate_limit_raises_429_when_over_quota(monkeypatch):
    """Over quota: raises HTTPException(429) with the retry-after detail."""
    monkeypatch.setattr(api, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(api, "get_tenant_quota", lambda _tenant: 5)
    limiter = _StubLimiter(allowed=False, retry_after_seconds=9)
    monkeypatch.setattr(api, "_rate_limiter", limiter)

    with pytest.raises(HTTPException) as excinfo:
        api._enforce_rate_limit(_principal(), endpoint="/query")

    assert excinfo.value.status_code == 429
    assert "Retry in 9s" in excinfo.value.detail
    assert limiter.calls[0]["key"] == "tenant-a:proj-x"


def test_enforce_rate_limit_noop_when_disabled(monkeypatch):
    """Disabled: returns immediately without consulting the limiter."""
    monkeypatch.setattr(api, "RATE_LIMIT_ENABLED", False)
    limiter = _StubLimiter(allowed=False)
    monkeypatch.setattr(api, "_rate_limiter", limiter)

    assert api._enforce_rate_limit(_principal(), endpoint="/query") is None
    assert limiter.calls == []


# ---------------------------------------------------------------------------
# _acquire_request_slot / _release_request_slot
# ---------------------------------------------------------------------------


def test_acquire_slot_returns_true_below_capacity(monkeypatch):
    """One slot free (cap=1): acquire succeeds and returns True."""
    monkeypatch.setattr(api, "_api_inflight_semaphore", asyncio.Semaphore(1))
    monkeypatch.setattr(api, "_api_overload_queue_timeout_ms", 50)

    acquired = asyncio.run(api._acquire_request_slot("/query"))

    assert acquired is True
    # One acquire consumed the only slot.
    assert api._api_inflight_semaphore.locked() is True


def test_acquire_slot_rejects_503_at_capacity(monkeypatch):
    """At capacity (semaphore exhausted): acquire raises HTTPException(503)."""
    monkeypatch.setattr(api, "_api_inflight_semaphore", asyncio.Semaphore(0))
    monkeypatch.setattr(api, "_api_overload_queue_timeout_ms", 1)

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(api._acquire_request_slot("/query"))

    assert excinfo.value.status_code == 503
    assert "overloaded" in excinfo.value.detail.lower()


def test_acquire_slot_returns_false_when_disabled(monkeypatch):
    """No semaphore configured (cap<=0): acquire is disabled, returns False."""
    monkeypatch.setattr(api, "_api_inflight_semaphore", None)

    assert asyncio.run(api._acquire_request_slot("/query")) is False


def test_release_slot_returns_capacity(monkeypatch):
    """Release after a real acquire restores the slot for the next caller."""
    sem = asyncio.Semaphore(1)
    monkeypatch.setattr(api, "_api_inflight_semaphore", sem)
    monkeypatch.setattr(api, "_api_overload_queue_timeout_ms", 50)

    async def _scenario():
        acquired = await api._acquire_request_slot("/query")
        assert sem.locked() is True  # cap consumed
        api._release_request_slot(acquired)
        assert sem.locked() is False  # slot returned
        # A fresh acquire now succeeds again because the slot was released.
        return await api._acquire_request_slot("/query")

    assert asyncio.run(_scenario()) is True


def test_release_slot_noop_when_not_acquired(monkeypatch):
    """Release with acquired=False must not touch the semaphore."""
    sem = asyncio.Semaphore(0)  # fully saturated
    monkeypatch.setattr(api, "_api_inflight_semaphore", sem)

    api._release_request_slot(False)

    # Still saturated — a non-acquired release must not have released a slot.
    assert sem.locked() is True


# ---------------------------------------------------------------------------
# add_request_id_middleware
# ---------------------------------------------------------------------------


def _request_id_app() -> FastAPI:
    """Minimal app with only the request-id middleware and one route."""
    app = FastAPI()
    app.middleware("http")(api.add_request_id_middleware)

    @app.get("/ping")
    async def _ping(request: Request) -> dict:
        return {"state_request_id": request.state.request_id}

    return app


def test_request_id_generated_when_absent():
    """No incoming header: a generated req- id is set on state and response."""
    client = TestClient(_request_id_app())
    resp = client.get("/ping")

    assert resp.status_code == 200
    header_id = resp.headers["x-request-id"]
    assert header_id.startswith("req-")
    # The response header and request.state id are the same generated value.
    assert resp.json()["state_request_id"] == header_id


def test_request_id_propagated_from_client_header():
    """Client-supplied x-request-id is echoed back unchanged."""
    client = TestClient(_request_id_app())
    resp = client.get("/ping", headers={"x-request-id": "client-supplied-42"})

    assert resp.status_code == 200
    assert resp.headers["x-request-id"] == "client-supplied-42"
    assert resp.json()["state_request_id"] == "client-supplied-42"


# ---------------------------------------------------------------------------
# Exception handlers (coroutine-level contract)
# ---------------------------------------------------------------------------


def _envelope(response) -> dict:
    import json

    return json.loads(bytes(response.body))


def test_http_exception_handler_string_detail_envelope():
    """A string-detail HTTPException maps to HTTP_<code> with the detail message."""
    request = _make_request(request_id="rid-1")
    exc = HTTPException(status_code=403, detail="Missing role")

    response = asyncio.run(api.http_exception_handler(request, exc))
    body = _envelope(response)

    assert response.status_code == 403
    assert body["ok"] is False
    assert body["error"]["code"] == "HTTP_403"
    assert body["error"]["message"] == "Missing role"
    assert body["request_id"] == "rid-1"


def test_http_exception_handler_500_scrubs_detail():
    """A 500 HTTPException is forced to the generic envelope (detail scrubbed)."""
    request = _make_request(request_id="rid-2")
    exc = HTTPException(status_code=500, detail="db password leaked here")

    response = asyncio.run(api.http_exception_handler(request, exc))
    body = _envelope(response)

    assert response.status_code == 500
    assert body["error"]["code"] == "INTERNAL_SERVER_ERROR"
    assert body["error"]["message"] == "Internal server error"
    assert "password" not in body["error"]["message"]


def test_request_validation_handler_returns_422():
    """RequestValidationError maps to a 422 envelope carrying the error list."""
    request = _make_request(request_id="rid-3")
    exc = RequestValidationError(
        [{"loc": ("body", "query"), "msg": "field required", "type": "missing"}]
    )

    response = asyncio.run(api.request_validation_exception_handler(request, exc))
    body = _envelope(response)

    assert response.status_code == 422
    assert body["error"]["code"] == "REQUEST_VALIDATION_ERROR"
    assert body["error"]["details"]["errors"][0]["msg"] == "field required"


def test_unhandled_exception_handler_returns_generic_500():
    """An unhandled exception is scrubbed to a generic 500 (no leak)."""
    request = _make_request(request_id="rid-4")
    exc = RuntimeError("internal stack secret xyz")

    response = asyncio.run(api.unhandled_exception_handler(request, exc))
    body = _envelope(response)

    assert response.status_code == 500
    assert body["error"]["code"] == "INTERNAL_SERVER_ERROR"
    assert body["error"]["message"] == "Internal server error"
    assert "secret" not in body["error"]["message"]


# ---------------------------------------------------------------------------
# lifespan
# ---------------------------------------------------------------------------


def test_lifespan_connects_and_closes_temporal(monkeypatch):
    """Lifespan connects Temporal on entry and closes it (plus sweeper) on exit."""
    events: list[str] = []

    class _FakeClient:
        async def close(self):
            events.append("client_closed")

    async def _fake_connect(target):
        events.append(f"connected:{target}")
        return _FakeClient()

    class _FakeTask:
        def cancel(self):
            events.append("sweeper_cancelled")

        def __await__(self):
            async def _coro():
                return None

            return _coro().__await__()

    monkeypatch.setattr(api.Client, "connect", _fake_connect)
    monkeypatch.setattr(api, "start_job_sweeper", lambda: _FakeTask())
    monkeypatch.setattr(api, "_validate_startup_config", lambda: None)
    monkeypatch.setattr(api, "_validate_all_config", lambda: None)
    monkeypatch.setattr(api, "_validate_optional_dependencies", lambda: None)
    monkeypatch.setattr(api, "_temporal_client", None)

    async def _run():
        async with api.lifespan(api.app):
            # Inside the context: the client is connected and stored.
            assert api._get_temporal_client() is not None
            events.append("inside")

    asyncio.run(_run())

    assert events == [
        f"connected:{api.TEMPORAL_TARGET_HOST}",
        "inside",
        "sweeper_cancelled",
        "client_closed",
    ]
    # After exit the module-level client is reset to None.
    assert api._get_temporal_client() is None
