from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from server import api
from src.platform.memory.schemas import ConversationMeta
from src.platform.security import auth
from src.platform.security.auth import authenticate_request, Principal


@dataclass
class _FakeMemory:
    """In-memory conversation memory; avoids Redis dependency in CI."""

    relevant: dict[str, list[str]] = field(default_factory=dict)
    ignored: dict[str, list[str]] = field(default_factory=dict)

    def _meta(self, conversation_id: str) -> ConversationMeta:
        return ConversationMeta(
            conversation_id=conversation_id or "conv_test",
            tenant_id="t",
            subject="s",
            project_id="",
        )

    def ensure_conversation(self, *, tenant_id, subject, project_id, conversation_id=None, title=""):
        return self._meta(conversation_id or "conv_test")

    def build_context(self, **_kwargs):  # pragma: no cover - trivial stub
        ctx = MagicMock()
        ctx.context_text = ""
        ctx.recent_turns = []
        return ctx

    def append_turn(self, **_kwargs):  # pragma: no cover - trivial stub
        return None

    def update_conversation_title(self, **_kwargs):  # pragma: no cover - trivial stub
        return None

    def compact_if_needed(self, **_kwargs):  # pragma: no cover - trivial stub
        return None


class _DummyWorkflowService:
    async def get_system_info(self):
        return {"status": "ok"}


class _DummyTemporalClient:
    def __init__(self, *, fail_query: bool = False):
        self.workflow_service = _DummyWorkflowService()
        self._fail_query = fail_query

    async def execute_workflow(self, *args, **kwargs):
        if self._fail_query:
            raise RuntimeError("simulated workflow failure")
        return {
            "query": "hello",
            "processed_query": "hello",
            "query_confidence": 0.9,
            "action": "search",
            "results": [],
            "stage_timings": [],
            "timing_totals": {"total_ms": 5.0},
            "budget_exhausted": False,
        }

    async def close(self):
        return None


def _set_auth_defaults() -> None:
    auth.AUTH_API_KEYS_REQUIRED = False
    auth.AUTH_JWT_ENABLED = False
    auth.AUTH_OIDC_ENABLED = False


def test_422_validation_error_uses_standard_envelope(monkeypatch):
    _set_auth_defaults()

    async def _fake_connect(_target: str):
        return _DummyTemporalClient()

    monkeypatch.setattr(api.Client, "connect", _fake_connect)
    with TestClient(api.app, raise_server_exceptions=False) as client:
        response = client.post("/query", json={})

    body = response.json()
    assert response.status_code == 422
    assert body["ok"] is False
    assert body["error"]["code"] == "REQUEST_VALIDATION_ERROR"
    assert "x-request-id" in response.headers
    assert body["request_id"] == response.headers["x-request-id"]


def test_401_auth_error_uses_standard_envelope(monkeypatch):
    auth.AUTH_API_KEYS_REQUIRED = True
    auth.AUTH_JWT_ENABLED = False
    auth.AUTH_OIDC_ENABLED = False

    async def _fake_connect(_target: str):
        return _DummyTemporalClient()

    monkeypatch.setattr(api.Client, "connect", _fake_connect)
    with TestClient(api.app, raise_server_exceptions=False) as client:
        response = client.post("/query", json={"query": "hello"})

    body = response.json()
    assert response.status_code == 401
    assert body["ok"] is False
    assert body["error"]["code"] == "HTTP_401"
    assert body["error"]["message"] == "Authentication required"


def test_403_role_error_uses_standard_envelope(monkeypatch):
    _set_auth_defaults()

    async def _fake_connect(_target: str):
        return _DummyTemporalClient()

    async def _viewer_principal():
        return Principal(
            subject="viewer",
            tenant_id="default",
            roles=["viewer"],
            auth_type="none",
        )

    monkeypatch.setattr(api.Client, "connect", _fake_connect)
    api.app.dependency_overrides[authenticate_request] = _viewer_principal
    try:
        with TestClient(api.app, raise_server_exceptions=False) as client:
            response = client.post("/query", json={"query": "hello"})
    finally:
        api.app.dependency_overrides.clear()

    body = response.json()
    assert response.status_code == 403
    assert body["ok"] is False
    assert body["error"]["code"] == "HTTP_403"
    assert "Missing role" in body["error"]["message"]


def test_500_query_failure_uses_standard_envelope(monkeypatch):
    _set_auth_defaults()

    async def _fake_connect(_target: str):
        return _DummyTemporalClient(fail_query=True)

    monkeypatch.setattr(api.Client, "connect", _fake_connect)
    with TestClient(api.app, raise_server_exceptions=False) as client:
        response = client.post("/query", json={"query": "hello"})

    body = response.json()
    assert response.status_code == 500
    assert body["ok"] is False
    assert body["error"]["code"] == "INTERNAL_SERVER_ERROR"
    # Production's unhandled-exception handler intentionally returns a
    # generic message and does NOT leak the upstream exception text — the
    # original "simulated workflow failure" is logged server-side but is
    # scrubbed from the client response (server/api.py:230-241).
    assert body["error"]["message"] == "Internal server error"


def test_404_not_found_uses_standard_envelope(monkeypatch):
    _set_auth_defaults()

    async def _fake_connect(_target: str):
        return _DummyTemporalClient()

    monkeypatch.setattr(api.Client, "connect", _fake_connect)
    with TestClient(api.app, raise_server_exceptions=False) as client:
        response = client.get("/does-not-exist")

    body = response.json()
    assert response.status_code == 404
    assert body["ok"] is False
    assert body["error"]["code"] == "HTTP_404"


def test_503_overload_uses_standard_envelope(monkeypatch):
    _set_auth_defaults()

    async def _fake_connect(_target: str):
        return _DummyTemporalClient()

    monkeypatch.setattr(api.Client, "connect", _fake_connect)
    monkeypatch.setattr(api, "_api_inflight_semaphore", asyncio.Semaphore(0))
    monkeypatch.setattr(api, "_api_overload_queue_timeout_ms", 1)
    with TestClient(api.app, raise_server_exceptions=False) as client:
        response = client.post("/query", json={"query": "hello"})

    body = response.json()
    assert response.status_code == 503
    assert body["ok"] is False
    assert body["error"]["code"] == "HTTP_503"
    assert "Server overloaded" in body["error"]["message"]


def _patched_client(monkeypatch, *, fail_query: bool = False):
    """Return a TestClient with stubbed Temporal + in-memory conversation memory."""
    _set_auth_defaults()

    async def _fake_connect(_target: str):
        return _DummyTemporalClient(fail_query=fail_query)

    monkeypatch.setattr(api.Client, "connect", _fake_connect)
    fake = _FakeMemory()
    monkeypatch.setattr("server.routes.query.get_conversation_memory", lambda: fake)
    return TestClient(api.app, raise_server_exceptions=False)


def test_every_error_response_has_required_fields(monkeypatch):
    """Every error response must include ok=false, non-empty error, non-empty request_id."""
    with _patched_client(monkeypatch) as client:
        resp_422 = client.post("/query", json={})
        resp_404 = client.get("/does-not-exist")

    for resp in (resp_422, resp_404):
        body = resp.json()
        assert body["ok"] is False, f"Expected ok=false, got {body}"
        error = body.get("error", {})
        assert error, "error field must be present and non-empty"
        assert isinstance(error.get("message"), str) and error["message"], (
            "error.message must be a non-empty string"
        )
        assert body.get("request_id"), "request_id must be present and non-empty"


def test_success_response_has_ok_true(monkeypatch):
    """Successful query response returns HTTP 200 with the QueryResponse schema."""
    with _patched_client(monkeypatch) as client:
        response = client.post("/query", json={"query": "hello"})

    assert response.status_code == 200
    body = response.json()
    # QueryResponse is returned directly (not wrapped in an ok-envelope).
    # A 200 response must include the 'query' field from QueryResponse.
    assert "query" in body


def test_500_error_does_not_leak_traceback(monkeypatch):
    """Internal server error must not expose raw Python tracebacks."""
    with _patched_client(monkeypatch, fail_query=True) as client:
        response = client.post("/query", json={"query": "hello"})

    assert response.status_code == 500
    error_msg = response.json()["error"]["message"]
    assert "Traceback" not in error_msg
    assert "File " not in error_msg
