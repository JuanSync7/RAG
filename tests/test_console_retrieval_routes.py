"""Console routes for the Retrieval tab: ignore / restore / doc-state."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, field
from unittest.mock import patch

from fastapi.testclient import TestClient

from server import api
from src.platform.memory.schemas import ConversationMeta
from src.platform.security import auth


_MEMORY_PATCHES = [
    "server.routes.query.get_conversation_memory",
    "server.console.routes.get_conversation_memory",
]


def _set_auth_defaults() -> None:
    auth.AUTH_API_KEYS_REQUIRED = False
    auth.AUTH_JWT_ENABLED = False
    auth.AUTH_OIDC_ENABLED = False


class _DummyWorkflowService:
    async def get_system_info(self):
        return {"status": "ok"}


class _DummyTemporalClient:
    def __init__(self):
        self.workflow_service = _DummyWorkflowService()

    async def execute_workflow(self, *args, **kwargs):
        return {
            "query": "x",
            "processed_query": "x",
            "query_confidence": 0.9,
            "action": "search",
            "results": [],
            "generated_answer": "",
            "stage_timings": [],
            "timing_totals": {"total_ms": 1.0},
            "budget_exhausted": False,
        }

    async def close(self):
        return None


def _patch_temporal(monkeypatch):
    async def _fake_connect(_target: str):
        return _DummyTemporalClient()

    monkeypatch.setattr(api.Client, "connect", _fake_connect)


@dataclass
class _FakeMemory:
    """Minimal in-memory memory provider sufficient for route round-trip tests."""

    relevant: dict[str, list[str]] = field(default_factory=dict)
    ignored: dict[str, list[str]] = field(default_factory=dict)

    def _meta(self, conversation_id: str) -> ConversationMeta:
        return ConversationMeta(
            conversation_id=conversation_id,
            tenant_id="t",
            subject="s",
            project_id="",
            relevant_doc_ids=list(self.relevant.get(conversation_id, [])),
            ignored_doc_ids=list(self.ignored.get(conversation_id, [])),
        )

    def ensure_conversation(self, *, tenant_id, subject, project_id, conversation_id=None, title=""):
        cid = conversation_id or "conv_test"
        self.relevant.setdefault(cid, [])
        self.ignored.setdefault(cid, [])
        return self._meta(cid)

    def mark_retrieved(self, *, tenant_id, subject, project_id, conversation_id, doc_ids):
        rel = self.relevant.setdefault(conversation_id, [])
        ign = self.ignored.setdefault(conversation_id, [])
        for d in doc_ids:
            if d not in rel and d not in ign:
                rel.append(d)
        return self._meta(conversation_id)

    def move_to_ignored(self, *, tenant_id, subject, project_id, conversation_id, doc_id):
        rel = self.relevant.setdefault(conversation_id, [])
        ign = self.ignored.setdefault(conversation_id, [])
        if doc_id in rel:
            rel.remove(doc_id)
        if doc_id not in ign:
            ign.append(doc_id)
        return self._meta(conversation_id)

    def restore_to_relevant(self, *, tenant_id, subject, project_id, conversation_id, doc_id):
        rel = self.relevant.setdefault(conversation_id, [])
        ign = self.ignored.setdefault(conversation_id, [])
        if doc_id in ign:
            ign.remove(doc_id)
        if doc_id not in rel:
            rel.append(doc_id)
        return self._meta(conversation_id)

    def get_seen_doc_ids(self, *, tenant_id, subject, project_id, conversation_id):
        meta = self._meta(conversation_id)
        return list(meta.relevant_doc_ids) + list(meta.ignored_doc_ids)


def _patch_memory(fake):
    stack = ExitStack()
    for target in _MEMORY_PATCHES:
        stack.enter_context(patch(target, return_value=fake))
    return stack


def test_doc_state_starts_empty(monkeypatch):
    _set_auth_defaults()
    _patch_temporal(monkeypatch)
    fake = _FakeMemory()
    with _patch_memory(fake):
        with TestClient(api.app, raise_server_exceptions=False) as client:
            resp = client.get("/console/conversations/conv_a/doc-state")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["relevant_doc_ids"] == []
    assert data["ignored_doc_ids"] == []


def test_ignore_then_doc_state_reflects_ignored(monkeypatch):
    _patch_temporal(monkeypatch)
    _set_auth_defaults()
    fake = _FakeMemory()
    fake.relevant["conv_a"] = ["d1", "d2", "d3"]
    fake.ignored["conv_a"] = []
    with _patch_memory(fake):
        with TestClient(api.app, raise_server_exceptions=False) as client:
            r1 = client.post(
                "/console/conversations/conv_a/ignore",
                json={"doc_id": "d2"},
            )
            r2 = client.get("/console/conversations/conv_a/doc-state")
    assert r1.status_code == 200, r1.text
    body = r1.json()["data"]
    assert body["ignored_doc_ids"] == ["d2"]
    assert "d2" not in body["relevant_doc_ids"]
    state = r2.json()["data"]
    assert state["ignored_doc_ids"] == ["d2"]


def test_ignore_batch_with_doc_ids_list(monkeypatch):
    _patch_temporal(monkeypatch)
    _set_auth_defaults()
    fake = _FakeMemory()
    fake.relevant["conv_a"] = ["d1", "d2", "d3"]
    with _patch_memory(fake):
        with TestClient(api.app, raise_server_exceptions=False) as client:
            r = client.post(
                "/console/conversations/conv_a/ignore",
                json={"doc_ids": ["d1", "d3"]},
            )
    assert r.status_code == 200
    data = r.json()["data"]
    assert sorted(data["ignored_doc_ids"]) == ["d1", "d3"]
    assert data["relevant_doc_ids"] == ["d2"]


def test_ignore_requires_doc_id_or_doc_ids(monkeypatch):
    _patch_temporal(monkeypatch)
    _set_auth_defaults()
    fake = _FakeMemory()
    with _patch_memory(fake):
        with TestClient(api.app, raise_server_exceptions=False) as client:
            r = client.post("/console/conversations/conv_a/ignore", json={})
    assert r.status_code == 400


def test_restore_moves_doc_back_to_relevant(monkeypatch):
    _patch_temporal(monkeypatch)
    _set_auth_defaults()
    fake = _FakeMemory()
    fake.relevant["conv_a"] = ["d1"]
    fake.ignored["conv_a"] = ["d2"]
    with _patch_memory(fake):
        with TestClient(api.app, raise_server_exceptions=False) as client:
            r = client.delete("/console/conversations/conv_a/ignore/d2")
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["ignored_doc_ids"] == []
    assert sorted(data["relevant_doc_ids"]) == ["d1", "d2"]
