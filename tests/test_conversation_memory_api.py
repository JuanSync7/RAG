from __future__ import annotations

import sys
from unittest.mock import patch

from fastapi.testclient import TestClient

from server import api
from src.platform.memory.provider import NoopConversationMemory, RedisConversationMemory
from src.platform.security import auth


class _DummyWorkflowService:
    async def get_system_info(self):
        return {"status": "ok"}


class _DummyTemporalClient:
    def __init__(self):
        self.workflow_service = _DummyWorkflowService()

    async def execute_workflow(self, *args, **kwargs):
        return {
            "query": "hello",
            "processed_query": "hello",
            "query_confidence": 0.9,
            "action": "search",
            "results": [],
            "generated_answer": "test answer",
            "stage_timings": [],
            "timing_totals": {"total_ms": 5.0},
            "budget_exhausted": False,
        }

    async def close(self):
        return None


_NOOP_MEMORY = NoopConversationMemory()


def _set_auth_defaults() -> None:
    auth.AUTH_API_KEYS_REQUIRED = False
    auth.AUTH_JWT_ENABLED = False
    auth.AUTH_OIDC_ENABLED = False


_MEMORY_PATCHES = [
    "server.routes.query.get_conversation_memory",
    "server.console.routes.get_conversation_memory",
]


def _patch_memory():
    """Return a context manager that patches all get_conversation_memory call sites."""
    from contextlib import ExitStack
    stack = ExitStack()
    for target in _MEMORY_PATCHES:
        stack.enter_context(patch(target, return_value=_NOOP_MEMORY))
    return stack


def test_query_returns_conversation_id(monkeypatch):
    _set_auth_defaults()

    async def _fake_connect(_target: str):
        return _DummyTemporalClient()

    monkeypatch.setattr(api.Client, "connect", _fake_connect)
    with _patch_memory():
        with TestClient(api.app, raise_server_exceptions=False) as client:
            response = client.post("/query", json={"query": "hello"})
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body.get("conversation_id"), str)
    assert body["conversation_id"]


def test_query_respects_provided_conversation_id(monkeypatch):
    _set_auth_defaults()

    async def _fake_connect(_target: str):
        return _DummyTemporalClient()

    monkeypatch.setattr(api.Client, "connect", _fake_connect)
    with _patch_memory():
        with TestClient(api.app, raise_server_exceptions=False) as client:
            response = client.post(
                "/query",
                json={"query": "follow up", "conversation_id": "conv_demo_123"},
            )
    assert response.status_code == 200
    body = response.json()
    assert body["conversation_id"] == "conv_demo_123"


def test_console_command_switch_and_compact_actions(monkeypatch):
    _set_auth_defaults()

    async def _fake_connect(_target: str):
        return _DummyTemporalClient()

    monkeypatch.setattr(api.Client, "connect", _fake_connect)
    with _patch_memory():
        with TestClient(api.app, raise_server_exceptions=False) as client:
            switch_resp = client.post(
                "/console/command",
                json={"mode": "query", "command": "switch", "arg": "conv_abc", "state": {}},
            )
            compact_resp = client.post(
                "/console/command",
                json={
                    "mode": "query",
                    "command": "compact",
                    "state": {"conversation_id": "conv_abc"},
                },
            )

    assert switch_resp.status_code == 200
    switch_data = switch_resp.json()["data"]
    assert switch_data["action"] == "switch_conversation"
    assert switch_data["data"]["conversation_id"] == "conv_abc"

    assert compact_resp.status_code == 200
    compact_data = compact_resp.json()["data"]
    assert compact_data["action"] == "compact_conversation"


# ---------------------------------------------------------------------------
# Memory compaction unit tests (direct provider, no HTTP layer)
# ---------------------------------------------------------------------------

class _FakeRedis:
    """Minimal in-memory Redis stand-in for compaction tests."""

    def __init__(self):
        self.kv = {}
        self.lists = {}
        self.zsets = {}

    def exists(self, key):
        return 1 if key in self.kv else 0

    def hgetall(self, key):
        return dict(self.kv.get(key, {}))

    def hset(self, key, mapping):
        cur = dict(self.kv.get(key, {}))
        cur.update(mapping)
        self.kv[key] = cur

    def zadd(self, key, mapping):
        cur = dict(self.zsets.get(key, {}))
        cur.update(mapping)
        self.zsets[key] = cur

    def zrevrange(self, key, start, end):
        cur = self.zsets.get(key, {})
        ordered = sorted(cur.items(), key=lambda it: it[1], reverse=True)
        ids = [k for k, _ in ordered]
        if end < 0:
            return ids[start:]
        return ids[start: end + 1]

    def lrange(self, key, start, end):
        cur = list(self.lists.get(key, []))
        if start < 0:
            start = max(0, len(cur) + start)
        if end < 0:
            end = len(cur) + end
        return cur[start: end + 1]

    def rpush(self, key, value):
        cur = list(self.lists.get(key, []))
        cur.append(value)
        self.lists[key] = cur


class _FakeRedisModule:
    def __init__(self, client):
        self._client = client

    def from_url(self, _url, decode_responses=True, **_kwargs):
        return self._client


def _make_provider(monkeypatch):
    fake = _FakeRedis()
    monkeypatch.setitem(sys.modules, "redis", _FakeRedisModule(fake))
    provider = RedisConversationMemory("redis://unused", "rag:compact:test")
    return provider


def _add_turns(provider, cid, n):
    for i in range(n):
        provider.append_turn(
            tenant_id="t1", subject="u1", project_id="p1",
            conversation_id=cid, role="user" if i % 2 == 0 else "assistant",
            content=f"turn content {i}",
        )


def test_compaction_triggers_after_threshold(monkeypatch):
    """After MEMORY_SUMMARY_TRIGGER_TURNS turns, compact_if_needed (force=True) produces a summary."""
    from config.settings import MEMORY_SUMMARY_TRIGGER_TURNS

    provider = _make_provider(monkeypatch)
    provider._llm_summarize = lambda turns, existing: "mock summary text"

    meta = provider.ensure_conversation(
        tenant_id="t1", subject="u1", project_id="p1", title="Compact test"
    )
    cid = meta.conversation_id

    _add_turns(provider, cid, MEMORY_SUMMARY_TRIGGER_TURNS)

    result = provider.compact_if_needed(
        tenant_id="t1", subject="u1", project_id="p1",
        conversation_id=cid, force=True,
    )
    assert result.text == "mock summary text"
    assert result.turns_compacted == MEMORY_SUMMARY_TRIGGER_TURNS


def test_post_compaction_summary_stored(monkeypatch):
    """After compaction the summary is persisted in meta and appears in build_context."""
    provider = _make_provider(monkeypatch)
    provider._llm_summarize = lambda turns, existing: "persisted summary"

    meta = provider.ensure_conversation(
        tenant_id="t1", subject="u1", project_id="p1", title="Persist test"
    )
    cid = meta.conversation_id
    _add_turns(provider, cid, 4)

    provider.compact_if_needed(
        tenant_id="t1", subject="u1", project_id="p1",
        conversation_id=cid, force=True,
    )

    ctx = provider.build_context(
        tenant_id="t1", subject="u1", project_id="p1",
        conversation_id=cid,
    )
    assert "persisted summary" in ctx.summary_text
    assert "persisted summary" in ctx.context_text


def test_compaction_not_triggered_below_threshold(monkeypatch):
    """compact_if_needed with force=False should not compact when count < threshold."""
    from config.settings import MEMORY_SUMMARY_TRIGGER_TURNS

    provider = _make_provider(monkeypatch)
    summarize_called = []
    provider._llm_summarize = lambda turns, existing: summarize_called.append(1) or "should not appear"

    meta = provider.ensure_conversation(
        tenant_id="t1", subject="u1", project_id="p1", title="No-compact test"
    )
    cid = meta.conversation_id
    # Add fewer turns than the trigger threshold
    _add_turns(provider, cid, max(1, MEMORY_SUMMARY_TRIGGER_TURNS - 1))

    provider.compact_if_needed(
        tenant_id="t1", subject="u1", project_id="p1",
        conversation_id=cid, force=False,
    )
    # Summarizer must NOT have been called
    assert summarize_called == []


def test_second_compaction_passes_existing_summary(monkeypatch):
    """A second forced compaction receives the first summary as context for the LLM call."""
    provider = _make_provider(monkeypatch)
    call_count = []

    def _mock_summarize(turns, existing):
        call_count.append(existing)
        return f"summary v{len(call_count)}"

    provider._llm_summarize = _mock_summarize

    meta = provider.ensure_conversation(
        tenant_id="t1", subject="u1", project_id="p1", title="Re-compact test"
    )
    cid = meta.conversation_id
    _add_turns(provider, cid, 4)

    # First compaction
    r1 = provider.compact_if_needed(
        tenant_id="t1", subject="u1", project_id="p1",
        conversation_id=cid, force=True,
    )
    assert r1.text == "summary v1"

    # Second compaction passes the existing summary
    r2 = provider.compact_if_needed(
        tenant_id="t1", subject="u1", project_id="p1",
        conversation_id=cid, force=True,
    )
    assert r2.text == "summary v2"
    # The existing summary from the first compaction was passed to the second call
    assert call_count[1] == "summary v1"
