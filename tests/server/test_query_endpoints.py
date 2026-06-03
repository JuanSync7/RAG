"""Orchestration + endpoint tests for ``server/routes/query.py``.

The pure helpers (``_sse``, ``_derive_title_from_query``, ``_autotitle_if_new``,
``_decode_page_bbox``, ``_source_refs``, ``_aggregate_stage_totals``) are covered
by ``tests/server/test_query_helpers.py`` and are NOT re-tested here.

This module covers the wiring that those tests do not touch:

- ``run_query``: the async non-stream orchestrator — its error branches (no
  Temporal client, RPCError 503, generic 500), the slot finally-release
  contract, the retrieval-mode payload flag, doc-id suppression / echo, and the
  memory turn-write guards (user turn always, assistant turn suppressed on
  block/flag).
- ``create_query_router``: every endpoint via ``TestClient`` —
  POST ``/query``, GET/POST/PATCH/DELETE conversation routes, the limit clamps,
  ``/metrics`` content-type passthrough, and the SSE ``/query/stream`` error and
  no-chunks paths.

Assertions use exact values and distinct per-field data so each one bites a
specific mutation.
"""
from __future__ import annotations

import asyncio
import json
import logging

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from temporalio.service import RPCError, RPCStatusCode

import server.routes.query as query_mod
from server.routes.query import create_query_router, run_query
from server.schemas import QueryRequest
from src.platform.memory.schemas import (
    ConversationMeta,
    ConversationSummary,
    ConversationTurn,
    MemoryContext,
)
from src.platform.security import Principal, authenticate_request


LOGGER = logging.getLogger("test")


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------
def _principal() -> Principal:
    return Principal(
        subject="user-1",
        tenant_id="tenant-a",
        roles=["query"],
        auth_type="none",
        project_id="proj-x",
    )


def _meta(
    *,
    conversation_id="conv-1",
    message_count=0,
    title="New conversation",
    relevant_doc_ids=None,
    ignored_doc_ids=None,
) -> ConversationMeta:
    return ConversationMeta(
        conversation_id=conversation_id,
        tenant_id="tenant-a",
        subject="user-1",
        project_id="proj-x",
        title=title,
        created_at_ms=111,
        updated_at_ms=222,
        message_count=message_count,
        relevant_doc_ids=list(relevant_doc_ids or []),
        ignored_doc_ids=list(ignored_doc_ids or []),
    )


class FakeMemory:
    """Recording double implementing exactly the provider surface used by the source."""

    def __init__(
        self,
        *,
        ensure_meta=None,
        seen_doc_ids=None,
        context=None,
        mark_meta=None,
        list_items=None,
        turns=None,
        delete_result=True,
        title_result="sentinel",
    ):
        self._ensure_meta = ensure_meta if ensure_meta is not None else _meta()
        self._seen_doc_ids = list(seen_doc_ids or [])
        self._context = context or MemoryContext(
            conversation_id="conv-1",
            summary_text="prior summary",
            recent_turns=[ConversationTurn(role="user", content="earlier", timestamp_ms=1)],
            context_text="CTX-TEXT",
        )
        self._mark_meta = mark_meta if mark_meta is not None else _meta()
        self._list_items = list_items if list_items is not None else [_meta()]
        self._turns = turns if turns is not None else [
            ConversationTurn(role="user", content="hi", timestamp_ms=7, query_id="q1")
        ]
        self._delete_result = delete_result
        # "sentinel" means: return the ensure_meta (a valid meta); None forces 404 path.
        self._title_result = self._ensure_meta if title_result == "sentinel" else title_result
        self.calls: list[tuple] = []

    def ensure_conversation(self, *, tenant_id, subject, project_id, conversation_id, title=""):
        self.calls.append(("ensure_conversation", dict(
            tenant_id=tenant_id, subject=subject, project_id=project_id,
            conversation_id=conversation_id, title=title)))
        return self._ensure_meta

    def update_conversation_title(self, *, tenant_id, subject, project_id, conversation_id, title):
        self.calls.append(("update_conversation_title", dict(
            tenant_id=tenant_id, subject=subject, project_id=project_id,
            conversation_id=conversation_id, title=title)))
        return self._title_result

    def get_seen_doc_ids(self, *, tenant_id, subject, project_id, conversation_id):
        self.calls.append(("get_seen_doc_ids", dict(
            tenant_id=tenant_id, subject=subject, project_id=project_id,
            conversation_id=conversation_id)))
        return list(self._seen_doc_ids)

    def build_context(self, *, tenant_id, subject, project_id, conversation_id, turn_window):
        self.calls.append(("build_context", dict(
            tenant_id=tenant_id, subject=subject, project_id=project_id,
            conversation_id=conversation_id, turn_window=turn_window)))
        return self._context

    def mark_retrieved(self, *, tenant_id, subject, project_id, conversation_id, doc_ids):
        self.calls.append(("mark_retrieved", dict(
            tenant_id=tenant_id, subject=subject, project_id=project_id,
            conversation_id=conversation_id, doc_ids=list(doc_ids))))
        return self._mark_meta

    def append_turn(self, *, tenant_id, subject, project_id, conversation_id,
                    role, content, query_id, sources=None):
        self.calls.append(("append_turn", dict(
            tenant_id=tenant_id, subject=subject, project_id=project_id,
            conversation_id=conversation_id, role=role, content=content,
            query_id=query_id, sources=list(sources or []))))

    def compact_if_needed(self, *, tenant_id, subject, project_id, conversation_id, force):
        self.calls.append(("compact_if_needed", dict(
            tenant_id=tenant_id, subject=subject, project_id=project_id,
            conversation_id=conversation_id, force=force)))
        return ConversationSummary(text="SUMMARY", updated_at_ms=999)

    def list_conversations(self, *, tenant_id, subject, project_id, limit):
        self.calls.append(("list_conversations", dict(
            tenant_id=tenant_id, subject=subject, project_id=project_id, limit=limit)))
        return list(self._list_items)

    def get_turns(self, *, tenant_id, subject, project_id, conversation_id, limit):
        self.calls.append(("get_turns", dict(
            tenant_id=tenant_id, subject=subject, project_id=project_id,
            conversation_id=conversation_id, limit=limit)))
        return list(self._turns)

    def delete_conversation(self, *, tenant_id, subject, project_id, conversation_id):
        self.calls.append(("delete_conversation", dict(
            tenant_id=tenant_id, subject=subject, project_id=project_id,
            conversation_id=conversation_id)))
        return self._delete_result

    # ---- assertion helpers -------------------------------------------------
    def names(self) -> list[str]:
        return [c[0] for c in self.calls]

    def args_for(self, name: str) -> list[dict]:
        return [c[1] for c in self.calls if c[0] == name]


class FakeTemporalClient:
    """Awaitable execute_workflow returning a canned result (or raising)."""

    def __init__(self, *, result=None, raises=None):
        self._result = result
        self._raises = raises
        self.calls: list[dict] = []

    async def execute_workflow(self, workflow_run, payload, *, id, task_queue):
        self.calls.append({"payload": payload, "id": id, "task_queue": task_queue})
        if self._raises is not None:
            raise self._raises
        return dict(self._result)


class Recorder:
    """A recording sync callable; optionally raises a preset exception."""

    def __init__(self, *, raises=None, returns=None):
        self.calls: list[tuple] = []
        self._raises = raises
        self._returns = returns

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self._raises is not None:
            raise self._raises
        return self._returns


class AsyncSlot:
    """Async acquire double returning a preset slot token; records calls."""

    def __init__(self, *, returns=True):
        self.calls: list = []
        self._returns = returns

    async def __call__(self, endpoint):
        self.calls.append(endpoint)
        return self._returns


def _ok_result(**overrides) -> dict:
    base = {
        "query": "q",
        "processed_query": "q",
        "query_confidence": 0.9,
        "action": "search",
        "results": [],
    }
    base.update(overrides)
    return base


# ===========================================================================
# run_query — direct async unit tests
# ===========================================================================
def test_run_query_no_temporal_client_raises_503_before_role_check(monkeypatch):
    """temporal_client=None short-circuits to 503; require_role is never reached."""
    fake_mem = FakeMemory()
    monkeypatch.setattr(query_mod, "get_conversation_memory", lambda: fake_mem)
    require_role = Recorder()

    with pytest.raises(HTTPException) as exc:
        asyncio.run(run_query(
            QueryRequest(query="hi"),
            _principal(),
            endpoint="/query",
            temporal_client=None,
            require_role=require_role,
            enforce_rate_limit=Recorder(),
            acquire_request_slot=AsyncSlot(),
            release_request_slot=Recorder(),
            logger=LOGGER,
        ))

    assert exc.value.status_code == 503
    assert require_role.calls == []
    assert fake_mem.calls == []


def test_run_query_happy_path_enriches_result_and_releases_slot(monkeypatch):
    """Happy path: gates called, workflow run with tenant/conv in payload, result enriched, slot released."""
    fake_mem = FakeMemory()
    monkeypatch.setattr(query_mod, "get_conversation_memory", lambda: fake_mem)
    temporal = FakeTemporalClient(result=_ok_result())
    require_role = Recorder()
    enforce = Recorder()
    acquire = AsyncSlot(returns=True)
    release = Recorder()

    resp = asyncio.run(run_query(
        QueryRequest(query="hi"),
        _principal(),
        endpoint="/query",
        temporal_client=temporal,
        require_role=require_role,
        enforce_rate_limit=enforce,
        acquire_request_slot=acquire,
        release_request_slot=release,
        logger=LOGGER,
    ))

    assert len(require_role.calls) == 1
    assert require_role.calls[0][0][1] == "query"
    assert enforce.calls[0][0][1] == "/query"
    assert acquire.calls == ["/query"]
    payload = temporal.calls[0]["payload"]
    assert payload["tenant_id"] == "tenant-a"
    assert payload["conversation_id"] == "conv-1"
    assert temporal.calls[0]["id"].startswith("rag-query-")
    assert resp.workflow_id.startswith("rag-query-")
    assert resp.conversation_id == "conv-1"
    assert resp.latency_ms is not None
    # finally-release fired with the acquired token.
    assert release.calls == [((True,), {})]


def test_run_query_rpc_error_maps_to_503_and_releases_slot(monkeypatch):
    """An RPCError from the workflow becomes a 503 and the slot is still released (finally)."""
    fake_mem = FakeMemory()
    monkeypatch.setattr(query_mod, "get_conversation_memory", lambda: fake_mem)
    temporal = FakeTemporalClient(raises=RPCError("boom", RPCStatusCode.UNAVAILABLE, b""))
    release = Recorder()

    with pytest.raises(HTTPException) as exc:
        asyncio.run(run_query(
            QueryRequest(query="hi"),
            _principal(),
            endpoint="/query",
            temporal_client=temporal,
            require_role=Recorder(),
            enforce_rate_limit=Recorder(),
            acquire_request_slot=AsyncSlot(returns=True),
            release_request_slot=release,
            logger=LOGGER,
        ))

    assert exc.value.status_code == 503
    assert release.calls == [((True,), {})]


def test_run_query_generic_error_maps_to_500_and_releases_slot(monkeypatch):
    """A non-RPC exception from the workflow becomes a 500 and the slot is still released."""
    fake_mem = FakeMemory()
    monkeypatch.setattr(query_mod, "get_conversation_memory", lambda: fake_mem)
    temporal = FakeTemporalClient(raises=RuntimeError("kaput"))
    release = Recorder()

    with pytest.raises(HTTPException) as exc:
        asyncio.run(run_query(
            QueryRequest(query="hi"),
            _principal(),
            endpoint="/query",
            temporal_client=temporal,
            require_role=Recorder(),
            enforce_rate_limit=Recorder(),
            acquire_request_slot=AsyncSlot(returns=True),
            release_request_slot=release,
            logger=LOGGER,
        ))

    assert exc.value.status_code == 500
    assert release.calls == [((True,), {})]


def test_run_query_retrieval_mode_sets_skip_generation(monkeypatch):
    """mode='retrieval' flips skip_generation=True in the workflow payload."""
    fake_mem = FakeMemory()
    monkeypatch.setattr(query_mod, "get_conversation_memory", lambda: fake_mem)
    temporal = FakeTemporalClient(result=_ok_result())

    asyncio.run(run_query(
        QueryRequest(query="hi", mode="retrieval"),
        _principal(),
        endpoint="/query",
        temporal_client=temporal,
        require_role=Recorder(),
        enforce_rate_limit=Recorder(),
        acquire_request_slot=AsyncSlot(),
        release_request_slot=Recorder(),
        logger=LOGGER,
    ))

    assert temporal.calls[0]["payload"]["skip_generation"] is True


def test_run_query_query_mode_omits_skip_generation(monkeypatch):
    """mode='query' (default) leaves skip_generation out of the payload (gives the flag teeth)."""
    fake_mem = FakeMemory()
    monkeypatch.setattr(query_mod, "get_conversation_memory", lambda: fake_mem)
    temporal = FakeTemporalClient(result=_ok_result())

    asyncio.run(run_query(
        QueryRequest(query="hi", mode="query"),
        _principal(),
        endpoint="/query",
        temporal_client=temporal,
        require_role=Recorder(),
        enforce_rate_limit=Recorder(),
        acquire_request_slot=AsyncSlot(),
        release_request_slot=Recorder(),
        logger=LOGGER,
    ))

    assert "skip_generation" not in temporal.calls[0]["payload"]


def test_run_query_seen_doc_ids_become_ignored_in_payload(monkeypatch):
    """Non-empty get_seen_doc_ids populates payload['ignored_doc_ids']."""
    fake_mem = FakeMemory(seen_doc_ids=["d-seen-1", "d-seen-2"])
    monkeypatch.setattr(query_mod, "get_conversation_memory", lambda: fake_mem)
    temporal = FakeTemporalClient(result=_ok_result())

    asyncio.run(run_query(
        QueryRequest(query="hi"),
        _principal(),
        endpoint="/query",
        temporal_client=temporal,
        require_role=Recorder(),
        enforce_rate_limit=Recorder(),
        acquire_request_slot=AsyncSlot(),
        release_request_slot=Recorder(),
        logger=LOGGER,
    ))

    assert temporal.calls[0]["payload"]["ignored_doc_ids"] == ["d-seen-1", "d-seen-2"]


def test_run_query_no_seen_doc_ids_omits_ignored_in_payload(monkeypatch):
    """Empty get_seen_doc_ids leaves ignored_doc_ids out of the payload."""
    fake_mem = FakeMemory(seen_doc_ids=[])
    monkeypatch.setattr(query_mod, "get_conversation_memory", lambda: fake_mem)
    temporal = FakeTemporalClient(result=_ok_result())

    asyncio.run(run_query(
        QueryRequest(query="hi"),
        _principal(),
        endpoint="/query",
        temporal_client=temporal,
        require_role=Recorder(),
        enforce_rate_limit=Recorder(),
        acquire_request_slot=AsyncSlot(),
        release_request_slot=Recorder(),
        logger=LOGGER,
    ))

    assert "ignored_doc_ids" not in temporal.calls[0]["payload"]


def test_run_query_memory_enabled_builds_context_and_writes_user_turn(monkeypatch):
    """memory_enabled: build_context feeds the payload and a user turn is appended."""
    fake_mem = FakeMemory()
    monkeypatch.setattr(query_mod, "get_conversation_memory", lambda: fake_mem)
    temporal = FakeTemporalClient(result=_ok_result(generated_answer="the answer"))

    asyncio.run(run_query(
        QueryRequest(query="my question", memory_enabled=True),
        _principal(),
        endpoint="/query",
        temporal_client=temporal,
        require_role=Recorder(),
        enforce_rate_limit=Recorder(),
        acquire_request_slot=AsyncSlot(),
        release_request_slot=Recorder(),
        logger=LOGGER,
    ))

    payload = temporal.calls[0]["payload"]
    assert payload["memory_context"] == "CTX-TEXT"
    assert payload["memory_recent_turns"] == [
        {"role": "user", "content": "earlier", "timestamp_ms": 1, "query_id": "", "sources": []}
    ]
    assert "build_context" in fake_mem.names()
    turns = fake_mem.args_for("append_turn")
    roles = [t["role"] for t in turns]
    assert "user" in roles
    user_turn = next(t for t in turns if t["role"] == "user")
    assert user_turn["content"] == "my question"


def test_run_query_block_action_suppresses_assistant_turn(monkeypatch):
    """post_guardrail_action='block' suppresses the assistant turn but keeps the user turn."""
    fake_mem = FakeMemory()
    monkeypatch.setattr(query_mod, "get_conversation_memory", lambda: fake_mem)
    temporal = FakeTemporalClient(
        result=_ok_result(generated_answer="blocked content", post_guardrail_action="block")
    )

    asyncio.run(run_query(
        QueryRequest(query="my question", memory_enabled=True),
        _principal(),
        endpoint="/query",
        temporal_client=temporal,
        require_role=Recorder(),
        enforce_rate_limit=Recorder(),
        acquire_request_slot=AsyncSlot(),
        release_request_slot=Recorder(),
        logger=LOGGER,
    ))

    roles = [t["role"] for t in fake_mem.args_for("append_turn")]
    assert "user" in roles
    assert "assistant" not in roles


def test_run_query_clean_action_writes_assistant_turn(monkeypatch):
    """A non-block/flag action writes the assistant turn (one-below partner for the guard)."""
    fake_mem = FakeMemory()
    monkeypatch.setattr(query_mod, "get_conversation_memory", lambda: fake_mem)
    temporal = FakeTemporalClient(
        result=_ok_result(generated_answer="the answer", post_guardrail_action="allow")
    )

    asyncio.run(run_query(
        QueryRequest(query="my question", memory_enabled=True),
        _principal(),
        endpoint="/query",
        temporal_client=temporal,
        require_role=Recorder(),
        enforce_rate_limit=Recorder(),
        acquire_request_slot=AsyncSlot(),
        release_request_slot=Recorder(),
        logger=LOGGER,
    ))

    turns = fake_mem.args_for("append_turn")
    roles = [t["role"] for t in turns]
    assert "assistant" in roles
    assistant_turn = next(t for t in turns if t["role"] == "assistant")
    assert assistant_turn["content"] == "the answer"


def test_run_query_doc_state_echo_from_mark_retrieved_meta(monkeypatch):
    """relevant/ignored/seen doc-id lists are echoed (deduped union) from mark_retrieved's meta."""
    mark_meta = _meta(relevant_doc_ids=["r1", "r2"], ignored_doc_ids=["r2", "i1"])
    fake_mem = FakeMemory(mark_meta=mark_meta)
    monkeypatch.setattr(query_mod, "get_conversation_memory", lambda: fake_mem)
    temporal = FakeTemporalClient(result=_ok_result())

    resp = asyncio.run(run_query(
        QueryRequest(query="hi"),
        _principal(),
        endpoint="/query",
        temporal_client=temporal,
        require_role=Recorder(),
        enforce_rate_limit=Recorder(),
        acquire_request_slot=AsyncSlot(),
        release_request_slot=Recorder(),
        logger=LOGGER,
    ))

    assert resp.relevant_doc_ids == ["r1", "r2"]
    assert resp.ignored_doc_ids == ["r2", "i1"]
    # Union with order preserved, duplicate "r2" collapsed.
    assert resp.seen_doc_ids == ["r1", "r2", "i1"]


def test_run_query_memory_disabled_skips_context_and_turns(monkeypatch):
    """memory_enabled=False: no build_context call and no turns appended."""
    fake_mem = FakeMemory()
    monkeypatch.setattr(query_mod, "get_conversation_memory", lambda: fake_mem)
    temporal = FakeTemporalClient(result=_ok_result(generated_answer="answer"))

    asyncio.run(run_query(
        QueryRequest(query="hi", memory_enabled=False),
        _principal(),
        endpoint="/query",
        temporal_client=temporal,
        require_role=Recorder(),
        enforce_rate_limit=Recorder(),
        acquire_request_slot=AsyncSlot(),
        release_request_slot=Recorder(),
        logger=LOGGER,
    ))

    names = fake_mem.names()
    assert "build_context" not in names
    assert "append_turn" not in names
    assert "memory_context" not in temporal.calls[0]["payload"]


# ===========================================================================
# Endpoints via TestClient
# ===========================================================================
def _build_app(
    *,
    temporal_client,
    emit=None,
    require_role=None,
    enforce=None,
    acquire=None,
    release=None,
):
    app = FastAPI()
    router = create_query_router(
        get_temporal_client=lambda: temporal_client,
        require_role=require_role or Recorder(),
        enforce_rate_limit=enforce or Recorder(),
        acquire_request_slot=acquire or AsyncSlot(),
        release_request_slot=release or Recorder(),
        emit_stream_observability=emit or Recorder(),
        logger=LOGGER,
    )
    app.include_router(router)
    app.dependency_overrides[authenticate_request] = _principal
    return app


def test_endpoint_post_query_happy(monkeypatch):
    """POST /query returns 200 with an enriched body (workflow_id present)."""
    fake_mem = FakeMemory()
    monkeypatch.setattr(query_mod, "get_conversation_memory", lambda: fake_mem)
    temporal = FakeTemporalClient(result=_ok_result())
    client = TestClient(_build_app(temporal_client=temporal))

    resp = client.post("/query", json={"query": "hello"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["workflow_id"].startswith("rag-query-")
    assert body["conversation_id"] == "conv-1"


def test_endpoint_post_query_no_temporal_client_503(monkeypatch):
    """POST /query with get_temporal_client()->None returns 503."""
    fake_mem = FakeMemory()
    monkeypatch.setattr(query_mod, "get_conversation_memory", lambda: fake_mem)
    client = TestClient(_build_app(temporal_client=None), raise_server_exceptions=False)

    resp = client.post("/query", json={"query": "hello"})

    assert resp.status_code == 503


def test_endpoint_list_conversations_maps_and_clamps_low(monkeypatch):
    """GET /conversations maps metas to responses; limit=0 is clamped up to 1."""
    fake_mem = FakeMemory(list_items=[_meta(conversation_id="c-A"), _meta(conversation_id="c-B")])
    monkeypatch.setattr(query_mod, "get_conversation_memory", lambda: fake_mem)
    client = TestClient(_build_app(temporal_client=object()))

    resp = client.get("/conversations", params={"limit": 0})

    assert resp.status_code == 200
    body = resp.json()
    assert [item["conversation_id"] for item in body] == ["c-A", "c-B"]
    assert fake_mem.args_for("list_conversations")[0]["limit"] == 1


def test_endpoint_list_conversations_clamps_high(monkeypatch):
    """GET /conversations clamps an over-large limit down to 100."""
    fake_mem = FakeMemory(list_items=[])
    monkeypatch.setattr(query_mod, "get_conversation_memory", lambda: fake_mem)
    client = TestClient(_build_app(temporal_client=object()))

    resp = client.get("/conversations", params={"limit": 999})

    assert resp.status_code == 200
    assert fake_mem.args_for("list_conversations")[0]["limit"] == 100


def test_endpoint_new_conversation_passes_title_and_id(monkeypatch):
    """POST /conversations/new forwards the body title + conversation_id to ensure_conversation."""
    fake_mem = FakeMemory(ensure_meta=_meta(conversation_id="brand-new"))
    monkeypatch.setattr(query_mod, "get_conversation_memory", lambda: fake_mem)
    client = TestClient(_build_app(temporal_client=object()))

    resp = client.post(
        "/conversations/new",
        json={"title": "My Title", "conversation_id": "brand-new"},
    )

    assert resp.status_code == 200
    assert resp.json()["conversation_id"] == "brand-new"
    args = fake_mem.args_for("ensure_conversation")[0]
    assert args["title"] == "My Title"
    assert args["conversation_id"] == "brand-new"


def test_endpoint_history_maps_turns_and_clamps_high(monkeypatch):
    """GET /conversations/{id}/history maps get_turns; limit clamps to 300."""
    turns = [ConversationTurn(role="user", content="u1", timestamp_ms=10, query_id="qq")]
    fake_mem = FakeMemory(turns=turns)
    monkeypatch.setattr(query_mod, "get_conversation_memory", lambda: fake_mem)
    client = TestClient(_build_app(temporal_client=object()))

    resp = client.get("/conversations/conv-1/history", params={"limit": 5000})

    assert resp.status_code == 200
    body = resp.json()
    assert body["conversation_id"] == "conv-1"
    assert body["turns"][0]["content"] == "u1"
    assert fake_mem.args_for("get_turns")[0]["limit"] == 300


def test_endpoint_history_clamps_low(monkeypatch):
    """GET history clamps limit=0 up to 1."""
    fake_mem = FakeMemory(turns=[])
    monkeypatch.setattr(query_mod, "get_conversation_memory", lambda: fake_mem)
    client = TestClient(_build_app(temporal_client=object()))

    resp = client.get("/conversations/conv-1/history", params={"limit": 0})

    assert resp.status_code == 200
    assert fake_mem.args_for("get_turns")[0]["limit"] == 1


def test_endpoint_compact_returns_summary(monkeypatch):
    """POST /conversations/{id}/compact returns the summary triple from compact_if_needed."""
    fake_mem = FakeMemory()
    monkeypatch.setattr(query_mod, "get_conversation_memory", lambda: fake_mem)
    client = TestClient(_build_app(temporal_client=object()))

    resp = client.post("/conversations/path-id/compact")

    assert resp.status_code == 200
    body = resp.json()
    assert body == {"conversation_id": "path-id", "summary": "SUMMARY", "updated_at_ms": 999}


def test_endpoint_compact_body_overrides_path_id(monkeypatch):
    """When a body provides conversation_id, it overrides the path param as the target."""
    fake_mem = FakeMemory()
    monkeypatch.setattr(query_mod, "get_conversation_memory", lambda: fake_mem)
    client = TestClient(_build_app(temporal_client=object()))

    resp = client.post("/conversations/path-id/compact", json={"conversation_id": "body-id"})

    assert resp.status_code == 200
    assert resp.json()["conversation_id"] == "body-id"
    assert fake_mem.args_for("compact_if_needed")[0]["conversation_id"] == "body-id"


def test_endpoint_delete_returns_flag(monkeypatch):
    """DELETE /conversations/{id} echoes the boolean delete result."""
    fake_mem = FakeMemory(delete_result=True)
    monkeypatch.setattr(query_mod, "get_conversation_memory", lambda: fake_mem)
    client = TestClient(_build_app(temporal_client=object()))

    resp = client.delete("/conversations/conv-9")

    assert resp.status_code == 200
    assert resp.json() == {"conversation_id": "conv-9", "deleted": True}


def test_endpoint_patch_title_happy(monkeypatch):
    """PATCH /conversations/{id} returns the updated meta on success."""
    fake_mem = FakeMemory(ensure_meta=_meta(conversation_id="conv-7", title="Renamed"))
    monkeypatch.setattr(query_mod, "get_conversation_memory", lambda: fake_mem)
    client = TestClient(_build_app(temporal_client=object()))

    resp = client.patch("/conversations/conv-7", json={"title": "Renamed"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["conversation_id"] == "conv-7"
    assert fake_mem.args_for("update_conversation_title")[0]["title"] == "Renamed"


def test_endpoint_patch_title_missing_returns_404(monkeypatch):
    """PATCH returns 404 when update_conversation_title returns None."""
    fake_mem = FakeMemory(title_result=None)
    monkeypatch.setattr(query_mod, "get_conversation_memory", lambda: fake_mem)
    client = TestClient(_build_app(temporal_client=object()), raise_server_exceptions=False)

    resp = client.patch("/conversations/conv-x", json={"title": "Nope"})

    assert resp.status_code == 404


def test_endpoint_metrics_passthrough(monkeypatch):
    """GET /metrics returns the render_metrics payload and content-type verbatim."""
    # Use a non-text/* media type so Starlette doesn't append a charset suffix,
    # letting us assert the content-type is passed through verbatim.
    monkeypatch.setattr(
        query_mod,
        "render_metrics",
        lambda: (b"# HELP custom_metric\n", "application/openmetrics-text; version=9.9"),
    )
    client = TestClient(_build_app(temporal_client=object()))

    resp = client.get("/metrics")

    assert resp.status_code == 200
    assert resp.content == b"# HELP custom_metric\n"
    assert resp.headers["content-type"] == "application/openmetrics-text; version=9.9"


# ===========================================================================
# /query/stream (SSE)
# ===========================================================================
def _parse_sse(text: str) -> list[tuple[str, dict]]:
    """Parse an SSE body into a list of (event, data-dict) frames."""
    frames = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        event = None
        data = None
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line[len("event: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
        if event is not None:
            frames.append((event, data))
    return frames


def test_stream_no_temporal_client_503(monkeypatch):
    """/query/stream with get_temporal_client()->None returns 503 before streaming."""
    fake_mem = FakeMemory()
    monkeypatch.setattr(query_mod, "get_conversation_memory", lambda: fake_mem)
    client = TestClient(_build_app(temporal_client=None), raise_server_exceptions=False)

    resp = client.post("/query/stream", json={"query": "hi"})

    assert resp.status_code == 503


def test_stream_rpc_error_emits_error_frame_and_observes_error(monkeypatch):
    """An RPCError during streaming emits an 'error' SSE frame and reports outcome='error'."""
    fake_mem = FakeMemory()
    monkeypatch.setattr(query_mod, "get_conversation_memory", lambda: fake_mem)
    temporal = FakeTemporalClient(raises=RPCError("down", RPCStatusCode.UNAVAILABLE, b""))
    emit = Recorder()
    release = Recorder()
    client = TestClient(_build_app(temporal_client=temporal, emit=emit, release=release))

    resp = client.post("/query/stream", json={"query": "hi"})

    assert resp.status_code == 200
    frames = _parse_sse(resp.text)
    events = [e for e, _ in frames]
    assert "error" in events
    assert emit.calls, "emit_stream_observability was not called"
    assert emit.calls[0][1]["outcome"] == "error"
    # finally still releases the slot.
    assert release.calls and release.calls[0][0][0] is True


def test_stream_no_chunks_emits_retrieval_then_done_no_tokens(monkeypatch):
    """A non-'search' action yields a retrieval frame then a done frame (0 tokens), no token frames."""
    fake_mem = FakeMemory()
    monkeypatch.setattr(query_mod, "get_conversation_memory", lambda: fake_mem)
    temporal = FakeTemporalClient(
        result=_ok_result(action="clarify", results=[], clarification_message="please clarify")
    )
    emit = Recorder()
    release = Recorder()
    client = TestClient(_build_app(temporal_client=temporal, emit=emit, release=release))

    resp = client.post("/query/stream", json={"query": "hi"})

    assert resp.status_code == 200
    frames = _parse_sse(resp.text)
    events = [e for e, _ in frames]
    assert events[0] == "retrieval"
    assert "done" in events
    assert "token" not in events
    done_data = next(d for e, d in frames if e == "done")
    assert done_data["token_count"] == 0
    assert release.calls and release.calls[0][0][0] is True
    assert emit.calls[0][1]["outcome"] == "completed"


def test_stream_non_search_action_with_chunks_still_early_returns(monkeypatch):
    """The early-return fires on action!='search' even when chunks are present.

    This discriminates the ``!=`` clause specifically: with non-empty results the
    ``not chunks`` guard is False, so only the action comparison can short-circuit
    to the no-generation done frame. If the comparison were flipped, generation
    would run and emit token frames — so we patch the LLM to a sentinel token and
    assert NO token frames appear and the done frame reports 0 tokens.
    """
    fake_mem = FakeMemory()
    monkeypatch.setattr(query_mod, "get_conversation_memory", lambda: fake_mem)

    class _Provider:
        def generate_stream(self, messages, **kwargs):
            yield "LEAK-TOKEN"

    import src.platform.llm as llm_mod
    import src.retrieval.generation.nodes as nodes_mod
    monkeypatch.setattr(llm_mod, "get_llm_provider", lambda: _Provider())
    monkeypatch.setattr(
        nodes_mod.OllamaGenerator, "build_messages",
        staticmethod(lambda **kwargs: [{"role": "user", "content": "x"}]),
    )

    temporal = FakeTemporalClient(
        result=_ok_result(
            action="clarify",
            results=[{"text": "chunk text", "score": 0.5, "metadata": {"document_id": "d9"}}],
            clarification_message="please clarify",
        )
    )
    emit = Recorder()
    client = TestClient(_build_app(temporal_client=temporal, emit=emit))

    resp = client.post("/query/stream", json={"query": "hi"})

    assert resp.status_code == 200
    frames = _parse_sse(resp.text)
    events = [e for e, _ in frames]
    assert "token" not in events
    done_data = next(d for e, d in frames if e == "done")
    assert done_data["token_count"] == 0
