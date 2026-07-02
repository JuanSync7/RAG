# @summary
# Endpoint + wiring tests for the turn-loop branch of the query routes
# (server/routes/query.py + server/turn_loop_runner.py): flag resolution
# precedence, SSE event ordering (loop events before token replay, done
# last), the clarify terminal shape, the non-stream metadata.turn_loop.trace,
# memory write-back kwargs (preview-capped chunk refs, action records,
# record_doc_studied), and non-loop path untouchedness when the flag is off.
# Exports: (tests)
# Deps: pytest, fastapi.testclient, server.routes.query,
#       server.turn_loop_runner, src.retrieval.pipeline.turn_loop
# @end-summary
"""Turn-loop endpoint tests.

Fakes follow the ``tests/server/test_query_endpoints.py`` pattern: a
recording FakeMemory (extended with the turn-context provider surface), a
fake Temporal client, and a fake ``run_turn_loop`` injected through the
runner's ``_get_run_turn_loop`` seam — the loop internals (Track A) are not
under test here, only the server wiring around them.
"""
from __future__ import annotations

import json
import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import server.routes.query as query_mod
import server.turn_loop_runner as runner_mod
from server.routes.query import create_query_router
from src.platform.memory.schemas import (
    ConversationMeta,
    ConversationSummary,
    MemoryContext,
)
from src.platform.security import Principal, authenticate_request
from src.retrieval.pipeline.turn_loop import (
    ClarificationOut,
    EvidenceChunk,
    TurnEvent,
    TurnEventType,
    TurnLoopResult,
)

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


def _meta(*, conversation_id="conv-1", message_count=1) -> ConversationMeta:
    # message_count=1 keeps the autotitle path quiet by default.
    return ConversationMeta(
        conversation_id=conversation_id,
        tenant_id="tenant-a",
        subject="user-1",
        project_id="proj-x",
        title="t",
        created_at_ms=1,
        updated_at_ms=2,
        message_count=message_count,
    )


class FakeMemory:
    """Recording double covering the loop-path provider surface."""

    def __init__(self, *, structured: dict | None = None):
        self._structured = dict(structured or {})
        self.calls: list[tuple] = []

    def ensure_conversation(self, *, tenant_id, subject, project_id, conversation_id, title=""):
        self.calls.append(("ensure_conversation", dict(conversation_id=conversation_id)))
        return _meta()

    def update_conversation_title(self, **kwargs):
        self.calls.append(("update_conversation_title", kwargs))
        return _meta()

    def get_seen_doc_ids(self, **kwargs):
        self.calls.append(("get_seen_doc_ids", kwargs))
        return []

    def build_context(self, *, tenant_id, subject, project_id, conversation_id, turn_window):
        self.calls.append(("build_context", dict(
            tenant_id=tenant_id, conversation_id=conversation_id, turn_window=turn_window)))
        return MemoryContext(
            conversation_id=conversation_id,
            summary_text="s",
            recent_turns=[],
            context_text="CTX",
            structured=dict(self._structured),
        )

    def mark_retrieved(self, **kwargs):
        self.calls.append(("mark_retrieved", kwargs))
        return _meta()

    def append_turn(self, *, tenant_id, subject, project_id, conversation_id, role,
                    content, query_id, sources=None, actions=None, chunk_refs=None,
                    answer_confidence=None, clarification=None):
        self.calls.append(("append_turn", dict(
            role=role, content=content, query_id=query_id,
            sources=list(sources or []), actions=list(actions or []),
            chunk_refs=list(chunk_refs or []),
            answer_confidence=answer_confidence, clarification=clarification)))

    def record_doc_studied(self, *, tenant_id, subject, project_id, conversation_id, entry):
        self.calls.append(("record_doc_studied", dict(entry=dict(entry))))
        return _meta()

    def compact_if_needed(self, *, tenant_id, subject, project_id, conversation_id, force):
        self.calls.append(("compact_if_needed", dict(force=force)))
        return ConversationSummary(text="S", updated_at_ms=9)

    # ---- assertion helpers -------------------------------------------------
    def names(self) -> list[str]:
        return [c[0] for c in self.calls]

    def args_for(self, name: str) -> list[dict]:
        return [c[1] for c in self.calls if c[0] == name]


class FakeTemporalClient:
    """Awaitable execute_workflow double (non-loop RAGQueryWorkflow path)."""

    def __init__(self, *, result=None):
        self._result = result or {
            "query": "q", "processed_query": "q", "query_confidence": 0.9,
            "action": "search", "results": [],
        }
        self.calls: list[dict] = []

    async def execute_workflow(self, workflow_run, payload, *, id, task_queue):
        self.calls.append({"payload": payload, "id": id, "task_queue": task_queue})
        return dict(self._result)


class Recorder:
    def __init__(self):
        self.calls: list[tuple] = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class AsyncSlot:
    async def __call__(self, endpoint):
        return True


def _chunk(
    *,
    chunk_id="uuid-1",
    document_id="doc-1",
    source_key="minio/doc-1.md",
    source="Doc One.pdf",
    heading="Overview",
    text="evidence text",
    score=0.83,
    start=10,
    end=90,
) -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        source_key=source_key,
        source=source,
        heading=heading,
        text=text,
        score=score,
        refactored_char_start=start,
        refactored_char_end=end,
        provenance=EvidenceChunk.PROVENANCE_RETRIEVE,
        round_added=0,
    )


def _events_to_trace(events: list[tuple[str, dict]]) -> list[TurnEvent]:
    return [TurnEvent.now(etype, dict(payload)) for etype, payload in events]


def _make_fake_loop(events: list[tuple[str, dict]], result: TurnLoopResult):
    """Async run_turn_loop double: emits events through deps, returns result."""

    async def fake_loop(query, context, deps, budget):
        fake_loop.calls.append({"query": query, "context": context, "budget": budget})
        for etype, payload in events:
            await deps.emit(TurnEvent.now(etype, dict(payload)))
        return result

    fake_loop.calls = []
    return fake_loop


ANSWER_EVENTS: list[tuple[str, dict]] = [
    (TurnEventType.TURN_ACTION, {"index": 1, "action": "RETRIEVE", "reason": "gap", "confidence": 0.8}),
    (TurnEventType.HYDE_QUERY, {"round": 1, "hypothetical_answer": "hypo", "search_terms": ["a"]}),
    (TurnEventType.RETRIEVE_RESULT, {"round": 1, "added": 5, "dup": 1, "pool_size": 5, "top": []}),
    (TurnEventType.JUDGE_VERDICT, {"round": 1, "kept": 4, "sufficient": True, "confidence": 0.72,
                                   "missing_information": ""}),
    (TurnEventType.LLM_CALL, {"alias": "controller", "purpose": "controller", "ms": 12}),
    (TurnEventType.TURN_ACTION, {"index": 2, "action": "ANSWER", "reason": "enough", "confidence": 0.9}),
    (TurnEventType.DRAFT, {"attempt": 1, "kind": "reasoning", "text_delta": "let me think"}),
    (TurnEventType.DRAFT, {"attempt": 1, "kind": "content", "text_delta": "draft text"}),
    (TurnEventType.LLM_CALL, {"alias": "default", "purpose": "draft", "ms": 30}),
    (TurnEventType.GATE, {"attempt": 1, "score": 0.71, "threshold": 0.62, "passed": True,
                          "weakest": "citation"}),
]


def _answered_result(*, pool=None, events=ANSWER_EVENTS) -> TurnLoopResult:
    return TurnLoopResult(
        answer="final answer here",
        action=TurnLoopResult.ACTION_ANSWERED,
        confidence=0.71,
        pool=list(pool if pool is not None else [_chunk()]),
        trace=_events_to_trace(events),
        stop_reason="gate_passed",
        llm_calls=2,
        actions_taken=2,
        elapsed_ms=1234,
    )


CLARIFY_EVENTS: list[tuple[str, dict]] = [
    (TurnEventType.TURN_ACTION, {"index": 1, "action": "CLARIFY", "reason": "ambiguous", "confidence": 0.4}),
    (TurnEventType.CLARIFY, {"question": "Which project?", "hints": ["Alpha", "Beta"],
                             "scoping_questions": ["Do you mean the flow for Alpha?"]}),
]


def _clarify_result() -> TurnLoopResult:
    return TurnLoopResult(
        answer="",
        action=TurnLoopResult.ACTION_ASK_USER,
        clarification=ClarificationOut(
            question="Which project?",
            hints=["Alpha", "Beta"],
            scoping_questions=["Do you mean the flow for Alpha?"],
        ),
        confidence=0.0,
        pool=[],
        trace=_events_to_trace(CLARIFY_EVENTS),
        stop_reason="clarify",
        llm_calls=2,
        actions_taken=1,
        elapsed_ms=456,
    )


@pytest.fixture()
def loop_env(monkeypatch):
    """Wire the runner seams to fakes; return a mutable env dict."""
    env: dict = {}

    def install(events, result, *, memory=None, temporal=None):
        fake_loop = _make_fake_loop(events, result)
        env["loop"] = fake_loop
        env["memory"] = memory or FakeMemory()
        env["temporal"] = temporal or FakeTemporalClient()
        monkeypatch.setattr(runner_mod, "_get_run_turn_loop", lambda: fake_loop)
        monkeypatch.setattr(runner_mod, "_get_llm_provider", lambda: object())
        # Pre-loop safety: pass-through (the real gates are unit-tested below).
        monkeypatch.setattr(
            runner_mod,
            "prepare_turn_input",
            lambda query, tenant_id: runner_mod.PreparedTurnInput(query=query),
        )
        monkeypatch.setattr(query_mod, "get_conversation_memory", lambda: env["memory"])
        return env

    return install


def _build_app(*, temporal_client, emit=None, db_client=None):
    app = FastAPI()
    router = create_query_router(
        get_temporal_client=lambda: temporal_client,
        require_role=Recorder(),
        enforce_rate_limit=Recorder(),
        acquire_request_slot=AsyncSlot(),
        release_request_slot=Recorder(),
        emit_stream_observability=emit or Recorder(),
        logger=LOGGER,
        db_client=db_client,
    )
    app.include_router(router)
    app.dependency_overrides[authenticate_request] = _principal
    return app


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


# ===========================================================================
# Flag resolution — request override beats env default
# ===========================================================================
def test_request_true_overrides_env_false(loop_env, monkeypatch):
    """turn_loop=true forces the loop path even with RAG_TURN_LOOP_ENABLED=false."""
    import config.settings as settings_mod
    monkeypatch.setattr(settings_mod, "RAG_TURN_LOOP_ENABLED", False, raising=False)
    env = loop_env(ANSWER_EVENTS, _answered_result())
    client = TestClient(_build_app(temporal_client=env["temporal"]))

    resp = client.post("/query", json={"query": "hello", "turn_loop": True})

    assert resp.status_code == 200
    assert len(env["loop"].calls) == 1
    # RAGQueryWorkflow was never dispatched.
    assert env["temporal"].calls == []


def test_request_false_overrides_env_true(loop_env, monkeypatch):
    """turn_loop=false forces the workflow path even with the env default on."""
    import config.settings as settings_mod
    monkeypatch.setattr(settings_mod, "RAG_TURN_LOOP_ENABLED", True, raising=False)
    env = loop_env(ANSWER_EVENTS, _answered_result())
    client = TestClient(_build_app(temporal_client=env["temporal"]))

    resp = client.post("/query", json={"query": "hello", "turn_loop": False})

    assert resp.status_code == 200
    assert env["loop"].calls == []
    assert len(env["temporal"].calls) == 1


def test_env_default_true_applies_when_request_omits_flag(loop_env, monkeypatch):
    """turn_loop=None falls back to the RAG_TURN_LOOP_ENABLED env default."""
    import config.settings as settings_mod
    monkeypatch.setattr(settings_mod, "RAG_TURN_LOOP_ENABLED", True, raising=False)
    env = loop_env(ANSWER_EVENTS, _answered_result())
    client = TestClient(_build_app(temporal_client=env["temporal"]))

    resp = client.post("/query", json={"query": "hello"})

    assert resp.status_code == 200
    assert len(env["loop"].calls) == 1
    assert env["temporal"].calls == []


# ===========================================================================
# SSE stream — ordering, token replay, done payload
# ===========================================================================
def test_stream_event_ordering_events_then_tokens_done_last(loop_env):
    """Loop events stream first, reasoning precedes the token replay, done is last."""
    env = loop_env(ANSWER_EVENTS, _answered_result())
    client = TestClient(_build_app(temporal_client=env["temporal"]))

    resp = client.post("/query/stream", json={"query": "hello", "turn_loop": True})

    assert resp.status_code == 200
    frames = _parse_sse(resp.text)
    events = [e for e, _ in frames]
    # Every emitted loop event appears as its own SSE event, in order.
    loop_event_names = [e for e, _ in ANSWER_EVENTS]
    assert [e for e in events if e in TurnEventType.ALL] == loop_event_names
    # All loop events precede the retrieval frame and the token replay.
    first_token = events.index("token")
    last_loop_event = max(i for i, e in enumerate(events) if e in TurnEventType.ALL)
    assert last_loop_event < events.index("retrieval") < first_token
    # Captured draft reasoning is re-emitted BEFORE the token replay.
    assert events.index("reasoning") < first_token
    reasoning = next(d for e, d in frames if e == "reasoning")
    assert reasoning["text"] == "let me think"
    # The accepted answer is replayed as standard token events.
    answer = "".join(d["token"] for e, d in frames if e == "token")
    assert answer == "final answer here"
    # done is the final frame and carries loop metadata + indexed sources.
    assert events[-1] == "done"
    done = frames[-1][1]
    assert done["token_count"] == 3
    tl = done["metadata"]["turn_loop"]
    assert tl["confidence"] == 0.71
    assert tl["stop_reason"] == "gate_passed"
    assert tl["actions"] == 2
    assert tl["llm_calls"] == 2
    assert done["sources"][0]["citation_index"] == 1
    assert done["sources"][0]["document_id"] == "doc-1"


def test_stream_answered_retrieval_frame_uses_standard_shape(loop_env):
    """The retrieval frame carries pool results + turn_loop metadata."""
    env = loop_env(ANSWER_EVENTS, _answered_result())
    client = TestClient(_build_app(temporal_client=env["temporal"]))

    resp = client.post("/query/stream", json={"query": "hello", "turn_loop": True})

    frames = _parse_sse(resp.text)
    retrieval = next(d for e, d in frames if e == "retrieval")
    assert retrieval["action"] == "search"
    assert retrieval["results"][0]["metadata"]["document_id"] == "doc-1"
    assert retrieval["metadata"]["turn_loop"]["terminal"] == "answered"
    assert retrieval["conversation_id"] == "conv-1"


def test_stream_honors_stream_events_flag_off(loop_env, monkeypatch):
    """RAG_TURN_LOOP_STREAM_EVENTS=false suppresses loop event frames only."""
    import config.settings as settings_mod
    monkeypatch.setattr(settings_mod, "RAG_TURN_LOOP_STREAM_EVENTS", False, raising=False)
    env = loop_env(ANSWER_EVENTS, _answered_result())
    client = TestClient(_build_app(temporal_client=env["temporal"]))

    resp = client.post("/query/stream", json={"query": "hello", "turn_loop": True})

    frames = _parse_sse(resp.text)
    events = [e for e, _ in frames]
    assert not any(e in TurnEventType.ALL for e in events)
    # Answer replay + done still stream.
    assert "token" in events
    assert events[-1] == "done"


# ===========================================================================
# Clarify terminal
# ===========================================================================
def test_stream_clarify_terminal_shape(loop_env):
    """CLARIFY: ask_user retrieval frame + hints/scoping in metadata; no tokens."""
    env = loop_env(CLARIFY_EVENTS, _clarify_result())
    client = TestClient(_build_app(temporal_client=env["temporal"]))

    resp = client.post("/query/stream", json={"query": "hello", "turn_loop": True})

    frames = _parse_sse(resp.text)
    events = [e for e, _ in frames]
    assert "clarify" in events
    assert "token" not in events
    retrieval = next(d for e, d in frames if e == "retrieval")
    assert retrieval["action"] == "ask_user"
    assert retrieval["clarification_message"] == "Which project?"
    tl = retrieval["metadata"]["turn_loop"]
    assert tl["hints"] == ["Alpha", "Beta"]
    assert tl["scoping_questions"] == ["Do you mean the flow for Alpha?"]
    assert events[-1] == "done"
    assert frames[-1][1]["token_count"] == 0


def test_nonstream_clarify_terminal_shape(loop_env):
    """Non-stream CLARIFY: standard ask_user fields + metadata hints."""
    env = loop_env(CLARIFY_EVENTS, _clarify_result())
    client = TestClient(_build_app(temporal_client=env["temporal"]))

    resp = client.post("/query", json={"query": "hello", "turn_loop": True})

    assert resp.status_code == 200
    body = resp.json()
    assert body["action"] == "ask_user"
    assert body["clarification_message"] == "Which project?"
    assert body["generated_answer"] is None
    assert body["metadata"]["turn_loop"]["hints"] == ["Alpha", "Beta"]
    assert body["metadata"]["turn_loop"]["terminal"] == "ask_user"


# ===========================================================================
# Non-stream trace + answer fields
# ===========================================================================
def test_nonstream_answer_and_trace_in_metadata(loop_env):
    """POST /query returns the answer plus the full typed trace in metadata."""
    env = loop_env(ANSWER_EVENTS, _answered_result())
    client = TestClient(_build_app(temporal_client=env["temporal"]))

    resp = client.post("/query", json={"query": "hello", "turn_loop": True})

    assert resp.status_code == 200
    body = resp.json()
    assert body["generated_answer"] == "final answer here"
    assert body["action"] == "search"
    assert body["workflow_id"].startswith("rag-turn-")
    assert body["results"][0]["metadata"]["chunk_id"] == "uuid-1"
    trace = body["metadata"]["turn_loop"]["trace"]
    assert [t["type"] for t in trace] == [e for e, _ in ANSWER_EVENTS]
    assert all(set(t) == {"type", "payload", "ts_ms"} for t in trace)
    assert trace[0]["payload"]["action"] == "RETRIEVE"


# ===========================================================================
# Memory write-back
# ===========================================================================
def test_memory_writeback_kwargs_and_preview_cap(loop_env):
    """append_turn gets preview-capped chunk refs, action records, confidence."""
    from config import settings

    long_text = "x" * (settings.RAG_TURN_CONTEXT_PREVIEW_CHARS + 200)
    pool = [_chunk(text=long_text)]
    events = ANSWER_EVENTS + [
        (TurnEventType.DEEP_STUDY, {"document_id": "doc-9", "title": "Spec", "window": 0,
                                    "of_windows": 3, "notes_preview": "first"}),
        (TurnEventType.DEEP_STUDY, {"document_id": "doc-9", "title": "Spec", "window": 1,
                                    "of_windows": 3, "notes_preview": "conclusion notes"}),
    ]
    env = loop_env(events, _answered_result(pool=pool, events=events))
    client = TestClient(_build_app(temporal_client=env["temporal"]))

    resp = client.post("/query", json={"query": "hello", "turn_loop": True})

    assert resp.status_code == 200
    memory = env["memory"]
    turns = memory.args_for("append_turn")
    roles = [t["role"] for t in turns]
    assert roles == ["user", "assistant"]
    assistant = turns[1]
    assert assistant["content"] == "final answer here"
    assert assistant["answer_confidence"] == 0.71
    # Chunk refs: preview capped, full text NOT stored.
    ref = assistant["chunk_refs"][0]
    assert set(ref) == {
        "chunk_id", "document_id", "source_key", "heading", "score",
        "refactored_char_start", "refactored_char_end", "preview",
    }
    assert len(ref["preview"]) == settings.RAG_TURN_CONTEXT_PREVIEW_CHARS
    assert ref["preview"] == long_text[: settings.RAG_TURN_CONTEXT_PREVIEW_CHARS]
    # Action records derived from turn_action events (with llm-call counts).
    actions = assistant["actions"]
    assert [a["action"] for a in actions] == ["RETRIEVE", "ANSWER"]
    assert actions[0]["llm_calls"] == 1
    assert actions[1]["llm_calls"] == 1
    # Studied-doc ledger written once per document.
    studied = memory.args_for("record_doc_studied")
    assert len(studied) == 1
    entry = studied[0]["entry"]
    assert entry["document_id"] == "doc-9"
    assert entry["windows_read"] == [0, 1]
    assert entry["conclusion"] == "conclusion notes"
    # Compaction pass still runs for loop turns.
    assert "compact_if_needed" in memory.names()


def test_memory_writeback_clarification_stored_on_turn(loop_env):
    """A clarify terminal persists the pending clarification on the turn."""
    env = loop_env(CLARIFY_EVENTS, _clarify_result())
    client = TestClient(_build_app(temporal_client=env["temporal"]))

    client.post("/query", json={"query": "hello", "turn_loop": True})

    turns = env["memory"].args_for("append_turn")
    assistant = next(t for t in turns if t["role"] == "assistant")
    assert assistant["content"] == "Which project?"
    assert assistant["clarification"] == {"question": "Which project?", "hints": ["Alpha", "Beta"]}


def test_loop_path_never_suppresses_docs(loop_env):
    """Design §5: no ignored_doc_ids injection and no mark_retrieved on loop turns."""
    env = loop_env(ANSWER_EVENTS, _answered_result())
    client = TestClient(_build_app(temporal_client=env["temporal"]))

    client.post("/query", json={"query": "hello", "turn_loop": True})
    client.post("/query/stream", json={"query": "hello", "turn_loop": True})

    names = env["memory"].names()
    assert "get_seen_doc_ids" not in names
    assert "mark_retrieved" not in names


def test_memory_disabled_skips_writeback_and_context(loop_env):
    """memory_enabled=false: loop runs with empty context and writes nothing."""
    env = loop_env(ANSWER_EVENTS, _answered_result())
    client = TestClient(_build_app(temporal_client=env["temporal"]))

    resp = client.post("/query", json={"query": "hello", "turn_loop": True,
                                       "memory_enabled": False})

    assert resp.status_code == 200
    names = env["memory"].names()
    assert "build_context" not in names
    assert "append_turn" not in names
    assert "record_doc_studied" not in names


def test_structured_memory_context_feeds_turn_context(loop_env):
    """MemoryContext.structured populates the TurnContext handed to the loop."""
    structured = {
        "rolling_summary": "we talked about X",
        "recent_turns": [{"query": "q1", "answer": "a1"}],
        "chunk_refs": [{"chunk_id": "c1", "document_id": "d1", "source_key": "k",
                        "heading": "H", "score": 0.5, "preview": "p"}],
        "docs_studied": [{"document_id": "d1", "conclusion": "done"}],
        "pending_clarification": {"question": "which?", "hints": ["a"]},
    }
    env = loop_env(ANSWER_EVENTS, _answered_result(),
                   memory=FakeMemory(structured=structured))
    client = TestClient(_build_app(temporal_client=env["temporal"]))

    client.post("/query", json={"query": "hello", "turn_loop": True})

    context = env["loop"].calls[0]["context"]
    assert context.conversation_id == "conv-1"
    assert context.rolling_summary == "we talked about X"
    assert context.chunk_refs[0]["chunk_id"] == "c1"
    assert context.pending_clarification == {"question": "which?", "hints": ["a"]}


# ===========================================================================
# Non-loop path untouched when the flag is off
# ===========================================================================
def test_flag_off_stream_path_untouched(loop_env, monkeypatch):
    """Flag off: workflow dispatch, mark_retrieved, and no loop frames."""
    import config.settings as settings_mod
    monkeypatch.setattr(settings_mod, "RAG_TURN_LOOP_ENABLED", False, raising=False)
    env = loop_env(ANSWER_EVENTS, _answered_result())
    client = TestClient(_build_app(temporal_client=env["temporal"]))

    resp = client.post("/query/stream", json={"query": "hello"})

    assert resp.status_code == 200
    frames = _parse_sse(resp.text)
    events = [e for e, _ in frames]
    assert not any(e in TurnEventType.ALL for e in events)
    assert events[0] == "retrieval"
    assert "done" in events
    # The classic path still ran: workflow dispatched, doc-state marked.
    assert len(env["temporal"].calls) == 1
    assert env["temporal"].calls[0]["id"].startswith("rag-stream-")
    assert "mark_retrieved" in env["memory"].names()
    assert env["loop"].calls == []


def test_flag_off_nonstream_response_has_no_metadata_key(loop_env, monkeypatch):
    """Flag off: /query body stays byte-identical (no metadata key added)."""
    import config.settings as settings_mod
    monkeypatch.setattr(settings_mod, "RAG_TURN_LOOP_ENABLED", False, raising=False)
    env = loop_env(ANSWER_EVENTS, _answered_result())
    client = TestClient(_build_app(temporal_client=env["temporal"]))

    resp = client.post("/query", json={"query": "hello"})

    assert resp.status_code == 200
    assert "metadata" not in resp.json()
    assert env["loop"].calls == []


# ===========================================================================
# Runner unit coverage: sanitizer terminal, evidence mapping, token replay
# ===========================================================================
def test_sanitizer_reject_short_circuits_to_ask_user(loop_env, monkeypatch):
    """A pre-loop input rejection terminates without running the loop."""
    env = loop_env(ANSWER_EVENTS, _answered_result())
    monkeypatch.setattr(
        runner_mod,
        "prepare_turn_input",
        lambda query, tenant_id: runner_mod.PreparedTurnInput(
            query=query, rejected=True, rejection_message="Your query appears to be empty."
        ),
    )
    client = TestClient(_build_app(temporal_client=env["temporal"]))

    resp = client.post("/query", json={"query": "hello", "turn_loop": True})

    assert resp.status_code == 200
    body = resp.json()
    assert body["action"] == "ask_user"
    assert body["ask_user_reason"] == "sanitizer_reject"
    assert body["clarification_message"] == "Your query appears to be empty."
    assert body["metadata"]["turn_loop"]["stop_reason"] == "input_rejected"
    assert env["loop"].calls == []


def test_prepare_turn_input_passes_clean_query(monkeypatch):
    """The real Stage-1 sanitizer passes a benign query through unchanged."""
    import config.settings as settings_mod
    monkeypatch.setattr(settings_mod, "GUARDRAIL_BACKEND", "", raising=False)

    prepared = runner_mod.prepare_turn_input("how do I create a verification env?", "t")

    assert prepared.rejected is False
    assert prepared.query == "how do I create a verification env?"


def test_evidence_from_dicts_maps_and_defaults():
    """Workflow chunk dicts map to EvidenceChunk with tolerant defaults."""
    chunks = runner_mod.evidence_from_dicts([
        {"chunk_id": "c1", "document_id": "d1", "source_key": "k", "source": "s",
         "heading": "h", "text": "t", "score": "0.5", "refactored_char_start": 0,
         "refactored_char_end": 4},
        {"text": "minimal"},
        "not-a-dict",
    ])

    assert len(chunks) == 2
    assert chunks[0].score == 0.5
    assert chunks[0].refactored_char_start == 0  # zero preserved, not sentineled
    assert chunks[1].refactored_char_start == -1
    assert chunks[1].provenance == EvidenceChunk.PROVENANCE_RETRIEVE


def test_iter_answer_tokens_roundtrips_text():
    """Token replay chunks reassemble to the exact answer text."""
    text = "line one\nsecond  spaced word"
    tokens = list(runner_mod.iter_answer_tokens(text))
    assert "".join(tokens) == text
    assert all(tokens)
    assert list(runner_mod.iter_answer_tokens("")) == []


def test_loop_crash_degrades_to_explained_ask_user(loop_env, monkeypatch):
    """A crashing loop yields an explained ask_user terminal, never a 500."""
    env = loop_env(ANSWER_EVENTS, _answered_result())

    async def exploding_loop(query, context, deps, budget):
        raise RuntimeError("boom")

    monkeypatch.setattr(runner_mod, "_get_run_turn_loop", lambda: exploding_loop)
    client = TestClient(_build_app(temporal_client=env["temporal"]))

    resp = client.post("/query", json={"query": "hello", "turn_loop": True})

    assert resp.status_code == 200
    body = resp.json()
    assert body["action"] == "ask_user"
    assert body["clarification_message"]
    assert body["metadata"]["turn_loop"]["stop_reason"] == "error"


def test_validate_event_payload_fills_defaults_and_keeps_extras():
    """SSE payload models default-fill known fields and pass extras through."""
    payload = runner_mod.validate_event_payload(
        TurnEventType.DRAFT, {"attempt": 2, "legacy_extra": "hmm"}
    )
    assert payload["attempt"] == 2
    assert payload["kind"] == "content"
    assert payload["text_delta"] == ""
    assert payload["legacy_extra"] == "hmm"
    # Unknown event types pass through untouched.
    raw = {"whatever": 1}
    assert runner_mod.validate_event_payload("future_event", raw) is raw
