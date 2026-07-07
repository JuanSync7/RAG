# @summary
# Tests for the turn-context memory transfer (TURN_LOOP_DESIGN.md §7): old-record
# decode tolerance, append_turn turn-loop kwargs round-trip, the docs_studied
# ledger (record_doc_studied accumulation/cap/persistence), structured context
# assembly incl. the pending-clarification lifecycle, preview-cap growth control
# on/off, and compaction grounding (docs studied + clarifications reach the
# summarizer; pre-migration 2-arg summarize stubs keep working).
# Exports: (test module)
# Deps: pytest, orjson, src.platform.memory.provider, src.platform.memory.schemas
# @end-summary
"""Turn-context memory transfer tests (design §7 — extend, don't fork).

Uses the repo's established in-memory Redis fake (see tests/test_memory_provider.py)
so no infrastructure is required. Config toggles are exercised by monkeypatching
the provider module's imported settings attributes — the same pattern the
existing factory-resilience tests use for ``MEMORY_ENABLED``.
"""

from __future__ import annotations

import sys

import orjson

from src.platform.memory import provider as pmod
from src.platform.memory.provider import NoopConversationMemory, RedisConversationMemory


# ---------------------------------------------------------------------------
# In-memory Redis fake (repo pattern from tests/test_memory_provider.py)
# ---------------------------------------------------------------------------


class _FakeRedis:
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
        return ids[start : end + 1]

    def lrange(self, key, start, end):
        cur = list(self.lists.get(key, []))
        if start < 0:
            start = max(0, len(cur) + start)
        if end < 0:
            end = len(cur) + end
        return cur[start : end + 1]

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
    """Create a RedisConversationMemory backed by a fresh _FakeRedis."""
    fake = _FakeRedis()
    monkeypatch.setitem(sys.modules, "redis", _FakeRedisModule(fake))
    provider = RedisConversationMemory("redis://unused", "rag:turnctx:test")
    return provider, fake


_SCOPE_KW = {"tenant_id": "t1", "subject": "u1", "project_id": "p1"}


def _new_conversation(provider):
    return provider.ensure_conversation(**_SCOPE_KW, title="Turn-context test").conversation_id


def _clarification():
    return {
        "question": "Which verification flow do you mean?",
        "hints": ["Formal flow", "Simulation flow"],
        "scoping_questions": ["How do I set up the simulation flow?"],
    }


# ---------------------------------------------------------------------------
# Old-record tolerance (pre-migration rows/hashes must keep loading)
# ---------------------------------------------------------------------------


def test_legacy_turn_row_without_new_fields_loads(monkeypatch):
    """A turn row written before the turn-loop fields existed loads with defaults."""
    provider, fake = _make_provider(monkeypatch)
    cid = _new_conversation(provider)
    scope = provider._scope(**_SCOPE_KW)
    legacy_row = orjson.dumps(
        {
            "role": "assistant",
            "content": "legacy answer",
            "timestamp_ms": 123,
            "query_id": "q-legacy",
            "sources": [{"source": "doc.md", "text": "some chunk text"}],
        }
    )
    fake.rpush(provider._turns_key(scope, cid), legacy_row)

    turns = provider.get_turns(**_SCOPE_KW, conversation_id=cid)
    assert len(turns) == 1
    turn = turns[0]
    assert turn.content == "legacy answer"
    assert turn.actions == []
    assert turn.chunk_refs == []
    assert turn.answer_confidence is None
    assert turn.clarification is None


def test_legacy_meta_hash_without_docs_studied_loads(monkeypatch):
    """A meta hash written before docs_studied existed decodes to an empty ledger."""
    provider, fake = _make_provider(monkeypatch)
    scope = provider._scope(**_SCOPE_KW)
    cid = "conv_pre_migration"
    # Pre-migration payload: exactly the fields the old writer stored.
    fake.hset(
        provider._meta_key(scope, cid),
        mapping={
            "tenant_id": "t1",
            "subject": "u1",
            "project_id": "p1",
            "title": "Old conversation",
            "created_at_ms": 1,
            "updated_at_ms": 1,
            "message_count": 0,
            "summary_text": "",
            "summary_updated_at_ms": 0,
            "summary_turns_compacted": 0,
            "relevant_doc_ids": "[]",
            "ignored_doc_ids": "[]",
        },
    )
    meta = provider.ensure_conversation(**_SCOPE_KW, conversation_id=cid)
    assert meta.docs_studied == []
    ctx = provider.build_context(**_SCOPE_KW, conversation_id=cid)
    assert ctx.structured["docs_studied"] == []
    assert ctx.structured["pending_clarification"] is None


def test_corrupt_docs_studied_value_decodes_empty(monkeypatch):
    """An unparsable docs_studied hash value degrades to [] instead of raising."""
    provider, fake = _make_provider(monkeypatch)
    cid = _new_conversation(provider)
    scope = provider._scope(**_SCOPE_KW)
    fake.hset(provider._meta_key(scope, cid), mapping={"docs_studied": "not-json"})
    meta = provider.ensure_conversation(**_SCOPE_KW, conversation_id=cid)
    assert meta.docs_studied == []


# ---------------------------------------------------------------------------
# append_turn turn-loop kwargs round-trip
# ---------------------------------------------------------------------------


def test_append_turn_turn_loop_kwargs_round_trip(monkeypatch):
    provider, _ = _make_provider(monkeypatch)
    cid = _new_conversation(provider)
    actions = [
        {"action": "RETRIEVE", "reason": "fresh question", "ms": 812, "llm_calls": 2},
        {"action": "ANSWER", "reason": "pool sufficient", "ms": 1490, "llm_calls": 3},
    ]
    chunk_refs = [
        {
            "chunk_id": "uuid-1",
            "document_id": "doc-1",
            "source_key": "docs/setup.md",
            "heading": "Setup",
            "score": 0.91,
            "refactored_char_start": 10,
            "refactored_char_end": 90,
            "preview": "short preview",
        }
    ]
    provider.append_turn(
        **_SCOPE_KW,
        conversation_id=cid,
        role="assistant",
        content="Here is the answer.",
        query_id="q1",
        actions=actions,
        chunk_refs=chunk_refs,
        answer_confidence=0.87,
        clarification=None,
    )
    turn = provider.get_turns(**_SCOPE_KW, conversation_id=cid)[-1]
    assert turn.actions == actions
    assert turn.chunk_refs == chunk_refs
    assert turn.answer_confidence == 0.87
    assert turn.clarification is None


def test_append_turn_persists_clarification_payload(monkeypatch):
    provider, _ = _make_provider(monkeypatch)
    cid = _new_conversation(provider)
    provider.append_turn(
        **_SCOPE_KW,
        conversation_id=cid,
        role="assistant",
        content="Which verification flow do you mean?",
        clarification=_clarification(),
    )
    turn = provider.get_turns(**_SCOPE_KW, conversation_id=cid)[-1]
    assert turn.clarification == _clarification()


def test_append_turn_with_only_structured_payload_is_stored(monkeypatch):
    """The empty-turn guard must not drop a turn that carries only chunk refs."""
    provider, _ = _make_provider(monkeypatch)
    cid = _new_conversation(provider)
    provider.append_turn(
        **_SCOPE_KW,
        conversation_id=cid,
        role="assistant",
        content="",
        chunk_refs=[{"chunk_id": "uuid-x", "preview": "p"}],
    )
    turns = provider.get_turns(**_SCOPE_KW, conversation_id=cid)
    assert len(turns) == 1
    assert turns[0].chunk_refs[0]["chunk_id"] == "uuid-x"


def test_append_turn_empty_turn_still_skipped(monkeypatch):
    provider, _ = _make_provider(monkeypatch)
    cid = _new_conversation(provider)
    provider.append_turn(**_SCOPE_KW, conversation_id=cid, role="user", content="   ")
    assert provider.get_turns(**_SCOPE_KW, conversation_id=cid) == []


# ---------------------------------------------------------------------------
# record_doc_studied: accumulation, cap, persistence
# ---------------------------------------------------------------------------


def test_record_doc_studied_accumulates_and_stamps_ts(monkeypatch):
    provider, _ = _make_provider(monkeypatch)
    cid = _new_conversation(provider)
    meta = provider.record_doc_studied(
        **_SCOPE_KW,
        conversation_id=cid,
        entry={
            "document_id": "doc-42",
            "windows_read": [0, 1],
            "sections": ["Install", "Configure"],
            "conclusion": "Use the module flow.",
        },
    )
    assert len(meta.docs_studied) == 1
    entry = meta.docs_studied[0]
    assert entry["document_id"] == "doc-42"
    assert entry["ts"] > 0  # stamped at the provider when absent

    meta = provider.record_doc_studied(
        **_SCOPE_KW,
        conversation_id=cid,
        entry={"document_id": "doc-43", "conclusion": "Second doc.", "ts": 777},
    )
    assert [d["document_id"] for d in meta.docs_studied] == ["doc-42", "doc-43"]
    assert meta.docs_studied[1]["ts"] == 777  # caller-provided ts preserved


def test_record_doc_studied_cap_keeps_newest(monkeypatch):
    monkeypatch.setattr(pmod, "RAG_TURN_CONTEXT_MAX_CHUNK_REFS", 3)
    provider, _ = _make_provider(monkeypatch)
    cid = _new_conversation(provider)
    for i in range(5):
        provider.record_doc_studied(
            **_SCOPE_KW,
            conversation_id=cid,
            entry={"document_id": f"doc-{i}", "conclusion": f"c{i}"},
        )
    meta = provider.ensure_conversation(**_SCOPE_KW, conversation_id=cid)
    assert [d["document_id"] for d in meta.docs_studied] == ["doc-2", "doc-3", "doc-4"]


def test_record_doc_studied_persists_across_meta_reads(monkeypatch):
    provider, _ = _make_provider(monkeypatch)
    cid = _new_conversation(provider)
    provider.record_doc_studied(
        **_SCOPE_KW,
        conversation_id=cid,
        entry={"document_id": "doc-p", "conclusion": "persisted"},
    )
    reloaded = provider.ensure_conversation(**_SCOPE_KW, conversation_id=cid)
    assert reloaded.docs_studied[0]["document_id"] == "doc-p"


def test_record_doc_studied_rejects_entry_without_document_id(monkeypatch):
    provider, _ = _make_provider(monkeypatch)
    cid = _new_conversation(provider)
    meta = provider.record_doc_studied(
        **_SCOPE_KW, conversation_id=cid, entry={"conclusion": "no id"}
    )
    assert meta.docs_studied == []


# ---------------------------------------------------------------------------
# Structured context assembly (MemoryContext.structured)
# ---------------------------------------------------------------------------


def test_structured_context_assembly(monkeypatch):
    provider, _ = _make_provider(monkeypatch)
    cid = _new_conversation(provider)
    provider.append_turn(
        **_SCOPE_KW, conversation_id=cid, role="user", content="How do I set up X?"
    )
    provider.append_turn(
        **_SCOPE_KW,
        conversation_id=cid,
        role="assistant",
        content="Set up X like this.",
        chunk_refs=[
            {"chunk_id": "uuid-a", "document_id": "doc-a", "preview": "alpha"},
            {"chunk_id": "uuid-b", "document_id": "doc-b", "preview": "beta"},
        ],
        answer_confidence=0.9,
    )
    provider.record_doc_studied(
        **_SCOPE_KW,
        conversation_id=cid,
        entry={"document_id": "doc-a", "conclusion": "covers install flow"},
    )
    ctx = provider.build_context(**_SCOPE_KW, conversation_id=cid)
    structured = ctx.structured
    assert structured["recent_turns"] == [
        {"question": "How do I set up X?", "answer": "Set up X like this."}
    ]
    assert [r["chunk_id"] for r in structured["chunk_refs"]] == ["uuid-a", "uuid-b"]
    assert structured["docs_studied"][0]["document_id"] == "doc-a"
    assert structured["pending_clarification"] is None
    # Prose context builder remains intact alongside the structured sibling.
    assert "How do I set up X?" in ctx.context_text


def test_structured_chunk_refs_capped_newest_turn_first(monkeypatch):
    monkeypatch.setattr(pmod, "RAG_TURN_CONTEXT_MAX_CHUNK_REFS", 2)
    provider, _ = _make_provider(monkeypatch)
    cid = _new_conversation(provider)
    for i in range(3):
        provider.append_turn(
            **_SCOPE_KW,
            conversation_id=cid,
            role="assistant",
            content=f"answer {i}",
            chunk_refs=[{"chunk_id": f"uuid-{i}", "preview": f"p{i}"}],
        )
    structured = provider.build_context(**_SCOPE_KW, conversation_id=cid).structured
    # Newest turns win the cap, newest turn first.
    assert [r["chunk_id"] for r in structured["chunk_refs"]] == ["uuid-2", "uuid-1"]


def test_pending_clarification_lifecycle(monkeypatch):
    """Set -> surfaced -> consumed by the next user turn (design §7)."""
    provider, _ = _make_provider(monkeypatch)
    cid = _new_conversation(provider)
    provider.append_turn(
        **_SCOPE_KW, conversation_id=cid, role="user", content="How do I verify it?"
    )
    # 1) The assistant turn ends ask_user: clarification is set on the turn.
    provider.append_turn(
        **_SCOPE_KW,
        conversation_id=cid,
        role="assistant",
        content="Which verification flow do you mean?",
        clarification=_clarification(),
    )
    # 2) Surfaced: no user answer yet, so the context carries it.
    structured = provider.build_context(**_SCOPE_KW, conversation_id=cid).structured
    assert structured["pending_clarification"] == _clarification()
    # 3) Consumed: a later user turn exists, so it is no longer pending.
    provider.append_turn(
        **_SCOPE_KW, conversation_id=cid, role="user", content="the second one"
    )
    structured = provider.build_context(**_SCOPE_KW, conversation_id=cid).structured
    assert structured["pending_clarification"] is None


def test_answered_assistant_turn_leaves_no_pending_clarification(monkeypatch):
    """A normal answered turn (clarification=None) never surfaces as pending."""
    provider, _ = _make_provider(monkeypatch)
    cid = _new_conversation(provider)
    provider.append_turn(**_SCOPE_KW, conversation_id=cid, role="user", content="Q?")
    provider.append_turn(
        **_SCOPE_KW, conversation_id=cid, role="assistant", content="Answered."
    )
    structured = provider.build_context(**_SCOPE_KW, conversation_id=cid).structured
    assert structured["pending_clarification"] is None


def test_noop_provider_turn_context_surface(monkeypatch):
    """Noop provider accepts the new kwargs/methods and returns empty state."""
    provider = NoopConversationMemory()
    provider.append_turn(
        tenant_id="t", subject="s", project_id=None,
        conversation_id="conv_noop", role="assistant", content="x",
        actions=[{"action": "ANSWER"}], chunk_refs=[{"chunk_id": "u"}],
        answer_confidence=0.5, clarification=_clarification(),
    )
    meta = provider.record_doc_studied(
        tenant_id="t", subject="s", project_id=None,
        conversation_id="conv_noop", entry={"document_id": "d"},
    )
    assert meta.docs_studied == []
    ctx = provider.build_context(
        tenant_id="t", subject="s", project_id=None, conversation_id="conv_noop"
    )
    assert ctx.structured == {}


# ---------------------------------------------------------------------------
# Growth control: preview capping at the provider write boundary
# ---------------------------------------------------------------------------


def test_preview_cap_applied_when_full_text_off(monkeypatch):
    monkeypatch.setattr(pmod, "RAG_TURN_CONTEXT_STORE_FULL_TEXT", False)
    monkeypatch.setattr(pmod, "RAG_TURN_CONTEXT_PREVIEW_CHARS", 8)
    provider, _ = _make_provider(monkeypatch)
    cid = _new_conversation(provider)
    provider.append_turn(
        **_SCOPE_KW,
        conversation_id=cid,
        role="assistant",
        content="answer",
        sources=[{"source": "doc.md", "text": "0123456789ABCDEF"}],
        chunk_refs=[{"chunk_id": "uuid-long", "preview": "0123456789ABCDEF"}],
    )
    turn = provider.get_turns(**_SCOPE_KW, conversation_id=cid)[-1]
    assert turn.sources[0]["text"] == "01234567"       # capped at the boundary
    assert turn.chunk_refs[0]["preview"] == "01234567"  # capped at the boundary


def test_preview_cap_disabled_when_full_text_on(monkeypatch):
    monkeypatch.setattr(pmod, "RAG_TURN_CONTEXT_STORE_FULL_TEXT", True)
    monkeypatch.setattr(pmod, "RAG_TURN_CONTEXT_PREVIEW_CHARS", 8)
    provider, _ = _make_provider(monkeypatch)
    cid = _new_conversation(provider)
    provider.append_turn(
        **_SCOPE_KW,
        conversation_id=cid,
        role="assistant",
        content="answer",
        sources=[{"source": "doc.md", "text": "0123456789ABCDEF"}],
        chunk_refs=[{"chunk_id": "uuid-long", "preview": "0123456789ABCDEF"}],
    )
    turn = provider.get_turns(**_SCOPE_KW, conversation_id=cid)[-1]
    assert turn.sources[0]["text"] == "0123456789ABCDEF"       # stored in full
    assert turn.chunk_refs[0]["preview"] == "0123456789ABCDEF"  # stored in full


def test_structured_context_caps_preview_even_for_full_text_records(monkeypatch):
    """Records stored with full text are still preview-capped in the context."""
    monkeypatch.setattr(pmod, "RAG_TURN_CONTEXT_STORE_FULL_TEXT", True)
    monkeypatch.setattr(pmod, "RAG_TURN_CONTEXT_PREVIEW_CHARS", 8)
    provider, _ = _make_provider(monkeypatch)
    cid = _new_conversation(provider)
    provider.append_turn(
        **_SCOPE_KW,
        conversation_id=cid,
        role="assistant",
        content="answer",
        chunk_refs=[{"chunk_id": "uuid-long", "preview": "0123456789ABCDEF"}],
    )
    structured = provider.build_context(**_SCOPE_KW, conversation_id=cid).structured
    assert structured["chunk_refs"][0]["preview"] == "01234567"


# ---------------------------------------------------------------------------
# Compaction grounding: structured fields reach the summarizer
# ---------------------------------------------------------------------------


class _FakeLLMResponse:
    def __init__(self, content):
        self.content = content


class _FakeLLMProvider:
    """Captures generate() messages and returns a fixed summary."""

    def __init__(self, content="grounded summary"):
        self.calls = []
        self._content = content

    def generate(self, messages, **kwargs):
        self.calls.append(messages)
        return _FakeLLMResponse(self._content)


def test_compaction_summary_receives_structured_grounding(monkeypatch):
    provider, _ = _make_provider(monkeypatch)
    fake_llm = _FakeLLMProvider()
    provider._llm_provider = fake_llm
    cid = _new_conversation(provider)
    provider.append_turn(
        **_SCOPE_KW, conversation_id=cid, role="user", content="How do I verify?"
    )
    provider.append_turn(
        **_SCOPE_KW,
        conversation_id=cid,
        role="assistant",
        content="Which verification flow do you mean?",
        clarification=_clarification(),
    )
    provider.record_doc_studied(
        **_SCOPE_KW,
        conversation_id=cid,
        entry={
            "document_id": "doc-42",
            "sections": ["Verification"],
            "conclusion": "Use the module flow.",
        },
    )
    result = provider.compact_if_needed(**_SCOPE_KW, conversation_id=cid, force=True)
    assert result.text == "grounded summary"
    assert len(fake_llm.calls) == 1
    prompt_text = fake_llm.calls[0][1]["content"]
    # Docs-studied ledger grounding present.
    assert "document_id=doc-42" in prompt_text
    assert "Use the module flow." in prompt_text
    # Clarification-asked grounding present (question + hints).
    assert "ASSISTANT ASKED CLARIFICATION" in prompt_text
    assert "Which verification flow do you mean?" in prompt_text
    assert "Formal flow" in prompt_text


def test_compaction_tolerates_pre_migration_two_arg_summarize_stub(monkeypatch):
    """Old-style 2-arg _llm_summarize overrides keep working (stub tolerance)."""
    provider, _ = _make_provider(monkeypatch)
    provider._llm_summarize = lambda turns, existing: "old-style summary"
    cid = _new_conversation(provider)
    provider.append_turn(
        **_SCOPE_KW, conversation_id=cid, role="user", content="hello"
    )
    result = provider.compact_if_needed(**_SCOPE_KW, conversation_id=cid, force=True)
    assert result.text == "old-style summary"


def test_compaction_passes_docs_studied_to_var_keyword_summarize(monkeypatch):
    """A **kwargs-accepting summarize override receives the structured ledger."""
    provider, _ = _make_provider(monkeypatch)
    captured = {}

    def _stub(turns, existing, **kwargs):
        captured.update(kwargs)
        return "kw summary"

    provider._llm_summarize = _stub
    cid = _new_conversation(provider)
    provider.append_turn(**_SCOPE_KW, conversation_id=cid, role="user", content="hi")
    provider.record_doc_studied(
        **_SCOPE_KW, conversation_id=cid, entry={"document_id": "doc-kw"}
    )
    result = provider.compact_if_needed(**_SCOPE_KW, conversation_id=cid, force=True)
    assert result.text == "kw summary"
    assert [d["document_id"] for d in captured["docs_studied"]] == ["doc-kw"]
