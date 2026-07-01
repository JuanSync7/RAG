from src.platform.memory.provider import NoopConversationMemory, RedisConversationMemory


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
        assert decode_responses is True
        return self._client


def test_redis_memory_roundtrip(monkeypatch):
    fake = _FakeRedis()
    fake_mod = _FakeRedisModule(fake)
    monkeypatch.setitem(__import__("sys").modules, "redis", fake_mod)

    provider = RedisConversationMemory("redis://unused", "rag:test:memory")
    provider._llm_summarize = lambda turns, existing: "summary updated"

    meta = provider.ensure_conversation(
        tenant_id="tenantA",
        subject="user1",
        project_id="projA",
        title="Chat 1",
    )
    assert meta.conversation_id.startswith("conv_")
    provider.append_turn(
        tenant_id="tenantA",
        subject="user1",
        project_id="projA",
        conversation_id=meta.conversation_id,
        role="user",
        content="What is RAG?",
        query_id="wf1",
    )
    provider.append_turn(
        tenant_id="tenantA",
        subject="user1",
        project_id="projA",
        conversation_id=meta.conversation_id,
        role="assistant",
        content="RAG combines retrieval and generation.",
        query_id="wf1",
    )

    rows = provider.list_conversations(
        tenant_id="tenantA",
        subject="user1",
        project_id="projA",
    )
    assert len(rows) == 1
    assert rows[0].title == "Chat 1"

    ctx = provider.build_context(
        tenant_id="tenantA",
        subject="user1",
        project_id="projA",
        conversation_id=meta.conversation_id,
        turn_window=4,
    )
    assert "RAG" in ctx.context_text

    compact = provider.compact_if_needed(
        tenant_id="tenantA",
        subject="user1",
        project_id="projA",
        conversation_id=meta.conversation_id,
        force=True,
    )
    assert compact.text == "summary updated"


def _make_provider(monkeypatch):
    """Helper: create a RedisConversationMemory backed by a fresh _FakeRedis."""
    fake = _FakeRedis()
    fake_mod = _FakeRedisModule(fake)
    monkeypatch.setitem(__import__("sys").modules, "redis", fake_mod)
    provider = RedisConversationMemory("redis://unused", "rag:test:memory")
    provider._llm_summarize = lambda turns, existing: "stubbed summary"
    return provider


def test_turn_count_increases(monkeypatch):
    provider = _make_provider(monkeypatch)
    meta = provider.ensure_conversation(
        tenant_id="t1", subject="u1", project_id="p1", title="Count test"
    )
    cid = meta.conversation_id

    for i in range(3):
        provider.append_turn(
            tenant_id="t1", subject="u1", project_id="p1",
            conversation_id=cid, role="user", content=f"message {i}",
        )

    turns = provider.get_turns(tenant_id="t1", subject="u1", project_id="p1", conversation_id=cid)
    assert len(turns) == 3


def test_retrieve_turns_in_chronological_order(monkeypatch):
    """Turns must be returned oldest-to-newest."""
    provider = _make_provider(monkeypatch)
    meta = provider.ensure_conversation(
        tenant_id="t1", subject="u1", project_id="p1", title="Order test"
    )
    cid = meta.conversation_id

    messages = ["first", "second", "third"]
    for msg in messages:
        provider.append_turn(
            tenant_id="t1", subject="u1", project_id="p1",
            conversation_id=cid, role="user", content=msg,
        )

    turns = provider.get_turns(
        tenant_id="t1", subject="u1", project_id="p1", conversation_id=cid
    )
    assert [t.content for t in turns] == ["first", "second", "third"]


def test_token_budget_limits_context_window(monkeypatch):
    """trim_turns_to_budget should drop older turns that exceed the token budget."""
    provider = _make_provider(monkeypatch)
    meta = provider.ensure_conversation(
        tenant_id="t1", subject="u1", project_id="p1", title="Budget test"
    )
    cid = meta.conversation_id

    # Each message is 40 chars → ~10 estimated tokens each.
    for i in range(5):
        provider.append_turn(
            tenant_id="t1", subject="u1", project_id="p1",
            conversation_id=cid, role="user",
            content="A" * 40,
        )

    ctx = provider.build_context(
        tenant_id="t1", subject="u1", project_id="p1",
        conversation_id=cid, turn_window=5,
    )
    # All 5 short turns fit within the default token budget; context must be non-empty.
    assert ctx.context_text
    assert len(ctx.recent_turns) > 0


def test_in_process_backend_same_behavior(monkeypatch):
    """NoopConversationMemory returns empty state but never raises."""
    provider = NoopConversationMemory()

    meta = provider.ensure_conversation(
        tenant_id="t1", subject="u1", project_id=None, title="Noop"
    )
    assert meta.conversation_id

    provider.append_turn(
        tenant_id="t1", subject="u1", project_id=None,
        conversation_id=meta.conversation_id, role="user", content="hello",
    )

    turns = provider.get_turns(
        tenant_id="t1", subject="u1", project_id=None,
        conversation_id=meta.conversation_id,
    )
    assert turns == []

    ctx = provider.build_context(
        tenant_id="t1", subject="u1", project_id=None,
        conversation_id=meta.conversation_id,
    )
    assert ctx.context_text == ""


def test_conversation_id_isolation(monkeypatch):
    """Two different conversation IDs must have independent turn histories."""
    provider = _make_provider(monkeypatch)

    meta_a = provider.ensure_conversation(
        tenant_id="t1", subject="u1", project_id="p1", title="Conv A"
    )
    meta_b = provider.ensure_conversation(
        tenant_id="t1", subject="u1", project_id="p1", title="Conv B"
    )
    assert meta_a.conversation_id != meta_b.conversation_id

    provider.append_turn(
        tenant_id="t1", subject="u1", project_id="p1",
        conversation_id=meta_a.conversation_id, role="user", content="Only in A",
    )
    provider.append_turn(
        tenant_id="t1", subject="u1", project_id="p1",
        conversation_id=meta_b.conversation_id, role="user", content="Only in B",
    )

    turns_a = provider.get_turns(
        tenant_id="t1", subject="u1", project_id="p1",
        conversation_id=meta_a.conversation_id,
    )
    turns_b = provider.get_turns(
        tenant_id="t1", subject="u1", project_id="p1",
        conversation_id=meta_b.conversation_id,
    )

    assert len(turns_a) == 1 and turns_a[0].content == "Only in A"
    assert len(turns_b) == 1 and turns_b[0].content == "Only in B"


# ---------------------------------------------------------------------------
# Retrieved / ignored doc-list lifecycle (S1)
# ---------------------------------------------------------------------------

def _ensure(provider):
    return provider.ensure_conversation(
        tenant_id="t1", subject="u1", project_id="p1", title="Doc-state test"
    ).conversation_id


def test_mark_retrieved_seeds_relevant_list(monkeypatch):
    provider = _make_provider(monkeypatch)
    cid = _ensure(provider)

    meta = provider.mark_retrieved(
        tenant_id="t1", subject="u1", project_id="p1",
        conversation_id=cid, doc_ids=["doc1", "doc2", "doc3"],
    )
    assert meta.relevant_doc_ids == ["doc1", "doc2", "doc3"]
    assert meta.ignored_doc_ids == []


def test_mark_retrieved_is_idempotent_and_skips_already_ignored(monkeypatch):
    provider = _make_provider(monkeypatch)
    cid = _ensure(provider)

    provider.mark_retrieved(
        tenant_id="t1", subject="u1", project_id="p1",
        conversation_id=cid, doc_ids=["doc1", "doc2"],
    )
    provider.move_to_ignored(
        tenant_id="t1", subject="u1", project_id="p1",
        conversation_id=cid, doc_id="doc2",
    )
    # Re-mark with overlapping ids — doc2 must NOT come back to relevant.
    meta = provider.mark_retrieved(
        tenant_id="t1", subject="u1", project_id="p1",
        conversation_id=cid, doc_ids=["doc1", "doc2", "doc3"],
    )
    assert meta.relevant_doc_ids == ["doc1", "doc3"]
    assert meta.ignored_doc_ids == ["doc2"]


def test_move_to_ignored_and_restore(monkeypatch):
    provider = _make_provider(monkeypatch)
    cid = _ensure(provider)

    provider.mark_retrieved(
        tenant_id="t1", subject="u1", project_id="p1",
        conversation_id=cid, doc_ids=["doc1", "doc2"],
    )
    meta = provider.move_to_ignored(
        tenant_id="t1", subject="u1", project_id="p1",
        conversation_id=cid, doc_id="doc1",
    )
    assert meta.relevant_doc_ids == ["doc2"]
    assert meta.ignored_doc_ids == ["doc1"]

    meta = provider.restore_to_relevant(
        tenant_id="t1", subject="u1", project_id="p1",
        conversation_id=cid, doc_id="doc1",
    )
    assert "doc1" in meta.relevant_doc_ids
    assert "doc1" not in meta.ignored_doc_ids


def test_get_seen_doc_ids_returns_union(monkeypatch):
    provider = _make_provider(monkeypatch)
    cid = _ensure(provider)

    provider.mark_retrieved(
        tenant_id="t1", subject="u1", project_id="p1",
        conversation_id=cid, doc_ids=["a", "b", "c"],
    )
    provider.move_to_ignored(
        tenant_id="t1", subject="u1", project_id="p1",
        conversation_id=cid, doc_id="b",
    )
    seen = provider.get_seen_doc_ids(
        tenant_id="t1", subject="u1", project_id="p1", conversation_id=cid,
    )
    assert set(seen) == {"a", "b", "c"}


def test_get_seen_doc_ids_empty_for_unknown_conversation(monkeypatch):
    provider = _make_provider(monkeypatch)
    seen = provider.get_seen_doc_ids(
        tenant_id="t1", subject="u1", project_id="p1",
        conversation_id="conv_never_created",
    )
    assert seen == []


def test_doc_state_persists_across_meta_reads(monkeypatch):
    """Round-trip via ensure_conversation: state stored in meta hash decodes back."""
    provider = _make_provider(monkeypatch)
    cid = _ensure(provider)

    provider.mark_retrieved(
        tenant_id="t1", subject="u1", project_id="p1",
        conversation_id=cid, doc_ids=["x", "y"],
    )
    provider.move_to_ignored(
        tenant_id="t1", subject="u1", project_id="p1",
        conversation_id=cid, doc_id="y",
    )
    # Re-fetch via ensure_conversation — should rebuild from the hash.
    reloaded = provider.ensure_conversation(
        tenant_id="t1", subject="u1", project_id="p1",
        conversation_id=cid,
    )
    assert reloaded.relevant_doc_ids == ["x"]
    assert reloaded.ignored_doc_ids == ["y"]


def test_noop_provider_doc_state_methods(monkeypatch):
    provider = NoopConversationMemory()
    meta = provider.mark_retrieved(
        tenant_id="t", subject="s", project_id=None,
        conversation_id="conv_noop", doc_ids=["a"],
    )
    assert meta.relevant_doc_ids == []
    assert meta.ignored_doc_ids == []
    assert provider.get_seen_doc_ids(
        tenant_id="t", subject="s", project_id=None, conversation_id="conv_noop",
    ) == []


# ---------------------------------------------------------------------------
# Newline preservation — stored turn content must keep its markdown structure
# (lists/paragraphs) so a reloaded conversation doesn't render as a wall of
# text. Regression for the chat-switch \n-collapse bug.
# ---------------------------------------------------------------------------

def test_sanitize_default_collapses_all_whitespace():
    from src.platform.memory.utils import sanitize_memory_text
    out = sanitize_memory_text("a:\n\n- x\n- y\n\nend")
    assert "\n" not in out
    assert out == "a: - x - y end"


def test_sanitize_preserve_newlines_keeps_structure():
    from src.platform.memory.utils import sanitize_memory_text
    md = "Intro:\n\n- a\n- b\n\nOutro."
    assert sanitize_memory_text(md, preserve_newlines=True) == "Intro:\n\n- a\n- b\n\nOutro."


def test_sanitize_preserve_newlines_collapses_midline_ws_caps_blanklines_keeps_indent():
    from src.platform.memory.utils import sanitize_memory_text
    # mid-line runs collapse, blank-line runs cap at one, trailing ws trimmed,
    # but LEADING indentation (nested list / code) is preserved.
    md = "a   b\t c\n\n\n\nd  \n  e"
    assert sanitize_memory_text(md, preserve_newlines=True) == "a b c\n\nd\n  e"


def test_sanitize_preserve_newlines_keeps_indented_code():
    from src.platform.memory.utils import sanitize_memory_text
    md = "Run:\n\n```bash\n    ./build.sh --opt\n```\n"
    # the 4-space code indentation survives (no leading-space stripping)
    assert (
        sanitize_memory_text(md, preserve_newlines=True)
        == "Run:\n\n```bash\n    ./build.sh --opt\n```"
    )


def test_append_turn_preserves_markdown_newlines(monkeypatch):
    provider = _make_provider(monkeypatch)
    meta = provider.ensure_conversation(
        tenant_id="t", subject="u", project_id="p", title="c",
    )
    md = "Steps:\n\n- one\n- two\n\nAll done."
    provider.append_turn(
        tenant_id="t", subject="u", project_id="p",
        conversation_id=meta.conversation_id, role="assistant",
        content=md, query_id="wf1",
    )
    turns = provider.get_turns(
        tenant_id="t", subject="u", project_id="p",
        conversation_id=meta.conversation_id,
    )
    assert turns and turns[-1].content == md  # newlines survive the round-trip


# ---------------------------------------------------------------------------
# Factory resilience: Redis-down must NOT permanently latch to no-op
# ---------------------------------------------------------------------------


class _PingRedis:
    """A redis client stub whose ``ping()`` succeeds or raises on demand."""

    def __init__(self, ok):
        self._ok = ok
        self.ping_calls = 0

    def ping(self):
        self.ping_calls += 1
        if not self._ok:
            raise ConnectionError("redis down")
        return True


class _TogglableRedisModule:
    """A fake ``redis`` module whose next client's ping is controlled by ``ok``.
    Records how many times ``from_url`` was called so the retry-throttle can be
    asserted."""

    def __init__(self):
        self.ok = True
        self.from_url_calls = 0

    def from_url(self, _url, decode_responses=True, **_kw):
        self.from_url_calls += 1
        return _PingRedis(self.ok)


def test_get_conversation_memory_recovers_when_redis_returns(monkeypatch):
    """Redis down at the first call must NOT latch to no-op for the process
    lifetime: the no-op is uncached, so once Redis is reachable again the next
    call reconnects and caches the Redis provider — no restart required."""
    from src.platform.memory import provider as pmod

    monkeypatch.setattr(pmod, "_MEMORY", None)
    monkeypatch.setattr(pmod, "_last_redis_attempt_at", None)
    monkeypatch.setattr(pmod, "MEMORY_ENABLED", True)
    monkeypatch.setattr(pmod, "MEMORY_PROVIDER", "redis")

    mod = _TogglableRedisModule()
    monkeypatch.setitem(__import__("sys").modules, "redis", mod)

    # 1) Redis DOWN -> uncached no-op (singleton not latched).
    mod.ok = False
    m1 = pmod.get_conversation_memory()
    assert isinstance(m1, NoopConversationMemory)
    assert pmod._MEMORY is None  # crucial: NOT cached, so it will retry

    # 2) Throttle: an immediate retry must skip the (blocking) connect attempt.
    calls = mod.from_url_calls
    m2 = pmod.get_conversation_memory()
    assert isinstance(m2, NoopConversationMemory)
    assert mod.from_url_calls == calls  # skipped by the retry interval

    # 3) Redis BACK and the throttle window has elapsed -> reconnect + cache.
    mod.ok = True
    monkeypatch.setattr(pmod, "_last_redis_attempt_at", None)
    m3 = pmod.get_conversation_memory()
    assert isinstance(m3, RedisConversationMemory)
    assert pmod._MEMORY is m3  # cached only on success

    # 4) Once connected it is the cached singleton (no further reconnects).
    calls = mod.from_url_calls
    m4 = pmod.get_conversation_memory()
    assert m4 is m3
    assert mod.from_url_calls == calls
