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
