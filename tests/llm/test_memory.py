"""Unit tests for ``src.common.llm.memory``.

Covers the in-process ``memory`` backend of ``ConversationSession``
(append by role, ordering, clear, ``messages`` snapshot isolation,
``to_dicts`` shape), the unknown-backend guard, and the ``conversation()``
factory's per-(session_id, backend) registry caching.  These paths are
pure in-process logic — no Redis provider is constructed for the
``memory`` backend, so no boundary mocking is required; the module-level
registry is reset between tests for isolation.
"""
from __future__ import annotations

import pytest

import src.common.llm.memory as memory_mod
from src.common.llm.memory import ConversationSession, conversation
from src.common.llm.schemas import ConversationMessage, MessageRole


@pytest.fixture(autouse=True)
def clear_registry():
    """Reset the module-level session registry before and after each test."""
    memory_mod._registry.clear()
    yield
    memory_mod._registry.clear()


def _session():
    return ConversationSession("s1", "memory")


# ── ConversationSession (memory backend) ────────────────────────────────


def test_add_user_appends_user_message():
    """add_user appends a single USER-role message."""
    s = _session()
    s.add_user("hello")
    assert s.messages == [ConversationMessage(role=MessageRole.USER, content="hello")]


def test_add_roles_preserve_insertion_order():
    """Messages retain insertion order across mixed roles."""
    s = _session()
    s.add_system("sys")
    s.add_user("u")
    s.add_assistant("a")
    assert [(m.role, m.content) for m in s.messages] == [
        (MessageRole.SYSTEM, "sys"),
        (MessageRole.USER, "u"),
        (MessageRole.ASSISTANT, "a"),
    ]


def test_clear_wipes_messages():
    """clear() empties the in-process history."""
    s = _session()
    s.add_user("x")
    s.clear()
    assert s.messages == []


def test_messages_returns_independent_copy():
    """The messages property returns a copy; mutating it cannot corrupt state."""
    s = _session()
    s.add_user("keep")
    snapshot = s.messages
    snapshot.append(ConversationMessage(role=MessageRole.USER, content="intruder"))
    assert len(s.messages) == 1


def test_to_dicts_emits_openai_shape():
    """to_dicts maps each message to {'role': value, 'content': ...}."""
    s = _session()
    s.add_user("q")
    s.add_assistant("a")
    assert s.to_dicts() == [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]


def test_unknown_backend_raises():
    """An unrecognised backend name raises ValueError at construction."""
    with pytest.raises(ValueError, match="Unknown backend"):
        ConversationSession("s", "sqlite")  # type: ignore[arg-type]


# ── conversation() factory ──────────────────────────────────────────────


def test_conversation_caches_same_session_id_and_backend():
    """Repeated calls with the same key return the identical instance."""
    a = conversation("sess", backend="memory")
    b = conversation("sess", backend="memory")
    assert a is b


def test_conversation_distinct_session_ids_are_separate():
    """Different session IDs yield distinct sessions."""
    a = conversation("one", backend="memory")
    b = conversation("two", backend="memory")
    assert a is not b


def test_conversation_persists_messages_across_factory_calls():
    """Because the instance is cached, history survives a second lookup."""
    conversation("chat", backend="memory").add_user("earlier")
    again = conversation("chat", backend="memory")
    assert [m.content for m in again.messages] == ["earlier"]
