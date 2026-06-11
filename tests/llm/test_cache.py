"""Unit tests for ``src.common.llm.cache``.

Covers the LangChain-cache bridge ``_RedisChatCache`` (deterministic key
derivation, generation (de)serialisation round-trip, lookup miss/hit,
TTL-passing update, no-op clear) and the public toggles ``enable_cache`` /
``disable_cache`` / ``clear_cache``.  The Redis boundary is replaced by an
in-memory fake ``CacheProvider``; the LangChain global cache registry
(``get_llm_cache`` / ``set_llm_cache``) is exercised for real and restored
after each test.
"""
from __future__ import annotations

import pytest

import src.common.llm.cache as cache_mod
from src.common.llm.cache import (
    _RedisChatCache,
    clear_cache,
    disable_cache,
    enable_cache,
)

# Route these through the cache module's own namespace so we share the exact
# langchain_core class/function identities that the source code uses. Sibling
# test modules evict and re-import langchain_core; a direct import here could
# bind a different module identity and desync the global-cache registry.
Generation = cache_mod.Generation
get_llm_cache = cache_mod.get_llm_cache
set_llm_cache = cache_mod.set_llm_cache


class _FakeCache:
    """In-memory stand-in for the platform CacheProvider."""

    def __init__(self):
        self.store: dict[str, object] = {}
        self.set_calls: list[tuple[str, object, int]] = []

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value, *, ttl_seconds):
        self.store[key] = value
        self.set_calls.append((key, value, ttl_seconds))


@pytest.fixture
def restore_global_cache():
    """Snapshot and restore the LangChain global LLM cache around a test."""
    original = get_llm_cache()
    yield
    set_llm_cache(original)


# ── _RedisChatCache._key ────────────────────────────────────────────────


def test_key_is_deterministic():
    """Same (prompt, llm_string) always yields the same key."""
    k1 = _RedisChatCache._key("hi", "model-x")
    k2 = _RedisChatCache._key("hi", "model-x")
    assert k1 == k2
    assert k1.startswith("llm_cache:")


def test_key_differs_on_prompt():
    """Different prompts produce different keys."""
    assert _RedisChatCache._key("a", "m") != _RedisChatCache._key("b", "m")


def test_key_differs_on_llm_string():
    """Different model strings produce different keys."""
    assert _RedisChatCache._key("p", "m1") != _RedisChatCache._key("p", "m2")


# ── serialise / deserialise round-trip ──────────────────────────────────


def test_serialise_preserves_text_and_info():
    """Serialisation captures text and generation_info per generation."""
    gens = [Generation(text="hello", generation_info={"finish": "stop"})]
    data = _RedisChatCache._serialise(gens)
    assert data == [{"text": "hello", "info": {"finish": "stop"}}]


def test_deserialise_round_trip():
    """A serialise→deserialise round-trip reconstructs the generations."""
    gens = [Generation(text="r1", generation_info={"k": 1}), Generation(text="r2")]
    data = _RedisChatCache._serialise(gens)
    restored = _RedisChatCache._deserialise(data)
    assert [g.text for g in restored] == ["r1", "r2"]
    assert restored[0].generation_info == {"k": 1}
    assert restored[1].generation_info is None


# ── lookup / update ─────────────────────────────────────────────────────


def test_lookup_miss_returns_none():
    """A key absent from the backing store yields None."""
    rc = _RedisChatCache(_FakeCache())
    assert rc.lookup("prompt", "model") is None


def test_update_then_lookup_round_trips_generations():
    """update() stores generations that lookup() reconstructs."""
    rc = _RedisChatCache(_FakeCache())
    rc.update("p", "m", [Generation(text="cached", generation_info={"a": 2})])
    hit = rc.lookup("p", "m")
    assert hit is not None
    assert hit[0].text == "cached"
    assert hit[0].generation_info == {"a": 2}


def test_update_passes_configured_ttl():
    """update() forwards the cache's TTL to the backing store."""
    fake = _FakeCache()
    rc = _RedisChatCache(fake, ttl=123)
    rc.update("p", "m", [Generation(text="x")])
    assert fake.set_calls[0][2] == 123


def test_lookup_uses_same_key_as_update():
    """lookup and update agree on the derived key for a given input."""
    fake = _FakeCache()
    rc = _RedisChatCache(fake)
    rc.update("same", "model", [Generation(text="v")])
    stored_key = fake.set_calls[0][0]
    assert stored_key == _RedisChatCache._key("same", "model")
    assert rc.lookup("same", "model") is not None


def test_clear_is_noop_and_does_not_touch_store():
    """clear() logs but leaves stored entries intact (TTL-driven expiry)."""
    fake = _FakeCache()
    rc = _RedisChatCache(fake)
    rc.update("p", "m", [Generation(text="keep")])
    rc.clear()
    assert rc.lookup("p", "m") is not None


# ── enable_cache / disable_cache / clear_cache ──────────────────────────


def test_enable_cache_memory_sets_inmemory_backend(restore_global_cache):
    """backend='memory' installs an InMemoryCache as the global cache."""
    enable_cache(backend="memory")
    # Use the cache module's own InMemoryCache reference: sibling test modules
    # evict/re-import langchain_core, so a fresh import here could be a
    # different class identity than the one enable_cache() instantiated.
    assert isinstance(get_llm_cache(), cache_mod.InMemoryCache)


def test_enable_cache_redis_installs_redis_bridge(restore_global_cache, monkeypatch):
    """backend='redis' installs a _RedisChatCache wired to the platform cache with TTL."""
    fake = _FakeCache()
    monkeypatch.setattr(cache_mod, "get_cache", lambda: fake)
    enable_cache(backend="redis", ttl=77)
    active = get_llm_cache()
    assert isinstance(active, _RedisChatCache)
    assert active._ttl == 77
    assert active._cache is fake


def test_enable_cache_unknown_backend_raises(restore_global_cache):
    """An unrecognised backend name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown cache backend"):
        enable_cache(backend="postgres")


def test_disable_cache_clears_global(restore_global_cache):
    """disable_cache sets the global LLM cache back to None."""
    enable_cache(backend="memory")
    disable_cache()
    assert get_llm_cache() is None


def test_clear_cache_calls_active_backend_clear(restore_global_cache):
    """clear_cache delegates to the active backend's clear()."""
    calls = []

    class _Spy:
        def clear(self, **kwargs):
            calls.append(True)

    set_llm_cache(_Spy())
    clear_cache()
    assert calls == [True]


def test_clear_cache_noop_when_disabled(restore_global_cache):
    """clear_cache does nothing when no cache is active."""
    set_llm_cache(None)
    clear_cache()  # must not raise
    assert get_llm_cache() is None
