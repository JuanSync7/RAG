# @summary
# LLM response caching — same input + same model = cached response.
# Delegates to the platform cache provider (Redis by default) for the
# application-level cache, and wraps langchain-core's global cache for
# LangChain-integrated LLM calls.
# Exports: enable_cache, disable_cache, clear_cache
# Deps: langchain_core.caches, langchain_core.globals, src.platform.cache.provider
# @end-summary
"""LLM response caching utilities.

Two layers of caching are available:

* **LangChain global cache** — transparently caches any ``BaseChatModel``
  call.  Uses an in-memory or SQLite backend via langchain-core (process-local).
* **Platform cache** — Redis-backed (``src.platform.cache``), shared across
  processes, with TTL.  Callers use this for application-level caching
  (e.g. deduplicating identical retrieval queries).

:func:`enable_cache` activates the LangChain layer.  The platform cache is
always available via :func:`get_platform_cache`.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Sequence

from langchain_core.caches import BaseCache, InMemoryCache
from langchain_core.globals import get_llm_cache, set_llm_cache
from langchain_core.outputs import Generation

from src.platform.cache import (
    CacheProvider,
    get_cache,
)
from config.settings import RAG_LLM_CACHE_DEFAULT_TTL_S

logger = logging.getLogger(__name__)


# ── Platform cache bridge ────────────────────────────────────────────────


def get_platform_cache() -> CacheProvider:
    """Return the project's Redis/in-memory cache provider.

    This is the same singleton exposed by ``src.platform.cache.get_cache()``,
    re-exported here so callers of ``src.common.llm`` don't need a separate
    import for application-level caching.

    Returns:
        The configured ``CacheProvider`` (Redis, in-memory TTL, or no-op).
    """
    return get_cache()


# ── LangChain global cache (for BaseChatModel calls) ────────────────────


class _RedisChatCache(BaseCache):
    """LangChain cache backend that delegates to the platform Redis cache.

    This plugs into langchain-core's global ``set_llm_cache()`` mechanism
    so that all ``BaseChatModel.invoke()`` calls are transparently cached
    in the same Redis instance the rest of the project uses.
    """

    def __init__(self, platform_cache: CacheProvider, ttl: int = RAG_LLM_CACHE_DEFAULT_TTL_S) -> None:
        self._cache = platform_cache
        self._ttl = ttl

    @staticmethod
    def _key(prompt: str, llm_string: str) -> str:
        raw = f"llm_cache:{prompt}|||{llm_string}"
        return f"llm_cache:{hashlib.sha256(raw.encode()).hexdigest()}"

    @staticmethod
    def _serialise(generations: Sequence[Generation]) -> list[dict[str, Any]]:
        return [{"text": g.text, "info": g.generation_info} for g in generations]

    @staticmethod
    def _deserialise(data: list[dict[str, Any]]) -> list[Generation]:
        return [
            Generation(text=d["text"], generation_info=d.get("info"))
            for d in data
        ]

    def lookup(self, prompt: str, llm_string: str) -> list[Generation] | None:
        """Return cached generations or *None* on miss."""
        hit = self._cache.get(self._key(prompt, llm_string))
        if hit is None:
            return None
        return self._deserialise(hit)

    def update(self, prompt: str, llm_string: str, return_val: Sequence[Generation]) -> None:
        """Store generations in Redis with TTL."""
        self._cache.set(
            self._key(prompt, llm_string),
            self._serialise(return_val),
            ttl_seconds=self._ttl,
        )

    def clear(self, **kwargs: Any) -> None:
        """Clear is a no-op — Redis entries expire via TTL."""
        logger.info("clear() called on Redis LLM cache — entries expire via TTL")


# ── Public API ───────────────────────────────────────────────────────────


def enable_cache(
    backend: str = "redis",
    *,
    ttl: int = RAG_LLM_CACHE_DEFAULT_TTL_S,
) -> None:
    """Activate LLM response caching globally for LangChain calls.

    Args:
        backend: ``"redis"`` (default — uses platform Redis cache),
                 or ``"memory"`` (in-process, no persistence).
        ttl: Cache entry TTL in seconds (Redis only).  Defaults to
             ``RAG_LLM_CACHE_DEFAULT_TTL_S`` (1 hour).

    Raises:
        ValueError: If *backend* is not recognised.
    """
    if backend == "redis":
        platform_cache = get_cache()
        cache = _RedisChatCache(platform_cache, ttl=ttl)
    elif backend == "memory":
        cache = InMemoryCache()
    else:
        raise ValueError(
            f"Unknown cache backend {backend!r}. Use 'redis' or 'memory'."
        )
    set_llm_cache(cache)
    logger.info("LLM cache enabled (backend=%s)", backend)


def disable_cache() -> None:
    """Turn off LLM caching (does not wipe existing entries)."""
    set_llm_cache(None)  # type: ignore[arg-type]


def clear_cache() -> None:
    """Wipe all cached responses from the active LangChain cache backend.

    No-op if caching is not currently enabled.
    """
    cache = get_llm_cache()
    if cache is not None:
        cache.clear()


__all__ = [
    "enable_cache",
    "disable_cache",
    "clear_cache",
    "get_platform_cache",
]
