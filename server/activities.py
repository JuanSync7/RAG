# @summary
# Temporal activity that executes RAG queries against a preloaded RAGChain singleton.
# The singleton is initialized once by the worker at startup — not per request.
# Exports: execute_rag_query, init_rag_chain, shutdown_rag_chain
# Deps: temporalio, src.retrieval.pipeline.rag_chain, dataclasses
# @end-summary
"""Temporal activities for the RAG query pipeline.

The RAGChain singleton is loaded once at worker startup. Activities run
inference against models already in memory — no per-request init cost.
"""

import logging
import time
import hashlib
import orjson
from dataclasses import asdict
from typing import Optional

from temporalio import activity
from config.settings import RAG_RETRIEVAL_TIMEOUT_MS
from config.settings import RAG_AGENTIC_RETRIEVAL_ENABLED
from src.platform.cache import get_cache
from src.platform import (
    CACHE_HITS,
    CACHE_MISSES,
)

logger = logging.getLogger("rag.server.activities")

_rag_chain = None
_cache = get_cache()


def init_rag_chain() -> None:
    """Load RAGChain once at worker startup. Called before the worker loop starts."""
    global _rag_chain
    if _rag_chain is not None:
        logger.warning("RAGChain already initialized, skipping re-init")
        return

    from src.retrieval.pipeline import RAGChain
    logger.info("Initializing RAGChain (loading models into memory)...")
    start = time.time()
    _rag_chain = RAGChain()
    logger.info("RAGChain ready in %.1fs", time.time() - start)


def shutdown_rag_chain() -> None:
    """Release the RAGChain singleton and its resources (for graceful shutdown)."""
    global _rag_chain
    if _rag_chain is not None:
        _rag_chain.close()
    _rag_chain = None
    logger.info("RAGChain released")


def get_rag_chain():
    """Get the preloaded RAGChain. Raises if not initialized."""
    if _rag_chain is None:
        raise RuntimeError(
            "RAGChain not initialized. The worker must call init_rag_chain() at startup."
        )
    return _rag_chain


@activity.defn
def execute_rag_query(request: dict) -> dict:
    """Run a RAG query against the preloaded models.

    This activity runs synchronously in a ThreadPoolExecutor managed by the
    Temporal worker. The RAGChain and all GPU models are already in memory.
    """
    rag = get_rag_chain()

    query = request["query"]
    source_filter: Optional[str] = request.get("source_filter")
    heading_filter: Optional[str] = request.get("heading_filter")
    alpha: float = request.get("alpha", 0.5)
    search_limit: int = request.get("search_limit", 10)
    rerank_top_k: int = request.get("rerank_top_k", 5)
    skip_generation: bool = request.get("skip_generation", False)
    tenant_id: Optional[str] = request.get("tenant_id")
    max_query_iterations: int = int(request.get("max_query_iterations", 3))
    fast_path: Optional[bool] = request.get("fast_path")
    overall_timeout_ms: int = int(request.get("overall_timeout_ms", RAG_RETRIEVAL_TIMEOUT_MS))
    stage_budget_overrides: dict = request.get("stage_budget_overrides", {}) or {}
    conversation_id: Optional[str] = request.get("conversation_id")
    memory_context: Optional[str] = request.get("memory_context")
    memory_recent_turns: list[dict] = request.get("memory_recent_turns", []) or []
    ignored_doc_ids: list[str] = list(request.get("ignored_doc_ids", []) or [])
    mode: str = str(request.get("mode") or "query")
    retrieval_sub_mode: str = str(request.get("retrieval_sub_mode") or "auto")
    extra_processing: bool = bool(request.get("extra_processing") or False)
    # Tree retrieval per-request override (TREE_RETRIEVAL_DESIGN.md §6).
    # Only treat as override when explicitly present and bool-typed; else None.
    _tree = request.get("tree_retrieval")
    tree_retrieval: Optional[bool] = _tree if isinstance(_tree, bool) else None
    deep_research: bool = bool(request.get("deep_research") or False)
    # Agentic-retrieval per-request override (AGENTIC_RETRIEVAL_DESIGN.md).
    # Only treat as override when explicitly present and bool-typed; else None
    # so the resolution falls back to the RAG_AGENTIC_RETRIEVAL_ENABLED config.
    _agentic = request.get("agentic_retrieval")
    agentic_retrieval: Optional[bool] = _agentic if isinstance(_agentic, bool) else None
    _max_rounds = request.get("max_agentic_rounds")
    max_agentic_rounds: Optional[int] = (
        int(_max_rounds) if isinstance(_max_rounds, int) else None
    )
    # Whether the agentic loop will actually run for this request — drives the
    # result-cache bypass below (the controller/judge loop is nondeterministic
    # by design, so caching one stochastic outcome under a deterministic key
    # would stale-serve).
    _agentic_active: bool = (
        agentic_retrieval
        if isinstance(agentic_retrieval, bool)
        else RAG_AGENTIC_RETRIEVAL_ENABLED
    )
    # Retrieval mode skips answer generation by definition. The route
    # handler may also set skip_generation directly (e.g. for streaming);
    # union both signals so neither path accidentally enables generation.
    if mode == "retrieval":
        skip_generation = True

    cache_payload = {
        "query": query,
        "source_filter": source_filter,
        "heading_filter": heading_filter,
        "alpha": alpha,
        "search_limit": search_limit,
        "rerank_top_k": rerank_top_k,
        "skip_generation": skip_generation,
        "tenant_id": tenant_id,
        "max_query_iterations": max_query_iterations,
        "fast_path": fast_path,
        "overall_timeout_ms": overall_timeout_ms,
        "stage_budget_overrides": stage_budget_overrides,
        "conversation_id": conversation_id,
        "memory_context": memory_context,
        "memory_recent_turns": memory_recent_turns,
        "ignored_doc_ids": sorted(ignored_doc_ids),
        "mode": mode,
        "retrieval_sub_mode": retrieval_sub_mode,
        "extra_processing": extra_processing,
        "tree_retrieval": tree_retrieval,
        "deep_research": deep_research,
        "agentic_retrieval": agentic_retrieval,
        "max_agentic_rounds": max_agentic_rounds,
    }
    cache_key = "rag:query:" + hashlib.sha256(
        orjson.dumps(cache_payload, option=orjson.OPT_SORT_KEYS)
    ).hexdigest()

    # The agentic loop is nondeterministic (controller/judge LLM calls), so a
    # cached result under a deterministic key would stale-serve. Bypass the
    # result cache entirely when the loop is active. The flag is still part of
    # cache_payload above so an agentic vs non-agentic run of the same query
    # can never collide on the non-agentic (cached) path.
    if not _agentic_active:
        cached = _cache.get(cache_key)
        if isinstance(cached, dict):
            CACHE_HITS.labels(layer="activity_result").inc()
            return cached
        CACHE_MISSES.labels(layer="activity_result").inc()

    activity.logger.info("Processing query: %s", query[:80])

    start = time.perf_counter()
    response = rag.run(
        query=query,
        alpha=alpha,
        search_limit=search_limit,
        rerank_top_k=rerank_top_k,
        source_filter=source_filter,
        heading_filter=heading_filter,
        skip_generation=skip_generation,
        tenant_id=tenant_id,
        max_query_iterations=max_query_iterations,
        fast_path=fast_path,
        overall_timeout_ms=overall_timeout_ms,
        stage_budget_overrides=stage_budget_overrides,
        conversation_id=conversation_id,
        memory_context=memory_context,
        memory_recent_turns=memory_recent_turns,
        ignored_doc_ids=ignored_doc_ids,
        mode=mode,
        retrieval_sub_mode=retrieval_sub_mode,
        extra_processing=extra_processing,
        tree_retrieval=tree_retrieval,
        deep_research=deep_research,
        agentic_retrieval=agentic_retrieval,
        max_agentic_rounds=max_agentic_rounds,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    result = asdict(response)
    result["latency_ms"] = round(elapsed_ms, 1)

    activity.logger.info(
        "Query complete in %.0fms — %d results, action=%s, generation=%s",
        elapsed_ms,
        len(response.results),
        response.action,
        "skipped" if skip_generation else "included",
    )
    # Do not cache nondeterministic agentic-loop results (see bypass above).
    if not _agentic_active:
        _cache.set(cache_key, result)
    return result
