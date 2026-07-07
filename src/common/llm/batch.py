# @summary
# Concurrent batch processing with throttling (sync and async).
# Exports: batch, abatch
# Deps: asyncio, concurrent.futures, src.common.llm.schemas
# @end-summary
"""Batch processing utilities for the LLM composition layer.

Provides ``batch()`` for synchronous thread-pool concurrency and
``abatch()`` for native async concurrency, both with configurable
throttling via *max_concurrency*.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Sequence

from config.settings import RAG_LLM_BATCH_MAX_CONCURRENCY
from src.common.llm.schemas import BatchResult


def batch(
    fn: Callable[[Any], Any],
    items: Sequence[Any],
    *,
    max_concurrency: int = RAG_LLM_BATCH_MAX_CONCURRENCY,
) -> BatchResult:
    """Process *items* through *fn* using a bounded thread pool.

    Each item is submitted as ``fn(item)``.  Successes are collected in
    ``result.succeeded``; failures in ``result.failed`` as
    ``(item, exception)`` tuples.
    """
    result = BatchResult(total=len(items), concurrency=max_concurrency)

    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        future_to_item = {pool.submit(fn, item): item for item in items}
        for future in as_completed(future_to_item):
            item = future_to_item[future]
            try:
                result.succeeded.append(future.result())
            except Exception as exc:
                result.failed.append((item, exc))

    return result


async def abatch(
    fn: Callable[[Any], Any],
    items: Sequence[Any],
    *,
    max_concurrency: int = RAG_LLM_BATCH_MAX_CONCURRENCY,
) -> BatchResult:
    """Async variant of :func:`batch` using a semaphore for backpressure.

    *fn* must be an awaitable callable (async function).
    """
    sem = asyncio.Semaphore(max_concurrency)
    result = BatchResult(total=len(items), concurrency=max_concurrency)

    async def _run(item: Any) -> None:
        async with sem:
            try:
                result.succeeded.append(await fn(item))
            except Exception as exc:
                result.failed.append((item, exc))

    await asyncio.gather(*(_run(item) for item in items))
    return result


__all__ = ["batch", "abatch"]
