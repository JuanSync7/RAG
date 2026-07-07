"""Tests for src.common.llm.batch — sync ``batch`` and async ``abatch``.

Covers happy path, partial failure (item↔exception pairing), empty input,
concurrency recording, and — for the async variant — that the semaphore
actually bounds in-flight concurrency.
"""

from __future__ import annotations

import asyncio

from src.common.llm.batch import abatch, batch
from src.common.llm.schemas import BatchResult


# ── batch (sync, ThreadPoolExecutor) ────────────────────────────────────


def test_batch_happy_all_succeed():
    items = [1, 2, 3, 4]
    result = batch(lambda x: x * 10, items, max_concurrency=4)

    assert isinstance(result, BatchResult)
    # as_completed → order is non-deterministic; compare as a set.
    assert set(result.succeeded) == {10, 20, 30, 40}
    assert result.failed == []
    assert result.total == 4
    assert result.concurrency == 4


def test_batch_partial_failure_pairs_item_to_exception():
    # Even items succeed; odd items raise ValueError(str(item)).
    def fn(x: int) -> int:
        if x % 2 == 1:
            raise ValueError(str(x))
        return x * 100

    items = [1, 2, 3, 4, 5]
    result = batch(fn, items, max_concurrency=5)

    assert set(result.succeeded) == {200, 400}
    assert result.total == 5

    # Build a {item: exc} map and assert the pairing is exactly right.
    failed_map = {item: exc for item, exc in result.failed}
    assert set(failed_map) == {1, 3, 5}
    for item, exc in failed_map.items():
        assert isinstance(exc, ValueError)
        # The exception message is str(item) — gives the future_to_item
        # mapping teeth: a swapped/dropped item would mismatch here.
        assert str(exc) == str(item)


def test_batch_empty_items():
    result = batch(lambda x: x, [], max_concurrency=10)

    assert result.total == 0
    assert result.succeeded == []
    assert result.failed == []


def test_batch_concurrency_recorded():
    result = batch(lambda x: x, [1, 2], max_concurrency=3)

    assert result.concurrency == 3


# ── abatch (async, asyncio.Semaphore) ───────────────────────────────────


def test_abatch_happy_all_succeed():
    async def fn(x: int) -> int:
        return x + 1

    result = asyncio.run(abatch(fn, [10, 20, 30], max_concurrency=2))

    assert set(result.succeeded) == {11, 21, 31}
    assert result.failed == []
    assert result.total == 3
    assert result.concurrency == 2


def test_abatch_partial_failure_pairs_item_to_exception():
    async def fn(x: int) -> int:
        if x < 0:
            raise ValueError(str(x))
        return x * 2

    items = [-1, 2, -3, 4]
    result = asyncio.run(abatch(fn, items, max_concurrency=4))

    assert set(result.succeeded) == {4, 8}
    assert result.total == 4

    failed_map = {item: exc for item, exc in result.failed}
    assert set(failed_map) == {-1, -3}
    for item, exc in failed_map.items():
        assert isinstance(exc, ValueError)
        assert str(exc) == str(item)


def test_abatch_respects_semaphore_bound():
    max_concurrency = 2
    state = {"in_flight": 0, "observed_max": 0}

    async def fn(x: int) -> int:
        state["in_flight"] += 1
        state["observed_max"] = max(state["observed_max"], state["in_flight"])
        # Yield control so other coroutines can interleave if the semaphore
        # allowed them to. Multiple sleeps widen the overlap window.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        state["in_flight"] -= 1
        return x

    result = asyncio.run(
        abatch(fn, list(range(6)), max_concurrency=max_concurrency)
    )

    assert set(result.succeeded) == set(range(6))
    assert result.total == 6
    # The semaphore must never let more than max_concurrency run at once.
    assert state["observed_max"] <= max_concurrency
