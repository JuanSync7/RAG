"""Unit tests for ``src.common.llm.fallback``.

Covers the strategy-level failover combinators ``fallback_chain`` (sync)
and ``afallback_chain`` (async): first-success wins, correct
``strategy_used`` / ``strategies_tried`` accounting, args/kwargs
forwarding, and last-exception re-raise when every strategy fails.  No
external boundaries — strategies are plain callables supplied by the test.
"""
from __future__ import annotations

import pytest

from src.common.llm.fallback import afallback_chain, fallback_chain
from src.common.llm.schemas import FallbackResult


# ── fallback_chain (sync) ───────────────────────────────────────────────


def test_first_strategy_succeeds_index_zero():
    """First strategy wins: strategy_used=0, strategies_tried=1."""
    run = fallback_chain(lambda: "ok")
    res = run()
    assert isinstance(res, FallbackResult)
    assert res.result == "ok"
    assert res.strategy_used == 0
    assert res.strategies_tried == 1


def test_second_strategy_used_after_first_raises():
    """When the first strategy raises, the second wins with index 1."""

    def bad():
        raise ValueError("first fails")

    run = fallback_chain(bad, lambda: "rescued")
    res = run()
    assert res.result == "rescued"
    assert res.strategy_used == 1
    assert res.strategies_tried == 2


def test_third_strategy_used_after_two_raise():
    """Two failures then success: strategy_used=2, strategies_tried=3."""

    def fail(_msg):
        def _f():
            raise RuntimeError(_msg)

        return _f

    run = fallback_chain(fail("a"), fail("b"), lambda: "win")
    res = run()
    assert res.result == "win"
    assert res.strategy_used == 2
    assert res.strategies_tried == 3


def test_args_and_kwargs_forwarded_to_strategy():
    """Positional and keyword args reach the winning strategy unchanged."""
    run = fallback_chain(lambda a, *, b: a + b)
    res = run(10, b=5)
    assert res.result == 15


def test_all_strategies_fail_reraises_last_exception():
    """When every strategy raises, the LAST exception propagates."""

    def first():
        raise ValueError("first")

    def last():
        raise KeyError("last")

    run = fallback_chain(first, last)
    with pytest.raises(KeyError, match="last"):
        run()


def test_falsy_result_still_counts_as_success():
    """A strategy returning a falsy value (0) is a success, not a failure."""
    run = fallback_chain(lambda: 0, lambda: 99)
    res = run()
    assert res.result == 0
    assert res.strategy_used == 0


# ── afallback_chain (async) ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_async_first_succeeds():
    """Async first strategy wins with index 0."""

    async def ok():
        return "async-ok"

    run = afallback_chain(ok)
    res = await run()
    assert res.result == "async-ok"
    assert res.strategy_used == 0
    assert res.strategies_tried == 1


@pytest.mark.asyncio
async def test_async_second_used_after_first_raises():
    """Async failover advances to the second strategy on failure."""

    async def bad():
        raise ValueError("nope")

    async def good():
        return "async-rescued"

    run = afallback_chain(bad, good)
    res = await run()
    assert res.result == "async-rescued"
    assert res.strategy_used == 1
    assert res.strategies_tried == 2


@pytest.mark.asyncio
async def test_async_all_fail_reraises_last():
    """Async: last exception propagates when all strategies fail."""

    async def first():
        raise ValueError("first")

    async def last():
        raise KeyError("last")

    run = afallback_chain(first, last)
    with pytest.raises(KeyError, match="last"):
        await run()


@pytest.mark.asyncio
async def test_async_forwards_args():
    """Async strategy receives forwarded args/kwargs."""

    async def add(a, *, b):
        return a + b

    run = afallback_chain(add)
    res = await run(7, b=8)
    assert res.result == 15
