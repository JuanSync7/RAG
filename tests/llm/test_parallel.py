"""Tests for src.common.llm.parallel — ``parallel`` and ``aparallel``.

Covers the thread-pool path (plain callables), the LangChain Runnable
path, the ``_is_runnable`` predicate, empty input, and the async variant
mixing sync + async tasks.

NOTE: these tests live under tests/llm/ on purpose — the real
``langchain_core`` is bound there (the suite-wide stub is evicted), which
the Runnable path depends on.
"""

from __future__ import annotations

import asyncio

from langchain_core.runnables import RunnableLambda

from src.common.llm.parallel import _is_runnable, aparallel, parallel
from src.common.llm.schemas import ParallelResult


# ── parallel (sync, thread path) ────────────────────────────────────────


def test_parallel_happy_thread_path():
    result = parallel(
        a=lambda: "alpha",
        b=lambda: 42,
    )

    assert isinstance(result, ParallelResult)
    assert result.results == {"a": "alpha", "b": 42}
    assert result.errors == {}
    assert set(result.timings) == {"a", "b"}
    for name in ("a", "b"):
        assert isinstance(result.timings[name], float)
        assert result.timings[name] >= 0.0


def test_parallel_partial_failure_thread_path():
    def boom():
        raise RuntimeError("kaboom")

    result = parallel(
        ok=lambda: "fine",
        bad=boom,
    )

    assert result.results == {"ok": "fine"}
    assert "bad" not in result.results
    assert set(result.errors) == {"bad"}
    assert isinstance(result.errors["bad"], RuntimeError)
    assert str(result.errors["bad"]) == "kaboom"
    # The failing task is still timed (exercises _timed_safe_call's exc branch).
    assert set(result.timings) == {"ok", "bad"}
    assert result.timings["bad"] >= 0.0


def test_parallel_empty():
    result = parallel()

    assert isinstance(result, ParallelResult)
    assert result.results == {}
    assert result.timings == {}
    assert result.errors == {}


# ── _is_runnable predicate ──────────────────────────────────────────────


def test_is_runnable_true_for_runnable_lambda():
    assert _is_runnable(RunnableLambda(lambda _: 1)) is True


def test_is_runnable_false_for_plain_lambda():
    assert _is_runnable(lambda: 1) is False


# ── parallel (Runnable path) ────────────────────────────────────────────


def test_parallel_runnable_path():
    # Mixing a Runnable with a plain callable forces has_runnable=True so the
    # whole batch routes through _run_langchain (not _run_threaded).
    #
    # LATENT BUG (documented, not worked around to keep source byte-identical):
    # _run_langchain wraps a Runnable as
    #   RunnableLambda(lambda _input, _t=task: _invoke_and_record(_n, _t.invoke, ...))
    # and _invoke_and_record calls fn() with NO argument — but Runnable.invoke
    # requires a positional ``input``. So a real Runnable task ALWAYS lands in
    # ``errors`` via the langchain path. The wrapper receives ``_input`` but
    # never forwards it to ``_t.invoke``. This test pins that observed behavior.
    def boom():
        raise RuntimeError("runnable-path-boom")

    result = parallel(
        r=RunnableLambda(lambda _: "from-runnable"),
        plain=lambda: "from-plain",
        bad=boom,
    )

    # The plain callable, wrapped & invoked with no args inside the langchain
    # path, succeeds and lands in results.
    assert result.results["plain"] == "from-plain"

    # Both the Runnable task and the deliberately-failing plain task are
    # captured in errors and filtered OUT of results (exercises the
    # `if name not in errors` guard in _run_langchain — M7 teeth).
    assert "r" not in result.results
    assert "bad" not in result.results
    assert set(result.errors) == {"r", "bad"}

    # The Runnable failure carries the langchain-path-specific message. The
    # thread path (has_runnable=False mutation) would instead report
    # "'RunnableLambda' object is not callable" — so this assertion gives the
    # has_runnable routing decision teeth (M6).
    assert isinstance(result.errors["r"], TypeError)
    assert "input" in str(result.errors["r"])

    assert isinstance(result.errors["bad"], RuntimeError)
    assert str(result.errors["bad"]) == "runnable-path-boom"

    # All three tasks are timed regardless of success/failure.
    assert set(result.timings) == {"r", "plain", "bad"}


# ── aparallel (async, mixed sync + async) ───────────────────────────────


def test_aparallel_happy_mixed():
    async def async_task():
        await asyncio.sleep(0)
        return "async-val"

    def sync_task():
        return "sync-val"

    result = asyncio.run(
        aparallel(asy=async_task, syn=sync_task)
    )

    assert result.results == {"asy": "async-val", "syn": "sync-val"}
    assert result.errors == {}
    assert set(result.timings) == {"asy", "syn"}
    for name in ("asy", "syn"):
        assert isinstance(result.timings[name], float)
        assert result.timings[name] >= 0.0


def test_aparallel_partial_failure():
    async def good():
        await asyncio.sleep(0)
        return "good-val"

    async def bad():
        await asyncio.sleep(0)
        raise ValueError("async-boom")

    result = asyncio.run(aparallel(good=good, bad=bad))

    assert result.results == {"good": "good-val"}
    assert "bad" not in result.results
    assert set(result.errors) == {"bad"}
    assert isinstance(result.errors["bad"], ValueError)
    assert str(result.errors["bad"]) == "async-boom"
    assert set(result.timings) == {"good", "bad"}
