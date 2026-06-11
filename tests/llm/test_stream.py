"""Unit tests for ``src.common.llm.stream``.

Covers ``stream`` / ``astream``: the Runnable-vs-plain-callable branch,
token chunk assembly (single chunk vs summed chunks), STEP_START /
LLM_TOKEN / STEP_END event emission via the optional callback, elapsed_ms
metadata, and the async-specific coroutine-function branch.  The only
boundary is the LangChain ``Runnable`` interface, exercised with a thin
real subclass of the live ``Runnable`` class so the ``isinstance`` dispatch
and chunk-assembly path are wired for real rather than mocked.
"""
from __future__ import annotations

import sys

from src.common.llm.schemas import StreamEvent
from src.common.llm.stream import astream, stream

# Capture the exact ``Runnable`` class that the imported ``stream`` / ``astream``
# functions check against, AT IMPORT TIME. The conftest swaps the langchain
# stub for the real package while collecting tests/llm and swaps it back for
# other packages, so sys.modules["src.common.llm.stream"] may be gone by the
# time these tests *run*. The bound ``stream`` function still closes over its
# original module globals, so we must subclass the Runnable captured here for
# the source's ``isinstance(x, Runnable)`` dispatch to hold at run time.
_RUNNABLE_BASE = sys.modules["src.common.llm.stream"].Runnable


def _live_runnable_base():
    """Return the ``Runnable`` class the imported ``stream`` dispatches on."""
    return _RUNNABLE_BASE


class _Chunk:
    """Addable chunk standing in for an AIMessageChunk.

    ``stream`` accumulates multi-chunk output via ``sum(chunks[1:], chunks[0])``,
    which requires ``__add__`` (plain ``str`` is rejected by ``sum``). This
    mirrors how LangChain message chunks concatenate.
    """

    def __init__(self, text):
        self.text = text

    def __add__(self, other):
        return _Chunk(self.text + other.text)

    def __eq__(self, other):
        return isinstance(other, _Chunk) and self.text == other.text

    def __repr__(self):  # pragma: no cover - debug aid
        return f"_Chunk({self.text!r})"


def _make_streaming_runnable(chunks):
    """A Runnable whose .stream() yields the given chunks in order."""

    class _StreamingRunnable(_live_runnable_base()):
        def invoke(self, _input, config=None, **kwargs):  # pragma: no cover - unused
            return chunks[-1]

        def stream(self, _input, config=None, **kwargs):
            yield from chunks

    return _StreamingRunnable()


def _make_async_streaming_runnable(chunks):
    """A Runnable whose .astream() yields the given chunks in order."""

    class _AsyncStreamingRunnable(_live_runnable_base()):
        def invoke(self, _input, config=None, **kwargs):  # pragma: no cover - unused
            return chunks[-1]

        async def astream(self, _input, config=None, **kwargs):
            for c in chunks:
                yield c

    return _AsyncStreamingRunnable()


# ── stream (sync) ───────────────────────────────────────────────────────


def test_stream_plain_callable_returns_result():
    """A plain callable is invoked directly and its result returned."""
    result = stream(lambda x: x * 2, 21)
    assert result == 42


def test_stream_plain_callable_emits_start_and_end_only():
    """Plain callables emit STEP_START then STEP_END (no LLM_TOKEN)."""
    events = []
    stream(lambda x: x + 1, 5, on_event=events.append, step_name="inc")
    assert [f.event for f in events] == [
        StreamEvent.STEP_START,
        StreamEvent.STEP_END,
    ]
    assert events[-1].data == 6
    assert events[-1].step_name == "inc"


def test_stream_no_callback_does_not_error():
    """Omitting on_event is allowed; the result still comes back."""
    assert stream(lambda x: x, "hi") == "hi"


def test_stream_runnable_single_chunk_returns_that_chunk():
    """A single streamed chunk is returned as-is (no summing)."""
    runnable = _make_streaming_runnable(["solo"])
    result = stream(runnable, "in")
    assert result == "solo"


def test_stream_runnable_multiple_chunks_are_summed():
    """Multiple chunks are concatenated via summation (chunk __add__)."""
    runnable = _make_streaming_runnable([_Chunk("a"), _Chunk("b"), _Chunk("c")])
    result = stream(runnable, "in")
    assert result == _Chunk("abc")


def test_stream_runnable_numeric_chunks_summed():
    """Numeric chunks accumulate by addition, proving the reduce path."""
    runnable = _make_streaming_runnable([1, 2, 3, 4])
    assert stream(runnable, "in") == 10


def test_stream_runnable_emits_one_token_per_chunk():
    """Each streamed chunk produces exactly one LLM_TOKEN frame."""
    events = []
    runnable = _make_streaming_runnable([_Chunk("x"), _Chunk("y")])
    stream(runnable, "in", on_event=events.append)
    kinds = [f.event for f in events]
    assert kinds == [
        StreamEvent.STEP_START,
        StreamEvent.LLM_TOKEN,
        StreamEvent.LLM_TOKEN,
        StreamEvent.STEP_END,
    ]
    token_data = [f.data for f in events if f.event == StreamEvent.LLM_TOKEN]
    assert token_data == [_Chunk("x"), _Chunk("y")]


def test_stream_frame_has_nonnegative_elapsed_ms():
    """Emitted frames carry an elapsed_ms float >= 0."""
    events = []
    stream(lambda x: x, "v", on_event=events.append)
    assert all(isinstance(f.elapsed_ms, float) and f.elapsed_ms >= 0.0 for f in events)


# ── astream (async) ─────────────────────────────────────────────────────


async def test_astream_sync_callable_branch():
    """A plain (non-coroutine) callable is invoked synchronously."""
    result = await astream(lambda x: x + 100, 1)
    assert result == 101


async def test_astream_coroutine_function_awaited():
    """A coroutine function is awaited (the iscoroutinefunction branch)."""

    async def coro(x):
        return x * 3

    result = await astream(coro, 4)
    assert result == 12


async def test_astream_runnable_multiple_chunks_summed():
    """astream concatenates multiple async chunks via __add__."""
    runnable = _make_async_streaming_runnable([_Chunk("foo"), _Chunk("bar")])
    result = await astream(runnable, "in")
    assert result == _Chunk("foobar")


async def test_astream_runnable_single_chunk():
    """A single async chunk is returned as-is."""
    runnable = _make_async_streaming_runnable(["only"])
    assert await astream(runnable, "in") == "only"


async def test_astream_emits_token_events():
    """astream emits one LLM_TOKEN per async chunk, bookended by step events."""
    events = []
    runnable = _make_async_streaming_runnable([_Chunk("p"), _Chunk("q")])
    await astream(runnable, "in", on_event=events.append)
    assert [f.event for f in events] == [
        StreamEvent.STEP_START,
        StreamEvent.LLM_TOKEN,
        StreamEvent.LLM_TOKEN,
        StreamEvent.STEP_END,
    ]
