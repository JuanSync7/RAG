"""Unit tests for ``src.common.llm.utils``.

Covers the deterministic, side-effect-light helpers shared across the LLM
composition layer: ``build_messages`` (message-list assembly with optional
system / history), ``timed`` (elapsed-time context manager), and
``safe_call`` (exception-swallowing call wrapper).  No external boundaries
are touched — these helpers are pure aside from a monotonic clock read.
"""
from __future__ import annotations

import time

import pytest

from src.common.llm.utils import build_messages, safe_call, timed


# ── build_messages ──────────────────────────────────────────────────────


def test_build_messages_prompt_only():
    """Bare prompt yields a single user message."""
    assert build_messages("hello") == [{"role": "user", "content": "hello"}]


def test_build_messages_with_system_prepends_system_first():
    """System message is prepended before the user message, in order."""
    result = build_messages("question", system="be terse")
    assert result == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "question"},
    ]


def test_build_messages_empty_system_is_skipped():
    """A falsy system string adds no system message (only user)."""
    assert build_messages("p", system="") == [{"role": "user", "content": "p"}]


def test_build_messages_history_between_system_and_user():
    """History sits after system and before the trailing user message."""
    history = [
        {"role": "user", "content": "prev-q"},
        {"role": "assistant", "content": "prev-a"},
    ]
    result = build_messages("now", system="sys", history=history)
    assert result == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "prev-q"},
        {"role": "assistant", "content": "prev-a"},
        {"role": "user", "content": "now"},
    ]


def test_build_messages_history_without_system():
    """History is included even when no system message is supplied."""
    history = [{"role": "assistant", "content": "earlier"}]
    result = build_messages("latest", history=history)
    assert result == [
        {"role": "assistant", "content": "earlier"},
        {"role": "user", "content": "latest"},
    ]


def test_build_messages_does_not_mutate_history_arg():
    """The caller's history list is copied via extend, not aliased."""
    history = [{"role": "user", "content": "a"}]
    build_messages("b", history=history)
    assert history == [{"role": "user", "content": "a"}]


# ── timed ───────────────────────────────────────────────────────────────


def test_timed_populates_elapsed_on_exit():
    """elapsed is 0.0 inside the block and positive after the block exits."""
    with timed("work") as t:
        assert t["elapsed"] == 0.0
        time.sleep(0.01)
    assert t["elapsed"] >= 0.01


def test_timed_records_elapsed_even_on_exception():
    """elapsed is populated in the finally branch when the body raises."""
    captured: dict[str, float] = {}
    with pytest.raises(RuntimeError):
        with timed("boom") as t:
            captured = t
            raise RuntimeError("fail")
    assert captured["elapsed"] >= 0.0


# ── safe_call ───────────────────────────────────────────────────────────


def test_safe_call_success_returns_result_and_none_error():
    """On success, returns (result, None)."""
    result, err = safe_call(lambda x: x + 1, 41)
    assert result == 42
    assert err is None


def test_safe_call_passes_kwargs():
    """Keyword arguments are forwarded to the wrapped callable."""
    result, err = safe_call(lambda *, a, b: a * b, a=3, b=4)
    assert result == 12
    assert err is None


def test_safe_call_swallows_exception_and_returns_it():
    """On failure, returns (None, exception) without raising."""
    boom = ValueError("nope")

    def explode():
        raise boom

    result, err = safe_call(explode)
    assert result is None
    assert err is boom
