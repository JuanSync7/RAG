"""Unit tests for LocalRetryProvider.execute (exponential-backoff retry logic).

These tests monkeypatch `src.platform.reliability.local_retry.time.sleep` with a
recorder so they run instantly and can assert the exact backoff sequence. They
cover success-first, retry-then-succeed, exhaustion/re-raise, the backoff cap,
non-retryable propagation, default-policy behavior, and last-error identity.
"""
from __future__ import annotations

import pytest

import src.platform.reliability.local_retry as local_retry_mod
from src.platform.reliability.local_retry import LocalRetryProvider
from src.platform.schemas import RetryPolicy


@pytest.fixture
def sleep_recorder(monkeypatch):
    """Patch the module-level time.sleep and record the args passed to it."""
    recorded: list[float] = []

    def _fake_sleep(seconds):
        recorded.append(seconds)

    monkeypatch.setattr(local_retry_mod.time, "sleep", _fake_sleep)
    return recorded


def _counting_fn(behaviors):
    """Build a fn that, on call i, executes behaviors[i].

    Each behavior is either a callable returning a value, or an exception
    instance to raise. Tracks the number of calls on the returned fn via
    `.calls`.
    """
    state = {"calls": 0}

    def fn():
        idx = state["calls"]
        state["calls"] += 1
        behavior = behaviors[idx]
        if isinstance(behavior, BaseException):
            raise behavior
        return behavior

    fn.state = state
    return fn


def test_success_on_first_attempt(sleep_recorder):
    """Returns the value, never sleeps, calls fn exactly once."""
    fn = _counting_fn(["OK-1"])
    provider = LocalRetryProvider()

    result = provider.execute("op", fn)

    assert result == "OK-1"
    assert fn.state["calls"] == 1
    assert sleep_recorder == []


def test_succeeds_on_second_attempt(sleep_recorder):
    """Fail once (retryable) then succeed: fn called twice, one sleep at initial backoff."""
    policy = RetryPolicy()  # initial_backoff_seconds == 0.5
    fn = _counting_fn([RuntimeError("boom-1"), "OK-2"])
    provider = LocalRetryProvider()

    result = provider.execute("op", fn, policy=policy)

    assert result == "OK-2"
    assert fn.state["calls"] == 2
    assert sleep_recorder == [0.5]


def test_exhausts_all_attempts_reraises_last(sleep_recorder):
    """Always fails: re-raises last exception; fn called max_attempts; sleep max_attempts-1 times."""
    policy = RetryPolicy(max_attempts=3)
    errors = [RuntimeError("e-1"), RuntimeError("e-2"), RuntimeError("e-3")]
    fn = _counting_fn(list(errors))
    provider = LocalRetryProvider()

    with pytest.raises(RuntimeError) as excinfo:
        provider.execute("op", fn, policy=policy)

    assert str(excinfo.value) == "e-3"
    assert fn.state["calls"] == 3
    assert len(sleep_recorder) == 2


def test_backoff_sequence_with_cap(sleep_recorder):
    """Backoff grows geometrically but is capped at max_backoff_seconds."""
    policy = RetryPolicy(
        max_attempts=5,
        initial_backoff_seconds=1.0,
        max_backoff_seconds=4.0,
        backoff_multiplier=2.0,
    )
    errors = [RuntimeError(f"e-{i}") for i in range(5)]
    fn = _counting_fn(errors)
    provider = LocalRetryProvider()

    with pytest.raises(RuntimeError):
        provider.execute("op", fn, policy=policy)

    # 5 attempts => 4 sleeps. Raw would be 1,2,4,8 but cap at 4.0 => 1,2,4,4.
    assert sleep_recorder == [1.0, 2.0, 4.0, 4.0]
    assert fn.state["calls"] == 5


def test_non_retryable_propagates_immediately(sleep_recorder):
    """An exception not in retryable_exceptions propagates without retry/sleep."""
    policy = RetryPolicy(retryable_exceptions=(ValueError,))
    fn = _counting_fn([KeyError("not-retryable")])
    provider = LocalRetryProvider()

    with pytest.raises(KeyError):
        provider.execute("op", fn, policy=policy)

    assert fn.state["calls"] == 1
    assert sleep_recorder == []


def test_default_policy_when_none(sleep_recorder):
    """policy=None uses RetryPolicy() defaults: 3 attempts on always-fail."""
    errors = [RuntimeError(f"d-{i}") for i in range(3)]
    fn = _counting_fn(errors)
    provider = LocalRetryProvider()

    with pytest.raises(RuntimeError):
        provider.execute("op", fn, policy=None)

    assert fn.state["calls"] == 3
    assert len(sleep_recorder) == 2


def test_last_error_is_most_recent(sleep_recorder):
    """When each attempt raises a distinct exception, the final raise is the last one."""
    policy = RetryPolicy(max_attempts=3)
    e1 = RuntimeError("first")
    e2 = RuntimeError("second")
    e3 = RuntimeError("third-and-last")
    fn = _counting_fn([e1, e2, e3])
    provider = LocalRetryProvider()

    with pytest.raises(RuntimeError) as excinfo:
        provider.execute("op", fn, policy=policy)

    assert excinfo.value is e3
    assert str(excinfo.value) == "third-and-last"
