"""Unit tests for :mod:`src.platform.timing`.

These tests pin every public behavior of :func:`measure_ms` and
:class:`TimingPool`. Time is made deterministic by monkeypatching
``src.platform.timing.time.perf_counter`` with a stateful fake whose return
values are controlled between calls, so elapsed-millisecond computations are
exact and assertable. The Prometheus side effect is intercepted with a fake
histogram so the observe contract (labels + observed value) is verified
without a real registry.

Construction note (verified by reading the source): ``TimingPool.__init__``
calls ``perf_counter()`` once to capture ``pipeline_start``. Tests that drive
the pool account for that first call in the fake's value sequence.
"""

from __future__ import annotations

import logging

import pytest

from src.platform import timing


class FakeClock:
    """Deterministic ``perf_counter`` replacement returning queued values.

    Each call pops the next value from ``values``; once exhausted it keeps
    returning the last value so trailing calls (if any) stay defined.
    """

    def __init__(self, values):
        self._values = list(values)
        self._last = self._values[0] if self._values else 0.0
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        if self._values:
            self._last = self._values.pop(0)
        return self._last


def _install_clock(monkeypatch, values):
    clock = FakeClock(values)
    monkeypatch.setattr(timing.time, "perf_counter", clock)
    return clock


class FakeLabeled:
    """Records the single ``observe`` call made on a labeled child."""

    def __init__(self, parent, labels):
        self._parent = parent
        self._labels = labels

    def observe(self, value):
        self._parent.observations.append((self._labels, value))


class FakeHistogram:
    """Fake Prometheus histogram capturing ``labels(...).observe(...)`` calls."""

    def __init__(self):
        self.observations = []

    def labels(self, **kwargs):
        return FakeLabeled(self, kwargs)


# ---------------------------------------------------------------------------
# measure_ms
# ---------------------------------------------------------------------------


def test_measure_ms_exact_elapsed(monkeypatch):
    """measure_ms returns the exact elapsed ms for a known start/now delta."""
    _install_clock(monkeypatch, [1.0005])
    assert timing.measure_ms(0.0) == 1000.5


def test_measure_ms_rounds_to_tenth(monkeypatch):
    """measure_ms rounds the elapsed milliseconds to 0.1 ms."""
    # 0.0123456 s -> 12.3456 ms -> rounds to 12.3
    _install_clock(monkeypatch, [0.0123456])
    assert timing.measure_ms(0.0) == 12.3


def test_measure_ms_zero_elapsed(monkeypatch):
    """measure_ms returns 0.0 when now equals started_at."""
    _install_clock(monkeypatch, [5.0])
    assert timing.measure_ms(5.0) == 0.0


# ---------------------------------------------------------------------------
# record — argument validation
# ---------------------------------------------------------------------------


def test_record_requires_started_at_or_ms(monkeypatch):
    """record raises ValueError when neither started_at nor ms is given."""
    _install_clock(monkeypatch, [0.0])
    pool = timing.TimingPool()
    with pytest.raises(ValueError, match="Provide either 'started_at' or 'ms'"):
        pool.record("stage_a", "retrieval")


# ---------------------------------------------------------------------------
# record — duration sources
# ---------------------------------------------------------------------------


def test_record_with_started_at_uses_measure_ms(monkeypatch):
    """record with started_at computes ms via measure_ms (now - started_at)."""
    # First perf_counter call -> pipeline_start (construction); second -> record's now.
    _install_clock(monkeypatch, [0.0, 2.5])
    pool = timing.TimingPool()
    entry = pool.record("stage_a", "retrieval", started_at=0.0)
    assert entry["ms"] == 2500.0


def test_record_with_ms_rounds_and_stores_all_keys(monkeypatch):
    """record with ms rounds to 0.1 and stores an entry with all five keys."""
    _install_clock(monkeypatch, [0.0])
    pool = timing.TimingPool()
    entry = pool.record("stage_a", "generation", ms=12.36)
    assert entry == {
        "stage": "stage_a",
        "bucket": "generation",
        "ms": 12.4,
        "budget_ms": None,
        "within_budget": True,
    }


# ---------------------------------------------------------------------------
# record — budget resolution
# ---------------------------------------------------------------------------


def test_record_budget_arg_overrides_stage_budgets(monkeypatch):
    """budget_ms argument takes precedence over stage_budgets for the entry."""
    _install_clock(monkeypatch, [0.0])
    pool = timing.TimingPool(stage_budgets={"stage_a": 500.0})
    entry = pool.record("stage_a", "retrieval", ms=50.0, budget_ms=100.0)
    assert entry["budget_ms"] == 100.0
    assert entry["within_budget"] is True


def test_record_uses_stage_budget_when_no_arg(monkeypatch):
    """stage_budgets value is used as the budget when no budget_ms arg is given."""
    _install_clock(monkeypatch, [0.0])
    pool = timing.TimingPool(stage_budgets={"stage_a": 100.0})
    entry = pool.record("stage_a", "retrieval", ms=150.0)
    assert entry["budget_ms"] == 100.0
    assert entry["within_budget"] is False


def test_record_no_budget_anywhere_is_within(monkeypatch):
    """With no budget arg and no stage budget, budget_ms is None and within is True."""
    _install_clock(monkeypatch, [0.0])
    pool = timing.TimingPool()
    entry = pool.record("stage_a", "retrieval", ms=99999.0)
    assert entry["budget_ms"] is None
    assert entry["within_budget"] is True


# ---------------------------------------------------------------------------
# record — budget boundary (<= semantics)
# ---------------------------------------------------------------------------


def test_record_ms_equal_to_budget_is_within(monkeypatch):
    """ms exactly equal to the budget is within budget (<=)."""
    _install_clock(monkeypatch, [0.0])
    pool = timing.TimingPool()
    entry = pool.record("stage_a", "retrieval", ms=100.0, budget_ms=100.0)
    assert entry["within_budget"] is True


def test_record_ms_one_step_above_budget_is_over(monkeypatch):
    """ms one tenth above the budget is over budget."""
    _install_clock(monkeypatch, [0.0])
    pool = timing.TimingPool()
    entry = pool.record("stage_a", "retrieval", ms=100.1, budget_ms=100.0)
    assert entry["within_budget"] is False


# ---------------------------------------------------------------------------
# record — Prometheus observe
# ---------------------------------------------------------------------------


def test_record_observes_on_prometheus_histogram(monkeypatch):
    """record observes ms on the histogram with stage/bucket labels."""
    _install_clock(monkeypatch, [0.0])
    pool = timing.TimingPool()
    fake = FakeHistogram()
    pool._prometheus = fake
    pool.record("stage_a", "retrieval", ms=42.3)
    assert fake.observations == [({"stage": "stage_a", "bucket": "retrieval"}, 42.3)]


def test_record_with_none_prometheus_does_not_crash(monkeypatch):
    """record completes normally when the histogram is unavailable (None)."""
    _install_clock(monkeypatch, [0.0])
    pool = timing.TimingPool()
    pool._prometheus = None
    entry = pool.record("stage_a", "retrieval", ms=42.3)
    assert entry["ms"] == 42.3


# ---------------------------------------------------------------------------
# totals
# ---------------------------------------------------------------------------


def test_totals_sums_buckets_sorted_with_total(monkeypatch):
    """totals sums per-bucket ms in sorted key order and includes total_ms."""
    _install_clock(monkeypatch, [0.0])
    pool = timing.TimingPool()
    # Two entries share "retrieval" so the bucket must SUM to 30.0 (not 10/20).
    pool.record("stage_a", "retrieval", ms=10.0)
    pool.record("stage_b", "retrieval", ms=20.0)
    pool.record("stage_c", "generation", ms=5.0)
    totals = pool.totals()
    assert totals == {"generation_ms": 5.0, "retrieval_ms": 30.0, "total_ms": 35.0}
    # Sorted order: "generation_ms" precedes "retrieval_ms" before total_ms.
    assert list(totals.keys()) == ["generation_ms", "retrieval_ms", "total_ms"]


# ---------------------------------------------------------------------------
# entries — copy isolation
# ---------------------------------------------------------------------------


def test_entries_returns_copy(monkeypatch):
    """entries returns a copy; mutating it does not affect the pool."""
    _install_clock(monkeypatch, [0.0])
    pool = timing.TimingPool()
    pool.record("stage_a", "retrieval", ms=10.0)
    returned = pool.entries()
    returned.append({"injected": True})
    assert len(pool.entries()) == 1


# ---------------------------------------------------------------------------
# elapsed_ms / is_overall_budget_exhausted
# ---------------------------------------------------------------------------


def test_elapsed_ms_uses_pipeline_start(monkeypatch):
    """elapsed_ms measures from the construction-time pipeline_start."""
    # Construction now=0.0, elapsed query now=1.5 -> 1500.0 ms.
    _install_clock(monkeypatch, [0.0, 1.5])
    pool = timing.TimingPool()
    assert pool.elapsed_ms() == 1500.0


def test_overall_budget_not_exhausted_below(monkeypatch):
    """is_overall_budget_exhausted is False when elapsed is below the budget."""
    # elapsed = 999.0 ms < 1000 ms budget.
    _install_clock(monkeypatch, [0.0, 0.999])
    pool = timing.TimingPool(overall_budget_ms=1000.0)
    assert pool.is_overall_budget_exhausted() is False


def test_overall_budget_not_exhausted_at_boundary(monkeypatch):
    """is_overall_budget_exhausted is False when elapsed equals the budget (strict >)."""
    # elapsed = 1000.0 ms == 1000 ms budget -> not exhausted.
    _install_clock(monkeypatch, [0.0, 1.0])
    pool = timing.TimingPool(overall_budget_ms=1000.0)
    assert pool.is_overall_budget_exhausted() is False


def test_overall_budget_exhausted_above(monkeypatch):
    """is_overall_budget_exhausted is True when elapsed exceeds the budget."""
    # elapsed = 1000.1 ms > 1000 ms budget.
    _install_clock(monkeypatch, [0.0, 1.0001])
    pool = timing.TimingPool(overall_budget_ms=1000.0)
    assert pool.is_overall_budget_exhausted() is True


# ---------------------------------------------------------------------------
# mark_budget_exhausted + properties
# ---------------------------------------------------------------------------


def test_mark_budget_exhausted_sets_flags_and_properties(monkeypatch):
    """mark_budget_exhausted sets the flag and stage exposed via properties."""
    _install_clock(monkeypatch, [7.0])
    pool = timing.TimingPool()
    assert pool.budget_exhausted is False
    assert pool.budget_exhausted_stage is None
    assert pool.pipeline_start == 7.0
    pool.mark_budget_exhausted("stage_a")
    assert pool.budget_exhausted is True
    assert pool.budget_exhausted_stage == "stage_a"


# ---------------------------------------------------------------------------
# check_stage_budget
# ---------------------------------------------------------------------------


def test_check_stage_budget_no_entry_returns_false(monkeypatch):
    """check_stage_budget returns False when no entry matches the stage."""
    _install_clock(monkeypatch, [0.0])
    pool = timing.TimingPool()
    pool.record("other", "retrieval", ms=10.0)
    assert pool.check_stage_budget("stage_a") is False


def test_check_stage_budget_reads_last_entry_over(monkeypatch):
    """check_stage_budget reads the LAST same-stage entry: first OK, last over -> True."""
    # Two perf_counter queries are needed by is_overall_budget_exhausted only if
    # within_budget is True; here the last entry is over so short-circuit -> no query.
    _install_clock(monkeypatch, [0.0])
    pool = timing.TimingPool()
    pool.record("stage_a", "retrieval", ms=10.0, budget_ms=100.0)  # within
    pool.record("stage_a", "retrieval", ms=200.0, budget_ms=100.0)  # over (last)
    assert pool.check_stage_budget("stage_a") is True


def test_check_stage_budget_reads_last_entry_within(monkeypatch):
    """check_stage_budget: first over, last within, overall OK -> False (reads last)."""
    # Construction now=0.0; is_overall_budget_exhausted queries now=0.5 -> 500 ms < budget.
    _install_clock(monkeypatch, [0.0, 0.5])
    pool = timing.TimingPool(overall_budget_ms=1000.0)
    pool.record("stage_a", "retrieval", ms=200.0, budget_ms=100.0)  # over (first)
    pool.record("stage_a", "retrieval", ms=10.0, budget_ms=100.0)  # within (last)
    assert pool.check_stage_budget("stage_a") is False


def test_check_stage_budget_or_logic_overall_exhausted(monkeypatch):
    """check_stage_budget returns True when last entry is within but overall is exhausted."""
    # Construction now=0.0; overall check now=5.0 -> 5000 ms > 1000 ms budget.
    _install_clock(monkeypatch, [0.0, 5.0])
    pool = timing.TimingPool(overall_budget_ms=1000.0)
    pool.record("stage_a", "retrieval", ms=10.0, budget_ms=100.0)  # within
    assert pool.check_stage_budget("stage_a") is True


# ---------------------------------------------------------------------------
# log_summary
# ---------------------------------------------------------------------------


def test_log_summary_empty_pool_logs_nothing(monkeypatch, caplog):
    """log_summary emits no log record when there are no entries."""
    _install_clock(monkeypatch, [0.0])
    pool = timing.TimingPool()
    with caplog.at_level(logging.INFO, logger="rag.timing"):
        pool.log_summary()
    assert caplog.records == []


def test_log_summary_non_empty_emits_one_info(monkeypatch, caplog):
    """log_summary emits one INFO record containing bucket/stage text."""
    _install_clock(monkeypatch, [0.0])
    pool = timing.TimingPool()
    pool.record("stage_a", "retrieval", ms=10.0)
    with caplog.at_level(logging.INFO, logger="rag.timing"):
        pool.log_summary()
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelno == logging.INFO
    message = record.getMessage()
    assert "retrieval.stage_a" in message
    assert "retrieval_ms" in message
