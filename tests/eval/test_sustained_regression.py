# @summary
# Tests for the sustained-regression detector over the per-pack run-history
# index (src.eval.runner.sustained_regression). Headline red:
# test_sustained_vs_transient_dip distinguishes a (qtype, metric) that regressed
# in >=K-of-N consecutive runs (SUSTAINED) from a one-off dip (TRANSIENT). Unit
# tests pin the min_regressed threshold, the epsilon boundary (strict
# delta < -epsilon regresses, exactly -epsilon does not; same predicate as
# baseline.py), the window tail-selection recency (most-recent runs, not oldest),
# the short-history available-window fallback, and the empty-history /
# empty-per_qtype_deltas degrade-to-() path.
# Exports: test_sustained_vs_transient_dip, test_below_min_regressed_not_flagged,
#          test_epsilon_boundary, test_window_selects_most_recent_runs_not_oldest,
#          test_short_history_uses_available_window,
#          test_empty_history_returns_empty
# Deps: src.eval.runner.sustained_regression
# @end-summary
"""Tests for the sustained-regression detector over run history."""
from __future__ import annotations


def _entry(timestamp: str, per_qtype_deltas: dict) -> dict:
    """Minimal run-history entry shape (mirrors index_run_report output)."""
    return {
        "timestamp": timestamp,
        "pack_name": "synpack",
        "k": 5,
        "gate_passed": True,
        "per_query_deltas": {},
        "per_qtype_deltas": per_qtype_deltas,
    }


# ---------------------------------------------------------------------------
# Headline RED — sustained vs transient
# ---------------------------------------------------------------------------


def test_sustained_vs_transient_dip() -> None:
    """A metric regressed in 2-of-3 runs is SUSTAINED; a 1-of-3 dip is TRANSIENT.

    (factoid, faithfulness) regresses (delta < -epsilon) in runs 1 and 3 (2 of
    3) → SUSTAINED at min_regressed=2. (factoid, recall) regresses only in run 2
    (1 of 3) → TRANSIENT. The detector must return EXACTLY the faithfulness
    regression with runs_regressed==2, and NOT recall.
    """
    from src.eval.runner.sustained_regression import detect_sustained_regressions

    history = [
        _entry(
            "2026-05-29T12:00:00+00:00",
            {"factoid": {"faithfulness": -0.1, "recall": 0.0}},
        ),
        _entry(
            "2026-05-30T12:00:00+00:00",
            {"factoid": {"faithfulness": 0.0, "recall": -0.2}},
        ),
        _entry(
            "2026-05-31T12:00:00+00:00",
            {"factoid": {"faithfulness": -0.08, "recall": 0.0}},
        ),
    ]

    out = detect_sustained_regressions(
        history, window=3, min_regressed=2, epsilon=0.05
    )

    assert len(out) == 1, f"only faithfulness is sustained: {out}"
    (sustained,) = out
    assert sustained.qtype == "factoid"
    assert sustained.metric == "faithfulness"
    assert sustained.runs_regressed == 2
    assert sustained.window == 3
    # latest_delta = delta in the most-recent entry that has this (qtype,metric).
    assert sustained.latest_delta == -0.08
    # recall (1-of-3) must NOT be flagged.
    assert all(s.metric != "recall" for s in out)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_below_min_regressed_not_flagged() -> None:
    """Exactly min_regressed - 1 regressed runs → () (gate has teeth)."""
    from src.eval.runner.sustained_regression import detect_sustained_regressions

    # 1 regressed run, min_regressed=2 → not flagged.
    history = [
        _entry("2026-05-29T12:00:00+00:00", {"factoid": {"faithfulness": -0.1}}),
        _entry("2026-05-30T12:00:00+00:00", {"factoid": {"faithfulness": 0.0}}),
        _entry("2026-05-31T12:00:00+00:00", {"factoid": {"faithfulness": 0.0}}),
    ]
    assert detect_sustained_regressions(
        history, window=3, min_regressed=2, epsilon=0.05
    ) == ()


def test_epsilon_boundary() -> None:
    """delta == -epsilon does NOT regress (strict <); just below -epsilon does.

    Single-sources the predicate with baseline.py, which uses strict
    ``current < baseline - epsilon`` (delta < -epsilon) and excludes exactly
    -epsilon. Two qtypes regressed in 2 of 2 runs: one at exactly -epsilon →
    NOT flagged; one just below (-epsilon - tiny) → flagged.
    """
    from src.eval.runner.sustained_regression import detect_sustained_regressions

    history = [
        _entry(
            "2026-05-30T12:00:00+00:00",
            {
                "atexact": {"recall": -0.05},
                "justbelow": {"recall": -0.05 - 1e-9},
            },
        ),
        _entry(
            "2026-05-31T12:00:00+00:00",
            {
                "atexact": {"recall": -0.05},
                "justbelow": {"recall": -0.05 - 1e-9},
            },
        ),
    ]
    out = detect_sustained_regressions(
        history, window=3, min_regressed=2, epsilon=0.05
    )
    flagged = {(s.qtype, s.metric) for s in out}
    assert ("justbelow", "recall") in flagged, (
        "delta below -epsilon must regress"
    )
    assert ("atexact", "recall") not in flagged, (
        "delta == -epsilon must NOT regress (strict <, mirrors baseline.py)"
    )


def test_window_selects_most_recent_runs_not_oldest() -> None:
    """len(history) > window: only the most-recent ``window`` runs count.

    Pins tail-selection (``history[-window:]``, NOT ``[:window]``). 5 entries,
    window=3: the OLDEST 2 runs regress faithfulness, the NEWEST 3 are clean.
    Correct tail-selection considers only the clean newest-3 → (). A
    head-selection bug would see the 2 old regressions and falsely flag, so this
    reds an ``[-window:]`` → ``[:window]`` mutation that the (history ≤ window)
    tests cannot catch.
    """
    from src.eval.runner.sustained_regression import detect_sustained_regressions

    clean = {"factoid": {"faithfulness": 0.0}}
    regress = {"factoid": {"faithfulness": -0.2}}
    history = [
        _entry("2026-05-27T12:00:00+00:00", regress),
        _entry("2026-05-28T12:00:00+00:00", regress),
        _entry("2026-05-29T12:00:00+00:00", clean),
        _entry("2026-05-30T12:00:00+00:00", clean),
        _entry("2026-05-31T12:00:00+00:00", clean),
    ]
    assert detect_sustained_regressions(
        history, window=3, min_regressed=2, epsilon=0.05
    ) == (), "only the newest 3 (clean) count; old regressions are out of window"


def test_short_history_uses_available_window() -> None:
    """2 entries with window=3 → use the 2 available; window field == 2."""
    from src.eval.runner.sustained_regression import detect_sustained_regressions

    history = [
        _entry("2026-05-30T12:00:00+00:00", {"factoid": {"recall": -0.2}}),
        _entry("2026-05-31T12:00:00+00:00", {"factoid": {"recall": -0.3}}),
    ]
    out = detect_sustained_regressions(
        history, window=3, min_regressed=2, epsilon=0.05
    )
    assert len(out) == 1
    (s,) = out
    assert s.runs_regressed == 2
    assert s.window == 2, "window field reflects the ACTUAL count considered"
    assert s.latest_delta == -0.3


def test_empty_history_returns_empty() -> None:
    """[] → (); a single entry with empty per_qtype_deltas → ()."""
    from src.eval.runner.sustained_regression import detect_sustained_regressions

    assert detect_sustained_regressions([]) == ()
    assert detect_sustained_regressions(
        [_entry("2026-05-31T12:00:00+00:00", {})]
    ) == ()
