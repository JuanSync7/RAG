# @summary
# Sustained-regression detector over the per-pack run-history index (#140).
# detect_sustained_regressions(history, *, window=3, min_regressed=2,
# epsilon=0.05) inspects the most-recent `window` timestamp-sorted history
# entries' per_qtype_deltas and flags each (qtype, metric) that regressed
# (delta < -epsilon, strict — same definition as baseline.py) in >= min_regressed of the
# considered runs — suppressing one-off judge/retrieval dips that a single-run
# baseline_diff would alarm on. Per-QTYPE granularity (parity with baseline_diff);
# per-QID sustained detection is future work. Returns a deterministically sorted
# tuple of frozen SustainedMetricRegression; empty history / no sustained → ().
# Sign convention: delta = current - baseline (negative = worse).
# Determinism: no datetime.now(), no randomness.
# Exports: SustainedMetricRegression, detect_sustained_regressions
# Deps: stdlib only (dataclasses).
# @end-summary
"""Sustained-regression detector over the per-pack run-history index.

Today's regression detection (:func:`~src.eval.runner.baseline.compute_baseline_diff`
+ ``fail_on_regression``) fires on a SINGLE current run vs ONE baseline — noisy:
a one-off judge or retrieval dip alarms even when the metric recovers next run.

This detector reads the append-only run-history index shipped in #140 (a
timestamp-sorted list of entries, each carrying ``per_qtype_deltas =
{qtype: {metric: delta}}``) and flags a ``(qtype, metric)`` only when it
regressed across **>= min_regressed of the most-recent N (``window``)
consecutive runs**, suppressing transient dips.

Sign convention and the regression predicate are single-sourced with
:mod:`src.eval.runner.baseline`: ``delta = current - baseline`` (negative =
worse), and a run "regressed" for a ``(qtype, metric)`` iff its delta is
STRICTLY below ``-epsilon`` (``delta < -epsilon``) — identical to
``compute_baseline_diff``'s ``current < baseline - epsilon``, so a delta of
exactly ``-epsilon`` is treated as unchanged (not a regression).

Granularity is per-QTYPE, for parity with the existing ``baseline_diff``.
Per-QID sustained detection (over ``per_query_deltas``) is explicitly OUT of
scope for this slice and noted as future work.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SustainedMetricRegression:
    """One ``(qtype, metric)`` that regressed across multiple consecutive runs.

    :param qtype: the golden question-type the metric is aggregated over.
    :param metric: the metric label (``recall`` / ``mrr`` / ``faithfulness``).
    :param runs_regressed: how many of the considered runs regressed for this
        ``(qtype, metric)`` (a run regresses iff ``delta < -epsilon``).
    :param window: the number of recent runs actually CONSIDERED (``<=`` the
        requested window when the history is shorter).
    :param latest_delta: the delta in the most-recent considered entry that has
        this ``(qtype, metric)`` (falling back to the most-recent considered
        entry overall when none carry it). Sign = ``current - baseline``.
    """

    qtype: str
    metric: str
    runs_regressed: int
    window: int
    latest_delta: float


def detect_sustained_regressions(
    history: list[dict],
    *,
    window: int = 3,
    min_regressed: int = 2,
    epsilon: float = 0.05,
) -> tuple[SustainedMetricRegression, ...]:
    """Flag ``(qtype, metric)`` pairs that regressed across consecutive runs.

    Inspects the most-recent ``window`` entries of the timestamp-sorted
    ``history`` (the last ``window`` of the ascending list). When fewer than
    ``window`` entries exist, all available entries are used and the emitted
    ``window`` field reflects the ACTUAL count considered.

    For each ``(qtype, metric)`` appearing in any considered entry's
    ``per_qtype_deltas``, a run "regressed" iff its delta ``< -epsilon`` (sign =
    ``current - baseline``; same predicate as :mod:`~src.eval.runner.baseline`).
    A pair is flagged iff it regressed in ``>= min_regressed`` of the considered
    runs.

    :param history: the run-history entries (timestamp-sorted ascending, as
        returned by :func:`~src.eval.runner.run_history.load_run_history`).
    :param window: how many recent runs to inspect (default ``3``).
    :param min_regressed: minimum regressed runs to flag a pair (default ``2``).
    :param epsilon: regression tolerance band (default ``0.05``); the predicate
        is strict (``delta < -epsilon``), so a delta of exactly ``-epsilon`` is
        treated as unchanged — not a regression (mirrors baseline.py).
    :returns: a tuple of :class:`SustainedMetricRegression`, sorted
        deterministically by ``(qtype, metric)``. Empty history or no sustained
        regression → ``()``. A missing/empty ``per_qtype_deltas`` for a run is
        treated as not-regressed for that run (never raises).
    """
    if not history or window <= 0:
        return ()

    # History is ascending by timestamp → the most-recent runs are the tail.
    considered = history[-window:]
    considered_count = len(considered)

    # Collect every (qtype, metric) that appears anywhere in the window.
    pairs: set[tuple[str, str]] = set()
    for entry in considered:
        per_qtype = entry.get("per_qtype_deltas") or {}
        for qtype, metric_map in per_qtype.items():
            for metric in (metric_map or {}):
                pairs.add((qtype, metric))

    results: list[SustainedMetricRegression] = []
    for qtype, metric in pairs:
        regressed_count = 0
        latest_delta: float | None = None
        # Walk most-recent-first so the first delta found is the latest.
        for entry in reversed(considered):
            metric_map = (entry.get("per_qtype_deltas") or {}).get(qtype) or {}
            if metric not in metric_map:
                continue
            delta = float(metric_map[metric])
            if latest_delta is None:
                latest_delta = delta
            if delta < -epsilon:
                regressed_count += 1

        if regressed_count >= min_regressed:
            # latest_delta is the delta in the most-recent entry that HAS this
            # pair; fall back to 0.0 only if (defensively) none carried it —
            # which cannot happen here since the pair came from the window.
            results.append(
                SustainedMetricRegression(
                    qtype=qtype,
                    metric=metric,
                    runs_regressed=regressed_count,
                    window=considered_count,
                    latest_delta=latest_delta if latest_delta is not None else 0.0,
                )
            )

    results.sort(key=lambda r: (r.qtype, r.metric))
    return tuple(results)
