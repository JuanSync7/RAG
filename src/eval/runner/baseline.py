# @summary
# P8b — pure baseline-diff engine. compute_baseline_diff(baseline_json,
# current_json, *, epsilon=0.05) diffs the three per-qtype metric maps
# (recall/mrr/faithfulness) from two persisted eval-report JSON payloads,
# classifying each (qtype, metric) into regressions / improvements / unchanged,
# and surfacing new/dropped qtypes plus a gate flip (pass -> fail). No I/O.
# A metric "regressed" iff current < baseline - epsilon (delta == -epsilon is
# NOT a regression); "improved" iff delta > epsilon; else unchanged.
# Exports: MetricDelta, BaselineDiff, compute_baseline_diff
# Deps: stdlib only (dataclasses).
# @end-summary
"""Pure baseline-diff engine (P8b).

:func:`compute_baseline_diff` is a deterministic, side-effect-free function
that consumes two persisted eval-report JSON payloads (the shape produced by
:func:`src.eval.cli._report_payload` + ``pack_name``/``timestamp``) and returns
an immutable :class:`BaselineDiff` describing how the *current* run moved
relative to the *baseline*.

Diff semantics
--------------
- The three per-qtype metric maps are diffed under stable metric labels:
  ``recall_by_qtype`` -> ``"recall"``, ``mrr_by_qtype`` -> ``"mrr"``,
  ``faithfulness_by_qtype`` -> ``"faithfulness"``.
- A ``(qtype, metric)`` pair present on BOTH sides yields a numeric
  ``delta = current - baseline`` and is classified:
    * ``regressed`` iff ``current < baseline - epsilon`` (strict; a delta of
      exactly ``-epsilon`` is NOT a regression),
    * ``improvement`` iff ``delta > epsilon``,
    * ``unchanged`` otherwise (``|delta| <= epsilon``).
- A pair present on only one side does NOT count as a regression/improvement;
  instead its qtype contributes to ``new_qtypes`` (present in current's maps
  only) or ``dropped_qtypes`` (present in baseline's maps only).
- ``gate_flipped_to_fail`` is ``True`` iff the baseline passed and the current
  failed.
"""
from __future__ import annotations

from dataclasses import dataclass

# Map the persisted per-qtype JSON map name to the stable metric label used in
# the diff (and downstream markdown). Order is load-bearing for determinism.
_METRIC_MAPS: tuple[tuple[str, str], ...] = (
    ("recall_by_qtype", "recall"),
    ("mrr_by_qtype", "mrr"),
    ("faithfulness_by_qtype", "faithfulness"),
)


@dataclass(frozen=True)
class MetricDelta:
    """One ``(qtype, metric)`` movement between baseline and current.

    ``baseline_value`` / ``current_value`` are ``None`` when the pair is absent
    on that side. ``delta`` is ``current - baseline`` (only meaningful — and
    only computed — when both sides are present; ``0.0`` otherwise). ``regressed``
    is ``True`` iff both present and ``current < baseline - epsilon``.
    """

    qtype: str
    metric: str
    baseline_value: float | None
    current_value: float | None
    delta: float
    regressed: bool


@dataclass(frozen=True)
class BaselineDiff:
    """Immutable result of :func:`compute_baseline_diff`."""

    pack_name: str
    baseline_timestamp: str
    current_timestamp: str
    passed_baseline: bool
    passed_current: bool
    regressions: tuple[MetricDelta, ...]
    improvements: tuple[MetricDelta, ...]
    unchanged: tuple[MetricDelta, ...]
    new_qtypes: tuple[str, ...]
    dropped_qtypes: tuple[str, ...]
    gate_flipped_to_fail: bool


def _qtypes_in(payload: dict) -> set[str]:
    """Union of qtypes across the three per-qtype metric maps of ``payload``."""
    qtypes: set[str] = set()
    for map_name, _label in _METRIC_MAPS:
        qtypes |= set(payload.get(map_name) or {})
    return qtypes


def compute_baseline_diff(
    baseline_json: dict,
    current_json: dict,
    *,
    epsilon: float = 0.05,
) -> BaselineDiff:
    """Diff two persisted eval-report payloads. See module docstring.

    :param baseline_json: the persisted baseline report payload.
    :param current_json: the persisted current report payload.
    :param epsilon: regression/improvement tolerance band (default ``0.05``).
    :returns: an immutable :class:`BaselineDiff`.
    """
    regressions: list[MetricDelta] = []
    improvements: list[MetricDelta] = []
    unchanged: list[MetricDelta] = []

    for map_name, label in _METRIC_MAPS:
        base_map = baseline_json.get(map_name) or {}
        cur_map = current_json.get(map_name) or {}
        # Iterate the union of qtypes for this metric, sorted for determinism.
        for qtype in sorted(set(base_map) | set(cur_map)):
            base_val = base_map.get(qtype)
            cur_val = cur_map.get(qtype)
            if base_val is None or cur_val is None:
                # One-sided pair: handled via new/dropped qtypes, not buckets.
                continue
            base_f = float(base_val)
            cur_f = float(cur_val)
            delta = cur_f - base_f
            regressed = cur_f < base_f - epsilon
            md = MetricDelta(
                qtype=qtype,
                metric=label,
                baseline_value=base_f,
                current_value=cur_f,
                delta=delta,
                regressed=regressed,
            )
            if regressed:
                regressions.append(md)
            elif delta > epsilon:
                improvements.append(md)
            else:
                unchanged.append(md)

    base_qtypes = _qtypes_in(baseline_json)
    cur_qtypes = _qtypes_in(current_json)
    new_qtypes = tuple(sorted(cur_qtypes - base_qtypes))
    dropped_qtypes = tuple(sorted(base_qtypes - cur_qtypes))

    passed_baseline = bool(baseline_json.get("passed"))
    passed_current = bool(current_json.get("passed"))
    gate_flipped_to_fail = passed_baseline and not passed_current

    return BaselineDiff(
        pack_name=current_json.get("pack_name", baseline_json.get("pack_name", "")),
        baseline_timestamp=baseline_json.get("timestamp", ""),
        current_timestamp=current_json.get("timestamp", ""),
        passed_baseline=passed_baseline,
        passed_current=passed_current,
        regressions=tuple(regressions),
        improvements=tuple(improvements),
        unchanged=tuple(unchanged),
        new_qtypes=new_qtypes,
        dropped_qtypes=dropped_qtypes,
        gate_flipped_to_fail=gate_flipped_to_fail,
    )
