# @summary
# Pure markdown formatter for the read-only `show-trend` CLI subcommand.
# render_trend_table(history, *, runs, window, min_regressed, epsilon) takes a
# run-history entry list (as loaded by run_history.load_run_history) and returns
# a deterministic per-(qtype, metric) markdown trend table — one delta column
# per the last `runs` timestamp-sorted entries, a `regressed (shown)` count +
# `latest` computed over those SHOWN runs, and a sustained YES/NO flag that is
# the detector's verdict over its own `--window` (sourced AS-IS from
# sustained_regression.detect_sustained_regressions, strict `delta < -epsilon`;
# may look back further than the shown runs). Empty history -> a friendly
# one-line message. NO I/O, no now(),
# no randomness; reuses ci._fmt for 3-decimal / em-dash formatting.
# Exports: render_trend_table
# Deps: stdlib only; src.eval.runner.ci (_fmt),
#       src.eval.runner.sustained_regression (detect_sustained_regressions).
# @end-summary
"""Pure markdown renderer for the read-only ``show-trend`` inspection surface.

``render_trend_table(history, ...)`` is a side-effect-free formatter: it maps a
run-history entry list to a markdown string and performs NO I/O, no pack load,
no ingest/retrieve/judge. The CLI (:mod:`src.eval.cli`) owns history loading and
stdout; this module owns presentation only.

The table has one row per ``(qtype, metric)`` pair seen in the most-recent
``runs`` entries (sorted by ``(qtype, metric)`` for determinism). Each row shows
a delta column per displayed run (em-dash when that run lacks the pair), then a
``regressed (shown)`` count and ``latest`` delta — BOTH computed over the
displayed runs so every number ties to a visible column — and a ``sustained``
YES/NO flag.

The ``sustained`` flag is the verdict of
:func:`~src.eval.runner.sustained_regression.detect_sustained_regressions` over
its own ``window`` (strict ``delta < -epsilon``, single-sourced — this module
never re-implements regression logic). Because ``window`` may look back FURTHER
than the displayed ``runs``, a pair can be flagged YES while its shown columns
look clean; that is an intentional cue to widen ``--runs``. When ``runs ==
window`` (the default) the shown count and the detector verdict coincide.

``runs`` (how many recent runs to DISPLAY as delta columns) is deliberately
distinct from ``window`` (how many recent runs the detector looks BACK over).
"""
from __future__ import annotations

from src.eval.runner.ci import _fmt
from src.eval.runner.sustained_regression import detect_sustained_regressions

_EMPTY_MESSAGE = "(no run history yet)"


def render_trend_table(
    history: list[dict],
    *,
    runs: int = 3,
    window: int = 3,
    min_regressed: int = 2,
    epsilon: float = 0.05,
) -> str:
    """Render a deterministic per-``(qtype, metric)`` markdown trend table.

    :param history: run-history entries, timestamp-sorted ascending (as returned
        by :func:`~src.eval.runner.run_history.load_run_history`). Each entry
        carries ``timestamp`` and ``per_qtype_deltas = {qtype: {metric: delta}}``.
    :param runs: how many most-recent runs to show as delta columns (default 3).
    :param window: detector lookback passed straight to
        :func:`~src.eval.runner.sustained_regression.detect_sustained_regressions`
        (default 3) — distinct from ``runs``.
    :param min_regressed: detector min regressed-run count to flag a pair
        (default 2).
    :param epsilon: detector tolerance band; the regression predicate is strict
        ``delta < -epsilon`` (default 0.05).
    :returns: a markdown table string, or :data:`_EMPTY_MESSAGE` when ``history``
        is empty (never an empty string, never a crash).
    """
    if not history:
        return _EMPTY_MESSAGE

    # Display window: the most-recent `runs` entries (history is ascending).
    displayed = history[-runs:] if runs > 0 else []
    if not displayed:
        return _EMPTY_MESSAGE

    # Collect every (qtype, metric) appearing in the displayed runs, sorted.
    pairs: set[tuple[str, str]] = set()
    for entry in displayed:
        per_qtype = entry.get("per_qtype_deltas") or {}
        for qtype, metric_map in per_qtype.items():
            for metric in (metric_map or {}):
                pairs.add((qtype, metric))
    sorted_pairs = sorted(pairs)

    # Sustained flags from the detector (single-sourced regression predicate).
    sustained = detect_sustained_regressions(
        history,
        window=window,
        min_regressed=min_regressed,
        epsilon=epsilon,
    )
    flagged = {(s.qtype, s.metric): s for s in sustained}

    # Run-delta columns: one per displayed run, labeled by timestamp.
    run_labels = [str(entry.get("timestamp", "")) for entry in displayed]

    header_cells = ["qtype", "metric", *run_labels, "regressed (shown)",
                    "latest", "sustained"]
    sep_cells = ["---"] * len(header_cells)

    lines: list[str] = []
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("| " + " | ".join(sep_cells) + " |")

    for qtype, metric in sorted_pairs:
        # Per-run delta cells (em-dash when a run lacks the pair).
        delta_cells: list[str] = []
        for entry in displayed:
            metric_map = (entry.get("per_qtype_deltas") or {}).get(qtype) or {}
            delta_cells.append(
                _fmt(metric_map[metric]) if metric in metric_map else _fmt(None)
            )

        # The `regressed (shown)` count and `latest` are ALWAYS computed over the
        # DISPLAYED runs, so every number in the row ties to a visible delta
        # column — never to a run not shown. The `sustained` flag, by contrast,
        # is the detector's verdict over its own `--window` (which may look back
        # FURTHER than the displayed `--runs`): single-sourced from
        # detect_sustained_regressions, so YES can hold even when the shown runs
        # look clean — a cue to widen `--runs`. When runs == window (the default)
        # the two bases coincide.
        regressed_count = sum(
            1 for entry in displayed if _regressed_in(entry, qtype, metric, epsilon)
        )
        regressed = f"{regressed_count}/{len(displayed)}"
        latest = _latest_delta_for(displayed, qtype, metric)
        flag = "YES" if (qtype, metric) in flagged else "NO"

        row = ["", qtype, metric, *delta_cells, regressed, latest, flag, ""]
        lines.append(" | ".join(row).strip())

    return "\n".join(lines)


def _regressed_in(entry: dict, qtype: str, metric: str, epsilon: float) -> bool:
    """Whether ``(qtype, metric)`` regressed in ``entry`` (strict delta < -eps).

    Mirrors the detector's predicate; absent pair → not regressed.
    """
    metric_map = (entry.get("per_qtype_deltas") or {}).get(qtype) or {}
    if metric not in metric_map:
        return False
    return float(metric_map[metric]) < -epsilon


def _latest_delta_for(
    displayed: list[dict], qtype: str, metric: str
) -> str:
    """Formatted latest delta for ``(qtype, metric)`` over the displayed runs.

    Walks the displayed runs most-recent-first and returns the first delta found
    for the pair (em-dash if none of the displayed runs carry it).
    """
    for entry in reversed(displayed):
        metric_map = (entry.get("per_qtype_deltas") or {}).get(qtype) or {}
        if metric in metric_map:
            return _fmt(metric_map[metric])
    return _fmt(None)
