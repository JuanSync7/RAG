# @summary
# P8b — baseline-diff markdown reporting. render_baseline_diff_markdown(
# current_json, baseline_json, *, epsilon=0.05) reconstructs the CURRENT
# EvalReport+GateResult from a persisted JSON payload, reuses the existing
# render_ci_summary (NO duplication of the per-qtype/failures tables), and
# APPENDS a deterministic "### Regressions vs Baseline" callout produced by
# render_regression_section(diff). _reconstruct_report_and_gate rebuilds the
# frozen dataclasses from the persisted payload keys. No I/O.
# Exports: render_baseline_diff_markdown, render_regression_section,
#          _reconstruct_report_and_gate
# Deps: stdlib only; src.eval.runner.ci (render_ci_summary, _fmt),
#       src.eval.runner.report (EvalReport), src.eval.runner.gate
#       (GateResult, GateFailure), src.eval.runner.baseline
#       (compute_baseline_diff, BaselineDiff).
# @end-summary
"""Baseline-diff markdown reporting (P8b).

This module renders a deterministic markdown report comparing a *current* eval
run against a promoted *baseline*. It REUSES the existing
:func:`src.eval.runner.ci.render_ci_summary` for the current-run tables (status
line, per-qtype metrics, failures, run-facts footer) and only ADDS a new
regression-callout section — so the per-qtype/failures rendering is never
duplicated.

The current ``EvalReport`` + ``GateResult`` are reconstructed from a persisted
report JSON payload (the shape produced by :func:`src.eval.cli._report_payload`)
via :func:`_reconstruct_report_and_gate`.
"""
from __future__ import annotations

from .baseline import BaselineDiff, compute_baseline_diff
from .ci import _fmt, render_ci_summary


def _reconstruct_report_and_gate(payload: dict):
    """Rebuild ``(EvalReport, GateResult)`` from a persisted report payload.

    Reads the additive-default fields off ``payload`` (the
    :func:`src.eval.cli._report_payload` shape). Missing optional maps default
    to empty so partial payloads still reconstruct.
    """
    from .gate import GateFailure, GateResult
    from .report import EvalReport

    report = EvalReport(
        collection_name=payload.get("collection_name", ""),
        k=int(payload.get("k", 0)),
        per_query_recall=dict(payload.get("per_query_recall") or {}),
        recall_by_qtype=dict(payload.get("recall_by_qtype") or {}),
        total_queries_scored=int(payload.get("total_queries_scored", 0)),
        total_queries_skipped=int(payload.get("total_queries_skipped", 0)),
        faithfulness_by_qtype=dict(payload.get("faithfulness_by_qtype") or {}),
        total_queries_judged=int(payload.get("total_queries_judged", 0)),
        mrr_by_qtype=dict(payload.get("mrr_by_qtype") or {}),
        per_query_mrr=dict(payload.get("per_query_mrr") or {}),
        per_query_faithfulness=dict(payload.get("per_query_faithfulness") or {}),
    )
    failures = tuple(
        GateFailure(
            qtype=f.get("qtype", ""),
            metric=f.get("metric", ""),
            expected=float(f.get("expected", 0.0)),
            actual=float(f.get("actual", 0.0)),
            qid=f.get("qid", ""),
        )
        for f in (payload.get("failures") or [])
    )
    gate = GateResult(passed=bool(payload.get("passed")), failures=failures)
    return report, gate


def render_regression_section(diff: BaselineDiff) -> str:
    """Render the deterministic ``### Regressions vs Baseline`` markdown section.

    Layout:
      * a ``### Regressions vs Baseline`` heading,
      * a baseline -> current provenance line (pack + timestamps),
      * a regressions table (or a "no regressions" line if empty),
      * a one-line gate-flip note when ``gate_flipped_to_fail``,
      * new/dropped qtype lines when present.

    Reuses :func:`src.eval.runner.ci._fmt` (3 decimals, em-dash for ``None``).
    """
    lines: list[str] = []
    lines.append("### Regressions vs Baseline")
    lines.append("")
    lines.append(
        f"Baseline `{diff.pack_name}` @ {diff.baseline_timestamp} → "
        f"current @ {diff.current_timestamp}"
    )
    lines.append("")

    if diff.regressions:
        lines.append("| qtype | metric | baseline | current | delta |")
        lines.append("| --- | --- | --- | --- | --- |")
        for d in diff.regressions:
            lines.append(
                f"| {d.qtype} | {d.metric} | {_fmt(d.baseline_value)} | "
                f"{_fmt(d.current_value)} | {d.delta:+.3f} |"
            )
    else:
        lines.append("No regressions vs baseline.")

    if diff.gate_flipped_to_fail:
        lines.append("")
        lines.append("Gate flipped PASS → FAIL vs baseline.")

    if diff.new_qtypes:
        lines.append("")
        lines.append(f"New qtypes (no baseline): {', '.join(diff.new_qtypes)}")
    if diff.dropped_qtypes:
        # No blank line when it directly follows a new-qtypes line; keep one
        # leading blank only when this is the first trailing block.
        if not diff.new_qtypes:
            lines.append("")
        lines.append(
            f"Dropped qtypes (in baseline, absent now): "
            f"{', '.join(diff.dropped_qtypes)}"
        )

    return "\n".join(lines)


def render_baseline_diff_markdown(
    current_json: dict,
    baseline_json: dict,
    *,
    epsilon: float = 0.05,
) -> str:
    """Render the full baseline-diff markdown report (deterministic).

    Reconstructs the current ``EvalReport`` + ``GateResult`` from
    ``current_json``, renders the standard CI summary via
    :func:`render_ci_summary`, then appends the regression-callout section for
    ``compute_baseline_diff(baseline_json, current_json, epsilon=epsilon)``.
    """
    report, gate = _reconstruct_report_and_gate(current_json)
    ci = render_ci_summary(report, gate)
    diff = compute_baseline_diff(baseline_json, current_json, epsilon=epsilon)
    return ci + "\n\n" + render_regression_section(diff)
