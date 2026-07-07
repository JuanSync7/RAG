# @summary
# P7d/P7e/P7f — pure CI summary renderer. render_ci_summary(report, gate)
# produces a deterministic GitHub-flavored-markdown PASS/FAIL summary: a status
# line, a per-qtype metrics table (recall@k + mrr + faithfulness + per-qtype
# status), a failures table sourced strictly from gate.failures with columns
# qtype/qid/metric/expected/actual (qid em-dash for qtype-level failures), and
# a run-facts footer. No I/O, no env reads — the write seam lives in
# src.eval.cli.
# Exports: render_ci_summary
# Deps: stdlib only; src.eval.runner.report.EvalReport (type only),
#       src.eval.runner.gate.GateResult (type only).
# @end-summary
"""CI eval-gate markdown summary renderer (P7d).

``render_ci_summary(report, gate)`` returns a GitHub-flavored-markdown string
suitable for a GitHub Actions job summary (``$GITHUB_STEP_SUMMARY``). It is a
pure function — deterministic, side-effect-free — so it is trivially testable
without live Weaviate/LLM. The CLI (``src.eval.cli``) owns the write seam.

Design invariants
-----------------
- The function NEVER invents "expected" thresholds for passing metrics. The
  gate only carries *failures*; passing actuals are shown from ``report``,
  while expected-vs-actual appears ONLY in the failures table.
- Per-qtype status is ``FAIL`` iff some ``gate.failure`` names that qtype.
- Faithfulness renders an em-dash (``—``) when absent for a qtype (judged-zero
  / not faithfulness-gated) — never ``0.000``. MRR renders the same em-dash
  when absent for a qtype.
- qtype rows and failure rows are emitted in sorted order for determinism.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — type-only imports.
    from src.eval.runner.gate import GateResult
    from src.eval.runner.report import EvalReport

_DASH = "—"


def _fmt(value: float | None) -> str:
    """Format a metric to 3 decimals, or an em-dash when absent."""
    if value is None:
        return _DASH
    return f"{value:.3f}"


def render_ci_summary(report: "EvalReport", gate: "GateResult") -> str:
    """Render a GitHub-flavored-markdown PASS/FAIL summary for an eval run.

    Parameters
    ----------
    report:
        The aggregated :class:`EvalReport` (actuals + run facts).
    gate:
        The :class:`GateResult`; ``gate.failures`` is the sole source of
        expected-vs-actual data and of per-qtype FAIL status.

    Returns
    -------
    str
        A deterministic markdown document (title, status line, per-qtype
        metrics table, failures section, run-facts footer).
    """
    status = "PASS" if gate.passed else "FAIL"
    failed_qtypes = {f.qtype for f in gate.failures}

    lines: list[str] = []
    lines.append("## Eval Gate Summary")
    lines.append("")
    lines.append(f"**Status: {status}**")
    lines.append("")

    # --- Per-qtype metrics table ---
    recall_key = f"recall@{report.k}"
    lines.append(f"| qtype | {recall_key} | mrr | faithfulness | status |")
    lines.append("| --- | --- | --- | --- | --- |")
    qtypes = sorted(
        set(report.recall_by_qtype)
        | set(report.mrr_by_qtype)
        | set(report.faithfulness_by_qtype)
    )
    for qt in qtypes:
        recall = report.recall_by_qtype.get(qt)
        mrr = report.mrr_by_qtype.get(qt)
        faith = report.faithfulness_by_qtype.get(qt)
        qt_status = "FAIL" if qt in failed_qtypes else "PASS"
        lines.append(
            f"| {qt} | {_fmt(recall)} | {_fmt(mrr)} | "
            f"{_fmt(faith)} | {qt_status} |"
        )
    lines.append("")

    # --- Failures section (expected-vs-actual sourced from gate only) ---
    lines.append("### Failures")
    lines.append("")
    if not gate.failures:
        lines.append("None — all declared thresholds met.")
    else:
        lines.append("| qtype | qid | metric | expected | actual |")
        lines.append("| --- | --- | --- | --- | --- |")
        for f in sorted(gate.failures, key=lambda x: (x.qtype, x.qid, x.metric)):
            lines.append(
                f"| {f.qtype} | {f.qid or _DASH} | {f.metric} | "
                f"{f.expected:.3f} | {f.actual:.3f} |"
            )
    lines.append("")

    # --- Run-facts footer ---
    lines.append(
        f"_Collection `{report.collection_name}` · k={report.k} · "
        f"scored={report.total_queries_scored} · "
        f"judged={report.total_queries_judged} · "
        f"skipped={report.total_queries_skipped}_"
    )

    return "\n".join(lines)
