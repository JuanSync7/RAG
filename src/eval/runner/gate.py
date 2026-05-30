# @summary
# P6a/P7e — threshold gating. Compares per-qtype recall@k, faithfulness, and
# MRR in an EvalReport against pack.thresholds.defaults floors. Pure logic,
# no I/O.
# Exports: GateFailure, GateResult, validate_eval_report.
# Deps: stdlib (dataclasses); src.eval.pack.schema.EvalPack;
#       src.eval.runner.report.EvalReport.
# @end-summary
"""Threshold gating (P6a).

``validate_eval_report(pack, report)`` compares the per-qtype metrics in an
:class:`~src.eval.runner.report.EvalReport` against the floors declared in
``pack.thresholds.defaults``.

Semantics
---------
- Floors only (not ceilings): ``actual >= expected`` passes.
- The recall metric key compared per qtype is ``f"recall_at_{report.k}"``.
  If that key is not declared for the qtype, the recall check is skipped.
- The faithfulness metric key is ``"faithfulness"``. If absent, skipped.
- The MRR metric key is ``"mrr"``. If absent, skipped. Unlike faithfulness
  it is NOT judged-guarded — MRR is retrieval-based, so it gates even when
  ``total_queries_judged == 0``.
- A qtype declared in thresholds but absent from the report's recall /
  faithfulness / mrr map is skipped (graceful — likely no goldens of that
  qtype were scored/judged this run).
- If ``report.total_queries_judged == 0`` all faithfulness checks are skipped
  (anti-gaming guard so a stale eval with no judged goldens can't pass
  trivially on faithfulness).
- ``passed`` is ``True`` iff ``failures == ()``. Failures are aggregated, not
  short-circuited.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.eval.pack.schema import EvalPack
from src.eval.runner.report import EvalReport


@dataclass(frozen=True)
class GateFailure:
    """A single floor breach: one (qtype, metric) pair below its threshold."""

    qtype: str
    metric: str
    expected: float
    actual: float


@dataclass(frozen=True)
class GateResult:
    """Aggregated outcome of :func:`validate_eval_report`.

    ``passed`` is ``True`` iff ``failures`` is the empty tuple.
    """

    passed: bool
    failures: tuple[GateFailure, ...]


def validate_eval_report(pack: EvalPack, report: EvalReport) -> GateResult:
    """Gate ``report`` against ``pack.thresholds.defaults``.

    See module docstring for the exact semantics.
    """
    failures: list[GateFailure] = []
    defaults = pack.thresholds.defaults
    recall_key = f"recall_at_{report.k}"
    faithfulness_enabled = report.total_queries_judged > 0

    for qtype, floors in defaults.items():
        # Recall floor check.
        if recall_key in floors:
            if qtype in report.recall_by_qtype:
                expected = float(floors[recall_key])
                actual = float(report.recall_by_qtype[qtype])
                if actual < expected:
                    failures.append(
                        GateFailure(
                            qtype=qtype,
                            metric=recall_key,
                            expected=expected,
                            actual=actual,
                        )
                    )

        # MRR floor check (retrieval-based — NOT judged-guarded).
        if "mrr" in floors:
            if qtype in report.mrr_by_qtype:
                expected = float(floors["mrr"])
                actual = float(report.mrr_by_qtype[qtype])
                if actual < expected:
                    failures.append(
                        GateFailure(
                            qtype=qtype,
                            metric="mrr",
                            expected=expected,
                            actual=actual,
                        )
                    )

        # Faithfulness floor check (only if anything was actually judged).
        if faithfulness_enabled and "faithfulness" in floors:
            if qtype in report.faithfulness_by_qtype:
                expected = float(floors["faithfulness"])
                actual = float(report.faithfulness_by_qtype[qtype])
                if actual < expected:
                    failures.append(
                        GateFailure(
                            qtype=qtype,
                            metric="faithfulness",
                            expected=expected,
                            actual=actual,
                        )
                    )

    failures_tuple = tuple(failures)
    return GateResult(passed=not failures_tuple, failures=failures_tuple)
