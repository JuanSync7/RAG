# @summary
# P6a/P7e/P7f — threshold gating. Compares per-qtype recall@k, faithfulness,
# and MRR in an EvalReport against pack.thresholds.defaults floors merged with
# pack.thresholds.overrides (qtype-level merge into effective_floors + per-qid
# individual-query checks). Pure logic, no I/O.
# Exports: GateFailure, GateResult, validate_eval_report.
# Deps: stdlib (dataclasses); src.eval.pack.schema.EvalPack;
#       src.eval.runner.report.EvalReport.
# @end-summary
"""Threshold gating (P6a / P7e / P7f).

``validate_eval_report(pack, report)`` compares the per-qtype metrics in an
:class:`~src.eval.runner.report.EvalReport` against the floors declared in
``pack.thresholds.defaults`` — as merged with ``pack.thresholds.overrides``.

Per-qtype semantics (defaults + qtype-overrides)
------------------------------------------------
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

Override semantics (P7f)
------------------------
- ``Thresholds.overrides`` is a list of :class:`ThresholdOverride`, each
  targeting EXACTLY ONE of a ``qtype`` or a ``qid`` (validated at pack load).
- **qtype-overrides** merge into ``effective_floors`` (a deep copy of
  ``defaults``): each override REPLACES the matching per-metric floor
  (last-wins on duplicate qtype+metric) and may introduce qtypes/metrics
  absent from ``defaults``. The aggregate loop iterates the UNION of
  defaults + qtype-override qtypes; the three check-bodies are unchanged and
  simply read the merged floors.
- **qid-overrides** check a single query's individual score against the
  floor. The metric key selects the per-query map:
  ``f"recall_at_{report.k}"`` → ``report.per_query_recall``;
  ``"mrr"`` → ``report.per_query_mrr``;
  ``"faithfulness"`` → ``report.per_query_faithfulness``;
  any other key is skipped (unknown metric / recall-k mismatch — mirrors the
  per-qtype skip-on-missing semantics). If the qid is absent from the
  selected map (query skipped / not judged) the check is skipped. A qid
  failure carries ``GateFailure.qid`` (and the qtype resolved from
  ``pack.goldens``, or ``""`` if the qid is not in any goldens file).

- ``passed`` is ``True`` iff ``failures == ()``. Failures (qtype-level and
  qid-level) are aggregated, not short-circuited.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.eval.pack.schema import EvalPack
from src.eval.runner.report import EvalReport


@dataclass(frozen=True)
class GateFailure:
    """A single floor breach: one (qtype, metric) pair below its threshold.

    ``qid`` is empty for qtype-level failures (defaults / qtype-overrides) and
    carries the offending query id for per-qid override failures (P7f).
    """

    qtype: str
    metric: str
    expected: float
    actual: float
    qid: str = ""


@dataclass(frozen=True)
class GateResult:
    """Aggregated outcome of :func:`validate_eval_report`.

    ``passed`` is ``True`` iff ``failures`` is the empty tuple.
    """

    passed: bool
    failures: tuple[GateFailure, ...]


def validate_eval_report(pack: EvalPack, report: EvalReport) -> GateResult:
    """Gate ``report`` against ``pack.thresholds`` (defaults + overrides).

    See module docstring for the exact semantics.
    """
    failures: list[GateFailure] = []
    defaults = pack.thresholds.defaults
    recall_key = f"recall_at_{report.k}"
    faithfulness_enabled = report.total_queries_judged > 0

    # --- Split overrides into qtype-level and qid-level (P7f). ---
    qtype_overrides = [
        ov for ov in pack.thresholds.overrides if ov.qtype is not None
    ]
    qid_overrides = [
        ov for ov in pack.thresholds.overrides if ov.qid is not None
    ]

    # --- effective_floors = deep copy of defaults, merged with qtype-overrides
    #     (replace per metric, last-wins, union introduces new qtypes/metrics).
    effective_floors: dict[str, dict[str, float]] = {
        qt: dict(metrics) for qt, metrics in defaults.items()
    }
    for ov in qtype_overrides:
        effective_floors.setdefault(ov.qtype, {}).update(ov.floors())

    for qtype, floors in effective_floors.items():
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

    # --- Per-qid overrides: check individual query scores (P7f). ---
    if qid_overrides:
        qid_to_qtype: dict[str, str] = {
            g.qid: qtype
            for qtype, glist in pack.goldens.items()
            for g in glist
        }
        per_query_maps = {
            recall_key: report.per_query_recall,
            "mrr": report.per_query_mrr,
            "faithfulness": report.per_query_faithfulness,
        }
        for ov in qid_overrides:
            for metric, floor in ov.floors().items():
                per_query = per_query_maps.get(metric)
                if per_query is None:
                    continue  # unknown metric / recall-k mismatch → skip.
                if ov.qid not in per_query:
                    continue  # query skipped / not judged → skip.
                actual = float(per_query[ov.qid])
                expected = float(floor)
                if actual < expected:
                    failures.append(
                        GateFailure(
                            qtype=qid_to_qtype.get(ov.qid, ""),
                            metric=metric,
                            expected=expected,
                            actual=actual,
                            qid=ov.qid,
                        )
                    )

    failures_tuple = tuple(failures)
    return GateResult(passed=not failures_tuple, failures=failures_tuple)
