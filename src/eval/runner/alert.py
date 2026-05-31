# @summary
# P8a/P8c — eval alert payload + log-only sink. AlertPayload is the frozen,
# transport-agnostic shape of a nightly gate result (pack/collection/passed/
# failures/report_path/timestamp) plus an optional trailing baseline_diff
# (BaselineDiff | None) attached by the orchestrator when a baseline exists.
# send_alert is a LOG-ONLY stub: info on pass, warning (per-failure) on fail,
# AND a warning per regression when baseline_diff carries regressions (logged
# even on gate pass). Real Slack/webhook wiring is a later slice.
# Exports: AlertPayload, send_alert
# Deps: stdlib (dataclasses, logging); src.eval.runner.baseline (BaselineDiff).
# @end-summary
"""Eval alert payload + log-only sink (P8a).

:class:`AlertPayload` is the transport-agnostic shape the nightly orchestrator
builds from each pack's :class:`~src.eval.runner.gate.GateResult`.
:func:`send_alert` is the current sink — a log-only stub so the orchestrator's
alert seam is exercised end-to-end now while the real Slack/webhook delivery is
deferred to a later slice. Tests inject a capturing ``alert_fn`` instead.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from .baseline import BaselineDiff

logger = logging.getLogger("rag.eval.alert")


@dataclass(frozen=True)
class AlertPayload:
    """Immutable, transport-agnostic snapshot of one pack's gate outcome.

    :param pack_name: the eval pack's name (``pack.meta.name``).
    :param collection_name: the Weaviate collection the run targeted.
    :param gate_passed: ``True`` iff the gate had no failures.
    :param failures: the GateFailure dicts
        (``{qtype, metric, expected, actual, qid}``); empty on pass.
    :param report_path: filesystem path of the persisted report JSON.
    :param timestamp: ISO-8601 timestamp of the run.
    :param baseline_diff: the :class:`~src.eval.runner.baseline.BaselineDiff`
        the orchestrator computed against the pack's committed
        ``baseline.json`` sibling, or ``None`` when no baseline exists. Trailing
        additive field (defaults to ``None`` so prior call sites are unaffected).
    """

    pack_name: str
    collection_name: str
    gate_passed: bool
    failures: list[dict]
    report_path: str
    timestamp: str
    baseline_diff: BaselineDiff | None = None


def send_alert(payload: AlertPayload) -> None:
    """Emit ``payload`` to the log (the current, log-only alert sink).

    On pass: a single ``INFO`` line. On fail: a ``WARNING`` line naming the pack
    plus one ``WARNING`` line per failure (qtype/metric/expected/actual).

    Independently of the gate outcome, if ``payload.baseline_diff`` carries any
    regressions, a ``WARNING`` summary line plus one ``WARNING`` line per
    regressed metric are emitted — so a regression vs baseline is surfaced even
    when the absolute gate passed. Real delivery (Slack/webhook) is wired in a
    later slice; this keeps the orchestrator's alert seam observable today.
    """
    if payload.gate_passed:
        logger.info(
            "eval gate PASSED: pack=%s collection=%s report=%s",
            payload.pack_name,
            payload.collection_name,
            payload.report_path,
        )
    else:
        logger.warning(
            "eval gate FAILED: pack=%s collection=%s failures=%d report=%s",
            payload.pack_name,
            payload.collection_name,
            len(payload.failures),
            payload.report_path,
        )
        for failure in payload.failures:
            logger.warning(
                "  pack=%s FAIL qtype=%s metric=%s expected>=%s actual=%s qid=%s",
                payload.pack_name,
                failure.get("qtype", ""),
                failure.get("metric", ""),
                failure.get("expected"),
                failure.get("actual"),
                failure.get("qid", ""),
            )

    diff = payload.baseline_diff
    if diff is not None and diff.regressions:
        logger.warning(
            "eval REGRESSION vs baseline: pack=%s regressions=%d report=%s",
            payload.pack_name, len(diff.regressions), payload.report_path,
        )
        for md in diff.regressions:
            logger.warning(
                "  pack=%s REGRESSION qtype=%s metric=%s baseline=%s current=%s delta=%s",
                payload.pack_name, md.qtype, md.metric,
                md.baseline_value, md.current_value, md.delta,
            )
