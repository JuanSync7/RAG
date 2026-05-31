# @summary
# P9 — one-shot judge-calibration loop orchestrator. run_calibration composes
# the existing pure pieces: it loads the pinned ground-truth fixture (or accepts
# injected examples), scores each example with the supplied JudgeClient, computes
# Cohen's kappa vs the reference labels (compute_calibration), detects drift
# strictly below min_kappa (detect_drift), persists a CalibrationRecord
# (persist_calibration_record, FIRST so the alert can carry the path), then fires
# a calibration alert (send_calibration_alert by default). Pooled single-threshold
# calibration over the whole fixture, labelled with qtype (default "all"). All
# side effects (judge, output_dir, alert_fn, now) are injected so the loop is a
# pure orchestrator.
# Exports: run_calibration.
# Deps: stdlib (collections.abc, datetime, pathlib, typing); .calibration
#       (compute_calibration, detect_drift, persist_calibration_record,
#       send_calibration_alert, CalibrationAlert, CalibrationRecord),
#       .calibration_fixture (CalibrationExample, load_calibration_fixture),
#       .judge (JudgeClient).
# @end-summary
"""P9 — one-shot judge-calibration loop.

This module wires the calibration *primitives* into a single end-to-end run.
Given a :class:`~src.eval.runner.judge.JudgeClient` and a (pack, threshold,
min_kappa) configuration, :func:`run_calibration`:

1. loads the versioned ground-truth fixture (or uses injected ``examples``),
2. scores every example with the judge,
3. computes Cohen's kappa of the binarized judge labels vs the references,
4. detects drift (strictly below ``min_kappa``),
5. persists a deterministic :class:`CalibrationRecord`,
6. fires a calibration alert carrying the persisted record's path.

Design rationale
----------------
* **Pooled, single-threshold** calibration over the *whole* fixture. The
  fixture is small and cross-qtype, so a per-qtype kappa would be statistically
  vacuous; the run is recorded with a ``qtype`` label (default ``"all"``).
* ``threshold`` and ``min_kappa`` are **required keyword-only** parameters — no
  hardcoded floors. The caller decides, which respects the project's
  "behaviour controlled by typed config, fail fast" rule.
* All side effects are injected (``judge_client``, ``output_dir``, ``alert_fn``,
  ``now``) so the loop is a pure orchestrator, mirroring the seams in
  :func:`src.eval.orchestrator.run_nightly` (alert sink defaults to the real
  sink; ``now`` defaults are applied inside the composed pieces).

The persist step runs *before* the alert so the alert payload can carry the
written record's path.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

from .calibration import (
    CalibrationAlert,
    CalibrationRecord,
    compute_calibration,
    detect_drift,
    persist_calibration_record,
    send_calibration_alert,
)
from .calibration_fixture import CalibrationExample, load_calibration_fixture
from .judge import JudgeClient

# Deferred: there is currently NO faithfulness threshold / min_kappa field in the
# pack schema, so this loop takes both as explicit params. A pack-aware threshold
# reader is intentionally left to a future slice that adds faithfulness floors to
# packs (adding a reader for a non-existent key now would be dead code).


def _score_examples(
    judge_client: JudgeClient, examples: Sequence[CalibrationExample]
) -> list[float]:
    """Score each example with the judge, preserving order.

    Returns the continuous judge scores (one per example). Kept tiny and private
    so :func:`run_calibration` reads as a flat orchestration.
    """
    return [
        judge_client.score(example.to_judge_question()).score
        for example in examples
    ]


def run_calibration(
    judge_client: JudgeClient,
    *,
    pack_name: str,
    threshold: float,
    min_kappa: float,
    output_dir: str | Path,
    examples: Sequence[CalibrationExample] | None = None,
    qtype: str = "all",
    alert_fn: Callable[[CalibrationAlert], None] = send_calibration_alert,
    now: datetime | None = None,
) -> CalibrationRecord:
    """Run the one-shot judge-calibration loop end to end.

    :param judge_client: the judge to calibrate; its ``score`` is called once per
        example.
    :param pack_name: the eval pack name recorded on the result and alert.
    :param threshold: binarization threshold for judge scores (required; inclusive
        ``>=`` boundary inside :func:`compute_calibration`).
    :param min_kappa: the kappa floor; drift is ``kappa < min_kappa`` (required).
    :param output_dir: directory the :class:`CalibrationRecord` JSON is written to
        (created if absent).
    :param examples: calibration examples to score. When ``None`` (the production
        path) the pinned ground-truth fixture is loaded via
        :func:`load_calibration_fixture`.
    :param qtype: the question-type label recorded on the run (default ``"all"``;
        the run is pooled across qtypes).
    :param alert_fn: the alert sink (default: the real
        :func:`send_calibration_alert`). Injected so tests can capture alerts.
    :param now: injected timestamp for deterministic results/filenames; forwarded
        to the composed pieces, which default to ``datetime.now(timezone.utc)``.
    :returns: the persisted :class:`CalibrationRecord`.
    """
    if examples is None:
        examples = load_calibration_fixture()

    judge_scores = _score_examples(judge_client, examples)
    reference_labels = [ex.reference_label for ex in examples]

    result = compute_calibration(
        judge_scores, reference_labels, threshold=threshold, now=now
    )
    drift = detect_drift(result.kappa, min_kappa=min_kappa)

    record = CalibrationRecord(
        pack_name=pack_name,
        qtype=qtype,
        min_kappa=min_kappa,
        drift_detected=drift,
        result=result,
    )

    # Persist FIRST so the alert payload can carry the written record's path.
    path = persist_calibration_record(record, output_dir, now=now)

    alert = CalibrationAlert(
        pack_name=pack_name,
        qtype=qtype,
        kappa=result.kappa,
        min_kappa=min_kappa,
        drift_detected=drift,
        timestamp=result.timestamp,
        record_path=str(path),
    )
    alert_fn(alert)

    return record
