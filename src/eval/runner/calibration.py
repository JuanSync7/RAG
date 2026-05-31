# @summary
# P9 — judge calibration loop. Quantifies agreement between the automated judge
# and a categorical reference (human / Tier-3) via Cohen's kappa, binarizing the
# judge's continuous scores at a threshold. compute_calibration returns a frozen
# CalibrationResult (kappa, agreement_rate=po, confusion tp/tn/fp/fn vs the
# reference, n, threshold, timestamp). detect_drift flags kappa strictly below a
# floor; persist_calibration_record mirrors persistence.persist_eval_report's
# deterministic {pack}_{filesafe-ts}.json layout; send_calibration_alert is a
# log-only sink (WARNING on drift, INFO otherwise), mirroring alert.send_alert.
# Exports: cohens_kappa, binarize_score, detect_drift, compute_calibration,
#          persist_calibration_record, send_calibration_alert, CalibrationResult,
#          CalibrationRecord, CalibrationAlert.
# Deps: stdlib only (dataclasses, json, logging, datetime, pathlib, typing).
# @end-summary
"""P9 — judge calibration loop.

The automated eval judge emits continuous faithfulness scores; periodically we
re-check it against a categorical reference label set (human adjudication or a
stronger Tier-3 model). This module binarizes the judge scores at a threshold
and measures inter-rater reliability via Cohen's kappa, persists a deterministic
calibration record, and raises a log-only drift alert when kappa falls below a
configured floor.

All helpers (:func:`cohens_kappa`, :func:`binarize_score`, :func:`detect_drift`)
are pure and side-effect-free. I/O is confined to
:func:`persist_calibration_record`; logging to :func:`send_calibration_alert`.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

logger = logging.getLogger("rag.eval.calibration")


def cohens_kappa(labels_a: Sequence[str], labels_b: Sequence[str]) -> float:
    """Return Cohen's kappa for two equal-length categorical label sequences.

    ``kappa = (po - pe) / (1 - pe)`` where ``po`` is the observed agreement rate
    and ``pe`` is the chance agreement implied by each rater's marginals. Works
    for any categorical labels, not just pass/fail.

    EDGE: when ``pe == 1.0`` (both raters placed everything in a single class, so
    chance agreement is total and the denominator collapses to 0) kappa is
    defined as ``1.0`` to avoid a ``0/0``.

    :raises ValueError: if the two sequences differ in length.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError(
            f"label sequences differ in length: {len(labels_a)} != {len(labels_b)}"
        )

    n = len(labels_a)
    if n == 0:
        # No observations: define perfect (vacuous) agreement.
        return 1.0

    # Observed agreement (po): fraction of positions where the raters agree.
    po = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n

    # Chance agreement (pe): sum over classes of the product of each rater's
    # marginal probability for that class.
    classes = set(labels_a) | set(labels_b)
    pe = 0.0
    for cls in classes:
        p_a = sum(1 for a in labels_a if a == cls) / n
        p_b = sum(1 for b in labels_b if b == cls) / n
        pe += p_a * p_b

    if pe == 1.0:
        # Both raters in one class -> total chance agreement; define kappa = 1.0.
        return 1.0
    return (po - pe) / (1.0 - pe)


def binarize_score(score: float, threshold: float) -> str:
    """Map a continuous score to ``"pass"``/``"fail"`` at ``threshold``.

    The boundary is INCLUSIVE: ``score == threshold`` -> ``"pass"`` (uses ``>=``).
    """
    return "pass" if score >= threshold else "fail"


def detect_drift(kappa: float, *, min_kappa: float) -> bool:
    """Return ``True`` iff ``kappa`` has drifted below the floor.

    STRICT: drift is ``kappa < min_kappa``. Equality (``kappa == min_kappa``)
    is NOT drift.
    """
    return kappa < min_kappa


@dataclass(frozen=True)
class CalibrationResult:
    """Immutable outcome of one calibration computation.

    :param kappa: Cohen's kappa between binarized judge labels and the reference.
    :param n: number of paired observations.
    :param agreement_rate: observed agreement ``po`` (fraction of matching
        positions).
    :param confusion_counts: confusion vs the reference treating ``"pass"`` as
        positive — keys EXACTLY ``"tp"``, ``"tn"``, ``"fp"``, ``"fn"``.
    :param threshold: the binarization threshold applied to judge scores.
    :param timestamp: ISO-8601 timestamp of the computation.
    """

    kappa: float
    n: int
    agreement_rate: float
    confusion_counts: dict[str, int]
    threshold: float
    timestamp: str


@dataclass(frozen=True)
class CalibrationRecord:
    """Immutable, persistable calibration record for a (pack, qtype) pair.

    :param pack_name: the eval pack's name.
    :param qtype: the question type the calibration covers.
    :param min_kappa: the configured kappa floor used for drift detection.
    :param drift_detected: whether ``result.kappa`` fell below ``min_kappa``.
    :param result: the underlying :class:`CalibrationResult`.
    """

    pack_name: str
    qtype: str
    min_kappa: float
    drift_detected: bool
    result: CalibrationResult


@dataclass(frozen=True)
class CalibrationAlert:
    """Immutable, transport-agnostic snapshot of a calibration outcome.

    :param pack_name: the eval pack's name.
    :param qtype: the question type the calibration covers.
    :param kappa: the measured kappa.
    :param min_kappa: the configured kappa floor.
    :param drift_detected: whether drift was flagged.
    :param timestamp: ISO-8601 timestamp of the run.
    :param record_path: filesystem path of the persisted record, or ``None``.
        Trailing additive field (defaults to ``None`` so prior call sites are
        unaffected) — mirrors :class:`AlertPayload.baseline_diff`.
    """

    pack_name: str
    qtype: str
    kappa: float
    min_kappa: float
    drift_detected: bool
    timestamp: str
    record_path: str | None = None


def compute_calibration(
    judge_scores: Sequence[float],
    reference_labels: Sequence[str],
    *,
    threshold: float,
    now: datetime | None = None,
) -> CalibrationResult:
    """Compute calibration of judge scores against categorical reference labels.

    Each judge score is binarized at ``threshold`` to a ``"pass"``/``"fail"``
    label; ``kappa`` is :func:`cohens_kappa` of those judge labels vs the
    reference; ``agreement_rate`` is the observed agreement ``po``; the confusion
    counts treat ``"pass"`` as the positive class.

    :param judge_scores: the automated judge's continuous scores.
    :param reference_labels: categorical ``"pass"``/``"fail"`` reference labels.
    :param threshold: binarization threshold (inclusive ``>=`` boundary).
    :param now: injected timestamp for deterministic tests; defaults to
        ``datetime.now(timezone.utc)`` (timezone-aware, never ``utcnow()``).
    :raises ValueError: if the score/label sequences differ in length.
    """
    if len(judge_scores) != len(reference_labels):
        raise ValueError(
            "judge_scores and reference_labels differ in length: "
            f"{len(judge_scores)} != {len(reference_labels)}"
        )

    judge_labels = [binarize_score(s, threshold) for s in judge_scores]
    kappa = cohens_kappa(judge_labels, reference_labels)

    n = len(judge_labels)
    agree = sum(1 for j, r in zip(judge_labels, reference_labels) if j == r)
    agreement_rate = agree / n if n else 1.0

    # Confusion vs the reference, "pass" treated as positive.
    confusion = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for j, r in zip(judge_labels, reference_labels):
        if j == "pass" and r == "pass":
            confusion["tp"] += 1
        elif j == "pass" and r == "fail":
            confusion["fp"] += 1
        elif j == "fail" and r == "pass":
            confusion["fn"] += 1
        else:  # j == "fail" and r == "fail"
            confusion["tn"] += 1

    if now is None:
        now = datetime.now(timezone.utc)

    return CalibrationResult(
        kappa=kappa,
        n=n,
        agreement_rate=agreement_rate,
        confusion_counts=confusion,
        threshold=threshold,
        timestamp=now.isoformat(),
    )


def persist_calibration_record(
    record: CalibrationRecord,
    output_dir: str | Path,
    *,
    now: datetime | None = None,
) -> Path:
    """Write ``record`` to a deterministically named JSON file; return its path.

    Mirrors :func:`src.eval.runner.persistence.persist_eval_report`: the filename
    is ``{pack_name}_{filesafe_ts}.json`` where ``filesafe_ts`` is the ISO-8601
    timestamp with colons stripped (illegal on some filesystems). The directory
    is created if absent. The JSON payload fully captures the record, including a
    nested ``result`` object.

    :param record: the :class:`CalibrationRecord` to persist.
    :param output_dir: directory to write into (created if absent).
    :param now: injected timestamp for deterministic filenames; defaults to
        ``datetime.now(timezone.utc)`` (timezone-aware, never ``utcnow()``).
    :returns: the :class:`~pathlib.Path` written.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    result = record.result
    payload = {
        "pack_name": record.pack_name,
        "qtype": record.qtype,
        "min_kappa": record.min_kappa,
        "drift_detected": record.drift_detected,
        "result": {
            "kappa": result.kappa,
            "n": result.n,
            "agreement_rate": result.agreement_rate,
            "confusion_counts": result.confusion_counts,
            "threshold": result.threshold,
            "timestamp": result.timestamp,
        },
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    filesafe_ts = now.isoformat().replace(":", "")
    path = out_dir / f"{record.pack_name}_{filesafe_ts}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def send_calibration_alert(payload: CalibrationAlert) -> None:
    """Emit ``payload`` to the log (the current, log-only calibration sink).

    On drift: one ``WARNING`` line naming the pack/qtype, kappa and floor. On no
    drift: one ``INFO`` line. Log-only; real delivery is a later slice. Mirrors
    the style of :func:`src.eval.runner.alert.send_alert`.
    """
    if payload.drift_detected:
        logger.warning(
            "calibration DRIFT: pack=%s qtype=%s kappa=%.3f min_kappa=%.3f "
            "timestamp=%s record=%s",
            payload.pack_name,
            payload.qtype,
            payload.kappa,
            payload.min_kappa,
            payload.timestamp,
            payload.record_path,
        )
    else:
        logger.info(
            "calibration OK: pack=%s qtype=%s kappa=%.3f min_kappa=%.3f "
            "timestamp=%s record=%s",
            payload.pack_name,
            payload.qtype,
            payload.kappa,
            payload.min_kappa,
            payload.timestamp,
            payload.record_path,
        )
