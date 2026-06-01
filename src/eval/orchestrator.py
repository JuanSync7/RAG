# @summary
# P8a/P8c — nightly eval orchestrator (testable core; the scripts/ shim is thin).
# discover_packs globs pack dirs (those containing pack.yaml) under a root.
# run_nightly iterates packs sequentially: run_eval → persist_eval_report →
# (P8c) if a committed `baseline.json` sibling exists, load it + read the
# just-persisted report back, compute_baseline_diff, attach to AlertPayload →
# alert_fn; resilient batch (one pack erroring does not abort the rest). Returns
# 0 iff EVERY pack ran AND passed, else 1; when fail_on_regression=True a
# regression vs baseline flips the exit to 1 even if the absolute gate passed.
# Packs with no baseline behave exactly as pre-P8c (baseline_diff=None, no flip).
# P10: run_nightly accepts + forwards execute_fn/retrieve_fn/judge_client_factory
# to run_eval so the offline CI smoke (src.eval.smoke) drives the full loop with
# no live infra; each defaults to None (live behaviour unchanged).
# After each persist+baseline-diff, the orchestrator also appends the run to a
# per-pack append-only run-history index (per-qid + per-qtype deltas vs the
# loaded baseline) via run_history.index_run_report; this is best-effort and
# never fails the run or changes run_nightly's return value/alert payload.
# Then (also best-effort) it reloads that history — now including the just-
# appended current run — and runs detect_sustained_regressions to flag any
# (qtype, metric) that regressed across >=K-of-N consecutive runs, attaching the
# verdict to AlertPayload.sustained_regressions. Detection failure is caught
# narrowly and never fails the run or alters any other field.
# Exports: discover_packs, run_nightly
# Deps: stdlib (json, logging, pathlib, datetime); src.eval.cli.run_eval;
#       src.eval.runner.persistence.persist_eval_report;
#       src.eval.runner.alert (AlertPayload, send_alert);
#       src.eval.runner.baseline.compute_baseline_diff;
#       src.eval.runner.run_history (index_run_report, load_run_history);
#       src.eval.runner.sustained_regression.detect_sustained_regressions.
# @end-summary
"""Nightly eval orchestrator core (P8a).

Two public functions, both importable and unit-testable (the
``scripts/run_nightly_eval.py`` shim only does argparse + delegation):

* :func:`discover_packs` — find eval pack directories under a root.
* :func:`run_nightly` — run the full eval loop over a list of packs, persist
  each report, alert on each gate result, and return a binary batch exit code.

Per-pack granularity (which pack passed / failed / errored) goes to the logs and
the persisted JSON files; the batch return value is binary pass/fail. ``alert_fn``
is the injection seam tests use to capture the shaped :class:`AlertPayload`.

Baseline diff (P8c): after persisting each pack's report, the orchestrator looks
for a committed ``baseline.json`` sibling inside the pack directory. When present
it loads the baseline + reads the just-persisted current report back and calls
:func:`~src.eval.runner.baseline.compute_baseline_diff`, attaching the result to
``AlertPayload.baseline_diff``. With ``fail_on_regression=True`` any regression
flips the batch exit to ``1`` even if the absolute gate passed. Packs with no
baseline — or with an UNUSABLE (corrupt/unparseable) ``baseline.json`` — behave
exactly as before (``baseline_diff=None``, no exit flip); a bad baseline file is
a baseline problem, not an eval error, so it is logged and skipped rather than
marking the pack errored.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.eval.cli import run_eval
from src.eval.runner.alert import AlertPayload, send_alert
from src.eval.runner.baseline import compute_baseline_diff
from src.eval.runner.persistence import persist_eval_report
from src.eval.runner.run_history import index_run_report, load_run_history
from src.eval.runner.sustained_regression import detect_sustained_regressions

logger = logging.getLogger("rag.eval.orchestrator")


def discover_packs(pack_root: str | Path, pack: str = "*") -> list[Path]:
    """Return eval pack directories under ``pack_root`` matching ``pack``.

    A pack directory is one that contains a ``pack.yaml``. ``pack`` may be an
    exact name or a glob (default ``"*"`` = all). Results are sorted
    deterministically by path. Returns ``[]`` if none match — the caller decides
    whether an empty result is an error.
    """
    root = Path(pack_root)
    if not root.is_dir():
        return []
    candidates = [
        child
        for child in root.glob(pack)
        if child.is_dir() and (child / "pack.yaml").is_file()
    ]
    return sorted(candidates)


def run_nightly(
    pack_paths: list[Path | str],
    *,
    output_dir: str | Path,
    k: int = 5,
    fresh: bool = True,
    chat_model_factory: Callable[..., Any] | None = None,
    alert_fn: Callable[[AlertPayload], None] = send_alert,
    now: datetime | None = None,
    fail_on_regression: bool = False,
    regression_epsilon: float = 0.05,
    stderr: Any = None,
    execute_fn: Callable[..., Any] | None = None,
    retrieve_fn: Callable[..., Any] | None = None,
    judge_client_factory: Callable[..., Any] | None = None,
) -> int:
    """Run the full eval loop over ``pack_paths`` and return a batch exit code.

    For EACH pack (sequentially):

    1. :func:`~src.eval.cli.run_eval` — ingest → retrieve → judge → gate.
    2. :func:`~src.eval.runner.persistence.persist_eval_report` — write the
       report JSON under ``output_dir``.
    3. Build an :class:`AlertPayload` from the gate result and hand it to
       ``alert_fn`` (defaults to the real log-only :func:`send_alert`; tests
       inject a capturing function).
    4. Log a one-line summary.

    **Resilient batch:** each pack is wrapped in try/except, so one pack raising
    in ``run_eval`` does NOT abort the rest — the error is logged, the pack is
    marked errored, and the loop continues.

    Between steps 2 and 3, when a committed ``baseline.json`` sibling exists in
    the pack directory, the orchestrator loads it + reads the just-persisted
    report back and computes a :class:`~src.eval.runner.baseline.BaselineDiff`
    (attached to the :class:`AlertPayload` as ``baseline_diff``).

    **Exit code:** ``0`` iff EVERY pack ran AND passed its gate; ``1`` if any
    pack's gate failed OR any pack errored. When ``fail_on_regression`` is set,
    any pack whose baseline diff carries regressions ALSO flips the exit to
    ``1`` — even if that pack's absolute gate passed.

    :param now: injected timestamp for deterministic persisted filenames.
    :param fail_on_regression: when ``True``, a regression vs the pack's
        committed ``baseline.json`` flips the batch exit to ``1`` even if the
        absolute gate passed. Default ``False`` (regressions are logged/attached
        but do not affect the exit code).
    :param regression_epsilon: tolerance band passed to
        :func:`~src.eval.runner.baseline.compute_baseline_diff` (default
        ``0.05``).
    :param execute_fn: optional ingest seam forwarded verbatim to
        :func:`~src.eval.cli.run_eval` (default ``None`` → live ``execute_plan``).
    :param retrieve_fn: optional retrieval seam forwarded to ``run_eval``
        (default ``None`` → live ``retrieve_for_goldens``).
    :param judge_client_factory: optional judge-client seam forwarded to
        ``run_eval`` (default ``None`` → live ``build_judge_client``). These
        three seams let the offline CI smoke (:mod:`src.eval.smoke`) drive the
        full loop with no live infra and no monkeypatch.
    """
    # Resolve ``now`` ONCE so the persisted-filename timestamp and the alert
    # timestamp are the SAME value (and timezone-aware) on the production
    # default path. Threading the same resolved ``now`` to both
    # persist_eval_report and AlertPayload avoids desync / tz-naive drift.
    if now is None:
        now = datetime.now(timezone.utc)

    all_passed = True

    for pack_path in pack_paths:
        pack_label = str(pack_path)
        try:
            result = run_eval(
                str(pack_path),
                k=k,
                fresh=fresh,
                chat_model_factory=chat_model_factory,
                stderr=stderr,
                execute_fn=execute_fn,
                retrieve_fn=retrieve_fn,
                judge_client_factory=judge_client_factory,
            )
            report_path = persist_eval_report(result, output_dir, now=now)

            baseline_diff = None
            baseline_json: dict | None = None
            baseline_path = Path(pack_path) / "baseline.json"
            if baseline_path.is_file():
                # An UNUSABLE committed baseline (corrupt JSON, merge-conflict
                # markers, truncation) must degrade like an ABSENT baseline — it
                # is a baseline-file problem, not an eval-pipeline error, so it
                # is caught HERE (narrow) rather than by the per-pack handler,
                # which would wrongly mark the pack errored and suppress its
                # alert even though the eval itself succeeded.
                try:
                    baseline_json = json.loads(
                        baseline_path.read_text(encoding="utf-8")
                    )
                    current_json = json.loads(
                        Path(report_path).read_text(encoding="utf-8")
                    )
                    baseline_diff = compute_baseline_diff(
                        baseline_json, current_json, epsilon=regression_epsilon
                    )
                except (json.JSONDecodeError, OSError) as exc:
                    logger.warning(
                        "pack=%s unusable baseline.json (%s: %s) — treating as "
                        "no baseline",
                        pack_label,
                        type(exc).__name__,
                        exc,
                    )
                    baseline_diff = None
                    baseline_json = None

            # Append this run to the per-pack run-history index (per-qid +
            # per-qtype deltas vs the loaded baseline). Best-effort: an indexing
            # failure must NOT fail the run or change the return value/alert
            # payload — it writes into the SAME output_dir the report uses. The
            # baseline dict is reused (None when absent OR unusable). Caught
            # NARROWLY (I/O + value/key shape) so real bugs are not swallowed.
            current_payload: dict | None = None
            try:
                current_payload = json.loads(
                    Path(report_path).read_text(encoding="utf-8")
                )
                index_run_report(
                    result,
                    baseline_payload=baseline_json,
                    current_payload=current_payload,
                    history_dir=output_dir,
                    now=now,
                )
            except (OSError, json.JSONDecodeError, ValueError, KeyError) as exc:
                logger.warning(
                    "pack=%s run-history indexing failed (%s: %s) — skipping",
                    pack_label,
                    type(exc).__name__,
                    exc,
                )

            # Sustained-regression verdict: reload the per-pack history (now
            # INCLUDING the run just appended above as the most-recent window
            # entry) and flag any (qtype, metric) that regressed across
            # >=K-of-N consecutive runs. Best-effort: a detection failure must
            # NOT fail the run, change ``rc``, or alter any other AlertPayload
            # field. Keyed on the SAME pack_name the history file is keyed on
            # (the persisted report's ``pack_name``). Caught NARROWLY so real
            # bugs are not swallowed. ``regression_epsilon`` is reused for
            # parity with the single-run baseline diff.
            sustained: tuple = ()
            history_pack_name = (current_payload or {}).get("pack_name")
            if history_pack_name:
                try:
                    history = load_run_history(history_pack_name, output_dir)
                    sustained = detect_sustained_regressions(
                        history, epsilon=regression_epsilon
                    )
                except (OSError, ValueError, KeyError) as exc:
                    logger.warning(
                        "pack=%s sustained-regression detection failed (%s: %s) "
                        "— skipping",
                        pack_label,
                        type(exc).__name__,
                        exc,
                    )

            failures = [
                {
                    "qtype": f.qtype,
                    "metric": f.metric,
                    "expected": f.expected,
                    "actual": f.actual,
                    "qid": f.qid,
                }
                for f in result.gate.failures
            ]
            payload = AlertPayload(
                pack_name=result.pack.meta.name,
                collection_name=result.collection_name,
                gate_passed=result.gate.passed,
                failures=failures,
                report_path=str(report_path),
                timestamp=now.isoformat(),
                baseline_diff=baseline_diff,
                sustained_regressions=sustained,
            )
            alert_fn(payload)

            if fail_on_regression and baseline_diff is not None and baseline_diff.regressions:
                all_passed = False

            logger.info(
                "pack=%s gate=%s failures=%d report=%s",
                result.pack.meta.name,
                "PASS" if result.gate.passed else "FAIL",
                len(failures),
                report_path,
            )
            if not result.gate.passed:
                all_passed = False
        except Exception as exc:  # noqa: BLE001 — resilient batch: log + continue.
            all_passed = False
            logger.error("pack=%s errored: %s: %s", pack_label, type(exc).__name__, exc)

    return 0 if all_passed else 1
