# @summary
# Per-pack append-only run-history index + per-qid regression deltas vs baseline.
# compute_per_query_deltas diffs the three per-query metric maps (recall/mrr/
# faithfulness) of two persisted eval-report payloads at qid granularity, sign
# convention = current - baseline (negative = worse), mirroring baseline.py;
# first run (baseline None/empty) → {} (never raises). index_run_report appends
# ONE plain-dict JSONL line (timestamp/pack_name/k/gate_passed/per_query_deltas/
# per_qtype_deltas) to <history_dir>/<pack_name>.history.jsonl (path keyed on
# pack_name ONLY — timestamp lives INSIDE the line), reusing compute_baseline_diff
# for the per-qtype layer; returns the index path. load_run_history reads+parses
# the JSONL and returns entries sorted by timestamp (missing file → []).
# Determinism: `now`/timestamps are explicit args; no inline datetime.now().
# Exports: compute_per_query_deltas, index_run_report, load_run_history
# Deps: stdlib (json, pathlib, datetime); src.eval.runner.baseline
#       (compute_baseline_diff — single-sourced per-qtype delta semantics).
# @end-summary
"""Per-pack run-history index + per-qid regression deltas vs baseline.

This layer sits one level BELOW the per-qtype :func:`compute_baseline_diff
<src.eval.runner.baseline.compute_baseline_diff>`: it adds a per-query (per-qid)
delta map plus an append-only history accumulation so a future
Ralph-on-regression auto-fixer can identify "the worst-regressing query across
runs".

Sign convention is mirrored from :mod:`src.eval.runner.baseline`:
``delta = current - baseline`` (negative = worse). The first run has no prior
baseline, so per-query deltas degrade to an empty map rather than raising.

The history index is keyed on ``pack_name`` ONLY — every run for a pack appends
one JSONL line to ``<history_dir>/<pack_name>.history.jsonl``; the per-run
timestamp lives INSIDE each line, never in the filename. Each line is a plain
JSON-serializable dict (NOT a frozen dataclass).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.eval.runner.baseline import compute_baseline_diff

# Map the persisted per-QUERY JSON map name to the stable metric label, mirroring
# baseline.py's _METRIC_MAPS (which maps the per-QTYPE maps). Order is
# load-bearing for determinism.
_PER_QUERY_MAPS: tuple[tuple[str, str], ...] = (
    ("per_query_recall", "recall"),
    ("per_query_mrr", "mrr"),
    ("per_query_faithfulness", "faithfulness"),
)


def compute_per_query_deltas(
    baseline_payload: dict | None,
    current_payload: dict,
    *,
    epsilon: float = 0.05,
) -> dict[str, dict[str, float]]:
    """Per-qid metric deltas (``current - baseline``) across the per-query maps.

    Diffs the three per-query metric maps (``per_query_recall`` → ``"recall"``,
    ``per_query_mrr`` → ``"mrr"``, ``per_query_faithfulness`` →
    ``"faithfulness"``) of two persisted eval-report payloads. A qid/metric pair
    present on BOTH sides yields ``delta = current - baseline`` (negative =
    worse, matching :class:`~src.eval.runner.baseline.MetricDelta`). Pairs
    present on only one side are skipped (a new/dropped query is not a delta).

    The ``epsilon`` argument is accepted for parity with
    :func:`~src.eval.runner.baseline.compute_baseline_diff` but does NOT filter
    the recorded deltas: every shared qid/metric delta is reported with its
    sign so a downstream auto-fixer can rank by magnitude. (Regression
    *classification* against ``epsilon`` is the caller's concern.)

    :param baseline_payload: the prior baseline report payload, or ``None``/empty
        on the first run (no prior) → returns ``{}`` (never raises).
    :param current_payload: the current run's report payload.
    :param epsilon: tolerance band, accepted for signature parity (unused here).
    :returns: ``{qid: {metric: delta}}`` for qid/metric pairs present on both
        sides; ``{}`` when there is no usable baseline.
    """
    if not baseline_payload:
        # First run (or empty baseline) has no prior to diff against.
        return {}

    deltas: dict[str, dict[str, float]] = {}
    for map_name, label in _PER_QUERY_MAPS:
        base_map = baseline_payload.get(map_name) or {}
        cur_map = current_payload.get(map_name) or {}
        for qid in base_map.keys() & cur_map.keys():
            base_val = base_map.get(qid)
            cur_val = cur_map.get(qid)
            if base_val is None or cur_val is None:
                continue
            delta = float(cur_val) - float(base_val)
            deltas.setdefault(qid, {})[label] = delta
    return deltas


def index_run_report(
    result: Any,
    *,
    baseline_payload: dict | None,
    current_payload: dict,
    history_dir: str | Path,
    now: datetime,
) -> Path:
    """Append one run's entry to the per-pack history JSONL and return its path.

    The index is keyed on ``pack_name`` ONLY:
    ``<history_dir>/<pack_name>.history.jsonl``. This function APPENDS exactly
    one line — a plain JSON-serializable dict — so the file accumulates one entry
    per run across invocations. The per-run timestamp lives INSIDE the line.

    Each entry carries:

    * ``timestamp`` — ISO-8601 string of ``now``,
    * ``pack_name`` — taken from ``current_payload``,
    * ``k`` — the eval ``k`` from ``current_payload``,
    * ``gate_passed`` — the absolute gate result (``current_payload["passed"]``),
    * ``per_query_deltas`` — ``{qid: {metric: delta}}`` from
      :func:`compute_per_query_deltas` (``{}`` on the first run),
    * ``per_qtype_deltas`` — ``{qtype: {metric: delta}}`` derived from the
      per-qtype :func:`~src.eval.runner.baseline.compute_baseline_diff` (reusing
      its single-sourced sign convention; ``{}`` on the first run).

    :param result: the :class:`~src.eval.cli.EvalRunResult` (accepted for
        signature parity with the orchestrator's other persistence calls; the
        serialized fields are read from ``current_payload`` so the JSONL shape is
        single-sourced with the persisted report).
    :param baseline_payload: the prior baseline payload, or ``None`` on the
        first run.
    :param current_payload: the just-persisted current report payload.
    :param history_dir: directory the JSONL index lives in (created if absent).
    :param now: injected timestamp (REQUIRED for determinism — never resolved
        inline).
    :returns: the :class:`~pathlib.Path` of the history index file.
    """
    pack_name = current_payload.get("pack_name", "")

    per_query_deltas = compute_per_query_deltas(baseline_payload, current_payload)
    per_qtype_deltas = _per_qtype_deltas(baseline_payload, current_payload)

    entry = {
        "timestamp": now.isoformat(),
        "pack_name": pack_name,
        "k": current_payload.get("k"),
        "gate_passed": bool(current_payload.get("passed")),
        "per_query_deltas": per_query_deltas,
        "per_qtype_deltas": per_qtype_deltas,
    }

    out_dir = Path(history_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / f"{pack_name}.history.jsonl"
    # Append one compact JSONL line per run.
    with index_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    return index_path


def load_run_history(pack_name: str, history_dir: str | Path) -> list[dict]:
    """Read+parse the per-pack history JSONL, sorted by ``timestamp``.

    :param pack_name: the pack whose history to load.
    :param history_dir: directory the JSONL index lives in.
    :returns: the parsed entries sorted ascending by ``timestamp``; ``[]`` when
        the index file is absent (graceful — a pack with no history yet).
    """
    index_path = Path(history_dir) / f"{pack_name}.history.jsonl"
    if not index_path.is_file():
        return []
    entries: list[dict] = []
    for line in index_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        entries.append(json.loads(line))
    entries.sort(key=lambda e: e.get("timestamp", ""))
    return entries


def _per_qtype_deltas(
    baseline_payload: dict | None,
    current_payload: dict,
) -> dict[str, dict[str, float]]:
    """Derive a ``{qtype: {metric: delta}}`` map from the per-qtype baseline diff.

    Reuses :func:`~src.eval.runner.baseline.compute_baseline_diff` so the
    per-qtype sign convention is single-sourced (delta = current - baseline).
    Returns ``{}`` when there is no usable baseline.
    """
    if not baseline_payload:
        return {}
    diff = compute_baseline_diff(baseline_payload, current_payload)
    out: dict[str, dict[str, float]] = {}
    for md in (*diff.regressions, *diff.improvements, *diff.unchanged):
        out.setdefault(md.qtype, {})[md.metric] = md.delta
    return out
