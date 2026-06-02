# @summary
# P8a.0 — report persistence primitive. persist_eval_report writes an
# EvalRunResult's report+gate to a JSON file under output_dir, named
# <pack_name>_<filesafe-iso-timestamp>.json, carrying the same eval payload
# shape _emit_json produces (single-sourced via cli._report_payload) PLUS
# top-level pack_name + timestamp. The orchestrator (P8a) consumes these files
# to persist and alert on eval runs. P11a adds persist_ralph_report, which
# mirrors the same filesafe-timestamp + mkdir(parents=True) convention to write
# a RalphReport (proposal-only investigation) to <output_dir>/ralph/.
# Exports: persist_eval_report, persist_ralph_report
# Deps: stdlib (datetime, json, pathlib); src.eval.cli._report_payload (shared
#       payload builder — single source for the JSON shape).
# @end-summary
"""Eval report persistence (P8a.0).

``persist_eval_report`` is a thin I/O primitive: it serializes the
report+gate payload (single-sourced from :func:`src.eval.cli._report_payload`)
plus a ``pack_name`` and ISO-8601 ``timestamp`` to a deterministically named
JSON file, and returns the written path. It performs no gating, emission, or
exit-code logic — the next slice (P8a orchestrator) layers diff/alert on top.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def persist_eval_report(
    result: Any,
    output_dir: str | Path,
    *,
    pack_name: str | None = None,
    now: datetime | None = None,
) -> Path:
    """Write ``result``'s report+gate to a JSON file and return its path.

    The JSON payload is exactly the shape :func:`src.eval.cli._emit_json`
    produces (built via the shared ``_report_payload`` helper so the two
    cannot drift) extended with two top-level keys:

    * ``pack_name`` — defaults to ``result.pack.meta.name``.
    * ``timestamp`` — ISO-8601 string of ``now``.

    :param result: an :class:`~src.eval.cli.EvalRunResult`.
    :param output_dir: directory to write into (created if absent).
    :param pack_name: override for the pack name (filename + payload).
    :param now: injected timestamp for deterministic tests; defaults to
        ``datetime.now(timezone.utc)`` (timezone-aware, not the deprecated
        ``datetime.utcnow()``).
    :returns: the :class:`~pathlib.Path` written.
    """
    # Local import to avoid a module-import cycle (cli imports run_eval which
    # this module's sibling package re-exports); resolved lazily at call time.
    from src.eval.cli import _report_payload

    if now is None:
        now = datetime.now(timezone.utc)
    if pack_name is None:
        pack_name = result.pack.meta.name

    payload = _report_payload(result.report, result.gate)
    payload["pack_name"] = pack_name
    payload["timestamp"] = now.isoformat()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filesafe timestamp: colons are illegal on some filesystems, so drop them.
    filesafe_ts = now.isoformat().replace(":", "")
    path = out_dir / f"{pack_name}_{filesafe_ts}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def persist_ralph_report(
    report: Any,
    output_dir: str | Path,
    *,
    now: datetime | None = None,
) -> Path:
    """Write a :class:`~src.eval.runner.ralph.RalphReport` to a JSON file.

    Mirrors :func:`persist_eval_report`'s conventions: a filesafe ISO timestamp
    (colons dropped) and ``mkdir(parents=True, exist_ok=True)``. The file lands
    under ``<output_dir>/ralph/`` named
    ``<pack_name>_ralph_<filesafe-iso-ts>.json`` and carries all report fields
    plus the flattened investigations list.

    This is the ONLY write in the Ralph-A surface — the loop itself
    (:func:`~src.eval.runner.ralph.run_ralph_a`) writes nothing, so persistence
    is an explicit, opt-in step the CLI performs.

    :param report: the :class:`~src.eval.runner.ralph.RalphReport` to persist.
    :param output_dir: base directory; the report is written under a ``ralph/``
        subdirectory (created if absent).
    :param now: injected timestamp for the filesafe filename; defaults to
        ``datetime.now(timezone.utc)``. (The payload ``timestamp`` always comes
        from ``report.timestamp`` — single-sourced with the run.)
    :returns: the :class:`~pathlib.Path` written.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    payload = {
        "pack_name": report.pack_name,
        "timestamp": report.timestamp,
        "sustained_count": report.sustained_count,
        "max_iterations": report.max_iterations,
        "investigations": [
            {
                "qtype": inv.qtype,
                "metric": inv.metric,
                "runs_regressed": inv.runs_regressed,
                "window": inv.window,
                "latest_delta": inv.latest_delta,
                "diagnosis": inv.diagnosis,
                "suggested_fix": inv.suggested_fix,
            }
            for inv in report.investigations
        ],
    }

    out_dir = Path(output_dir) / "ralph"
    out_dir.mkdir(parents=True, exist_ok=True)

    filesafe_ts = now.isoformat().replace(":", "")
    path = out_dir / f"{report.pack_name}_ralph_{filesafe_ts}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
