# @summary
# Offline scorer + comparison report for the turn_loop multi-mode basket run
# (rows produced by run_basket.collect / `run_basket compare --json`). Scores
# each answer on the axis its query calls for — coverage for answer queries,
# correct-decline for adversarial/out-of-corpus, clarification for vague ones —
# then aggregates per (mode, qtype) and renders the Phase-4 gate: does turn_loop
# match or beat every retired mode within the judge noise floor?
# Exports: score_rows, aggregate, gate, render, load_rows, main
# Deps: evals.common.answer_quality (judge, lazy); stdlib only otherwise
# @end-summary
"""Score a multi-mode basket run and render the turn_loop-vs-retired-modes gate.

Pipeline: ``run_basket compare --json run.json`` (collect) → ``score_basket
--collected run.json`` (this module). Scoring is separated from collection so a
slow live run is done once and re-scored freely.

Each query is scored on the axis its ``expected_behavior`` implies:
  * ``answer``               → keyword-coverage judge in [0,1] (higher better)
  * ``refuse_or_state_unknown`` → 1.0 iff the answer correctly declines
  * ``clarify``              → 1.0 iff the response asks for clarification
All three land in [0,1] so they aggregate together. The gate asks, per qtype:
is turn_loop's mean >= the best retired mode's mean minus the judge noise floor?
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Callable, Optional

from evals.common.answer_quality import (
    JUDGE_NOISE_FLOOR,
    refusal_correct,
    score_answer,
)

LEGACY_MODES = ("linear", "agentic", "deep_research", "tree")

QualityFn = Callable[[str, Optional[str], list], float]
RefusalFn = Callable[[str, Optional[str]], bool]


def _looks_like_clarify(route: Optional[str], answer: Optional[str]) -> bool:
    """A clarifying response asks the user a question rather than answering."""
    if route == "clarify":
        return True
    if not answer:
        return False
    return answer.strip().endswith("?")


def _metric_for(row: dict, *, quality_fn: QualityFn, refusal_fn: RefusalFn) -> dict:
    """Score one collected row on the axis its expected_behavior implies."""
    spec = row.get("spec") or {}
    behavior = spec.get("expected_behavior") or "answer"
    query = row.get("query") or ""
    answer = row.get("answer") or ""
    anchors = spec.get("anchor_terms") or []
    errored = bool(row.get("error"))

    if behavior == "refuse_or_state_unknown":
        kind = "refuse"
        metric = 1.0 if refusal_fn(query, answer) else 0.0
    elif behavior == "clarify":
        kind = "clarify"
        metric = 1.0 if _looks_like_clarify(row.get("route"), answer) else 0.0
    else:
        kind = "answer"
        metric = quality_fn(query, answer, anchors)

    return {
        "qid": row.get("qid"),
        "qtype": row.get("qtype"),
        "mode": row.get("mode"),
        "kind": kind,
        "metric": round(float(metric), 4),
        "answered": bool(answer.strip()),
        "errored": errored,
        "ttd_s": row.get("ttd_s"),
    }


def score_rows(rows: list[dict], *,
               quality_fn: QualityFn = score_answer,
               refusal_fn: RefusalFn = refusal_correct) -> list[dict]:
    """Score every collected row; returns lightweight scored rows."""
    return [_metric_for(r, quality_fn=quality_fn, refusal_fn=refusal_fn) for r in rows]


def _mean(xs: list[float]) -> Optional[float]:
    return round(statistics.fmean(xs), 4) if xs else None


def aggregate(scored: list[dict]) -> dict:
    """Aggregate scored rows into per-(mode, qtype) and per-mode summaries.

    Pure: no I/O, no model calls — the unit-testable core of the report.
    """
    modes = sorted({s["mode"] for s in scored})
    qtypes = sorted({s["qtype"] for s in scored})

    by_mode_qtype: dict[str, dict[str, Optional[float]]] = {}
    for m in modes:
        by_mode_qtype[m] = {}
        for qt in qtypes:
            vals = [s["metric"] for s in scored if s["mode"] == m and s["qtype"] == qt]
            by_mode_qtype[m][qt] = _mean(vals)

    per_mode: dict[str, dict[str, Any]] = {}
    for m in modes:
        ms = [s for s in scored if s["mode"] == m]
        lat = [s["ttd_s"] for s in ms if isinstance(s["ttd_s"], (int, float))]
        per_mode[m] = {
            "overall": _mean([s["metric"] for s in ms]),
            "error_rate": round(sum(1 for s in ms if s["errored"]) / len(ms), 4) if ms else None,
            "mean_ttd_s": _mean(lat),
            "n": len(ms),
        }

    return {"modes": modes, "qtypes": qtypes,
            "by_mode_qtype": by_mode_qtype, "per_mode": per_mode}


def gate(agg: dict, *, noise_floor: float = JUDGE_NOISE_FLOOR,
         latency_factor: float = 1.5) -> dict:
    """Verdict: does turn_loop match/beat every retired mode within the floor?

    Quality gate is the hard PASS/FAIL (per the blueprint: turn_loop must not
    regress any class). Latency is reported as an advisory warning, not a hard
    fail — the blueprint treats latency as a separately-measured GO signal.
    """
    modes = agg["modes"]
    if "turn_loop" not in modes:
        return {"verdict": "N/A", "reason": "no turn_loop rows", "regressions": [],
                "latency_warnings": []}
    legacy = [m for m in modes if m in LEGACY_MODES]

    regressions: list[dict] = []
    for qt in agg["qtypes"]:
        tl = agg["by_mode_qtype"]["turn_loop"].get(qt)
        if tl is None:
            continue
        legacy_scores = [(m, agg["by_mode_qtype"][m].get(qt)) for m in legacy]
        legacy_scores = [(m, v) for m, v in legacy_scores if v is not None]
        if not legacy_scores:
            continue
        best_mode, best_val = max(legacy_scores, key=lambda kv: kv[1])
        if tl + noise_floor < best_val:
            regressions.append({
                "qtype": qt, "turn_loop": tl,
                "best_legacy_mode": best_mode, "best_legacy": best_val,
                "delta": round(tl - best_val, 4),
            })

    latency_warnings: list[dict] = []
    tl_ttd = agg["per_mode"]["turn_loop"].get("mean_ttd_s")
    legacy_ttds = [(m, agg["per_mode"][m].get("mean_ttd_s")) for m in legacy]
    legacy_ttds = [(m, v) for m, v in legacy_ttds if v is not None]
    if tl_ttd is not None and legacy_ttds:
        fastest_mode, fastest = min(legacy_ttds, key=lambda kv: kv[1])
        if fastest > 0 and tl_ttd > latency_factor * fastest:
            latency_warnings.append({
                "turn_loop_ttd_s": tl_ttd, "fastest_legacy_mode": fastest_mode,
                "fastest_legacy_ttd_s": fastest, "factor": round(tl_ttd / fastest, 2),
            })

    return {
        "verdict": "PASS" if not regressions else "FAIL",
        "regressions": regressions,
        "latency_warnings": latency_warnings,
    }


def _cell(v: Optional[float]) -> str:
    """Right-aligned 15-wide cell: a 2-dp number or '-' for a missing value."""
    return f"{'-' if v is None else format(v, '.2f'):>15}"


def render(agg: dict, verdict: dict) -> str:
    """Human-readable comparison matrix + gate verdict."""
    modes = agg["modes"]
    lines: list[str] = []
    lines.append("Answer-quality by qtype x mode (mean metric in [0,1]):")
    lines.append(f"  {'qtype':<14}" + "".join(f"{m:>15}" for m in modes))
    for qt in agg["qtypes"]:
        cells = "".join(_cell(agg["by_mode_qtype"][m].get(qt)) for m in modes)
        lines.append(f"  {qt:<14}{cells}")
    overall = "".join(_cell(agg["per_mode"][m]["overall"]) for m in modes)
    lines.append(f"  {'OVERALL':<14}{overall}")

    lines.append("")
    lines.append("Per-mode: error-rate / mean TTD(s):")
    for m in modes:
        pm = agg["per_mode"][m]
        lines.append(f"  {m:<14} err={pm['error_rate']}  ttd={pm['mean_ttd_s']}  n={pm['n']}")

    lines.append("")
    lines.append(f"GATE (turn_loop >= best retired mode - {JUDGE_NOISE_FLOOR}): {verdict['verdict']}")
    for r in verdict["regressions"]:
        lines.append(f"  REGRESSION {r['qtype']}: turn_loop={r['turn_loop']:.2f} "
                     f"< {r['best_legacy_mode']}={r['best_legacy']:.2f} (Δ{r['delta']:+.2f})")
    for w in verdict["latency_warnings"]:
        lines.append(f"  LATENCY WARN: turn_loop ttd={w['turn_loop_ttd_s']}s is "
                     f"{w['factor']}x {w['fastest_legacy_mode']}={w['fastest_legacy_ttd_s']}s")
    return "\n".join(lines)


def load_rows(path: Path) -> list[dict]:
    """Load collected rows from a `compare --json` payload (or a bare row list)."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return data["rows"] if isinstance(data, dict) and "rows" in data else data


def main(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Score a multi-mode basket run")
    ap.add_argument("--collected", required=True,
                    help="JSON from `run_basket compare --json` (or a bare row list)")
    ap.add_argument("--json", dest="json_out", default="",
                    help="optional path to write the aggregate+verdict as JSON")
    args = ap.parse_args(argv)

    rows = load_rows(Path(args.collected))
    scored = score_rows(rows)
    agg = aggregate(scored)
    verdict = gate(agg)
    print(render(agg, verdict))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump({"aggregate": agg, "verdict": verdict, "scored": scored},
                      fh, indent=2)
        print(f"\nwrote aggregate+verdict -> {args.json_out}")
    return 0 if verdict["verdict"] in ("PASS", "N/A") else 1


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
