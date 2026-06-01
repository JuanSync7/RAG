<!-- @summary
Offline eval loop: validates an eval_pack, ingests its corpus, retrieves against goldens, judges chunk-level faithfulness, gates on per-qtype thresholds, and exits with a deterministic code.
@end-summary -->

# eval

## Overview

This package implements the offline eval loop. A self-contained **eval pack** on disk (`pack.yaml`, `corpus/`, `goldens/`, `thresholds.yaml`, optional `prompts/`) is the configuration surface. The runner chains:

```
load_pack → plan_pack_ingest → execute_plan → retrieve_for_goldens
          → score_goldens → build_eval_report → validate_eval_report
```

into a single CLI subcommand. End-to-end answer-faithfulness for production runtime lives in `src/guardrails/` — this package is the offline pack-driven eval surface only.

## Subdirectories

| Directory / File | Purpose |
| --- | --- |
| `pack/` | Eval-pack format: typed schema (Pydantic), structural validator, loader, single error type. |
| `runner/` | Pure planner, live ingest executor, retrieval + recall@k, judge contract, faithfulness executor, gate. |
| `cli.py` | `python -m src.eval run <pack>` orchestrator. Deterministic exit codes. `run_eval(...)` exposes three default-preserving DI seams — `execute_fn` / `retrieve_fn` / `judge_client_factory` (resolved to the live `runner.*` symbols at call time) — so the offline smoke can drive the loop with no live infra. |
| `orchestrator.py` | Nightly batch core: `run_nightly(...)` runs the loop per pack, persists, alerts, and returns a binary pass/fail code. Forwards the three `run_eval` DI seams. After each persist+baseline-diff it appends the run to a per-pack append-only run-history index via `runner/run_history.py`, then (best-effort) reloads that history and runs `runner/sustained_regression.py` to attach a multi-run sustained-regression verdict to the alert (never fails the run). |
| `runner/run_history.py` | Per-pack append-only run-history index + per-qid regression deltas vs baseline. `compute_per_query_deltas` diffs the three per-query metric maps at qid granularity (sign = current − baseline, mirroring `baseline.py`); `index_run_report` appends one JSONL line (timestamp/pack_name/k/gate_passed/per_query_deltas/per_qtype_deltas) to `<output_dir>/<pack>.history.jsonl`; `load_run_history` reads it back sorted by timestamp. Feeds a future Ralph-on-regression auto-fixer. |
| `runner/sustained_regression.py` | Multi-run sustained-regression detector over the run-history index. `detect_sustained_regressions(history, *, window=3, min_regressed=2, epsilon=0.05)` flags each `(qtype, metric)` that regressed (delta ≤ −epsilon, mirroring `baseline.py`) in ≥`min_regressed` of the most-recent `window` runs — suppressing one-off judge/retrieval dips a single-run `baseline_diff` would alarm on. Per-QTYPE granularity (parity with `baseline_diff`); per-QID sustained detection is future work. Returns a deterministically sorted tuple of frozen `SustainedMetricRegression`; surfaced on `AlertPayload.sustained_regressions`. |
| `runner/show_trend.py` | Pure markdown formatter for the read-only `show-trend` subcommand. `render_trend_table(history, *, runs=3, window=3, min_regressed=2, epsilon=0.05)` maps run-history entries to a per-`(qtype, metric)` trend table (one delta column per the last `runs` runs, plus `regressed/window`, `latest`, and a `sustained` YES/NO flag from `detect_sustained_regressions`). NO I/O; empty history → a friendly one-line message. Reuses `ci._fmt`. |
| `smoke.py` | Offline PR-gate smoke (P10): `python -m src.eval.smoke` drives the FULL loop (`run_nightly → run_eval → persist → gate → alert`) against a tiny synthetic pack via the DI seams — no live Weaviate/embeddings/LLM. Runs a clean (rc 0) and an injected-regression (rc 1) scenario; exits 0 iff BOTH behave. |
| `__main__.py` | Module entry — delegates to `cli.main`. |

## Quick Start

```bash
# Run the OpenTitan reference pack with k=5 and JSON output.
python -m src.eval run evals/packs/opentitan_riscv --k 5 --format json

# Override the judge's samples_per_claim for a quick smoke run.
python -m src.eval run evals/packs/opentitan_riscv --samples-per-claim 1

# Incremental update mode (do NOT drop the collection first).
python -m src.eval run evals/packs/opentitan_riscv --no-fresh

# READ-ONLY: print a markdown trend table from a pack's run-history index.
# No ingest/retrieve/judge, no writes; missing history → friendly message, exit 0.
python -m src.eval show-trend opentitan_riscv --reports-dir eval_reports --runs 3
```

Exit codes: `0` pass, `1` gate fail, `2` pack load/validation error, `3` runtime/infra error.

## Engineering Documentation

- [`docs/eval/EVAL_LOOP_ENGINEERING_GUIDE.md`](../../docs/eval/EVAL_LOOP_ENGINEERING_GUIDE.md): implementation-oriented walkthrough — architecture, stage contracts, pack schema, CLI flags, multi-sample judge semantics, gating + anti-gaming guard, extension steps, troubleshooting.
