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
| `orchestrator.py` | Nightly batch core: `run_nightly(...)` runs the loop per pack, persists, alerts, and returns a binary pass/fail code. Forwards the three `run_eval` DI seams. |
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
```

Exit codes: `0` pass, `1` gate fail, `2` pack load/validation error, `3` runtime/infra error.

## Engineering Documentation

- [`docs/eval/EVAL_LOOP_ENGINEERING_GUIDE.md`](../../docs/eval/EVAL_LOOP_ENGINEERING_GUIDE.md): implementation-oriented walkthrough — architecture, stage contracts, pack schema, CLI flags, multi-sample judge semantics, gating + anti-gaming guard, extension steps, troubleshooting.
