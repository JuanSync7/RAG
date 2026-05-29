---
slice_id: P6b
title: Eval CLI runner — python -m src.eval run <pack-path>
validable_outcome: >
  `python -m src.eval run <pack-path>` chains load_pack → plan_pack_ingest →
  execute_plan → retrieve_for_goldens → score_goldens → build_eval_report →
  validate_eval_report and exits 0 (gate pass) / 1 (gate fail) / 2 (pack
  load/validation error) / 3 (runtime/infra error). Both --format json and
  --format text emit deterministic output; --k propagates through retrieval
  and the gate's recall_at_<k> key.
touches:
  - src/eval/cli.py
  - src/eval/__main__.py
  - tests/eval/test_cli.py
depends_on:
  - P6a   # threshold gating (validate_eval_report)
  - P5    # faithfulness scoring (score_goldens)
  - P4    # retrieval + recall metrics
  - P3    # execute_plan
  - P3.0  # plan_pack_ingest
  - P0.5  # eval_pack format + loader
---

# P6b — Eval CLI runner

## Scope

A thin orchestrator over the existing runner surface. No new business logic;
the CLI's job is to wire the chain together and translate outcomes into exit
codes + stdout/stderr output.

## Architecture

- `src/eval/__main__.py`: `from src.eval.cli import main; sys.exit(main())`.
- `src/eval/cli.py`:
  - `_build_parser()` → argparse with subcommand `run`, positional `pack_path`,
    optional `--k INT` (default 5), `--format {json,text}` (default `text`),
    `--verbose`, `--fresh/--no-fresh` (default fresh=True — matches
    `execute_plan`'s default).
  - `run_cli(pack_path, *, k=5, format="text", verbose=False, fresh=True,
    chat_model_factory=None) -> int` — pure orchestrator returning exit code.
  - `main(argv=None) -> int` — argparse dispatch.
  - All heavy imports (execute_plan, retrieve, judge) are LAZY inside `run_cli`
    so the parser stays cheap and the langchain_core stub timing in conftest
    is not perturbed.

## Exit codes

| code | meaning |
| ---- | ------- |
| 0    | gate pass |
| 1    | gate fail (one or more threshold breaches) |
| 2    | PackValidationError on load |
| 3    | any other Exception during ingest/retrieve/score/report |

## Output

- `--format text` (default): per-qtype recall + faithfulness rows, then a
  `PASS`/`FAIL` summary; on FAIL, each `GateFailure` printed to stderr.
- `--format json`: single JSON object on stdout with `collection_name`, `k`,
  `passed`, `failures[]`, `recall_by_qtype`, `faithfulness_by_qtype`,
  `total_queries_scored`, `total_queries_skipped`, `total_queries_judged`.

## Test list (acceptance criteria) — `tests/eval/test_cli.py`

1. `test_cli_pack_not_found_exits_2` — nonexistent path → 2.
2. `test_cli_invalid_pack_exits_2` — malformed pack → 2.
3. `test_cli_gate_pass_returns_zero` — full chain mocked, gate passes → 0.
   *(Mutation A target.)*
4. `test_cli_gate_fail_returns_one` — recall below floor → 1; stderr lists failures.
5. `test_cli_runtime_error_exits_3` — execute_plan raises → 3.
6. `test_cli_k_parameter_propagates` — `--k 3` calls retrieve with k=3
   and gate compares `recall_at_3`. *(Mutation B target.)*
7. `test_cli_json_format_valid` — `format="json"` emits parseable JSON
   with required keys.
8. `test_cli_text_format_human_readable` — text output contains per-qtype
   rows + pass/fail line.
9. `test_cli_main_argv_parsing` — `main(["run", path, "--k", "3",
   "--format", "json"])` dispatches correctly (discriminating partner).
10. `test_cli_chat_model_factory_injected` — fake factory plumbed through
    to build_judge_client.

## Mutation probes

- **A — gate inversion**: in `run_cli`, return `1` on `gate.passed=True`.
  Must red `test_cli_gate_pass_returns_zero`.
- **B — k propagation**: hardcode `k=5` in `run_cli`, ignoring the kwarg.
  Must red `test_cli_k_parameter_propagates`.

## Risks (avoided)

- No litellm/platform.llm imports at module-top.
- `retrieve_for_goldens` is called by name — we monkeypatch its module-level
  bindings on `src.eval.runner.retrieve`, not local re-imports.
- `fresh=True` default matches `execute_plan`.
