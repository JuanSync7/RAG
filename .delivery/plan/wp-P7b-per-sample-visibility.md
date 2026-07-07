---
slice_id: P7b
title: Per-sample visibility for multi-sample judge
validable_outcome: |
  Operators can opt into per-sample judge outputs in CLI JSON output to debug
  judge variance. QueryFaithfulnessResult carries a `samples` tuple of
  JudgmentScore (always populated when score_goldens runs; default () for
  manual construction). CLI exposes `--show-samples` flag (JSON-only) that
  injects `per_query_samples: {qid: [{score, reasoning}, ...]}` into the
  payload. Backwards compatible — all P5/P6/P6a/P6b/P6c tests stay green.
touches:
  - src/eval/runner/faithfulness.py (add samples field; populate from sampling loop)
  - src/eval/cli.py (add --show-samples; thread show_samples through run_cli; extend _emit_json)
  - tests/eval/test_per_sample_visibility.py (new, 8 fast tests)
depends_on:
  - P7a  # base branch only — no code dep
  - P6c  # multi-sample judge sampling loop
  - P6b  # CLI runner
  - P5   # QueryFaithfulnessResult / score_goldens
---

# P7b — Per-sample visibility

## Motivation

P6c averages N judge samples into a single mean score per golden, retaining
only the best-scored reasoning string. When variance is high (e.g., judge
flipping 0.2/0.8/0.5 on the same chunk), operators have no way to inspect the
distribution without re-running with logging. This slice surfaces the raw
per-sample `JudgmentScore` array via opt-in CLI JSON output.

## End state

1. `QueryFaithfulnessResult.samples: tuple[JudgmentScore, ...] = ()` — final
   field, default empty tuple (backwards-compatible for manual construction).
2. `score_goldens` always passes `samples=tuple(samples)` regardless of N.
3. CLI: `--show-samples` boolean flag on the `run` subcommand (default False).
   When set AND `--format json`, payload includes `per_query_samples`.
   Silently ignored under `--format text`.
4. `_emit_json` gains a `faithfulness_results` kwarg + `show_samples` kwarg.
   `run_cli` threads `show_samples` from argparse / kwarg.

## Tests (8 fast)

1. `test_score_goldens_populates_samples_field_with_N_judgments` — Mutation A target.
2. `test_score_goldens_samples_is_one_tuple_for_default_N`.
3. `test_query_faithfulness_result_samples_defaults_to_empty_tuple`.
4. `test_query_faithfulness_result_samples_frozen` — discriminating partner.
5. `test_cli_show_samples_off_by_default_in_json`.
6. `test_cli_show_samples_flag_includes_per_query_samples_in_json` — Mutation B target.
7. `test_cli_show_samples_flag_silently_ignored_in_text_format`.
8. `test_cli_argv_parses_show_samples_flag`.

## Mutation probes

- **Mutation A** — collapse `samples=tuple(samples)` → `samples=()` in score_goldens.
  Reds test 1.
- **Mutation B** — `_emit_json` ignores `show_samples` and never injects
  `per_query_samples`. Reds test 6.
