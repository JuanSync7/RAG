---
slice_id: P6c
validable_outcome: |
  When pack.meta.judge.samples_per_claim > 1 (or CLI --samples-per-claim N
  overrides), the eval loop invokes the judge N times per golden and stores
  the mean score; reasoning comes from the highest-scored sample. Default
  samples_per_claim=1 preserves exact P5 behavior. All prior tests pass.
touches:
  - src/eval/runner/faithfulness.py
  - src/eval/runner/judge.py
  - src/eval/cli.py
  - src/eval/runner/__init__.py
  - src/eval/__init__.py
  - tests/eval/test_multi_sample_judge.py
depends_on: [P6b, P5.0, P5, P0.5]
---

# P6c — Multi-sample judge (averaged faithfulness scoring)

## Surface changes

1. `score_goldens(..., *, samples_per_claim: int = 1) -> FaithfulnessResults`
   - For each judged golden, call `judge_client.score(question)` N times.
   - `result.score = statistics.mean(samples)`.
   - `result.reasoning = max(samples, key=lambda s: s.score).reasoning`
     (first-wins tie-break — deterministic).
   - `samples_per_claim < 1` raises ValueError.

2. `read_samples_per_claim(pack: EvalPack) -> int` (new in `judge.py`).
   - Returns `pack.meta.judge.samples_per_claim` or 1 if missing.
   - Raises ValueError on < 1.

3. `cli.run_cli(..., samples_per_claim: int | None = None)`:
   - If None, derive from `read_samples_per_claim(pack)`.
   - Otherwise use the override.
   - Threads through to `score_goldens(...)`.

4. argparse `run` subcommand: `--samples-per-claim N` (default None).

5. Additive exports: `read_samples_per_claim` from
   `src/eval/runner/__init__.py` and `src/eval/__init__.py`.

## Non-goals

- No change to `JudgeClient.score` (still single-shot).
- No change to `build_judge_client` signature.
- No change to ingest/retrieval/gate/report contracts.
