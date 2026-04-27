<!-- @summary
Auto-research artifacts for the retrieval query pipeline latency optimization loop. Contains the research program, benchmark fixtures, iteration log, and a final changelog from a completed auto-research run that achieved an 87% p95 latency reduction (2266 ms → 294 ms).
@end-summary -->

# research/retrieval

This directory holds the durable artifacts from the auto-research loop that optimized retrieval-only p95 latency in the `RAGChain.ask(...)` pipeline (stages 1–5: query processing through reranking). The loop ran 5 iterations and met its success condition at iter 005, achieving a settled p95 of 294 ms against a target of ≤ 1813 ms.

## Contents

| Path | Purpose |
| --- | --- |
| `PROGRAM.md` | Research brief: objective, scoring mode, keep/discard rules, mutable vs. immutable file lists, exploration strategies, and stop conditions. Authoritative spec for the loop. |
| `iterations.tsv` | One row per iteration: commit, settled p95, p50, drift count, lint failures, keep/discard status, and reasoning. The durable record of every loop decision. |
| `benchmark_queries.json` | Fixed query set used by the scoring harness — 15 KG queries (regression-guarded) and 5 cold queries (latency-only). Immutable. |
| `baseline_outputs.json` | Per-query `(action, top-K doc SHA1s)` snapshot used as the regression guard oracle. Unchanged throughout the run. |
| `changelog.md` | Post-loop summary: headline results, per-stage evolution, per-iteration narrative, what worked/didn't, unexplored levers, and operational notes for the final state. |
