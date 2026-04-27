<!-- @summary
Raw log output from scaling sweep load tests against the query API. Each file captures a multi-worker, multi-concurrency sweep run, recording per-run latency percentiles, error rates, and SLO pass/fail results.
@end-summary -->

# scripts/results

Output directory for scaling sweep load test logs. Files are timestamped at creation so successive runs append rather than overwrite.

## Contents

| Path | Purpose |
| --- | --- |
| `scaling_sweep_20260312_185307.log` | Scaling sweep run at 18:53 on 2026-03-12. Workers 1–N × concurrency levels 20/40/60; all requests failed (connection refused — server not running). |
| `scaling_sweep_20260312_185614.log` | Follow-up sweep run three minutes later at 18:56. Same parameter matrix, connection refused throughout. |
| `scaling_sweep_bounded_20260312_190400.log` | Bounded sweep run at 19:04. Reduced total requests per step; includes a warmup phase; server came up partway through the run. |
