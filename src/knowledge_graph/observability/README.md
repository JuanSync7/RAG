<!-- @summary
AST-based observability coverage scorer for the knowledge_graph package.
Checks that module-level loggers and timing instrumentation are present in five target files.
@end-summary -->

# knowledge_graph/observability

This directory contains the observability coverage program for `src/knowledge_graph`. It uses static AST analysis to verify that structured logging and latency timing have been added to key files — without executing any production code.

## Contents

| Path | Purpose |
| --- | --- |
| `score_observability.py` | Scorer: checks 7 AST targets across 5 files and prints `SCORE: N/7 (X%)` |
| `PROGRAM.md` | Research program definition — objectives, mutable files, scoring criteria, stop conditions |
| `research/` | Iteration history — changelog and per-iteration results |
