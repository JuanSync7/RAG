<!-- @summary
Standalone retrieval eval suite. Loads domain-specific golden query sets and
evaluates retrieval quality independently of the knowledge-graph pipeline.
@end-summary -->

# evals/retrieval/

This package contains the standalone retrieval evaluation suite. It is separate from
the KG-focused evals in `evals/knowledge_graph/` and focuses purely on retrieval
quality: given a golden query set, verify that the expected chunks are returned.

The suite is currently a stub — `conftest.py` wires up the fixture loader and skips
tests gracefully when golden query files have not yet been populated.

## Contents

| Path | Purpose |
| --- | --- |
| `conftest.py` | Pytest fixtures; loads domain golden query sets from `fixtures/` (skips if unpopulated) |
| `fixtures/` | Placeholder directory for per-domain golden query JSON files (see its own README) |
