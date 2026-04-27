<!-- @summary
Top-level evaluation harness for RagWeave. Contains domain-specific eval suites
(knowledge graph and retrieval) plus the shared conftest with common fixture helpers.
@end-summary -->

# evals/

The `evals/` package is the top-level evaluation harness for RagWeave. It houses
domain-specific eval suites as sub-packages alongside a shared `conftest.py` that
provides common fixture utilities (such as `load_json_fixture`) consumed by all suites.

## Contents

| Path | Purpose |
| --- | --- |
| `conftest.py` | Shared pytest fixtures and helpers used across all eval sub-packages |
| `knowledge_graph/` | Eval suite for knowledge-graph extraction, entity resolution, relationship quality, and KG-based retrieval |
| `retrieval/` | Standalone retrieval eval suite (golden-query harness, separate from KG evals) |
