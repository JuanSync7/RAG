<!-- @summary
Test suite for the knowledge graph retrieval subsystem, covering backend
equivalence, path matching and context formatting, query expander integration,
schema validation, and typed traversal of NetworkXBackend.
@end-summary -->

# tests/knowledge_graph

Tests for `src/knowledge_graph` retrieval components. The suite validates
the full retrieval stack: backend CRUD round-trips and deduplication, path
pattern matching and token-budget-aware context formatting, the
`GraphQueryExpander` integration (typed dispatch, untyped fallback, error
degradation), schema/validation contracts, and the typed-neighbor traversal
API of `NetworkXBackend`.

## Contents

| Path | Purpose |
| --- | --- |
| `test_kg_backend_equivalence.py` | Cross-backend equivalence, insert/query round-trips, relation queries, deletes, deduplication, case-insensitive normalization, clear-all, path pattern matching, max-hop fanout, and invalid pattern edge cases |
| `test_kg_retrieval_expander_integration.py` | `GraphQueryExpander` unit and integration tests: typed dispatch, untyped fallback, error degradation, no-seed early return, and backward-compat iteration |
| `test_kg_retrieval_path_formatter.py` | `PathMatcher` and `GraphContextFormatter` tests: single/multi-hop patterns, cycle guards, deduplication, result structure, marker styles, and token-budget truncation |
| `test_kg_retrieval_schemas_validation_backend.py` | Three-part coverage: `ExpansionResult`/`PathResult` schema contracts, `validate_edge_types`/`validate_path_patterns` validators, and `NetworkXBackend.query_neighbors_typed` typed traversal |
