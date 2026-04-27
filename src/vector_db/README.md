<!-- @summary
Public API and backend abstraction layer for vector store operations. Exposes a
config-driven facade for client lifecycle, collection management, document CRUD,
hybrid search, multi-collection fan-out, aggregation, and visual page operations.
@end-summary -->

# vector_db

The `vector_db` package is the single import surface for all vector store
operations in RagWeave. Pipeline code imports only from this package; the
concrete backend is selected at runtime via the `VECTOR_DB_BACKEND` config key,
making it straightforward to swap implementations without touching call sites.

Multi-collection search is available via `multi_search()`, which fans out across
named collections in parallel, deduplicates results by object identity, and
returns a single ranked list. Visual page operations (used by the visual
embedding pipeline) are also re-exported here.

## Contents

| Path | Purpose |
| --- | --- |
| `__init__.py` | Public facade — all exports, backend dispatch, `multi_search` fan-out |
| `backend.py` | `VectorBackend` abstract base class defining the swappable backend contract |
| `common/` | Backend-agnostic data contracts (`DocumentRecord`, `SearchResult`, `SearchFilter`) |
| `weaviate/` | Weaviate implementation of `VectorBackend` |
