<!-- @summary
Weaviate implementation of the VectorBackend contract. Provides the backend
class, low-level embedded-client helpers for text collections, and a separate
store for the visual page collection used by the visual embedding pipeline.
@end-summary -->

# vector_db/weaviate

This directory contains the Weaviate-specific implementation of the
`VectorBackend` abstract base class. All Weaviate-specific logic is isolated
here; the rest of the codebase interacts only through `src.vector_db`.

`backend.py` is a thin delegation layer. The actual Weaviate operations live in
`store.py` (text collections) and `visual_store.py` (visual page collection).
`SearchFilter` objects are translated to native Weaviate filter objects inside
`WeaviateBackend._translate_filters`.

## Contents

| Path | Purpose |
| --- | --- |
| `backend.py` | `WeaviateBackend` — `VectorBackend` implementation, delegates to `store` and `visual_store` |
| `store.py` | Low-level helpers: embedded client connection, collection schema, CRUD, hybrid search, aggregation |
| `visual_store.py` | Visual page collection (`RAGVisualPages`): schema, batch insert, near-vector search, deletion |
| `__init__.py` | Re-exports `WeaviateBackend`, `build_chunk_id`, and visual store helpers |
