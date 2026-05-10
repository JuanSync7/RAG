<!-- @summary
Concrete graph storage backend implementations behind the GraphStorageBackend ABC.
@end-summary -->

# backends/ — Graph Storage Implementations

Each backend implements the `GraphStorageBackend` ABC defined in `backend.py` (parent directory).

## Files

| File | Phase | Purpose |
|------|-------|---------|
| `networkx_backend.py` | 1 | NetworkX DiGraph implementation — default, file-based |
| `neo4j_backend.py` | 2 | Neo4j sync-driver implementation — server-based |

## ABC Contract

All backends must implement 11 abstract methods:

- **Write**: `add_node()`, `add_edge()`, `upsert_entities()`, `upsert_triples()`, `upsert_descriptions()`
- **Read**: `query_neighbors()`, `get_entity()`, `get_predecessors()`
- **Persistence**: `save()`, `load()`
- **Diagnostics**: `stats()`

Plus 4 concrete methods with default implementations (override for efficiency):
`get_all_entities()`, `get_all_node_names_and_aliases()`, `get_outgoing_edges()`, `get_incoming_edges()`

## Backend Selection

Controlled by `RAG_KG_BACKEND` env var → `KGConfig.backend` field → `get_graph_backend()` dispatcher.
