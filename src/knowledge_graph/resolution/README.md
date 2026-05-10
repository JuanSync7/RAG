<!-- @summary
Phase 3 entity resolution: embedding-based dedup and YAML alias-table merges.
@end-summary -->

# resolution/ — Entity Resolution (Phase 3)

Deduplicates entities in the knowledge graph using a two-phase approach.

## Files

| File | Purpose |
|------|---------|
| `schemas.py` | `MergeCandidate` and `ResolutionReport` dataclasses |
| `alias_resolver.py` | YAML alias-table-based deterministic merges |
| `embedding_resolver.py` | Embedding cosine similarity-based fuzzy merges |
| `resolver.py` | `EntityResolver` orchestrator — runs alias first, then embedding |

## How It Works

1. **Alias merges** (deterministic): reads `config/kg_aliases.yaml`, matches entity names case-insensitively
2. **Embedding merges** (fuzzy): computes BGE-M3 embeddings for entity names, finds pairs above threshold
3. Both phases are type-constrained — only entities of the same type are compared

Runs as a post-ingestion step, after extraction and before community detection.
