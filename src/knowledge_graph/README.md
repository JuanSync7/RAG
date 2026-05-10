<!-- @summary
Modular knowledge graph subsystem with ABC backend pattern, multi-extractor pipeline,
community detection, and configurable query expansion.
@end-summary -->

# Knowledge Graph Subsystem

Replaces the monolithic `src/core/knowledge_graph.py` with a modular package.
Uses the ABC backend pattern (like `src/guardrails/`) for swappable graph storage
behind a stable public API with lazy singleton dispatch.

## Public API

All callers import from `src.knowledge_graph`:

```python
from src.knowledge_graph import (
    get_graph_backend,    # Lazy singleton — returns configured backend
    get_query_expander,   # Query expansion with optional community awareness
    export_obsidian,      # Obsidian vault export
    KGConfig,             # Runtime configuration dataclass
)
```

Backend selection is controlled by `RAG_KG_BACKEND` env var (`networkx` or `neo4j`).

## Directory Layout

| Directory | Purpose |
|-----------|---------|
| `common/` | Shared schemas, config types, utilities, description manager |
| `extraction/` | Entity/relationship extractors (regex, GLiNER, LLM, SV parser, Python AST, Bash regex) |
| `backends/` | Graph storage implementations (NetworkX, Neo4j) |
| `query/` | Entity matching, query expansion, query sanitization |
| `community/` | Leiden community detection, LLM summarization (Phase 2) |
| `export/` | Graph export formats (Obsidian markdown vault) |

## Key Files

| File | Purpose |
|------|---------|
| `__init__.py` | Public API facade — `get_graph_backend()`, `get_query_expander()` |
| `backend.py` | `GraphStorageBackend` ABC — 11 abstract + 4 concrete methods |

## Configuration

Runtime config is driven by `KGConfig` dataclass (`common/types.py`), populated from
`RAG_KG_*` environment variables defined in `config/settings.py`.

Schema-driven node/edge types are defined in `config/kg_schema.yaml` with phase tags
(`phase_1`, `phase_1b`, `phase_2`).

## Phase Model

- **Phase 1**: NetworkX backend, regex + GLiNER extractors, basic query expansion
- **Phase 1b**: LLM structured-output extractor, tree-sitter SV parser, entity descriptions
- **Phase 2**: Leiden community detection, LLM summarization, Neo4j backend, global retrieval, Python/Bash parsers

See `docs/knowledge_graph/KNOWLEDGE_GRAPH_ENGINEERING_GUIDE.md` for the full engineering guide.
