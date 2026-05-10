<!-- @summary
Shared contracts, configuration types, and utility functions for the KG subsystem.
@end-summary -->

# common/ — Shared Contracts and Configuration

Foundational types and helpers used by all other KG sub-packages.

## Files

| File | Purpose |
|------|---------|
| `schemas.py` | `Entity`, `Triple`, `ExtractionResult`, `EntityDescription` — core data contracts |
| `types.py` | `KGConfig` (runtime config), `SchemaDefinition`, `NodeTypeDefinition`, `EdgeTypeDefinition`, `load_schema()` |
| `utils.py` | `normalize_alias()`, `validate_type()`, `derive_gliner_labels()`, `is_phase_active()` |
| `shared.py` | Cross-node shared heuristics and helpers |
| `description_manager.py` | Token-budgeted entity description accumulation |

## Key Types

- **`KGConfig`**: Dataclass with all runtime settings (backend selection, extractor toggles, community params, Neo4j credentials). Constructed by `_build_kg_config()` in `__init__.py` from `RAG_KG_*` env vars.
- **`Entity`**: Canonical entity with name, type, sources, aliases, mentions, and description.
- **`Triple`**: Directed edge (subject → predicate → object) with source and weight.
- **`SchemaDefinition`**: Parsed `config/kg_schema.yaml` with phase-aware query methods.
