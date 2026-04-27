<!-- @summary
Data lifecycle management for the ingest subsystem: four-store orphan detection,
garbage collection, schema migration, cross-store validation, and report
formatting. All engines are read-only by default; mutations require explicit
confirmation.
@end-summary -->

# lifecycle

Engines and typed contracts for managing the long-term health of the four
ingest stores (Weaviate, MinIO, Neo4j, manifest). All operations are
idempotent and isolated per store failure by default.

## Contents

| Path | Purpose |
| --- | --- |
| `__init__.py` | Stable public API — re-exports all engines, dataclasses, and factory helpers |
| `schemas.py` | Typed contracts: `OrphanReport`, `GCReport`, `StoreInventory`, `LifecycleConfig`, `MigrationPlan`, `ValidationReport`, and related dataclasses |
| `sync.py` | `SyncEngine` — read-only four-store inventory and orphan detection |
| `gc.py` | `GCEngine` — soft and hard garbage collection across all four stores |
| `migration.py` | `MigrationEngine` — per-`trace_id` schema migration with dry-run and execute modes |
| `changelog.py` | Schema changelog loader and minimum-cost migration strategy resolver |
| `validation.py` | `E2EValidator` — cross-store consistency checks by `trace_id`; exposes a CLI entry point |
| `orphan_report.py` | Pure formatting helpers (`format_text`, `format_json`) for `OrphanReport` and `GCReport` |
