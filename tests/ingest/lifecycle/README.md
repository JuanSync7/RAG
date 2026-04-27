<!-- @summary
Unit tests for the ingest lifecycle subsystem: garbage collection, schema
migration, cross-store synchronisation, end-to-end validation, orphan
reporting, and the migrate/validate CLI entry points.
@end-summary -->

# tests/ingest/lifecycle

Unit tests for the `src/ingest/lifecycle` subsystem. All external stores
(Weaviate, MinIO, Neo4j) are replaced with mocks, making these pure-unit
tests with no I/O dependencies.

## Contents

| Path | Purpose |
| --- | --- |
| `conftest.py` | Shared fixtures for lifecycle tests |
| `test_changelog.py` | Schema changelog parser — `load_changelog`, `get_required_migrations`, `determine_migration_strategy` |
| `test_gc_engine.py` | `GCEngine` — soft vs hard delete paths, retention window, hard-delete confirmation guard |
| `test_lifecycle_cli.py` | `run_migration_cli` and `run_validation_cli` — dry-run, from/to flags, confirm gating, JSON/text output, argparse validation |
| `test_migration.py` | `MigrationEngine` — plan/execute happy path, per-entry failure isolation, idempotency, confirm-guard |
| `test_orphan_report.py` | `orphan_report` formatters — text and JSON formatting roundtrip, machine-parseable JSON output |
| `test_sync_engine.py` | `SyncEngine` — store inventory collection and cross-store diff logic |
| `test_validation.py` | `E2EValidator` — per-trace and bulk validation, store consistency, disabled-store handling, sample-size support |
