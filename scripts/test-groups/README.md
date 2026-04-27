<!-- @summary
Named test-group configuration files consumed by the test-runner skill. Each `.conf` file defines a pytest scope (paths, markers, timeouts, validation overrides) for a specific subsystem, enabling targeted test runs without manually specifying pytest arguments.
@end-summary -->

# scripts/test-groups

Configuration files for named test groups. Each file is a key=value manifest read by the test-runner skill to invoke pytest with the correct scope, timeouts, and sandbox overrides for a particular subsystem.

## Contents

| Path | Purpose |
| --- | --- |
| `all.conf` | Full test suite (`tests/`). |
| `guardrails.conf` | Guardrails tests (`tests/guardrails/`). NeMo-dependent safety checks; 180 s run timeout. |
| `import-check.conf` | Import checker tests (`tests/import_check/`). Requires `subprocess:allow` override for `sys.modules` surgery. |
| `ingest.conf` | Ingestion pipeline tests (`tests/ingest/`). Document processing, embedding, and orchestration; 300 s run timeout. |
| `observability.conf` | Observability tests (`tests/observability/`). Langfuse backend stubs. |
| `retrieval.conf` | Retrieval pipeline tests (`tests/retrieval/`). Query processing, generation, and routing; 120 s run timeout. |
| `root.conf` | Root-level tests (`tests/test_*.py`). Mixed: API, cache, security, RAG chain. |
| `server.conf` | Server tests (`tests/server/`). Schema contracts and API routes. |
