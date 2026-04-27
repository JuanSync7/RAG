<!-- @summary
Config-driven document store subsystem with a stable public API, a swappable backend abstraction, and a MinIO implementation. All pipeline code that stores or retrieves documents imports exclusively from this package.
@end-summary -->

# src/db

Provides a single import surface for document persistence. The active backend is selected at runtime via the `DATABASE_BACKEND` config key; only `"minio"` is currently supported. All document operations (create, read, delete, exist-check, presigned URL, list) are exposed as module-level functions that delegate to the configured backend singleton.

## Contents

| Path | Purpose |
| --- | --- |
| `__init__.py` | Public API: client lifecycle helpers, bucket management, and all document CRUD functions; re-exports `StoredDocument` and `build_document_id` |
| `backend.py` | `DocumentBackend` ABC defining the contract every backend must implement |
| `common/` | Shared data contracts (`StoredDocument` dataclass) used across backends |
| `minio/` | MinIO backend implementation: `MinioBackend` adapter and low-level store helpers |
