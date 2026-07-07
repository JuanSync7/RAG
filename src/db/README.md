<!-- @summary
Config-driven document store subsystem with a stable public API, a swappable backend abstraction, a MinIO implementation, and a shared clean-document resolver (id fallback chain across both MinIO layouts). All pipeline code that stores or retrieves documents imports exclusively from this package.
@end-summary -->

# src/db

Provides a single import surface for document persistence. The active backend is selected at runtime via the `DATABASE_BACKEND` config key; only `"minio"` is currently supported. All document operations (create, read, delete, exist-check, presigned URL, list) are exposed as module-level functions that delegate to the configured backend singleton.

## Contents

| Path | Purpose |
| --- | --- |
| `__init__.py` | Public API: client lifecycle helpers, bucket management, all document CRUD functions, and `resolve_clean_document`; re-exports `StoredDocument` and `build_document_id` |
| `backend.py` | `DocumentBackend` ABC defining the contract every backend must implement |
| `common/` | Shared data contracts (`StoredDocument` dataclass) used across backends |
| `minio/` | MinIO backend implementation: `MinioBackend` adapter and low-level store helpers |

## Clean-document resolution

`resolve_clean_document(client, document_id=None, source_key=None, source=None, bucket=None)`
is the shared resolver used by the console document viewer
(`server/console/services.py`) and the turn loop's DEEP_STUDY fetch
(`docs/retrieval/TURN_LOOP_DESIGN.md` §5). It walks the identity fallback
chain `document_id` → `build_document_id(source_key)` →
`build_document_id(source)` against the document store layout
(`<document_id>.md`, written by normal ingest), then falls back to the
lifecycle-populated `MinioCleanStore` layout (`clean/{safe_key}.md`) keyed by
`source_key` / `source`. Library semantics: returns a `StoredDocument` (or
`None` on any miss or storage error) and never raises — HTTP mapping belongs
to the callers.
