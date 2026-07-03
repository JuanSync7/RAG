<!-- @summary
Tests for the document-store layer (`src/db/`): the storage backend contract,
the MinIO store implementation, the facade factory, and the clean-document
resolver used by the turn loop's DEEP_STUDY fetch seam.
@end-summary -->

# tests/db/

Tests for `src/db/` — the document store facade and its MinIO backend.

## Files

| File | Purpose |
| --- | --- |
| `test_backend_contract.py` | Storage backend contract invariants shared by all backends |
| `test_minio_store.py` | MinIO-backed document store behavior |
| `test_facade_factory.py` | Store facade construction/factory wiring |
| `test_document_resolver.py` | `resolve_clean_document`: document_id → source_key → source fallback chain across both MinIO layouts, never-raise library semantics |

## Running

```bash
uv run --extra dev python -m pytest tests/db -q
```
