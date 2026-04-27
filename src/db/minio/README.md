<!-- @summary
MinIO backend for the document store: a `DocumentBackend` adapter and low-level helpers for document CRUD, presigned URLs, and page-image storage used by the visual embedding pipeline.
@end-summary -->

# src/db/minio

Implements the `DocumentBackend` contract against a MinIO (S3-compatible) object store. Each document is stored as two objects: a `.md` content file and a `.meta.json` sidecar. Page images for the visual retrieval pipeline are stored under the `pages/{document_id}/` key prefix.

## Contents

| Path | Purpose |
| --- | --- |
| `__init__.py` | Re-exports `MinioBackend` and the store-level helpers used by `src/db/__init__.py` |
| `backend.py` | `MinioBackend` — thin `DocumentBackend` subclass that delegates to `store.py` and resolves the default bucket from config |
| `store.py` | Low-level helpers: client creation, bucket management, document CRUD, presigned URL generation, deterministic ID builder, and page-image upload/delete for the visual pipeline |
