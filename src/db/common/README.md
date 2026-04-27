<!-- @summary
Backend-agnostic data contracts for the db subsystem. Contains the `StoredDocument` dataclass shared by all backend implementations and the public db API.
@end-summary -->

# src/db/common

Holds the shared type definitions that cross the boundary between the public db API and backend implementations. Keeping contracts here prevents circular imports and ensures all backends return the same data shapes.

## Contents

| Path | Purpose |
| --- | --- |
| `__init__.py` | Re-exports `StoredDocument` as the package's public surface |
| `schemas.py` | `StoredDocument` dataclass — `document_id`, `content`, `metadata` fields |
