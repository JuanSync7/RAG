<!-- @summary
Backend-agnostic data contracts for the vector_db subsystem. Defines the shared
typed dataclasses used at the boundaries of every VectorBackend implementation.
@end-summary -->

# vector_db/common

This directory holds the shared, backend-independent data contracts for the
`vector_db` subsystem. No store-specific imports belong here. All
`VectorBackend` implementations accept and return these types at their public
boundaries, keeping the abstraction layer clean.

## Contents

| Path | Purpose |
| --- | --- |
| `schemas.py` | `DocumentRecord`, `SearchResult`, and `SearchFilter` dataclasses |
| `__init__.py` | Re-exports the three dataclasses for convenient import from `src.vector_db.common` |
