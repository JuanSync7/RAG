<!-- @summary
Operator guide for backfilling the table-aware + page-provenance properties
(PR #100) onto Weaviate collections created before that change.
@end-summary -->

# Weaviate Schema Migration — table-aware + page-provenance fields

## Why this exists

PR #100 added 12 properties to the Weaviate collection schema declared in
`src/vector_db/weaviate/store.py::ensure_collection`:

```
chunk_type, table_id, table_group_id, table_row_index,
table_num_rows, table_num_cols, table_has_header,
table_caption, table_markdown,
page_no, page_label, page_bbox
```

Collections **created before PR #100** lack these properties. The store
reads work (missing properties return `None`), but `add_documents` filters
writes against the live schema, so any table or page-provenance metadata on
new chunks is **silently dropped**. See memory note
`project_weaviate_schema_drop.md` for the underlying behaviour.

## What the migration does

The migration script — `scripts/migrate_weaviate_table_schema.py` — diffs
the live collection schema against
`src.vector_db.weaviate.store.TABLE_AWARE_PROPERTIES` and adds any missing
property via the Weaviate v4 `collections.config.add_property` API. It is:

- **Non-destructive.** Existing data is not touched. The collection itself
  is not recreated. Properties that already exist (even with the "wrong"
  flags) are left alone.
- **Idempotent.** Re-running reports `[OK] all 12 properties present` and
  does nothing.
- **Schema-pinned.** The property list is the same constant
  `ensure_collection` consumes — the two cannot drift.

## When to run it

- **Once, on every existing collection** after deploying PR #100 (or any
  later change that pre-dates table-aware chunking).
- After any environment where Weaviate persistence has been wiped and you
  are not certain the freshly-created collection went through the current
  `ensure_collection`.
- As part of CI smoke-testing for an environment whose collection was
  bootstrapped by an older RagWeave build.

## Usage

```bash
# Dry-run a single collection (no writes; prints the [MISSING] diff)
uv run python scripts/migrate_weaviate_table_schema.py \
    --collection RagWeave --dry-run

# Apply against a single collection
uv run python scripts/migrate_weaviate_table_schema.py --collection RagWeave

# Apply against every collection visible to the client
uv run python scripts/migrate_weaviate_table_schema.py --all-collections
```

The script uses the existing client builder
(`src.vector_db.weaviate.store.get_weaviate_client`) so it honours
`RAG_WEAVIATE_MODE`, `RAG_WEAVIATE_HOST`, `RAG_WEAVIATE_HTTP_PORT`, and
`RAG_WEAVIATE_GRPC_PORT` exactly like the application — no separate
connection config.

## Sample output

Dry-run against a collection missing 11 of the 12 properties:

```
=== RagWeave ===
[MISSING] table_id (TEXT, filterable)
[MISSING] table_group_id (TEXT, filterable)
[MISSING] table_row_index (INT, filterable)
[MISSING] table_num_rows (INT, default-index)
[MISSING] table_num_cols (INT, default-index)
[MISSING] table_has_header (BOOL, default-index)
[MISSING] table_caption (TEXT, searchable)
[MISSING] table_markdown (TEXT, searchable)
[MISSING] page_no (INT, filterable)
[MISSING] page_label (TEXT, filterable)
[MISSING] page_bbox (TEXT, default-index)
[SUMMARY] 11 missing / 12 total
```

Apply mode against the same collection:

```
=== RagWeave ===
[ADDED] table_id
[ADDED] table_group_id
...
[SUMMARY] added=11 already_present=1 errors=0
```

Second run on the same collection (idempotency check):

```
=== RagWeave ===
[OK] all 12 properties present
```

## Exit codes

- `0` — every targeted collection is now fully migrated (or was already).
- `1` — at least one `add_property` call failed; partial migration applied.
- `2` — a requested collection does not exist on the server.

## Tests

Coverage lives in `tests/vector_db/weaviate/test_schema_migration.py` and
covers the diff/apply/idempotency/error-surface behaviour against a
`MagicMock`-backed Weaviate stub. Run with:

```bash
uv run pytest tests/vector_db/weaviate/ -x
```
