<!-- @summary
Shared typed contracts and content-hash utilities for the Embedding Pipeline's
cross-document deduplication subsystem.
@end-summary -->

# embedding/common

Provides the deduplication building blocks reused across embedding pipeline nodes.
`types.py` defines the `MergeEvent` contract; `dedup_utils.py` supplies SHA-256
content hashing, text normalisation, Weaviate helper functions for exact-match
lookups, and source-document provenance helpers.

## Contents

| Path | Purpose |
| --- | --- |
| `types.py` | `MergeEvent` TypedDict and `create_merge_event` factory — the canonical dedup event schema |
| `dedup_utils.py` | Content hash infrastructure: text normalisation, SHA-256 hashing, Weaviate lookup helpers, merge revert helper, and fuzzy-fingerprint builder |
| `card_builder.py` | Pure, deterministic `build_document_card` — turns one document's chunks into the §6.4 baseline document-routing card record (title + ordered/deduped section headings + compact `card_text`); no LLM, no I/O, headings capped per §11 |
| `card_backfill.py` | `backfill_cards_from_corpus` — DESIGN §9 rollout step 1: populate `RAGDocumentCards` from chunks ALREADY in `RAGDocuments` (no re-ingest). Enumerates documents (`iter_document_ids`, server-side group-by) or takes explicit ids, fetches each document's chunks cursor-paginated (`fetch_chunks_by_document_id`), builds the baseline card via `build_document_card`, batch-embeds `card_text` with the corpus embedder, and upserts via `ensure_card_collection` + `add_document_cards`. Per-document errors are isolated into a stats dict; idempotent + resumable. Run via `scripts/backfill_document_cards.py` |
| `role_backfill.py` | `backfill_roles_from_corpus` — nav-classify rollout (Slice D): tag `chunk_role` on chunks ALREADY in the corpus (no re-ingest). Enumerates documents (`iter_document_ids`) or takes explicit ids, fetches each document's chunks WITH text (`fetch_chunks_by_document_id(include_text=True)`), classifies the page with the SAME shared LLM classifier ingest uses (`classify_roles_from_config`), and updates each chunk in place (`update_chunk_role`). FAIL-OPEN to `content` on any classifier error; `--dry-run` writes nothing; per-chunk update errors are isolated; idempotent + resumable. Run via `scripts/backfill_chunk_roles.py` (prerequisite: `scripts/migrate_weaviate_table_schema.py` to add the `chunk_role` property) |
| `__init__.py` | Package marker |
