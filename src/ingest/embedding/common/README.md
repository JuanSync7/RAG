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
| `__init__.py` | Package marker |
