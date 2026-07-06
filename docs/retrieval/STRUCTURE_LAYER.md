<!-- @summary
Design backlog from the _v2 validation discussion: the "structure/navigation layer"
(nav chunks + cards + tree nodes as one family), unifying the doc summary into the
card, one canonical ingest config, and an image/visual backfill. Captures decisions
and empirical findings; not yet implemented.
@end-summary -->

# Structure / Navigation Layer — design backlog

Captured 2026-07-02 during `Sanity_testcase_v2` validation. These are **design
directions**, not yet built. Empirical findings that motivate them are noted inline.

## 1. Navigation is a *layer*, not junk

Nav/boilerplate chunks were being treated as noise to suppress. Better model: there
is a **content layer** (answer-bearing chunks) and a **structure/navigation layer**
that helps you *find* content but isn't the answer — and it has three members that
are the same kind of thing:

- **document cards** — which *document* (routing)
- **tree section nodes** — which *section* (hierarchical descent)
- **navigation chunks** — explicit ToC / index / cross-references

Queries split by **intent**: "what *are* the coherency features?" → content; "*which
chapter* covers coherency / where is X / list the appendix" → navigation.

**Current state:** the query-side role filter (`RAG_RETRIEVAL_ROLE_SCHEMA_PRESENT=true`)
is a *blanket* exclusion of `navigation`+`boilerplate`. Proven net-win (BM25 top-20:
"proprietary notice" 11 boilerplate→0; "contents chapter appendix" 9 nav→0; content
backfills the freed slots).

**Bounded casualty:** a *pure* structure query ("give me the table of contents",
"what's in the appendix") now has its answer hidden. But "which chapter covers X"
still works — answered from **content chunks' `heading_path` breadcrumb**, not the ToC
(validated: "Which chapter covers cache coherency?" → correctly "Chapter B1 / B1.5").

**Future:** detect navigation-intent and route those queries to a navigation index
(the nav chunks) instead of hard-excluding; keep `boilerplate` hard-excluded (no
navigation value). This is an *enhancement*, not an urgent fix — the filter is safe on.

## 2. The doc summary belongs in the card (stop calling it "metadata")

`metadata_generation` conflates two different things:
- **Metadata** = deterministic, factual attributes (`document_id`/uuid5, `content_hash`,
  `source_key`, `chunk_index`, `heading_path`, `page_no`, `table_*`). Always correct,
  cheap, and what backfill/dedup/provenance rely on. Keep as "metadata".
- **Summary + keywords** = a *generated*, lossy LLM product. Misnamed as "metadata".

Worse, that summary is **orphaned**: it's built from only the first ~10k chars (so for
a 716-page spec it summarizes the front matter, not the substance), and it's **not in
the `_v2` chunk schema** (no `document_summary`/`document_keywords` property) so it's
dropped on write. Meanwhile the **document card has a `summary` field that's hardcoded
`""`** (the `CARD_LLM_SUMMARY` path is unwired). Two half-built doc-summary mechanisms
that don't connect.

**Future:** rename the deterministic half to "metadata"; make the LLM doc-summary a
separate "summary" concern that **populates the card's summary field** (one doc-level
representation: deterministic headings + LLM summary, embedded once, actually used by
routing). Consider map-reduce over the whole doc instead of the first 10k chars. Until
then, `metadata_generation` is an LLM call/doc that contributes ~nothing to retrieval —
decide deliberately (fix or disable).

## 3. One canonical ingest config (no fast/slow drift)

There is **one** ingest code path; "fast/slow" is just which flags are flipped. The
sensible canonical config = the **code defaults**: nav-classify ON, TableFormer
ACCURATE (v2), metadata ON — with the heavy stages OFF *unless their retrieval consumer
is on* (cards only with routing; tree nodes only with tree retrieval; VLM only for
diagram queries; ColQwen visual only for image-native queries). `_v2` deviated
(TableFormer FAST, nav backfilled instead of inline) for iteration speed — acceptable
for the testbed, but prod should standardize on the defaults so dev==prod.

Cost reality (measured): the only per-doc LLM costs are metadata (1), nav-classify
(`ceil(chunks/40)`), and VLM (≤4). Tree nodes and cards are **not** LLM-based
(deterministic/embedding-only). "Contextual chunking = 1 LLM/chunk" does **not** exist
here (it's Docling's deterministic breadcrumb).

## 4. Image / visual backfill (roadmap)

Build `image_backfill` next to `role_backfill`/`card_backfill`: for an existing
corpus, extract each figure/page image → store JPEG in MinIO → VLM-summarize (caption +
verbatim OCR + tags) → write that text + `figure_image_uri` (the `_v2` schema already
has `figure_image_uri`/`caption_label`/`page_no`/`page_bbox`/`tags`) onto the chunk →
optionally add a ColQwen page-vector for image-native retrieval. **Caveat:** unlike the
role backfill (pure Weaviate-text), this needs the **source images** — either re-parse
with Docling or a persisted DoclingDocument — so it re-touches the source, not just the
index. Visual embedding is expected to become default-on later.
