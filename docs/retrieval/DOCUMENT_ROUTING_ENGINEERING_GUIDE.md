# RAPTOR-lite Document Routing — Engineering Guide

> **Audience:** engineers operating, tuning, or extending document routing.
> **Status:** implemented on branch `feat/raptor-lite-doc-routing`, **gated OFF by
> default** (no behaviour change unless explicitly enabled).
> **Companion docs:** [`DOCUMENT_ROUTING_DESIGN.md`](DOCUMENT_ROUTING_DESIGN.md)
> (rationale / design intent) and
> [`DOCUMENT_ROUTING_IMPLEMENTATION_PLAN.md`](DOCUMENT_ROUTING_IMPLEMENTATION_PLAN.md)
> (the slice plan this was built from). Read the design first for *why*; this
> guide is *what was built and how to run it*.

This guide documents the as-built RAPTOR-lite document-routing feature: a
two-stage hierarchical retrieval layer that first decides **which documents to
look in** (Stage-1 "route") and then does the usual **chunk retrieval blended
with the flat path** (Stage-2 "retrieve"). Routing is **soft** — it only widens
the candidate pool, never hard-filters — and document "cards"/summaries are used
**only for routing and are NEVER sent to the LLM**.

---

## 1. Architecture & module layout

The feature adds a thin routing package on the query side, a card-emission path
on the ingest side, a dedicated card collection in the vector store, and one new
filter operator. Nothing on the existing flat/tree retrieval path was removed;
routing is layered *in front of* it.

```
                          ┌──────────────────────────────────────────────┐
   INGEST (per document)  │  build_document_card → embed card_text →       │
   when cards enabled     │  stage staged_card_records → atomic commit     │
                          │  into the card collection (RAGDocumentCards)   │
                          └──────────────────────────────────────────────┘
                                              │  (one routing-only card per doc)
                                              ▼
                          ┌──────────────────────────────────────────────┐
   QUERY                  │  Stage 3.5 ROUTE:                              │
   when routing enabled   │    decompose_query → per sub-query             │
                          │    route_documents over the card index →       │
                          │    union routed document_ids (soft hint)        │
                          └──────────────────────────────────────────────┘
                                              │  routed_doc_ids (or None)
                                              ▼
                          ┌──────────────────────────────────────────────┐
   QUERY                  │  Stage 4 RETRIEVE: _collect_candidates          │
   (always)               │    flat → descent → lift → [routed `in` search] │
                          │    union + dedup, bound to MAX_CANDIDATES       │
                          │  → ONE rerank pass → top-K → LLM                │
                          └──────────────────────────────────────────────┘
```

### 1.1 New / changed modules

**Query side — `src/retrieval/routing/` (new package):**

| File | Responsibility | Key exports |
| --- | --- | --- |
| `__init__.py` | Stable public facade (thin; imports nothing heavy) | `route_documents`, `decompose_query`, `DecompositionError`, `PROTOCOL_GLOSSARY`, `detect_comparison_intent`, `canonicalize_terms`, `RoutingResult`, `DecompositionResult`, `DocCard` |
| `schemas.py` | Typed cross-slice contracts (pure dataclasses, no I/O) | `DocCard`, `RoutingResult`, `DecompositionResult` |
| `glossary.py` | `PROTOCOL_GLOSSARY` (canonical AMBA entities + informal-alias maps) + lexical comparison-intent detector + term canonicalizer | `PROTOCOL_GLOSSARY`, `detect_comparison_intent`, `canonicalize_terms` |
| `decomposition.py` | Tier-1 regex (`regex_decompose`), Tier-2 glossary-LLM (`llm_decompose`), and the gated orchestrator (`decompose_query`) | `regex_decompose`, `llm_decompose`, `decompose_query`, `DecompositionError` |
| `router.py` | Stage-1 card-index router: query embedding → routed `document_id`s | `route_documents` |

**Query side — `src/retrieval/pipeline/rag_chain.py` (extended):**

- `RAGChain._route_documents_stage1(...)` — the Stage-3.5 orchestration helper:
  decompose → route each sub-query → union the routed `document_id`s. Wrapped in
  a broad `try/except` so it can never break `run()` (any failure → `None`).
- `RAGChain._embed_query_cached(...)` — embeds a sub-query via the same LRU cache
  used at Stage-3, so the identity sub-query reuses the already-computed vector
  and decomposition sub-queries are embedded at most once.
- `RAGChain._collect_candidates(..., routed_doc_ids=...)` — the candidate union
  gains an optional 4th source: a hybrid search restricted to the routed docs via
  the new `in` filter operator, diversity-capped per doc and unioned with the
  existing `flat → descent → lift` sources.
- `RAGChain.run(...)` — Stage 3.5 calls `_route_documents_stage1` (gated on
  `RAG_DOCUMENT_ROUTING_ENABLED`, skipped under deep-research) and threads
  `routed_doc_ids` into `_collect_candidates`.

**Ingest side:**

| File | Responsibility |
| --- | --- |
| `src/ingest/embedding/common/card_builder.py` | Pure, deterministic `build_document_card(chunks, *, max_headings, with_summary, summary)` → card-record dict (`document_id`, `title`, `source`, `section_headings`, `card_text`, `summary`, `num_chunks`). No LLM, no I/O. |
| `src/ingest/embedding/nodes/document_card.py` | `document_card_emission_node` — LangGraph node that (when enabled) groups `state["chunks"]` by `document_id`, builds one card per doc, batch-embeds `card_text`, and stages `staged_card_records`. Strict no-op when disabled. |
| `src/ingest/embedding/nodes/commit_node.py` | Extended so the atomic commit also writes staged cards (`ensure_card_collection` + `add_document_cards`) **after** the chunk write, and rolls them back with the rest on failure. |
| `src/ingest/embedding/workflow.py` | Wires `document_card_emission` into the graph between `tree_node_synthesis` and `metadata_generation` (always in the graph; the node short-circuits when disabled). |
| `src/ingest/embedding/state.py` / `src/ingest/common/types.py` | `staged_card_records` state field; `IngestionConfig.build_document_cards / card_llm_summary / card_max_headings / card_collection`. |

**Vector store side:**

| File | Responsibility |
| --- | --- |
| `src/vector_db/weaviate/card_store.py` | The `RAGDocumentCards` collection store: `ensure_card_collection` (idempotent schema, vectorizer `none`), `add_document_cards` (deterministic-UUID upsert with caller-supplied vectors), `delete_document_cards` (rollback by the same `uuid5(document_id)`). |
| `src/vector_db/weaviate/backend.py` | `_single_filter` gains the `in` operator → Weaviate `contains_any([...])` (mirrors the existing `not_in` empty-set handling). |
| `src/vector_db/__init__.py` / `src/vector_db/weaviate/__init__.py` | Re-export `ensure_card_collection`, `add_document_cards`, `delete_document_cards` on the stable facade. |
| `src/vector_db/common/schemas.py` | `SearchFilter` docstring lists `in` as a supported operator. |

### 1.2 Why a separate card collection

The card collection (`RAGDocumentCards`, vectorizer `none`) is deliberately
decoupled from the heavy chunk `DocumentRecord` path, exactly the way
`visual_store.py` decouples the visual page collection. It is small (one object
per document), its vectors are pre-computed by the ingest embedder, and it is
*never* read into answer context — only the router queries it.

---

## 2. Stage-by-stage flow & decision points

### 2.1 Ingest: building and committing cards

1. **Synthesis →** `tree_node_synthesis` produces `node_kind="section"` nodes
   (heading paths) alongside leaf chunks. These structural nodes are the free,
   no-LLM raw material for a card.
2. **Card emission (`document_card_emission_node`):** when
   `config.build_document_cards` is `True`, group `state["chunks"]` by
   `metadata["document_id"]` (first-seen order), and for each document call
   `build_document_card`:
   - **title** = `metadata["title"]` → first derived heading → `source`/
     `source_key`/`source_name` → `""`;
   - **section_headings** = ordered, de-duplicated headings (from `heading_path`
     / `heading` / `section_path`), **capped at `card_max_headings`** (the
     `.xlsx` guard, §11 of the design);
   - **card_text** = the title line, then each heading as a `"## "`-prefixed
     line, then (only when `with_summary` and a summary is supplied) the summary;
   - **num_chunks** = count of *leaf* chunks (diagnostic).
   Cards whose `card_text` is empty are dropped (never embedded, never staged).
   All `card_text` values are batch-embedded in **one** `embed_documents` call,
   each vector attached to its card, and the list staged as `staged_card_records`.
   When the flag is off (the default) the node is a **strict no-op**: no card
   work, no embedder call, no `staged_card_records` key.
3. **Atomic commit (`commit_node`):** within the same `try` that writes MinIO and
   the chunk records, **after** the chunk write succeeds, if
   `staged_card_records` is present the node calls `ensure_card_collection` then
   `add_document_cards(card_collection)`. A card-write failure raises into the
   shared `except` and triggers the **same rollback** as a chunk failure (chunks,
   MinIO, and the just-written cards are all undone — cards by their deterministic
   `uuid5(document_id)`). When no cards are staged, the card path is a strict
   no-op and chunk-commit behaviour is byte-identical to pre-feature.

**Idempotency:** each card's object UUID is `uuid5(NAMESPACE_DNS, document_id)`,
so re-ingesting a document **replaces** its card (insert-or-replace) rather than
duplicating it — mirroring the chunk store's deterministic-UUID upsert.

### 2.2 Query: route then retrieve

Stage 3.5 runs only when `RAG_DOCUMENT_ROUTING_ENABLED` is true and the request
is **not** deep-research (deep-research has its own recursive path and bypasses
Stage 4 entirely).

1. **Decompose (`decompose_query`, C3):** gated on `RAG_DECOMPOSITION_ENABLED`
   **and** lexical comparison intent. If either is false → identity result
   `[query]` (no tier invoked). When engaged, it tries the configured primary
   tier and falls back:
   - `RAG_DECOMPOSITION_LLM_PRIMARY=true` (default): `llm_decompose` first, then
     `regex_decompose` on any `DecompositionError`/exception;
   - otherwise: `regex_decompose` first, then `llm_decompose`.
   A tier "succeeds" only when it yields **≥ 2** non-empty sub-queries; otherwise
   the orchestrator falls through to the identity baseline. `decompose_query`
   **never raises**.
2. **Route each sub-query (`route_documents`, B1):** the identity sub-query
   reuses the already-computed query embedding (never re-embedded); other
   sub-queries are embedded once via `_embed_query_cached`. Each embedding does a
   **vector-first** (`alpha=1.0`) search over the card collection for up to
   `top_n` cards. The router returns a `RoutingResult`:
   - `used=True` with rank-ordered, de-duplicated `doc_ids` when the **top** card
     hit clears `min_score`;
   - `used=False` (empty) when there are no hits, the top hit is below
     `min_score`, the collection is missing/empty, or **any** backend error
     occurs — `route_documents` **never raises**.
3. **Union (`_route_documents_stage1`, B3):** only `used=True` results
   contribute. Their `doc_ids` are unioned (de-duped, first-seen order) across
   sub-queries. If **no** sub-query routed with confidence → returns `None`
   (→ pure flat retrieval). Decomposition makes each routing query single-topic,
   so e.g. "AXI4 vs CHI" routes "AXI4" → the AXI spec and "CHI" → the CHI spec,
   then unions both.
4. **Collect candidates (`_collect_candidates`, B2):** when `routed_doc_ids` is
   truthy a **4th** candidate source is added — a hybrid search filtered to the
   routed docs (`SearchFilter("document_id", "in", routed_doc_ids)`), with a
   per-doc diversity cap of `RAG_DOCUMENT_ROUTING_PER_DOC_LEAVES` so one routed
   doc cannot dominate. Sources are unioned in the order
   **flat → descent → lift → routed** (flat keeps priority; a chunk found by
   multiple sources dedupes to its first occurrence). The union is bounded to
   `RAG_DOCUMENT_ROUTING_MAX_CANDIDATES` **only when routing is active** — the
   `routed_doc_ids=None`/empty path is byte-identical to pre-routing.
5. **One rerank pass (unchanged):** the deduped union is reranked once against
   the query; the reranker is the **final authority** on relevance and returns
   the best top-K. `RAG_DOCUMENT_ROUTING_BOOST` (default 0.0) is a reserved
   *tiny* tie-break only — this slice does **not** apply a post-rerank score
   override.

### 2.3 Invariants (the safety contract)

- **Cards/summaries are routing-only and NEVER sent to the LLM.** Only Stage-4
  chunks become answer context. (The thin heading/TOC chunk filter,
  `_drop_thin_chunks`, still keeps structural nodes out of the answer pool.)
- **Routing is soft:** route to **top-N, never top-1**; never a hard filter; a
  strong chunk from a non-routed doc still survives via the flat path.
- **Fall back to pure flat retrieval** on low confidence / empty routed set /
  missing collection / any failure (`used=False` → `routed_doc_ids=None`).
- **Conservative defaults = fully OFF / no-op.** With the default flags, ingest
  writes no cards and retrieval runs the exact pre-feature path.

---

## 3. Configuration keys & behaviour toggles

All keys live in `config/settings.py` (env-overridable). Ingest keys are mirrored
onto `IngestionConfig` fields in `src/ingest/common/types.py`. **Every default is
conservative (OFF / no-op).**

### 3.1 Retrieval — Stage-1 routing

| Key | Default | Effect |
| --- | --- | --- |
| `RAG_DOCUMENT_ROUTING_ENABLED` | `false` | Master switch. False → router never invoked; retrieval unchanged. |
| `RAG_DOCUMENT_ROUTING_TOP_N` | `6` | Routed docs per (sub-)query. **Validated ≥ 2** (design §7: never top-1). |
| `RAG_DOCUMENT_ROUTING_MIN_SCORE` | `0.0` | Card-similarity floor; the *top* card hit must clear it or routing yields nothing. `0.0` = accept any hit. |
| `RAG_DOCUMENT_ROUTING_PER_DOC_LEAVES` | `5` | Chunks fetched per routed doc in the routed-doc union; also the per-doc diversity cap. |
| `RAG_DOCUMENT_ROUTING_MAX_CANDIDATES` | `60` | Upper bound on the unioned candidate pool (design §6.2). **Validated ≥ `PER_DOC_LEAVES`.** |
| `RAG_DOCUMENT_ROUTING_BOOST` | `0.0` | Reserved tiny tie-break for routed candidates. `0.0` = no score influence (rerank decides). Soft only. |
| `RAG_DOCUMENT_CARD_COLLECTION` | `"RAGDocumentCards"` | Card collection name. Shared by ingest (emission) and query (router). |

### 3.2 Retrieval — comparison decomposition

| Key | Default | Effect |
| --- | --- | --- |
| `RAG_DECOMPOSITION_ENABLED` | `false` | Master switch. False → `decompose_query` is identity (`[query]`). |
| `RAG_DECOMPOSITION_LLM_PRIMARY` | `true` | When enabled, use Tier-2 LLM as primary (regex fallback); false = regex primary (LLM fallback). |
| `RAG_DECOMPOSITION_LLM_TIMEOUT_SECONDS` | `8` | Per-call timeout for the Tier-2 LLM. On timeout/error → fall back. |
| `RAG_DECOMPOSITION_MIN_SUBQUERIES` | `2` | Minimum sub-queries a valid decomposition must yield. **Validated ≥ 2 and ≤ MAX.** |
| `RAG_DECOMPOSITION_MAX_SUBQUERIES` | `5` | Maximum sub-queries a valid decomposition may yield. |

### 3.3 Ingest — document cards

| Key (`config.settings`) | `IngestionConfig` field | Default | Effect |
| --- | --- | --- | --- |
| `RAG_INGESTION_BUILD_DOCUMENT_CARDS` | `build_document_cards` | `false` | When true, ingest emits one card per doc into the card collection. False = no card collection written. |
| `RAG_INGESTION_CARD_LLM_SUMMARY` | `card_llm_summary` | `false` | When true, card text would include an LLM summary. False = baseline title + headings (no LLM call). *(Baseline card path is wired; the LLM-summary upgrade is an extension — see §5.)* |
| `RAG_INGESTION_CARD_MAX_HEADINGS` | `card_max_headings` | `60` | Cap on headings per card (the `.xlsx` guard, design §11). |
| `RAG_DOCUMENT_CARD_COLLECTION` | `card_collection` | `"RAGDocumentCards"` | Card collection name (shared with retrieval). |

### 3.4 Config validation (fail-fast)

`config.settings.validate_document_routing_config()` raises a descriptive
`ValueError` for contradictory settings (invoked lazily by callers, matching
`validate_visual_retrieval_config`'s fail-fast-at-use pattern — importing
`config.settings` with default env never raises):

- `RAG_DOCUMENT_ROUTING_TOP_N < 2` (design §7: never top-1);
- `RAG_DECOMPOSITION_MIN_SUBQUERIES < 2`, or `MIN > MAX`;
- `RAG_DOCUMENT_ROUTING_MAX_CANDIDATES < RAG_DOCUMENT_ROUTING_PER_DOC_LEAVES`
  (the union bound cannot be smaller than the per-doc leaf fetch).

### 3.5 How to enable end-to-end

Routing needs cards in the index, so it is a **two-part** enablement:

1. **Populate cards (ingest, requires re-ingest):**
   ```bash
   export RAG_INGESTION_BUILD_DOCUMENT_CARDS=true
   # optional knobs:
   # export RAG_INGESTION_CARD_MAX_HEADINGS=60
   # export RAG_DOCUMENT_CARD_COLLECTION=RAGDocumentCards
   ```
   Then re-ingest the corpus. This creates/populates `RAGDocumentCards` with one
   card per document. (Until cards exist, routing is a safe no-op — `used=False`.)

2. **Turn routing on at query time:**
   ```bash
   export RAG_DOCUMENT_ROUTING_ENABLED=true
   # to also split comparison queries per entity:
   export RAG_DECOMPOSITION_ENABLED=true   # (+ RAG_DECOMPOSITION_LLM_PRIMARY=true by default)
   ```

With both off (the default) the system behaves exactly as before. With routing
on but cards absent, retrieval falls back to flat (no error). Enabling
decomposition without routing simply affects nothing downstream (the routed
union is what consumes the sub-queries).

### 3.6 Populate cards WITHOUT re-ingest (design §9 step 1)

When the corpus is already in `RAGDocuments` but the source files are **not** on
the box (so a re-ingest is impossible), build the card index directly from the
existing title + heading chunks — design §9 rollout step 1 ("build doc cards
from existing title+heading nodes — no re-ingest").

```bash
# Inside rag-ingest-worker-dev (picks up RAG_TEI_EMBED_URL,
# RAG_DOCUMENT_CARD_COLLECTION, RAG_WEAVIATE_* from the container env):
uv run python scripts/backfill_document_cards.py            # all documents
uv run python scripts/backfill_document_cards.py --dry-run  # build + count, no write
uv run python scripts/backfill_document_cards.py --document-ids docA,docB
```

The tool (`backfill_cards_from_corpus` in
`src/ingest/embedding/common/card_backfill.py`) enumerates documents via a
server-side group-by on `document_id` (`iter_document_ids` — no full-object
scan), fetches each document's chunks cursor-paginated
(`fetch_chunks_by_document_id` — memory bounded at one page, safe for a
~475k-object collection), builds the **same** baseline card as the ingest node
(`build_document_card`, no LLM), embeds `card_text` with the **same** embedder as
the corpus (`get_embedding_provider(tier="ingest")` → TEI qwen3-embed), and
upserts via `ensure_card_collection` + `add_document_cards`. It is idempotent
(deterministic `uuid5(document_id)` upsert) and resumable: a single bad document
is recorded in the returned `errors` list and skipped, never fatal. This is the
preferred way to populate cards on the live dev box; re-ingest (§3.5 step 1) is
only needed when you also want fresh chunks (or the LLM-summary upgrade).

---

## 4. Extension steps

### 4.1 Add the LLM doc-summary upgrade (design §6.4 / §9 step 2)

The baseline card is title + headings (no LLM). To enrich routing with one
concise LLM summary per document at ingest:

1. In `document_card_emission_node`, when `config.card_llm_summary` is true,
   generate a summary per document (e.g. via `get_llm_provider()` over the
   document's leaf text) and pass it to `build_document_card(..., with_summary=
   True, summary=<text>)`. `build_document_card` already accepts and stores
   `summary` and appends it to `card_text` when `with_summary` is set — so the
   builder needs **no** change, only the node's summary-generation step and
   honouring `RAG_INGESTION_CARD_LLM_SUMMARY`.
2. Re-ingest to repopulate cards (the deterministic UUID makes this an
   insert-or-replace). The card schema already has a `summary` (TEXT) property.
3. Cards remain routing-only — the summary still must never reach the LLM at
   query time.

### 4.2 Tune routing breadth and blend

- **N (routed docs):** `RAG_DOCUMENT_ROUTING_TOP_N` (≥ 2). Higher N → broader
  routing (better recall on scattered/comparison queries, more rerank work).
- **Per-doc leaves:** `RAG_DOCUMENT_ROUTING_PER_DOC_LEAVES` — how many chunks each
  routed doc contributes (and the per-doc diversity cap).
- **Union bound:** `RAG_DOCUMENT_ROUTING_MAX_CANDIDATES` — the reranker is what
  prevents dilution, so a larger pool is safe up to GPU-reranker cost. Keep
  ≥ `PER_DOC_LEAVES`.
- **Confidence floor:** `RAG_DOCUMENT_ROUTING_MIN_SCORE` — raise to make routing
  more conservative (only engage on strong card matches).
- **Blend bias:** `RAG_DOCUMENT_ROUTING_BOOST` — a reserved tiny tie-break; keep
  near 0.0 and let the rerank arbitrate. Do **not** use it as a hard override.

### 4.3 Extend the glossary / decomposition

Add a canonical entity (with informal aliases, optional spec IDs, description) to
`PROTOCOL_GLOSSARY` in `glossary.py`; the canonicalizer, detector, and the LLM
prompt's rendered glossary all derive from this single source of truth.

---

## 5. Troubleshooting & failure modes

| Symptom | Likely cause | Resolution |
| --- | --- | --- |
| Routing seems to do nothing (results identical to flat) | Card collection empty/absent → `route_documents` returns `used=False`; `routed_doc_ids=None` | Re-ingest with `RAG_INGESTION_BUILD_DOCUMENT_CARDS=true`, or (when source files are off-box) run `scripts/backfill_document_cards.py` to populate `RAGDocumentCards` from the existing chunks (§3.6). Confirm the collection has objects. |
| Routing engages but adds the wrong docs | Card text too thin (some docs have few headings) | Consider the LLM-summary upgrade (§4.1); raise `RAG_DOCUMENT_ROUTING_MIN_SCORE` so weak matches fall back to flat. |
| Routing never engages on strong queries | `RAG_DOCUMENT_ROUTING_MIN_SCORE` too high for the embedding model's score scale | Lower `MIN_SCORE` (default `0.0` accepts any hit). |
| Comparison query not split | `RAG_DECOMPOSITION_ENABLED=false`, or intent not detected, or both tiers failed | Enable decomposition; check phrasing against `detect_comparison_intent` cues; regex guards intentionally refuse ambiguous splits (a miss is safe). |
| Decomposition mis-split fear | Regex on natural language is brittle | `regex_decompose` is conservative: it only splits unambiguous structures, guards against aspect-text leaking into an entity ("AHB for a high-performance SoC") and against bare-`and` splits, and returns `[]` on any doubt → caller keeps the original query. |
| LLM decomposition slow / times out | Shared/contended dev vLLM (reasoning model) | `RAG_DECOMPOSITION_LLM_TIMEOUT_SECONDS` bounds the call; on timeout/error it raises `DecompositionError` and the orchestrator falls back to regex, then to the identity query. |
| `.xlsx` document routing dominated by giant tables | Spreadsheets have hundreds of sheet/column headings | `RAG_INGESTION_CARD_MAX_HEADINGS` (default 60) caps a card's heading list at ingest (design §11). |
| Re-ingest created duplicate cards | — (should not happen) | Cards upsert by deterministic `uuid5(document_id)`; a re-ingest **replaces** the card. If you see duplicates, the `document_id` changed between runs. |
| `ValueError` from `validate_document_routing_config` at use | Contradictory config (TOP_N < 2, MIN > MAX subqueries, MAX_CANDIDATES < PER_DOC_LEAVES) | Fix the offending key named in the message. |
| Deep-research queries ignore routing | By design | Routing is skipped under deep-research (it has its own recursive retrieval path). |

**General principle:** every routing/decomposition failure path is engineered to
**degrade to the safe pre-feature behaviour**, never to error. `route_documents`
and `decompose_query` never raise; `_route_documents_stage1` is wrapped so it can
never break `run()`.

---

## 6. Testing

All tests run with `uv run pytest <path> -p no:cacheprovider -q` from the
worktree root.

| Area | Test file(s) |
| --- | --- |
| Config knobs / defaults / validation | `tests/retrieval/test_routing_config_defaults.py`, `tests/ingest/test_card_config_fields.py` |
| `in` filter operator | `tests/vector_db/test_filter_in_operator.py` |
| Glossary + comparison-intent + canonicalization | `tests/retrieval/routing/test_glossary.py` |
| Tier-1 regex decomposition (incl. mis-split guards) | `tests/retrieval/routing/test_decomposition_regex.py` |
| Tier-2 glossary-LLM decomposition (mock provider) | `tests/retrieval/routing/test_decomposition_llm.py` |
| C3 gated orchestrator (gate + tier fallback order) | `tests/retrieval/routing/test_decomposition_orchestrator.py` |
| Stage-1 card-index router (B1) | `tests/retrieval/routing/test_router.py` |
| Routed-doc candidate union (B2) | `tests/retrieval/test_collect_candidates_routing.py` |
| Stage-1 wiring into `run()` (B3) | `tests/retrieval/test_run_document_routing.py` |
| Card builder (pure) | `tests/ingest/test_card_builder.py` |
| Card emission node (toggle on/off) | `tests/ingest/test_document_card_node.py` |
| Atomic card commit + rollback | `tests/ingest/test_commit_node_cards.py` |
| Card collection store (mock client) | `tests/vector_db/test_card_store.py` |
| End-to-end (design §10 validation set, integration-marked) | `tests/integration/test_document_routing_e2e.py` |

The **§10 validation set** (asserted by the e2e test in both directions):

- **Should improve:** "compare AXI4 / CHI" surfaces **both** authoritative spec
  docs (the dedicated per-entity sub-queries route to the AXI spec and the CHI
  spec respectively, and the routed-doc union pulls a CHI chunk in past the
  keyword-overlapping distractors).
- **Must NOT regress:** a needle register fact (offset/reset value) and a
  single-document project question return identical-or-better results vs
  routing-off — the needle's card does not obviously match, so the flat path
  still carries it, and routing adds nothing it can subtract.

---

## 7. Relationship to the design

This guide is the as-built counterpart to
[`DOCUMENT_ROUTING_DESIGN.md`](DOCUMENT_ROUTING_DESIGN.md). Section mapping:

- Design §3 (2-stage retrieval) → §2 here (Stage 3.5 route → Stage 4 retrieve).
- Design §6.1 (cards/headings are routing/navigation, never answer content) →
  §2.3 invariants.
- Design §6.2 (union → dedup → one rerank, bounded pool) → §2.2 steps 4–5.
- Design §6.3 / §6.5 (decomposition feeds routing; LLM-primary, regex fallback)
  → §2.2 step 1 + §5 troubleshooting.
- Design §6.4 (optional LLM summaries) → §4.1 extension.
- Design §7 (soft, never top-1, fall back) → §2.3 + §3.4 validation.
- Design §11 (`.xlsx` card heading cap) → §3.3 (`card_max_headings`) + §5.
