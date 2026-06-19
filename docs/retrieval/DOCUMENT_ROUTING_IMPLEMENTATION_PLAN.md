# RAPTOR-lite Document Routing — Implementation Plan & Slice Briefs

> Companion to `docs/retrieval/DOCUMENT_ROUTING_DESIGN.md` (the design).
> This file is the **single source of truth** for the implementation. Every
> implementation subagent must read the design + this file before starting.

- **Branch:** `feat/raptor-lite-doc-routing`
- **Worktree:** `/var/tmp/Juan.Kok/ragweave-docrouting`
- **Goal:** implement everything the design lists as a weakness in the current
  RagWeave retrieval implementation, via narrow vertical slices, each TDD-driven
  with a validable end goal and proper end-to-end testing.

## Non-negotiable conventions (ALL subagents)

1. **Run tests with** `uv run pytest <path> -p no:cacheprovider -q` from the
   worktree root (this is what `make test` uses; the worktree `.venv` is already
   synced with `--extra dev`). `-p no:cacheprovider` avoids `.pytest_cache` races
   between parallel subagents. Examples:
   - `uv run pytest tests/retrieval/routing/test_glossary.py -p no:cacheprovider -q`
   - `uv run pytest tests/ingest/test_card_builder.py -p no:cacheprovider -q`
   - Whole subsuite: `uv run pytest tests/retrieval/ -p no:cacheprovider -q`
   > NOTE: `scripts/run-tests.sh` is currently broken in this worktree (its
   > safety-validator blocks even committed tests and its plugin gate checks a
   > wrong module name `pytest_json_report`). Do NOT use it; use `uv run pytest`.
2. **No `curl`/`wget`** for any local check (EDR may SIGKILL). Use Python
   `urllib.request`/`http.client` or in-process calls.
3. **TDD + ralph loop**: write the failing test FIRST, then implement, then
   `scripts/run-tests.sh`, then fix, and repeat until green. Do not stop until
   your slice's gate is green. Report the exact command + final result.
4. **Safety philosophy from the design (§6.2, §7):**
   - Routing is **soft**: top-N (never top-1), **never a hard filter**, fall
     back to pure flat retrieval when confidence is low / routed set is small.
   - Summaries/cards are **routing-only, NEVER sent to the LLM**.
   - Blend = **union → dedup → ONE rerank pass** (do not rerank sources
     separately; do not merge by raw hybrid score).
5. **Config-driven**: every new behaviour is gated by a typed config flag with a
   **conservative default (OFF / no-op)**. No behaviour change unless explicitly
   enabled. Validate contradictory settings, fail fast with a clear message.
6. **Project conventions** (CLAUDE.md): `@summary` block on every new/changed
   source file; docstrings; thin facades; typed contracts in `schemas.py`;
   keep public import surfaces stable; update directory `README.md`.
7. **Stay on the worktree** at `/var/tmp/Juan.Kok/ragweave-docrouting`. Do not
   touch the other checkouts. Do not commit (the orchestrator commits per slice).

## Key code anchors (verified)

- `src/retrieval/pipeline/rag_chain.py`
  - `RAGChain.run(...)` — orchestrator. Hybrid-search/candidate stage ~1703-1756.
  - `_collect_candidates(...)` 788-862 — union+dedup of `(leaf, descent, lift)`.
  - `_do_search(...)` 580-621 — calls `src.vector_db.search`, applies `SearchFilter`s.
  - `_run_tree_descent` 681-737, `_run_tree_lift` 739-786.
  - reranker call ~1862-1877 (`self.reranker.rerank(query, documents, top_k)`).
- `src/vector_db/common/schemas.py` — `SearchResult(text,score,metadata,object_id,collection)`,
  `SearchFilter(property,operator,value)`; ops eq/ne/gt/lt/gte/lte/like/not_in.
- `src/vector_db/weaviate/backend.py::_single_filter` 253-286 — operator translation.
- `src/vector_db/weaviate/store.py` — `ensure_collection` (236-361), `add_documents`
  (443-607, collection-agnostic via `collection=` kwarg), `TABLE_AWARE_PROPERTIES`.
- `src/vector_db/weaviate/visual_store.py` — template for a SECOND collection.
- `src/vector_db/__init__.py` — `search(...)` 251-276, `add_documents(...)` 154-169
  (backend-agnostic facades).
- `src/ingest/embedding/workflow.py` — `build_embedding_graph()` 35-111 (12-node DAG;
  `tree_node_synthesis` → `metadata_generation` → ... → `embedding_storage` → `commit`).
- `src/ingest/embedding/nodes/tree_node_synthesis.py` — produces `node_kind="section"`
  nodes with `heading_path`, `section_path`, `parent_section_id`, `document_id`.
- `src/ingest/embedding/nodes/embedding_storage.py` / `commit_node.py` — stage + atomic commit.
- `src/ingest/common/types.py` — `IngestionConfig` (target_collection, embedding_batch_size,
  enable_tree_retrieval_ingest, enable_llm_metadata, ...); `Runtime.embedder`.
- `src/core/embeddings.py` — `get_embedding_provider(tier=...)`, `.embed_documents([text])`.
- `src/platform/llm/provider.py` — `get_llm_provider().json_completion(messages, temperature,
  max_tokens, timeout)` → `LLMResponse(content=...)`; `src/common.parse_json_object`.
- `config/settings.py` — module-level attrs; `VECTOR_COLLECTION_DEFAULT="RAGDocuments"`.
- Tests: `tests/retrieval/`, `tests/ingest/`, `tests/vector_db/`; root `tests/conftest.py`
  stubs heavy deps; markers: integration, slow, smoke.

> NOTE: the design §8 says `_decompose_comparison` "already exists" — it does NOT.
> Only `deep_research.py::_decompose` (the Tier-3 recursive path we must NOT use
> for comparisons) exists. The comparison-intent detector, `PROTOCOL_GLOSSARY`,
> and Tier-1/Tier-2 decomposition are all built fresh in Pillar C.

## New module layout (target)

```
src/retrieval/routing/
  __init__.py            # stable facade: route_documents, decompose_query, ...
  schemas.py             # RoutingResult, DecompositionResult, DocCard contracts
  glossary.py            # PROTOCOL_GLOSSARY + comparison-intent detector + term map
  decomposition.py       # Tier-1 regex + Tier-2 glossary-LLM + orchestrator
  router.py              # Stage-1 card-index routing (query -> routed doc_ids)
src/ingest/embedding/nodes/document_card.py   # card emission node
src/ingest/embedding/common/card_builder.py   # pure card-construction helpers
src/vector_db/weaviate/card_store.py          # ensure/add for the card collection
tests/retrieval/routing/                       # unit tests for routing+decomposition
tests/ingest/test_document_card*.py            # unit tests for card build/emit
tests/vector_db/test_card_store.py             # card collection schema/add tests
tests/integration/test_document_routing_e2e.py # end-to-end (integration-marked)
```

## Config knobs to add (conservative defaults)

Retrieval (config/settings.py):
- `RAG_DOCUMENT_ROUTING_ENABLED` (bool, default **false**)
- `RAG_DOCUMENT_ROUTING_TOP_N` (int, default 6) — routed docs, never 1
- `RAG_DOCUMENT_ROUTING_MIN_SCORE` (float, default 0.0) — card-sim floor for routing
- `RAG_DOCUMENT_ROUTING_PER_DOC_LEAVES` (int, default 5) — chunks fetched per routed doc
- `RAG_DOCUMENT_ROUTING_MAX_CANDIDATES` (int, default 60) — union bound (§6.2)
- `RAG_DOCUMENT_ROUTING_BOOST` (float, default 0.0) — tiny optional tie-break only
- `RAG_DOCUMENT_CARD_COLLECTION` (str, default "RAGDocumentCards")
- `RAG_DECOMPOSITION_ENABLED` (bool, default **false**)
- `RAG_DECOMPOSITION_LLM_PRIMARY` (bool, default true) — Tier-2 primary when enabled
- `RAG_DECOMPOSITION_LLM_TIMEOUT_SECONDS` (int, default 8)
- `RAG_DECOMPOSITION_MIN_SUBQUERIES` (int, default 2), `RAG_DECOMPOSITION_MAX_SUBQUERIES` (int, default 5)

Ingestion (config/settings.py + IngestionConfig):
- `RAG_INGESTION_BUILD_DOCUMENT_CARDS` → `IngestionConfig.build_document_cards` (bool, default **false**)
- `RAG_INGESTION_CARD_LLM_SUMMARY` → `IngestionConfig.card_llm_summary` (bool, default **false**; baseline = title+headings)
- `RAG_INGESTION_CARD_MAX_HEADINGS` → `IngestionConfig.card_max_headings` (int, default 60; §11 xlsx guard)
- `IngestionConfig.card_collection` (str, default = `RAG_DOCUMENT_CARD_COLLECTION`)

## Slices (each = one subagent, TDD, validable gate)

### Wave 0 — foundation (file-disjoint, parallel-safe)

- **S0a — `in`/`contains_any` filter operator.**
  Files: `src/vector_db/weaviate/backend.py` (`_single_filter`), docstring in
  `src/vector_db/common/schemas.py`. Test: `tests/vector_db/test_filter_in_operator.py`.
  Gate: a `SearchFilter("document_id","in",[a,b])` translates to a weaviate
  `contains_any([a,b])` (unit-test the translation; mock `weaviate.classes.query.Filter`),
  unknown-op error message lists `in`. `scripts/run-tests.sh --group import-check` green.

- **S0b — config knobs.**
  Files: `config/settings.py`, `src/ingest/common/types.py` (IngestionConfig fields).
  Test: `tests/retrieval/test_routing_config_defaults.py` (+ ingest config test).
  Gate: every knob above importable with the stated default; OFF by default;
  add a fail-fast validation (e.g. TOP_N>=2; MIN<=MAX subqueries). Tests green.

- **S0c — glossary + comparison-intent detector.**
  Files: `src/retrieval/routing/__init__.py`, `src/retrieval/routing/glossary.py`,
  `src/retrieval/routing/schemas.py`. Test: `tests/retrieval/routing/test_glossary.py`.
  Content: `PROTOCOL_GLOSSARY` (canonical entities incl. AXI/AXI4/AXI5/CHI/ACE/AHB/
  APB + informal→canonical maps like "hub-based"→CHI, "extension-based"→ACE),
  `detect_comparison_intent(query)->bool` (lexical: "vs", "versus", "compare",
  "A / B / C", "difference between", "or the … one"), `canonicalize_terms(query)`.
  Gate: detector flags the design's comparison examples and does NOT flag plain
  single-topic queries; glossary maps informal→canonical. Tests green.

### Wave 1 — Pillar A (ingest cards) ∥ Pillar C (decomposition)

A and C touch disjoint trees (ingest+vector_db vs retrieval/routing) → run in
lockstep parallel: (A1∥C1) then (A2∥C2) then (A3∥C3) then A4.

- **A1 — card builder (pure).** `src/ingest/embedding/common/card_builder.py`:
  `build_document_card(chunks, *, max_headings, with_summary=False) -> DocCard`
  (title from first heading/`metadata["title"]`/source; ordered unique section
  headings from `heading_path`/section nodes; `card_text` = title + headings
  [+ summary]; cap headings; §11 xlsx care). Test: `tests/ingest/test_card_builder.py`.
  Gate: deterministic card from synthetic chunks; xlsx-like (100s of headers)
  capped; no LLM. Green.

- **A2 — card collection store.** `src/vector_db/weaviate/card_store.py`:
  `ensure_card_collection(client, collection)` (props: document_id, title,
  source, section_headings[TEXT_ARRAY], card_text, summary, num_chunks; vectorizer
  none) + reuse `add_documents(..., collection=...)`. Facade exports in
  `src/vector_db/__init__.py`. Test: `tests/vector_db/test_card_store.py` (mock client).
  Gate: idempotent ensure; one object/doc upserted to the card collection. Green.

- **A3 — card emission node + workflow wiring.**
  `src/ingest/embedding/nodes/document_card.py` (`document_card_emission_node`):
  when `config.build_document_cards`, group `state["chunks"]` by `document_id`,
  build a card (A1), embed `card_text` via `runtime.embedder.embed_documents`,
  stage card records on state (`staged_card_records`); else no-op. Wire into
  `src/ingest/embedding/workflow.py` after `tree_node_synthesis`. Tests:
  `tests/ingest/test_document_card_node.py` + workflow-routing toggle test.
  Gate: toggle ON → state carries staged card records (mock embedder); OFF →
  byte-for-byte no-op. Green.

- **A4 — commit cards.** Extend `src/ingest/embedding/nodes/commit_node.py` to
  `ensure_card_collection` + `add_documents(staged_card_records, collection=card_collection)`
  inside the existing atomic commit (rollback-safe). Test: extend commit tests.
  Gate: staged cards committed to the card collection alongside chunks; rollback
  still consistent. Green.

- **C1 — Tier-1 regex decomposition (fallback).**
  `src/retrieval/routing/decomposition.py`: `regex_decompose(query)->list[str]`.
  Split on "vs/versus/compared to/ '/' lists/ difference between". MUST guard the
  observed mis-splits: do NOT let aspect text leak into an entity ("AHB for a
  high-performance SoC"), do NOT split on bare "and". Return `[]` (→ caller keeps
  original query) when uncertain. Test: `tests/retrieval/routing/test_decomposition_regex.py`
  incl. the mis-split cases (assert NOT split). Gate: safe splits only. Green.

- **C2 — Tier-2 glossary-LLM decomposition (primary).**
  `src/retrieval/routing/decomposition.py`: `llm_decompose(query, *, timeout)->list[str]`
  via `get_llm_provider().json_completion` grounded in `PROTOCOL_GLOSSARY`; tight
  prompt, capped tokens; validate output (2-5 non-empty short strings) else raise.
  Test: `tests/retrieval/routing/test_decomposition_llm.py` (mock provider: valid
  JSON → entities; invalid/timeout → raises so orchestrator falls back). Gate:
  parses + validates; maps informal→canonical. Green.

- **C3 — decomposition orchestrator (gated).**
  `decompose_query(query)->DecompositionResult`: if `RAG_DECOMPOSITION_ENABLED`
  and `detect_comparison_intent` → try `llm_decompose` (primary), on failure
  `regex_decompose`, on failure → `[query]` (safe baseline). Non-comparison or
  disabled → `[query]`. Facade export in `src/retrieval/routing/__init__.py`.
  Test: `tests/retrieval/routing/test_decomposition_orchestrator.py`. Gate: gate +
  tier fallback order verified; disabled → identity. Green.

### Wave 2 — Pillar B (routing + blend), sequential

- **B1 — card-index router.** `src/retrieval/routing/router.py`:
  `route_documents(query_embedding, *, top_n, min_score, collection)->RoutingResult`
  (doc_ids + scores) by vector search over the card collection via
  `src.vector_db.search`. Returns empty (→ pure flat) when below `min_score` or
  collection absent/empty. Test: `tests/retrieval/routing/test_router.py` (mock
  search). Gate: top-N doc_ids, never top-1 hard, empty on low-confidence. Green.

- **B2 — routed-doc candidate union.** Extend `_collect_candidates` (rag_chain)
  to accept `routed_doc_ids` + bounds; add a 4th source = hybrid search filtered
  to routed docs via the new `in` operator (`SearchFilter("document_id","in",ids)`),
  `RAG_DOCUMENT_ROUTING_PER_DOC_LEAVES` each; union+dedup with existing sources;
  cap to `RAG_DOCUMENT_ROUTING_MAX_CANDIDATES`. Test:
  `tests/retrieval/test_collect_candidates_routing.py`. Gate: routed-doc chunks
  represented, dedup correct, bound respected; `routed_doc_ids=None` → identical
  to today. Green.

- **B3 — wire Stage-1 into `run()`.** In `RAGChain.run`, gated by
  `RAG_DOCUMENT_ROUTING_ENABLED`: `decompose_query` (C3) → per-sub-query
  `route_documents` (B1) → union routed doc_ids → pass to `_collect_candidates`
  (B2) → single rerank (unchanged). Soft-boost only; OFF or low-confidence →
  today's path exactly. Test: `tests/retrieval/test_run_document_routing.py`
  (routing ON boosts routed docs; OFF identical; low-confidence → flat;
  needle/single-doc no-regression). Gate green.

### Wave 3 — end-to-end + docs

- **E1 — end-to-end test.** `tests/integration/test_document_routing_e2e.py`
  (integration-marked): exercise ingest→card→route→retrieve with a high-fidelity
  in-memory fake vector backend (or seeded real stack if available), on the
  validation set (§10): "compare AXI4/CHI" surfaces both spec docs (improve);
  a needle fact + a single-doc project query do NOT regress vs routing-off.
  Gate: e2e passes; both directions asserted.

- **E2 — docs + context-agent.** Flip `DOCUMENT_ROUTING_DESIGN.md` status to
  *implemented*; add engineering guide `docs/retrieval/DOCUMENT_ROUTING_ENGINEERING_GUIDE.md`
  (architecture, flow, config keys, extension, troubleshooting); update affected
  directory `README.md`s; run `context-agent update`. Gate: docs match code.

## Definition of done (whole effort)

- All slice gates green via `scripts/run-tests.sh`.
- `scripts/run-tests.sh --group retrieval` and `--group ingest` green.
- With all flags OFF: behaviour byte-identical to pre-change (no-regression).
- With flags ON: routed docs surface for broad/comparison queries; needle +
  single-doc queries unaffected (E1 asserts both directions).
- Docs + `@summary` blocks + READMEs updated; `context-agent update` run.
