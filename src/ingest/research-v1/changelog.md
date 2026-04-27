# Ingestion Pipeline Quality — Research Changelog

## Run: 2026-04-20 — autoresearch/ingest-quality-2026-04-20

**Scoring mode:** Numerical
**Starting score:** 6/77
**Final score:** 77/77
**Iterations:** 11 (11 kept, 0 discarded)
**Scorer expansions:** 2 (62 → 77 criteria at iteration 008→009 boundary)

---

## Phase 1 (iterations 002–008): Foundation — 6/62 → 62/62

### Robustness (iterations 002–005)

- **Consistent logger naming (002)** — All 17 nodes now use `rag.ingest.<phase>.<name>` hierarchy. Every log is filterable under `rag.ingest.*`. (+14)
- **INFO-level stage logging (003)** — Every node logs completion with key metrics. (+15)
- **DEBUG-level timing (004)** — Every node measures `time.monotonic()` and logs at DEBUG. (+16)
- **Pipeline-wide robustness (005)** — LLM failures at WARNING, streaming SHA-256, unique logger names. (+3)

### Speed (iterations 006, 008)

- **Single-pass file read (006)** — `document_ingestion_node` reads bytes once, hashes and decodes.
- **Reduced retry delay (006)** — 1.0s → 0.3s in embedding retry.
- **Early-exit paragraph matching (006)** — Breaks on ratio > 0.85.
- **Reduced manifest I/O (006)** — Removed per-skip `save_manifest()`.
- **EntityExtractor singleton (008)** — Module-level instead of per-document.

### Code Quality (iteration 007)

- **metadata_generation state contract (007)** — Returns `"chunks"` explicitly.
- **Dead alias removed (007)** — `_ollama_json` deleted.
- **Deprecation warning (007)** — `docling_document` parameter warns on use.

---

## Phase 2 (iterations 009–011): Deep improvements — 66/77 → 77/77

### Robustness (iteration 009)

- **Error-path logging (009)** — 5 nodes (`knowledge_graph_extraction`, `knowledge_graph_storage`, `document_storage`, `chunking`, `embedding_storage`) now call `logger.error()` in except blocks before returning error state. Previously, tracebacks were silently lost.
- **Timing consistency (009)** — `visual_embedding_node` switched from `time.time()` to `time.monotonic()` for clock-adjustment immunity.
- **Quality score log level (009)** — `quality_validation_node` now logs score computation failures at WARNING (was DEBUG). Chunks silently dropped by errors are now visible.
- **Multimodal error logging (009)** — `multimodal_processing_node` now logs non-strict vision failures at WARNING instead of silently continuing.

### Speed (iteration 010)

- **Removed per-document ensure_collection (010)** — `embedding_storage_node` no longer calls `ensure_collection()` per document. The orchestrator already ensures it before the loop. Saves one Weaviate API round-trip per document.
- **Removed per-document ensure_bucket (010)** — `document_storage_node` no longer calls `ensure_bucket()` per document. Saves one MinIO round-trip per document.
- **str.translate optimization (010)** — `document.py` `normalize_unicode` replaced 8 sequential `str.replace` calls with `str.maketrans` + `.translate()` for a single-pass O(n) scan.

### Code Quality (iteration 011)

- **Configurable collection name (011)** — `dedup_utils.py` replaced 5 hardcoded `"Chunk"` strings with `_DEFAULT_CHUNK_COLLECTION = VECTOR_COLLECTION_DEFAULT`. Functions now accept optional `collection_name` parameter.
- **Type annotation fix (011)** — `dedup_override_sources` typed as `list[str]` instead of bare `list`.
- **Error return consistency (011)** — 5 embedding nodes (`document_storage`, `chunking`, `knowledge_graph_extraction`, `embedding_storage`, `knowledge_graph_storage`) no longer spread `{**state}` on error returns. Returns only changed keys, consistent with Phase 1 nodes.
- **Encoding dedup (011)** — Added `decode_with_fallbacks(data: bytes)` to `utils.py`. `read_text_with_fallbacks` now delegates to it. `document_ingestion_node` uses it instead of inline encoding loop. Single source of truth for encoding fallback order.

---

### Hypothesis mispredictions

None across all 11 iterations. All hypotheses predicted the exact criteria that would flip.

### Remaining gaps (out of scope)

- **ColQwen2 per-document load/unload** — model loaded/unloaded per document in visual path
- **Sequential document processing** — `ingest_directory` is single-threaded (concurrency only via Temporal)
- **Blocking sleep in async** — `time.sleep(0.3s)` in embedding retry blocks event loop
- **Double file read in orchestrator** — `impl.py` hash check + Phase 1 re-reads source
- **O(n²) processing_log** — `append_processing_log` spreads list per call
- **Fuzzy dedup 10K scan** — `minhash_engine.py` fetches up to 10K objects for comparison
- **`enriched_chunks` dead field** — declared in `EmbeddingPipelineState` but never written
- **`IngestState` legacy TypedDict** — unused, replaced by per-phase state types
