# Ingestion Pipeline Cleanup — Research Changelog (v2)

## Run: 2026-04-20 — autoresearch/ingest-cleanup-2026-04-20

**Scoring mode:** Numerical
**Starting score:** 0/8
**Final score:** 8/8
**Iterations:** 4 (1 baseline + 3 kept, 0 discarded)
**Predecessor:** autoresearch/ingest-quality-2026-04-20 (77/77 on v1 scorer)

---

## Speed (iteration 004)

- **Eliminated double file read (004)** — `ingest_directory()` now calls `source_path.read_bytes()` once and uses `sha256_bytes(raw_bytes)` for the idempotency hash check. The pre-read bytes are passed through `ingest_file()` → `run_document_processing()` → `DocumentProcessingState["raw_bytes"]` → `document_ingestion_node()`. The node checks `state.get("raw_bytes")` before falling back to disk read, preserving backward compatibility for direct callers. Saves one full file read per non-skipped document.

## Code Quality (iterations 002–003)

- **Removed legacy `IngestState` TypedDict (002)** — Deleted the 31-line `IngestState` class from `types.py` (replaced by `DocumentProcessingState` and `EmbeddingPipelineState`). Removed export from `common/__init__.py`. Updated `append_processing_log` type hint from `IngestState` to `dict[str, Any]` (the function accepts both Phase 1 and Phase 2 state dicts).

- **Removed dead `enriched_chunks` field (003)** — Deleted field declaration from `EmbeddingPipelineState`, removed initialization (`"enriched_chunks": []`) from `embedding/impl.py`, updated `visual_embedding.py` MUST-NOT-MODIFY comment, and removed test assertions referencing the field. The field was declared and initialized but never written to or read by any node.

---

### Hypothesis mispredictions

None across all 3 iterations. All hypotheses predicted the exact criteria that would flip.

### Deferred items (assessed and rejected)

- **Blocking `time.sleep(0.3s)` in embedding retry** — Not a real issue. LangGraph runs sync nodes in thread pool executors; `time.sleep` blocks the thread, not the event loop. Converting to async would cascade through the entire call chain for no practical gain.
- **O(n²) `processing_log` append** — Not a real issue. With ~15 entries per document, total work is O(120) element copies — nanoseconds. The spread pattern (`[*existing, new]`) is intentional for LangGraph's immutable state semantics (test explicitly verifies non-mutation).
- **Fuzzy dedup 10K scan** — Known limitation documented in code ("Acceptable for corpora under 100K chunks"). The real fix is an LSH forest index — a feature, not an optimization.
