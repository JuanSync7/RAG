# Ingestion Pipeline Test Coverage — Research Changelog (v3)

## Run: 2026-04-20 — autoresearch/ingest-coverage-2026-04-20

**Scoring mode:** Numerical
**Starting score:** 60%
**Final score:** 75%
**Iterations:** 5 (5 kept, 0 discarded)
**Tests added:** ~470 new tests (1121 → ~1548)

---

## What improved

### Iteration 002 — Pure logic tests for `support/document.py` (+1%)
- 151 tests covering all 10 public functions: `strip_boilerplate` (41 pattern tests), `normalize_unicode`, `clean_whitespace`, `strip_section_markers`, `strip_trailing_short_lines`, `clean_text`, `extract_metadata`, `metadata_to_dict`, `chunk_text`, `process_document`
- Coverage gain was modest (+1%) because `document.py` has only 56 missed statements out of 4785 total

### Iteration 003 — Pure logic tests for temporal + minhash (+5%)
- 81 tests across 3 new files: `test_temporal_constants.py` (trigger routing, env var overrides), `test_temporal_worker.py` (slot/queue resolution, validation), `test_minhash_engine.py` (word shingles, fingerprinting, similarity estimation, MinHashEngine class)
- Biggest single-iteration gain — these were all 0% coverage files with 171 combined statements
- Fixed a pre-existing conftest cross-test pollution bug where lifecycle's redis stub interfered with datasketch imports

### Iteration 004 — Edge case tests for parsers + validation (+2%)
- 51 tests: `test_parser_text.py` (HTML/RST conversion fallbacks, chunk-before-parse error), `test_parser_registry.py` (missing Docling/tree-sitter ImportError paths, unknown extension fallback, warmup non-fatal), validation.py extensions (Weaviate/KG/MinIO error paths, lazy manifest loading)
- Covered error/fallback paths that are critical for production resilience but rarely hit in happy-path testing

### Iteration 005 — Mock-based tests for vision + MinIO + markdown (+4%)
- 89 tests: `test_vision_support.py` (data-url decoding, file resolution, VLM mock), `test_minio_clean_store.py` (full CRUD with mock client), `test_markdown_support.py` (heading normalization, semantic splitting mock, full pipeline)
- All mock tests follow `test_mock_*` naming convention

### Iteration 006 — Mock/edge tests for GC + migration + parser_code (+3%)
- 100 tests: GC engine (58 tests covering all 4 store cleanup helpers, dry-run, purge_expired edge cases), migration (19 tests covering lazy loading, unknown strategy errors, manifest I/O), parser_code (23 tests covering tree-sitter parse+chunk pipeline, AST helper functions)
- Reached the 75% target

---

### Hypothesis mispredictions

- **Iteration 002**: predicted 63-68% from document.py tests alone; actual was 61%. The file's 56 missed statements represent only ~1.2% of total, making the per-file impact much smaller than expected. Lesson: file-level coverage % is misleading — absolute missed statement count determines impact on the overall number.
- All other hypotheses were within range of their predictions.

### Remaining gaps (not covered)

Files still below 75% that could be targeted in a future run:

| File | Coverage | Missed | Reason not covered |
|------|----------|--------|--------------------|
| `temporal/activities.py` | 0% | 86 | Temporal test harness needed for activity mocking |
| `temporal/workflows.py` | 0% | 77 | Temporal workflow sandbox needed |
| `cli.py` | ~50% | ~45 | CLI integration tests need careful mock wiring |
| `support/docling.py` | ~70% | ~50 | Docling HybridChunker mocking is complex |
| `impl.py` | ~83% | ~40 | Orchestrator edge cases need multi-service mocks |
| `embedding/impl.py` | ~55% | ~8 | LangGraph graph invoke mocking |

### Test infrastructure notes

- **Scorer**: `src/ingest/scorer_v3.py` — measures coverage % via `pytest-cov`, enforces `test_mock_*` naming convention, correctness guard (all tests must pass)
- **Dependency added**: `pytest-cov` (dev dependency)
- **Conftest fix**: `tests/ingest/conftest.py` modified to eagerly import datasketch before lifecycle conftest installs a redis stub, preventing cross-test pollution
- **Naming convention**: 2 pre-existing violations are grandfathered in the scorer; all new mock tests follow `test_mock_*`
