# Ingestion Pipeline Test Coverage — Research Changelog

## Run v5: 2026-04-20 — autoresearch/ingest-coverage-v5-2026-04-20

**Scoring mode:** Numerical
**Starting score:** 96% (after excluding scorer tooling files from measurement)
**Final score:** 98%
**Iterations:** 3 (1 baseline + 2 improvement, all kept)
**Tests added:** ~50 new tests (1831 → 1881)

---

## What improved

### Iteration 002 — parser_code AST path + migration CLI/strategy dispatchers (+1%)
- 19 tests: tree-sitter real parse/chunk path (parser_code 82→100%), migration _migrate_full_phase2, _migrate_kg_reextract, CLI helpers (migration 89→99%)
- Key: used real tree-sitter with Python source to exercise full chunk() body including KG extraction, docstrings, decorators

### Iteration 003 — Broad sweep: gc, impl, validation, structure_detection (+1%)
- 31 tests across 4 files: gc CLI helpers/manifest exception paths, impl fallback matching/verify_core_design/parser registry failure, validation CLI helpers/report, structure_detection strict/fallback paths
- Reached 98% target in only 2 improvement iterations

---

## Key learnings

1. **Scorer hygiene matters:** Excluding scorer_*.py tooling files (533 statements, 0% covered) from measurement revealed that production code was already at 96%. This reframed the effort from "cover 327 statements" to "cover 44 statements."

2. **Efficiency at high coverage:** With precise line-level targeting and broad-file strategy, 50 tests covered 44+ statements across 6 files in 2 iterations. Each iteration was surgically targeted using `--cov-report=term-missing` output.

3. **Real dependencies beat mocks for parser testing:** Using actual tree-sitter + tree_sitter_python to test parser_code.py covered 31 lines in one shot, where mock-based approaches in v4 only partially reached the AST chunk-building paths.

---

## Run v4: 2026-04-20 — autoresearch/ingest-coverage-v4-2026-04-20

**Scoring mode:** Numerical
**Starting score:** 75%
**Final score:** 85%
**Iterations:** 6 (1 baseline + 5 improvement, all kept)
**Tests added:** ~283 new tests (1548 → 1831)

---

## What improved

### Iteration 002 — Mock tests for CLI, embedding/impl, cross-document dedup (+1%)
- 24 tests: `test_cli_coverage.py`, `test_embedding_impl.py`, `test_cross_document_dedup.py`
- Lesson: per-file coverage % misleads — absolute missed statement count determines impact

### Iteration 003 — Broad mock tests across gc, migration, temporal, dedup (+3%)
- 102 tests across 7 files: gc CLI helpers, migration strategy executors, temporal paths, dedup_utils, common utils
- Confirmed that broad targeting (7+ files) outperforms deep targeting (1-2 files)

### Iteration 004 — Edge-case tests for parser_code, docling, temporal workflows (+3%)
- 78 tests: parser_code tree-sitter chunk paths, docling mock DocumentConverter, temporal workflow Phase 1/2
- Key challenge: temporalio.workflow sandbox attributes required direct setattr monkey-patching

### Iteration 005 — impl orchestrator, gc report, validation CLI, visual embedding (+2%)
- 57 tests: ingest_directory partial failure/cleanup, _emit_report text/JSON, validation CLI dispatch, visual embedding edge cases

### Iteration 006 — parser_code KG, colqwen, worker, activities (+1%)
- 46 tests: KG relationship extraction, ColQwen embed/query branches, run_worker() async, activity functions
- Reached 85% target

---

## Key learnings

1. **Diminishing returns above 80%:** Each percentage point requires ~49 statements. Remaining gaps are in deeply-branched code (orchestrator error paths, Temporal sandbox, GPU model branches).

2. **Temporal SDK testing:** `temporalio.workflow` sandbox attributes (`execute_activity`, `start_child_workflow`, `logger`, `info`) don't exist as normal module attrs under pytest. Use `setattr` before patching.

3. **Broad > deep:** Iterations targeting 4+ files consistently outperformed 1-2 file deep dives.

---

## Run v3: 2026-04-20 — autoresearch/ingest-coverage-2026-04-20

**Starting score:** 60%
**Final score:** 75%
**Iterations:** 5 (5 kept, 0 discarded)
**Tests added:** ~470 new tests (1121 → ~1548)

### Iteration 002 — Pure logic tests for support/document.py (+1%)
### Iteration 003 — Pure logic tests for temporal + minhash (+5%)
### Iteration 004 — Edge case tests for parsers + validation (+2%)
### Iteration 005 — Mock-based tests for vision + MinIO + markdown (+4%)
### Iteration 006 — Mock/edge tests for GC + migration + parser_code (+3%)
