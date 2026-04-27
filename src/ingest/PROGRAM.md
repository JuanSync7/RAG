# Ingestion Pipeline Test Coverage — Auto-Research Program (v5)

## Objective

Increase test coverage of `src/ingest/` from 96% to 98% by writing new tests (unit, mock, and integration). Each iteration adds tests targeting the remaining uncovered code paths. Scorer excludes scorer_*.py tooling files from measurement.

Baseline: 96% coverage (1831 tests passing, ~4331 production statements, ~183 missed).

## Scoring mode

Numerical

## Metric

Score = integer coverage percentage reported by scorer_v5.py (excludes scorer tooling files).
Higher is better. Target: 98%.

## Scoring mechanism

`uv run python src/ingest/scorer_v5.py --target 98` from the project root.

### Correctness guard

1. All existing tests must pass (`pytest tests/ingest/ -x -q --tb=short`).
2. Mock naming convention: any new test function whose name contains "mock" must start with `test_mock_`. Pre-existing violations are grandfathered.

## Mutable files

Only test files may be modified or created:

### Existing test files (modify to add tests)
- `tests/ingest/test_clean_store.py`
- `tests/ingest/test_cli_args.py`
- `tests/ingest/test_ingest_cli.py`
- `tests/ingest/test_shared_helpers.py`
- `tests/ingest/test_orchestrator.py`
- `tests/ingest/test_parser_integration.py`
- `tests/ingest/test_dedup_revert.py`
- `tests/ingest/test_dedup_workflow_integration.py`
- `tests/ingest/test_visual_embedding_node.py`
- `tests/ingest/test_visual_embedding_colqwen.py`
- `tests/ingest/test_visual_embedding_config_state.py`
- `tests/ingest/test_visual_embedding_minio.py`
- `tests/ingest/test_visual_embedding_docling.py`
- `tests/ingest/test_visual_store.py`
- `tests/ingest/test_document_support.py`
- `tests/ingest/test_markdown_support.py`
- `tests/ingest/test_parser_code.py`
- `tests/ingest/test_parser_text.py`
- `tests/ingest/test_parser_registry.py`
- `tests/ingest/test_vision_support.py`
- `tests/ingest/test_minhash_engine.py`
- `tests/ingest/test_minio_clean_store.py`
- `tests/ingest/test_temporal_constants.py`
- `tests/ingest/test_temporal_worker.py`
- `tests/ingest/test_temporal_workflows.py`
- `tests/ingest/test_temporal_activities.py`
- `tests/ingest/test_impl_coverage.py`
- `tests/ingest/test_cli_coverage.py`
- `tests/ingest/test_embedding_impl.py`
- `tests/ingest/test_cross_document_dedup.py`
- `tests/ingest/test_dedup_utils.py`
- `tests/ingest/test_common_utils.py`
- `tests/ingest/test_common_shared.py`
- `tests/ingest/test_colqwen_support.py`
- `tests/ingest/test_docling_support.py`
- `tests/ingest/test_visual_embedding_coverage.py`
- `tests/ingest/test_structure_detection_extra.py`
- `tests/ingest/test_parser_code_coverage.py`
- `tests/ingest/doc_processing/test_structure_detection_coverage.py`
- `tests/ingest/embedding/test_chunk_enrichment.py`
- `tests/ingest/embedding/test_chunking.py`
- `tests/ingest/embedding/test_cross_reference.py`
- `tests/ingest/embedding/test_document_storage.py`
- `tests/ingest/embedding/test_embedding_storage.py`
- `tests/ingest/embedding/test_quality_validation.py`
- `tests/ingest/lifecycle/test_gc_engine.py`
- `tests/ingest/lifecycle/test_migration.py`
- `tests/ingest/lifecycle/test_validation.py`
- `tests/ingest/lifecycle/test_changelog.py`
- `tests/ingest/lifecycle/test_sync_engine.py`
- `tests/ingest/lifecycle/test_orphan_report.py`

### New test files (create as needed)
- `tests/ingest/test_vlm_enrichment_coverage.py` — for `embedding/nodes/vlm_enrichment.py`
- `tests/ingest/test_sync_coverage.py` — for `lifecycle/sync.py`
- `tests/ingest/test_lifecycle_init_coverage.py` — for `lifecycle/__init__.py`
- `tests/ingest/test_parser_base_coverage.py` — for `support/parser_base.py`
- `tests/ingest/test_orphan_report_coverage.py` — for `lifecycle/orphan_report.py`

### Existing conftest files (modify to add shared fixtures)
- `tests/ingest/conftest.py`
- `tests/ingest/lifecycle/conftest.py`

## Immutable files (DO NOT MODIFY)

- `src/ingest/PROGRAM.md` — this file
- `src/ingest/scorer_v5.py` — the scoring script
- `src/ingest/scorer_v3.py` — previous scoring script
- `src/ingest/scorer_v2.py` — previous scoring script
- `src/ingest/scorer.py` — v1 scoring script
- All source files under `src/ingest/` — this run adds tests only, never changes production code
- All existing test files NOT listed in the mutable set above

## Exploration directions

Remaining gaps by missed statements (183 total across production code):

### 1. parser_code.py deep coverage (31 missed — biggest single file)
Lines 110-111, 179-280, 383-384. Targets: `_extract_kg_relationships`, `_walk_calls`, multi-language chunk() paths (Rust/Go imports), tree-sitter grammar fallback.

### 2. lifecycle/migration.py remaining paths (29 missed)
Lines 300, 302, 389-390, 422-424, 539, 553-555, 593-599, 619-621, 629-630, 638-640, 644-649, 718. Targets: `_migrate_full_phase2()`, `_migrate_kg_reextract()`, CLI helpers, `_emit_migration_report()`.

### 3. lifecycle/gc.py remaining paths (19 missed)
Lines 290, 400-408, 525, 565-567, 611-613, 632-634, 642-643, 654, 691. Targets: remaining `run_gc_cli()` paths, `_open_*_client` helpers, manifest/report edge cases.

### 4. impl.py orchestrator edge cases (14 missed)
Lines 258, 261, 266, 492, 498, 712, 718, 735, 781-787, 801, 822, 876. Targets: error handling in Phase 1/2 pipelines, skip logic, design warnings.

### 5. lifecycle/validation.py remaining paths (10 missed)
Lines 142, 432, 441-457, 467, 477, 533. Targets: CLI helpers, `_emit_validation_report()`.

### 6. structure_detection.py uncovered block (9 missed)
Lines 100-118. Targets: structure detection fallback paths.

### 7. Broad sweep of small gaps (8+ files, 1-8 missed each)
- `common/shared.py` (8 missed: 133, 156-157, 170, 212, 217-219)
- `lifecycle/sync.py` (8 missed: 98-100, 106-108, 183-187)
- `embedding/nodes/vlm_enrichment.py` (7 missed: 200, 211, 219, 225, 249, 265-271)
- `support/docling.py` (7 missed: 84-85, 245, 249-251, 354)
- `embedding/nodes/chunking.py` (5 missed: 117-125)
- `support/colqwen.py` (4 missed: 117-118, 223, 320)
- `support/parser_base.py` (4 missed: 148-151)
- `lifecycle/orphan_report.py` (4 missed: 89-92)
- `common/clean_store.py` (3 missed: 93-95)
- `common/utils.py` (3 missed: 86-87, 125)
- `embedding/nodes/quality_validation.py` (3 missed: 55-57)
- `embedding/support/minhash_engine.py` (3 missed: 181-187)
- `doc_processing/impl.py` (2 missed: 56-76)
- `lifecycle/changelog.py` (2 missed: 235, 248)
- `lifecycle/__init__.py` (2 missed: 95, 128)
- `temporal/worker.py` (2 missed: 120, 213)
- Various 1-line gaps: cli.py:323, markdown.py:201, parser_registry.py:97, clean_store.py:1

## Constraints

- **Never modify production source code** — only test files are mutable
- **Mock naming convention** — any test function using mocks whose name contains "mock" must start with `test_mock_`
- **No new external dependencies** — use only `pytest`, `pytest-cov`, `unittest.mock`, and existing project deps
- **Test isolation** — new tests must not depend on external services being running
- **Backward compatibility** — existing 1831 tests must continue to pass at all times
- **Test types allowed** — unit tests, mock-based tests, and integration tests (no live services)

## Stop conditions

- Coverage reaches 98% — success, stop
- 5 consecutive iterations with no coverage improvement — stop and report
- Correctness guard fails 3× in a row — stop, infrastructure broken
- 20 iterations reached — stop and report regardless of score
