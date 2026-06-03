# Test-Coverage Initiative — Vertical-Slice Ledger

**Goal:** Drive RagWeave's test suite toward completeness — unit coverage, true
end-to-end, and real-DB / real-frontend system integration. Sequential
subagent-driven vertical slices; each slice has a *validatable end goal*, runs a
TDD + ralph (mutation-teeth) loop, and must leave the offline CI command green.

**Methodology (per slice):**
- Author subagent reads the REAL source (no hallucinated signatures), writes
  comprehensive tests (happy + error + boundary + edge), runs them green, then
  runs a ralph/mutation loop: mutate each key behavior, confirm a test reds,
  revert by **inverse-edit only** (never `git checkout`/`restore`/`stash` a
  tracked file), and reports which mutations bit.
- Higher-risk slices (server routes, db store, real-DB, e2e, frontend) get a
  second **adversarial verifier** subagent (independent mutation + gaming/assertion
  -quality audit). Main commits author work BEFORE dispatching the verifier.
- Main runs the full offline CI command on the new tip before declaring a slice done.

**Offline CI command (the real green signal):**
```
/home/kok-shew-juan/RagWeave/.venv/bin/python -m pytest -m "not slow and not integration" \
  tests/ingest tests/test_rate_limiter.py tests/test_cache_provider.py \
  tests/test_security_auth.py tests/test_api_key_store.py tests/test_quota_store.py \
  tests/test_query_processor.py tests/test_generator.py tests/test_validation.py \
  tests/test_query_filters.py tests/test_rag_chain_budget.py \
  tests/test_rag_chain_integration.py tests/test_api_schemas.py tests/test_mcp_adapter.py \
  tests/test_api_error_envelope.py
```
New test dirs/files added by a slice must also be added to the CI path list (or a
new scoped job) so they actually gate. `python -m src.eval.smoke` is the eval gate.

**Env:** `python` not on PATH. Use `/home/kok-shew-juan/RagWeave/.venv/bin/python`.
Do NOT `uv run` in the worktree (spawns a stray empty `.venv`).

## Slice backlog (ranked; recon slice-1 "platform-validation" DROPPED — already covered by tests/test_validation.py)

| # | slice-id | target | layer | infra | size | status |
|---|----------|--------|-------|-------|------|--------|
| 1 | platform-command-runtime-unit | src/platform/command_runtime.py [104] | unit | none | S | 🚧 in progress |
| 2 | platform-metrics-export | src/platform/metrics.py [80] | unit/contract | none | S | ⏳ |
| 3 | platform-timing-unit | src/platform/timing.py [221] | unit | none | S | ⏳ |
| 4 | platform-cli-log-formatting-unit | src/platform/cli_log_formatting.py [240] | unit | none | S | ⏳ |
| 5 | platform-cli-interactive-unit | src/platform/cli_interactive.py [322] | unit | none | M | ⏳ |
| 6 | db-minio-store-mocked | src/db/minio/store.py [446] | mocked-integration | none (fake S3) | M | ⏳ |
| 7 | db-backend-contract | src/db/backend.py + minio/backend.py | contract | none | M | ⏳ |
| 8 | vector-db-backend-contract | src/vector_db/backend.py | contract | none | M | ⏳ |
| 9 | server-api-bootstrap-unit | server/api.py | unit | none | M | ⏳ |
| 10 | server-query-routes-unit | server/routes/query.py [895] | unit/contract | none (mock Temporal) | M | ⏳ |
| 11 | server-ingest-routes-unit | server/routes/ingest.py [578] | unit/contract | none | M | ⏳ |
| 12 | server-documents-routes-unit | server/routes/documents.py [365] | unit/contract | none | M | ⏳ |
| 13 | vector-db-weaviate-search-integration | src/vector_db/weaviate/store.py | real-integration | real-Weaviate | M | ⏳ |
| 14 | query-e2e-mocked | server query→workflow→activity | mocked-integration | Temporal | L | ⏳ |
| 15 | ingest-e2e-real-stack | ingest→MinIO→Weaviate→retrieve | real-integration | full stack | L | ⏳ |
| 16 | console-web-component-unit | server/console/web/src/*.ts | frontend unit | none (vitest) | M | ⏳ |
| 17 | console-web-e2e | console against running backend | frontend e2e | browser+stack | L | ⏳ |

Statuses: ⏳ pending · 🚧 in progress · ✅ shipped · ⏸️ deferred

## Lessons / next-moves (append per cycle)
- (recon) `tests/platform/` does not exist yet; platform module tests currently
  live at `tests/` root (e.g. test_validation.py). Decision: create `tests/platform/`
  for the new platform slices and add it to the CI path list.
