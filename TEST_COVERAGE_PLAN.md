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

**Env:** Initiative worktree = `/home/kok-shew-juan/RagWeave/.worktrees/test-coverage`
on branch `test/coverage-initiative` (off develop @ 9106490). `python` not on PATH —
use `/home/kok-shew-juan/RagWeave/.venv/bin/python`. Do NOT `uv run` in the worktree
(spawns a stray empty `.venv`). **Git worktrees do NOT share tracked files** — each has
its own working copy; subagents must edit/mutate the file inside THIS worktree path and
run pytest from THIS worktree dir, or mutations hit the wrong checkout (learned in slice 1).

## Slice backlog (ranked; recon slice-1 "platform-validation" DROPPED — already covered by tests/test_validation.py)

| # | slice-id | target | layer | infra | size | status |
|---|----------|--------|-------|-------|------|--------|
| 1 | platform-command-runtime-unit | src/platform/command_runtime.py [104] | unit | none | S | ✅ 9f950ad (20 tests, 5 mutations bit) |
| 2 | platform-metrics-export | src/platform/metrics.py [80] | unit/contract | none | S | ✅ 99a7435 (23 tests, 4 mutations bit) |
| 3 | platform-timing-unit | src/platform/timing.py [221] | unit | none | S | ✅ 18d4894 (26 tests, 7 mutations bit) |
| 4 | platform-cli-log-formatting-unit | src/platform/cli_log_formatting.py [240] | unit | none | S | ✅ 77b0ff7 (31 tests, 6 mutations bit) |
| 5 | platform-cli-interactive-unit | src/platform/cli_interactive.py [322] | unit | none | M | ✅ 3a1ddff (29 tests, 7 mutations bit) |
| 6 | db-minio-store-mocked | src/db/minio/store.py [446] | mocked-integration | none (fake S3) | M | ✅ cda1912 (36 tests, 8 mutations bit) |
| 7 | db-backend-contract | src/db/backend.py + minio/backend.py | contract | none | M | ✅ 4b2131a (24 tests, 5 mutations bit) |
| 7b | db-facade-factory | src/db/__init__.py (_get_db_backend singleton + DATABASE_BACKEND dispatch + unknown-backend ValueError + facade arg forwarding) | unit/contract | none | S | ⏳ NEW (found in slice 7) |
| 8 | vector-db-backend-contract | src/vector_db/backend.py + weaviate/backend.py | contract | none | M | ✅ 1908893 (44 tests, 8 mutations bit) |
| 9 | server-api-bootstrap-unit | server/api.py | unit | none | M | ✅ a756a99 (15 tests, 8 mutations bit; wired tests/server→CI, +64 existing) |
| 10 | server-query-helpers-unit | server/routes/query.py [895] pure helpers | unit | none | M | ✅ 2d4ea1a (38 tests, 8 mutations bit) |
| 10b | server-query-endpoints | server/routes/query.py endpoints (run_query, /query, /query/stream, conv CRUD) via TestClient + fake deps | contract | none (mock Temporal/RAGChain) | L | ⏳ NEW (split from slice 10) |
| 11 | server-ingest-jobstore-unit | server/routes/ingest.py [578] JobRegistry+sweeper+builders | unit | none | M | ✅ e90b504 (33 tests, 7 mutations bit) |
| 11b | server-ingest-endpoints | ingest.py endpoints (upload/check-path/url/dir/jobs/stream/cancel) + _run_workflow via TestClient + Temporal mock | contract | none | L | ⏳ NEW (split from slice 11) |
| 12 | server-documents-routes-unit | server/routes/documents.py [365] | unit/contract | none | M | ⏳ |
| 13 | vector-db-weaviate-search-integration | src/vector_db/weaviate/store.py | real-integration | real-Weaviate | M | ⏳ |
| 14 | query-e2e-mocked | server query→workflow→activity | mocked-integration | Temporal | L | ⏳ |
| 15 | ingest-e2e-real-stack | ingest→MinIO→Weaviate→retrieve | real-integration | full stack | L | ⏳ |
| 12fe | console-web-component-unit | server/console/web/src/*.ts | frontend unit | none (vitest) | M | ✅ aabaf89 (60 tests, vitest+jsdom stood up, 8 mutations bit) |
| 17 | console-web-e2e | console against running backend | frontend e2e | browser+stack | L | ⏳ |

Statuses: ⏳ pending · 🚧 in progress · ✅ shipped · ⏸️ deferred

## Progress tally
- **13 slices COMPLETE (~430 new tests: ~370 Python + 60 TS), all mutation-proven.**
  Latest: slice 13 (src/common/llm: 68 tests across utils/fallback/stream/output/cache/memory).
  Combined offline gate **independently re-run by main: 2737 passed, 4 skipped, 0 failed**
  (slice 13 touched shared tests/conftest.py — langchain stub-vs-real boundary mgmt via
  pytest_collectstart + pytest_runtest_setup, order-independent; verified green).
  CI path list now adds tests/llm. Branch commits 9f950ad..2e12f55.
  - **FINDING (pre-existing, NOT fixed here):** the checked-in esbuild bundle
    `server/console/static/user-console.js(.map)` is STALE vs its TS source (source already
    has `ask_user_reason` clarification labels the bundle lacks). Slice-12's `npm run build`
    regenerated it; I reverted that from this branch to keep it additive/test-only.
    FOLLOW-UP: rebuild + commit the console bundle in a separate chore (CI rebuilds fresh, so
    runtime is unaffected, but the committed artifact is out of sync).
- **INFRA FEASIBILITY (probed):** docker available; a LIVE shared `rag-weaviate` (1.28.0,
  healthy) is up on host **:8090 (HTTP→8080)** and **:50051 (gRPC)**; a healthy TEI embeddings
  container is up too. So real-Weaviate integration is EXECUTABLE NOW. ⚠️ SHARED instance —
  integration tests MUST use a unique isolated collection name + guaranteed teardown (never
  touch shared collections). MinIO/Temporal containers were NOT visible in `docker ps` (full
  ingest→retrieve e2e via Temporal is therefore partial until those are up). add_documents
  takes pre-computed embeddings, so a real-Weaviate round-trip test can use synthetic 1024-dim
  vectors (no TEI dependency). Template: existing `tests/vector_db/test_collection_selection.py`
  (integration-marked) for the live-connect approach.
- **NEXT CHAPTER = the goal's explicit e2e / real-DB / real-frontend pillars** (best started
  fresh, not at the tail of a long context): stand up the real stack (docker-compose:
  Weaviate+MinIO+Temporal) and write/execute slices 13(real-Weaviate search), 14(query e2e via
  Temporal), 15(ingest→MinIO→Weaviate→retrieve e2e), 17(console e2e browser+stack). Write these
  as dual-marked `slow+integration` tests (deselected offline) and confirm they run against a
  live stack. Remaining infra-FREE leftovers: 7b db-facade-factory, 10b/11b/12 route endpoints
  via TestClient, parallel.py/batch.py/graph, guardrails fill.
- **12 slices COMPLETE (~360 new tests: ~300 Python + 60 TS), all mutation-proven.**
  Branch `test/coverage-initiative` (off develop @ 9106490), commits 9f950ad..aabaf89.
  Python offline CI gate: **2661 passed**. Frontend: vitest stood up, 60 tests, tsc+build green.
  CI path list now: tests/platform, tests/db, tests/vector_db, tests/server (+ existing roots);
  console step runs `npm test`. NOT pushed / no PR — accumulating on the branch.
  - Done: platform(1-5), minio store(6), db backend(7), vector_db backend(8),
    server api.py(9), query helpers(10), ingest JobRegistry(11), frontend vitest(12fe).
- **REMAINING WORK (sequenced):**
  - Infra-FREE (land in offline CI): 7b db-facade-factory; 10b query endpoints (TestClient+mock
    Temporal/RAGChain); 11b ingest endpoints (TestClient+Temporal mock); documents.py routes+helpers;
    admin.py/system.py routes; **src/common/ llm stack (1818 lines, NONE coverage — biggest remaining
    Python gap: provider/parallel/memory/cache/output/stream/fallback)**; guardrails (PARTIAL→fill).
  - Infra-DEPENDENT (the goal's explicit "real db / e2e / real frontend" pillars — write as
    dual-marked `pytest.mark.slow + integration` tests per [[feedback_dual_marker_gating]], deselected
    offline; EXECUTION needs a stack-up step): 13 real-Weaviate search; 14 query e2e (Temporal);
    15 ingest→MinIO→Weaviate→retrieve e2e; 17 console e2e (browser+stack). docker-compose.yml exists;
    bringing the real stack up headless is the open feasibility question — approach fresh, not at
    the tail of a long context.
- **Slices 1–8 COMPLETE.** ~233 new tests, ~51 mutations proven. Branch
  `test/coverage-initiative` (off develop @ 9106490). Offline CI gate after slice 8:
  **2511 passed, 4 skipped, 12 deselected**. CI path list now includes tests/platform,
  tests/db, tests/vector_db. NOT pushed / no PR — accumulating; surface at a checkpoint.
  - db tier (6–7): minio store 36 + db backend contract 24. vector_db (8): 44.
  - Weaviate Filter isolation hazard: `tests/conftest.py` installs a stub
    `weaviate.classes.query.Filter` only when weaviate isn't already in sys.modules; the
    real Filter's introspection attrs aren't reliable in full-suite runs. Slice 8 used a
    recording-fake Filter (monkeypatch `sys.modules["weaviate.classes.query"].Filter`) for
    isolation-stable teeth. Reuse this pattern for any weaviate-Filter-touching test.
- **Platform tier (slices 1–5): 129 tests added across tests/platform/**
  (command_runtime 20, metrics 23, timing 26, cli_log_formatting 31, cli_interactive 29),
  30 mutations proven to bite. tests/platform wired into offline CI. Branch
  `test/coverage-initiative` (off develop @ 9106490): 9f950ad → 99a7435 → 18d4894 →
  77b0ff7 → 3a1ddff. Offline CI gate after slice 5: **2374 passed, 4 skipped, 10 deselected**.
  NOT pushed / no PR yet — accumulating on the branch; surface for review at a checkpoint.

## Lessons / next-moves (append per cycle)
- (recon) `tests/platform/` did not exist; platform tests lived at `tests/` root.
  Created `tests/platform/` and added it to the CI path list (slice 1).
- **Git worktrees do NOT share tracked files.** Slice-1 mutation spot-check edited the
  PRIMARY tree's source and saw NO red — because the worktree has its own checkout. Always
  edit/mutate + run pytest inside the SAME worktree (`.worktrees/test-coverage`).
- **Verification rhythm that works:** dispatch author (writes tests + internal ralph
  mutation loop, returns mutation table) → main runs the new file + `git diff --stat`
  (byte-identical) + ONE fresh independent mutation spot-check in the worktree → revert by
  inverse-edit → commit → ledger. 5/5 author mutation tables confirmed by spot-check.
- **Equivalent mutants are real.** Slice 5: the spec'd DOWN-clamp mutation
  (`min(len-1,sel+1)`→`sel+1`) is unkillable because a downstream re-clamp subsumes it.
  Author correctly substituted a load-bearing mutation. Don't force teeth on equivalent mutants.
- Author subagents must be handed the FULL source inline (or told to read the exact worktree
  path) + env facts + a concrete mutation list. This gives precise, high-teeth tests fast.
- **Latent weak test found in EXISTING suite (slice 6):** `tests/test_document_management_backend.py`
  builds `S3Error` POSITIONALLY (`S3Error("NoSuchKey", ...)`), but the real signature is
  `S3Error(response, code, message, resource, request_id, host_id, ...)` — so "NoSuchKey" lands
  in `response` and `.code` becomes "not found". Those tests may not exercise the code-branch
  they intend. FOLLOW-UP: audit + fix (build S3Error with keywords). Slice 6 built errors with kwargs.
- db tier: slice 6 done. NEXT: slice 7 (db backend ABC contract), slice 8 (vector_db backend
  contract), then server routes (9–12 — query/ingest/documents, the largest untested product
  surface), then real-infra (13–15) + frontend (16–17). Offline CI after slice 6: 2410 passed.
