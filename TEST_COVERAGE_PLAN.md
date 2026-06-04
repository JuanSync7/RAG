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
| 7b | db-facade-factory | src/db/__init__.py (_get_db_backend singleton + DATABASE_BACKEND dispatch + unknown-backend ValueError + facade arg forwarding) | unit/contract | none | S | ✅ dfed02b (19 tests, 5+1 mutations bit) |
| 8 | vector-db-backend-contract | src/vector_db/backend.py + weaviate/backend.py | contract | none | M | ✅ 1908893 (44 tests, 8 mutations bit) |
| 9 | server-api-bootstrap-unit | server/api.py | unit | none | M | ✅ a756a99 (15 tests, 8 mutations bit; wired tests/server→CI, +64 existing) |
| 10 | server-query-helpers-unit | server/routes/query.py [895] pure helpers | unit | none | M | ✅ 2d4ea1a (38 tests, 8 mutations bit) |
| 10b | server-query-endpoints | server/routes/query.py endpoints (run_query, /query, /query/stream, conv CRUD) via TestClient + fake deps | contract | none (mock Temporal/RAGChain) | L | ✅ 375f286 (30 tests, 11+1 mutations bit; caught equiv mutant) |
| 11 | server-ingest-jobstore-unit | server/routes/ingest.py [578] JobRegistry+sweeper+builders | unit | none | M | ✅ e90b504 (33 tests, 7 mutations bit) |
| 11b | server-ingest-endpoints | ingest.py endpoints (upload/check-path/url/dir/jobs/stream/cancel) + _run_workflow via TestClient + Temporal mock | contract | none | L | ✅ b741823 (41 tests, 12+1 mutations bit) |
| 12 | server-documents-routes-unit | server/routes/documents.py [365] | unit/contract | none | M | ✅ 2c79524 (24 tests via TestClient, 9+1 mutations bit) |
| 14rt | vector-db-weaviate-real-integration | src/vector_db/weaviate/store.py | real-integration | LIVE Weaviate ✅ | M | ✅ f1bdeac (4 live tests PASSED; isolated+clean) |
| 14 | query-e2e-mocked | server query→workflow→activity | mocked-integration | Temporal | L | ⏳ |
| 15 | ingest-e2e-real-stack | ingest→MinIO→Weaviate→retrieve | real-integration | full stack | L | ⏳ |
| 13b | llm-concurrency-helpers | src/common/llm/batch.py + parallel.py | unit | none | M | ✅ 7b52c00 (15 tests, 10+1 mutations bit; FOUND latent Runnable bug) |
| CIG | ci-gate-extended-suites | .github/workflows/ci.yml (separate step) + 3 colang quarantines | ci/infra | none | M | ✅ 2516eaa (gated ~1000 existing tests: retrieval/guardrails/core/contracts/observability) |
| GG | guardrails-granite-guardian | src/guardrails/models/granite_guardian.py | unit | none (fake httpx) | M | ✅ 1b6ba66 (23 tests, 10+1 mutations bit) |
| CS | console-services-unit | server/console/services.py [386] | unit | none | M | ✅ d8a4782 (46 tests, 12+1 mutations bit; 3 security guards) |
| TR | reliability-temporal-retry | src/platform/reliability/temporal_retry.py [134] | unit | none (fake Client) | S | ✅ 39cb336 (12 tests, 8+1 mutations bit) |
| GP | guardrails-gliner-pii | src/guardrails/shared/gliner_pii.py [184] | unit | none (fake model) | S | ✅ 084e98f (17 tests, 10+1 mutations bit; gliner not installed → ImportError path) |
| LR | reliability-local-retry | src/platform/reliability/local_retry.py [68] | unit | none | S | ✅ a348d68 (7 tests, mutations bit; sleep mocked) |
| MG | guardrails-merge-gate | src/guardrails/common/merge_gate.py [103] | unit | none | S | ✅ a348d68 (8 tests, priority-order teeth) |
| SC | guardrails-self-check | src/guardrails/models/self_check.py [159] | unit | none (mock call_oneshot) | S | ✅ f100161 (17 tests, 12+1 mutations bit) |
| — | ingest-scorer-v2/v3/v5 | src/ingest/scorer_v2.py/v3/v5 | — | — | — | ⛔ DEAD CODE — zero repo-wide refs; candidate for DELETION, not testing |
| 9b | server-admin-system-routes | server/routes/admin.py + system.py | contract | none | M | ✅ 4a2b80a (28 tests, 10+1 mutations bit) |
| 12fe | console-web-component-unit | server/console/web/src/*.ts | frontend unit | none (vitest) | M | ✅ aabaf89 (60 tests, vitest+jsdom stood up, 8 mutations bit) |
| 17 | console-web-e2e | console against running backend | frontend e2e | browser+stack | L | ⏳ |

Statuses: ⏳ pending · 🚧 in progress · ✅ shipped · ⏸️ deferred

## SESSION 2 — INFRA-FREE PILLAR COMPLETE (continued 2026-06-04 while awaiting e2e stack)
- **While blocked on the e2e stack (user bringing up MinIO+Temporal), squeezed the remaining
  infra-free surface:** gliner_pii (GP, 17t), local_retry+merge_gate (LR/MG, 15t), self_check
  (SC, 17t). Verified `scorer_v2/v3/v5` are DEAD (zero repo refs) — NOT tested (deletion candidates).
- **Infra-free surface is now EXHAUSTED:** the only untested ≥100-line src/server modules left are
  the dead scorer_v* files. Every live module with non-trivial logic has mutation-proven tests.
- **Session-2 total: 13 work items, ~287 new tests + ~1000 newly-gated.** Commits 2c79524..f100161.
  Extended gate: 1066 passed. Primary gate: 2959 passed. Frontend: 60.
- **REMAINING = ONLY the infra-dependent e2e pillars** (Temporal ingest→serve, browser console),
  blocked on stack/browser not available in this env. Watcher armed for instant resume on stack-up.

## SESSION 2 (2026-06-03) — infra-free pillar driven to close-to-complete
- **10 work items: ~238 new tests + ~1000 existing tests newly GATED.**
  Slices: documents routes (12, 24t), db facade (7b, 19t), query endpoints (10b, 30t),
  ingest endpoints (11b, 41t), llm concurrency batch+parallel (13b, 15t), admin+system
  routes (9b, 28t), **CI-gating fix (CIG)**, granite_guardian (GG, 23t), console services
  (CS, 46t, 3 security guards), temporal_retry (TR, 12t). Commits 2c79524..39cb336.
- **Primary CI gate: 2952 passed.** Extended-suites gate: 1023 passed. Frontend: 60.
- **2 latent product findings (test-pinned, fixes deferred):** parallel.py `_run_langchain`
  Runnable bug ([[project_parallel_runnable_bug]]); console static-asset guard uses str.startswith
  not Path.is_relative_to (sharp edge, not exploitable as built — noted in CS commit).
- **ENTIRE server/routes/ surface now covered** (query/ingest/documents/admin/system).
- **Remaining infra-FREE leftovers are LOW-VALUE:** gliner_pii.py (needs GLiNER ML model — partial
  only), scorer_v2/v3/v5.py (appear DEAD — only self-referencing; verify liveness before investing),
  plus assorted <120-line modules. Diminishing returns.
- **STILL OPEN = the goal's INFRA-DEPENDENT e2e pillars, BLOCKED in this env:**
  (1) full ingest→serve e2e via Temporal — MinIO + Temporal containers NOT up here;
  (2) browser-based console e2e — needs a browser + the running full stack.
  Real-DB pillar (live Weaviate) is DONE (slice 14rt). These two need the stack stood up
  (docker-compose: MinIO+Temporal) and a browser; cannot be EXECUTED/validated in this environment.
  Writing un-runnable e2e here would violate honest-CI-gating (synthetic-green is a yellow flag).

## CI-gating slice (2026-06-03) — closed a major protective-coverage gap
- **Finding:** the real `ci.yml` "Run tests" step gated only ~12 test paths; `tests/retrieval`
  (568), `tests/core` (6), `tests/contracts` (19), `tests/observability` (315), `tests/guardrails`
  (92) — **~1000 existing green tests — were NOT gated by CI** (regressions there went uncaught).
- **Action (user-approved "wire green dirs + keep authoring"):** added a SEPARATE ci.yml step
  `Run tests (extended suites)` running those 5 dirs in their OWN pytest invocation.
- **Why a separate step, NOT merged into the primary command:** merging them caused **115 failures**
  in `tests/ingest/test_visual_embedding_node.py` — a cross-area sys.modules pollution. Those suites
  stub heavy modules (numpy/PIL/docling/langchain) at COLLECTION time (collection is global per
  session), so `_extract_page_images` hit `unhandled error: __array_interface__` (a stubbed
  numpy/PIL leaking into ingest's real-array path). A separate process = isolated sys.modules = clean.
  THIS is why the dirs were excluded originally; the separate-step design gates them without the clash.
- **Quarantined (tracked skips):** the 3 `tests/guardrails` Colang tests (`test_colang_syntax.py`
  both, `test_colang_flows.py::test_all_co_files_parse`) import nemoguardrails which needs REAL
  `langchain_core`; the suite-wide langchain stub (tests/conftest.py) shadows it. Skip reason points
  at the stub boundary; re-enable via a real-langchain collection boundary like tests/llm.
- Extended-suites step verified green: **1000 passed, 6 skipped, 0 failed**. Primary command unchanged.

## Progress tally
- **14 slices COMPLETE (~434 new tests, all mutation-proven OR live-verified).**
  PILLAR STATUS vs goal: unit/contract foundation ✅ (platform/db/vector_db/server/common-llm);
  real frontend ✅ (vitest stood up + 60 component tests); **real DB ✅ (slice 14rt — live
  Weaviate round-trip executed & verified clean)**. STILL OPEN: full ingest→serve e2e via
  Temporal (MinIO+Temporal containers not up in this env) and browser-based console e2e
  (needs browser+stack). Branch commits 9f950ad..f1bdeac.
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
