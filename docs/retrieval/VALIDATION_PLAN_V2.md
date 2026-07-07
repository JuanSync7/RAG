<!-- @summary
Deep validation plan for the Sanity_testcase_v2 re-ingest: nav/boilerplate role
filtering, table de-duplication, HyDE quality, deep research on broad queries,
faithfulness rail on the instruct model, and multi-turn queries. Grounds every
test in the real dev topology and lists the concrete queries to run.
@end-summary -->

# RagWeave `_v2` Corpus — Deep Validation Plan

Status: **drafted while the `Sanity_testcase_v2` re-ingest runs** (dependency:
execution starts once ingest + role-backfill + cutover complete).

## 0. Dev topology (verified live, 2026-07-01)

| Alias / role | Model | Endpoint (dev tunnel → ai01) | Notes |
| --- | --- | --- | --- |
| `default` / `query` | `qwopus-9b` | `:18001` → ai01:8001 | reasoning/gen model |
| `judge` / **instruct** | `qwen2.5-7b-judge` | `:18005` → ai01:8005 | the "instruct" model the user means |
| embed | `qwen3-embed-4b` (2560-dim) | `:18002` → ai01:8002 | |
| cross-encoder rerank | `qwen3-reranker` | ai01 | measured net-negative vs judge (see memory) |

Live dev flags: `RAG_AGENTIC_RETRIEVAL_ENABLED=true`, `RAG_AGENTIC_CONTROLLER_MODEL_ALIAS=judge`,
`RAG_AGENTIC_LLM_JSON_MODE=true`, collection still `Sanity_testcase` (cutover to `_v2` pending).
Role filter dormant (`RAG_RETRIEVAL_ROLE_SCHEMA_PRESENT=false`); temporary heuristic
(`RAG_RERANK_DROP_NAVIGATIONAL=true`) is what filters nav today.

API: `POST http://localhost:8102/query` on ai03 (Cloudflare-fronted). **Probe with
Python `urllib` only — never curl/wget (Falcon SIGKILLs them).**

## 0b. CORPUS CORRECTION (2026-07-02)

The first `_v2` run was pointed at the WRONG directory — `documents/scratch_import`
(the **full corpus**, 1238 files → 856 docs / 83,482 chunks) instead of the reduced
**sanity** set. The sanity testbed is `documents/sanity_testcase/` (**116 files →
~99 ingestable docs**; old `Sanity_testcase` = 98 docs / 12,197 chunks). Fix applied:
`--fresh` re-ingest of `documents/sanity_testcase` into `Sanity_testcase_v2` (wipes the
bloated collection). Live `Sanity_testcase` untouched throughout.

## 1. Blocking prerequisite — chunk_role must be populated

The corrected re-ingest runs with `RAG_INGESTION_NAV_CLASSIFY=false` (fast), so `_v2`
lands with **no `chunk_role` metadata**. We deliberately populate it via **backfill**
(rather than inline nav-classify) — this both fills the roles AND exercises the
backfill tool end-to-end, the exact mechanism the full corpus / prod will need later.
Order:

1. Let the fast ingest finish (docling + embedding preserved).
2. Run **role-backfill** (`backfill_roles_from_corpus`, in-place, no re-embed, fail-open
   to `content`) over `Sanity_testcase_v2` → tags every chunk `content|navigation|boilerplate`.
3. Cutover dev to `_v2`; flip `RAG_RETRIEVAL_ROLE_SCHEMA_PRESENT=true`; disable
   `RAG_TABLE_GROUP_DEDUP_ENABLED` (HEAD tables are single-rep).
4. Verify role tally is sane (e.g. boilerplate/navigation a small minority) before trusting the filter.

## 2. Enabling code changes (corpus-independent, done up front)

| # | Change | Why |
| --- | --- | --- |
| A | Export `tried_hyde` in `AgenticResult.telemetry()` → response metadata | can't judge HyDE output that isn't observable |
| B | `RAG_DEEP_RESEARCH_MODEL_ALIAS` (default `default`) → thread into `DeepResearch` | point deep research at instruct |
| C | `RAG_FAITHFULNESS_MODEL_ALIAS` (default `judge`) → thread through `FaithfulnessChecker` → `call_oneshot` | rail on instruct + existing system prompt |
| D | role-backfill runner + urllib query harness | the workhorses of the loop |

Each ships with a unit test and config validation; none touches prod (main branch).

## 3. Test matrix

Corpus themes: AMBA CHI spec, AXI, DFT FLOWS.pdf, MBIST, IHI0050H, Vitec/G60 SoC,
DFT checklists (xlsx).

### 3.1 Nav / boilerplate role filtering
Goal: nav+boilerplate excluded **unless useful to the query**.
- **Exclusion**: queries whose pre-filter top hits were historically ToC/front-matter
  (generic CHI queries). Assert no retrieved chunk has `chunk_role in {navigation, boilerplate}`.
- **"Unless useful" escape** (the risk of a blanket hard-filter): structural/meta queries —
  "which chapter covers coherency?", "where is the AXI protocol described?". Does dropping
  nav lose the answer? If yes → add a query-aware escape (keep nav only when it out-scores
  content, i.e. Option B score-penalty — the generic fix, not a per-query hack).
- **Regression**: MBIST "steps before inserting MBIST for a block" (known ground truth in
  DFT FLOWS.pdf) — must still rank the answer chunk highly with roles filtered.

### 3.2 Tables
- **No duplicate**: query the formerly-duplicated Table C-3; assert no two results share
  `(table_group_id, table_row_block_index)`. Repeat for a DFT-checklist xlsx table and a
  wide multi-block table.
- **Answerable**: a lone table block still carries header+caption+breadcrumb; `table_markdown`
  present on ≥1 hit per group. Ask a value-lookup question and check the answer is grounded.

### 3.3 HyDE quality (agentic controller = instruct)
- Capture `tried_hyde` per query. Judge: is the hypothetical on-topic, specific, plausible
  (answer-space) vs empty/generic? `hyde_failures` must be 0 (nonzero = controller mis-provisioned).
- **Factoid** ("how many channels does AXI have?") should converge round 1.
- **Broad** ("describe DFT flow principles") should iterate, judge naming gaps for round 2 HyDE.
- Decide if HyDE needs a better prompt / more variants.

### 3.4 Deep research on broad queries (instruct model)
- `deep_research=true` with `RAG_DEEP_RESEARCH_MODEL_ALIAS=judge`.
- Queries: "compare CHI vs AXI", "summarize the DFT flow", "memory interface, bus protocol,
  and power specs of the G60 SoC". Observe topic decomposition, per-topic sub-queries,
  sufficiency checks, final synthesis. Compare instruct vs qwopus (latency, decomposition quality).

### 3.5 Faithfulness rail (instruct + system prompt)
- Grounded query (answer well-supported) → expect high faithfulness, no flag.
- Thin-context query (answer under-supported) → expect low score / flag.
- Compare instruct (`judge`) vs qwopus scoring on the same answers — is instruct as reliable + faster?

### 3.6 Multi-turn
- T1 "What is the AXI protocol?" → T2 "How many channels does it have?" (pronoun).
- T1 "Explain MBIST insertion." → T2 "What about the steps before that?" (elliptical).
- T1 "Compare CHI and AXI." → T2 "Which is better for coherency?" (referent + compare).
- Same `conversation_id`; assert T2 resolves the referent (retrieval + answer). Test both
  regimes: memory-into-generation (default) and retrieval auto-rewrite.

## 3b. Cutover runbook (exact steps — execute when ingest completes)

ai03 = `ssh 172.28.22.76` (tcsh — pipe bash via `'bash -s' < file`). `_v2` collections:
`Sanity_testcase_v2` (chunks) present; `_v2_cards` NOT built (card emission skipped —
routing is off, so fine; create an empty `_v2_cards` at cutover so a stray card read
can't 404). `chunk_role` property is declared in the store schema at collection
create, so it exists on `_v2` (null-valued) — backfill only populates it.

Both `/vols` and ai03 `ragweave-develop` are at commit `f1cea688` (clean) → **file-copy
deploy is safe** (no divergence). `recreate_dev.sh` recreates api+worker from the saved
CreateCommands + env-files (`/tmp/dev-{api,worker}.env`).

1. **Confirm done**: `rag-ingest-dev` Exited(0); tail of `/var/tmp/Juan.Kok/reingest_v2_sanity.log`
   shows `Ingestion complete. processed=~98`. Record `Sanity_testcase_v2` object count (~12k expected).
2. **Deploy code** — from `/vols`, tar the 7 files and extract into ai03 `/work` in place
   (preserve dir inode): `tar czf - -C <repo> config/settings.py src/retrieval/pipeline/rag_chain.py
   src/retrieval/pipeline/agentic/state.py src/retrieval/pipeline/agentic/orchestrator.py
   src/guardrails/shared/faithfulness.py src/ingest/embedding/common/role_backfill.py
   scripts/validate_v2.py | ssh ai03 tar xzf - -C /var/tmp/Juan.Kok/ragweave-develop`.
   (No restart needed yet — backfill runs via `podman exec` which re-imports; api/worker pick
   up code at the step-4 recreate.)
3. **Backfill roles** (dry-run to eyeball tally, then write):
   `podman exec rag-worker-dev python -m src.ingest.embedding.common.role_backfill --collection Sanity_testcase_v2 --dry-run`
   then without `--dry-run`. ~12k chunks ÷ 40 ≈ ~300 judge calls (~10 min). Expect `content`
   the large majority; `navigation`+`boilerplate` a minority.
4. **Flip env + recreate** — append to BOTH `/tmp/dev-api.env` and `/tmp/dev-worker.env`
   (replacing existing keys): `RAG_VECTOR_COLLECTION_DEFAULT=Sanity_testcase_v2`,
   `RAG_DOCUMENT_CARD_COLLECTION=Sanity_testcase_v2_cards`, `RAG_RETRIEVAL_ROLE_SCHEMA_PRESENT=true`,
   `RAG_TABLE_GROUP_DEDUP_ENABLED=false`, `RAG_DEEP_RESEARCH_MODEL_ALIAS=judge`,
   `RAG_FAITHFULNESS_MODEL_ALIAS=judge`. Create an empty `Sanity_testcase_v2_cards` (so a
   stray card read can't 404). Then `bash /var/tmp/Juan.Kok/recreate_dev.sh`.
5. **Verify + run harness**: `podman exec rag-worker-dev python /work/scripts/validate_v2.py`
   → expect ROLL-UP `nav/boilerplate-leaks=0 dup-table-queries=0 total-hyde-failures=0`.

## 3c. Validation findings (2026-07-02, on _v2 post-cutover)

- **Role filter — PASS (proven at DB level).** BM25 top-20: "proprietary notice" 11
  boilerplate→0; "contents chapter appendix" 9 nav→0; "coherency snoop" 3 nav/boiler→0.
  Freed slots backfill with content. Nav-intent query "which chapter covers coherency"
  still answers correctly (from content chunks' heading breadcrumb, not the ToC).
- **Table dedup — PASS (fixed at source).** `_v2` has `table_row`/`table` types, zero
  `table_summary`; "Table C-3" answers honestly with no duplicate reps.
- **MBIST regression — PASS.** DFT FLOWS.pdf ranks #3–4, grounded insertion-flow answer.
- **HyDE observability — PASS**, but **HyDE quality is a real problem**: the instruct
  controller, given a bare question with no context, confabulates a hypothetical in the
  WRONG DOMAIN (e.g. "DFT flow" → Density Functional Theory physics; "DFT coverage %" →
  crystallography; "AXI channels" → "32 channels/1 GS/s"). `hyde_failures=0` because the
  JSON is valid — the content is just wrong, and it biases the embedding. Every query ran
  `rounds=1` (no retry). This is the next quality lever to pull (constrain HyDE to the
  domain / use search terms more / retry on low judge confidence).
- **Deep research — FIXED (was 100% dead).** Two bugs: (1) `_rerank_deep_research` threw
  on a reranker 404 and zeroed the whole DR result; (2) DR LLM calls inherited
  `RAG_GENERATION_MAX_TOKENS=16384` → `ContextWindowExceededError` on the 16k-context
  instruct model. Fixes: fail-open rerank fallback to hybrid score order
  (`_rerank_pool_or_fallback`); `RAG_DEEP_RESEARCH_MAX_OUTPUT_TOKENS=2048`. DR now runs on
  the instruct model ("Summarize DFT flow" decomposed into 4 nodes / 5 LLM calls).
- **Reranker (cross-encoder) is NOT served on any dev tunnel** (all `/rerank` 404). Only
  DR consumed it; normal/agentic paths use judge-ranking. Given the CE is also measured
  net-negative, the hybrid-score fallback is the correct end state — do not stand up a CE.
- **Multi-turn — SPLIT result (memory helps generation, NOT retrieval).**
  - Pronoun ("What is AXI?" → "How many channels does **it** have?"): **PASS** — resolved
    it→AXI, answered "5 channels" correctly, and the context DISAMBIGUATED (the standalone
    "how many channels does AXI have" was confused by cRoCodile DMA's 8 channels).
  - Elliptical ("Explain MBIST insertion." → "What steps come before **that**?"): referent
    resolved in generation, but retrieval didn't fetch the pre-MBIST steps → partial.
  - Comparison ("What is CHI?" → "How does **it** compare to AXI?"): FAIL — memory carried
    CHI but this turn's retrieval pulled only AXI → "context lacks CHI, can't compare".
  - **Root cause:** in `mode=query` (default), conversation history is injected into the
    GENERATION prompt but does NOT rewrite the RETRIEVAL query (`history_decision=None` on
    every turn). Generation-side referent resolution works; retrieval-side doesn't. The
    query-rewrite/condensation path exists but only in `mode=retrieval`. **Fix direction:**
    condense follow-ups into a standalone retrieval query in query mode too (next step,
    not yet done — it's a feature change, not a bug fix).

## 3d. Follow-up fixes applied (2026-07-02)

- **HyDE domain-grounding — FIXED (validated).** Root cause: the controller had NO
  corpus-domain context AND the default `RAG_DOMAIN_DESCRIPTION` described an AI/ML corpus
  ("interpret acronyms in this domain") — so silicon-design acronyms confabulated in the
  wrong field (DFT→Density Functional Theory, "DFT coverage"→crystallography, "AXI
  channels"→signal-processing). Generic fix (not per-acronym): inject `{{ domain }}` into
  the HyDE prompt (`generate_hyde(domain=...)` ← orchestrator ← `DOMAIN_DESCRIPTION`) + set
  a correct silicon/SoC `RAG_DOMAIN_DESCRIPTION` in dev env. Result: all three HyDE
  hypotheticals now in-domain; "DFT flow" went from REFUSING to a full grounded answer;
  "AXI channels" → correct "5 channels (AR/AW/W/R/B)". Unit test guards the wiring.
- **Temporary query-side nav heuristic — REMOVED (surgical).** Deleted the redundant
  `_is_navigational` import + `_is_low_value_chunk` + `drop_nav`/`nav_max_chars` from
  `_filter_thin_candidates` and both call sites in rag_chain (now covered by the metadata
  role filter). KEPT the thin/heading-only floor (`_is_thin_or_heading`, `RERANK_MIN_CHARS`)
  — a distinct safeguard the role filter doesn't replace. The ingest-side shared
  `is_navigational` (legacy drop when nav_classify=false) is untouched. `test_nav_filter.py`
  rewritten to cover only the retained thin floor; 809 tests pass. (Leftover: the now-unused
  query-side settings `RERANK_DROP_NAVIGATIONAL`/`RERANK_NAV_MAX_CHARS` remain as harmless
  dead config, still referenced by ingest-side doc comments.)

## 3e. Faithfulness rail on the instruct model — VALIDATED (in isolation)

Answer to "can we use the instruct model + a system prompt for the faithfulness rail?":
**YES.** `RAG_FAITHFULNESS_MODEL_ALIAS=judge` routes the rail's `call_oneshot` self-check
to the instruct model, keeping the existing evaluator system prompt. Isolation test on the
instruct model (`judge`): grounded answer → score **1.00** PASS; fabricated answer ("5 GHz
over fiber, 64 DMA engines") → score **0.00**, PASS-with-**warning** (flag action). Clean
discrimination. NOTE: the rail does NOT fire in the live dev query path because
`GUARDRAIL_BACKEND` is unset (guardrails disabled in dev) — a deployment toggle, not a code
gap. To exercise it live, set `GUARDRAIL_BACKEND=nemo` (turns on ALL output rails
+ latency) — deferred as a deliberate choice.

## 4. Cleanup (after validation passes)
- Delete the temporary query-side nav heuristic (`_is_navigational`, `_filter_thin_candidates`,
  `_is_low_value_chunk`, `_is_thin_or_heading`, `_TOC_LEADER_RE`, `_NAV_PHRASE_RE`, exports,
  `RAG_RERANK_DROP_NAVIGATIONAL`/`RAG_RERANK_NAV_MAX_CHARS`) — only once the metadata filter
  is proven to cover the same cases.
- Keep `RAG_TABLE_GROUP_DEDUP_ENABLED=false` (HEAD single-rep tables make it unnecessary; leave
  code as a rollback switch).
- Commit + deploy to the Cloudflare dev instance; never disrupt prod.
