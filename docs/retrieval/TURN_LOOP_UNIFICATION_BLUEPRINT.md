<!-- @summary
Audited migration blueprint: collapse the 4 mutually-exclusive retrieval modes
(deep_research / agentic_retrieval / tree_retrieval / turn_loop) into ONE
self-routing orchestrator — turn_loop — with the other modes demoted to actions,
a confidence pre-flight router, and the user param zoo reduced to an effort dial.
Grounded in a 5-agent code audit across develop and feat/agentic-conversation-loop
(2026-07-03). File:line references are to those trees at audit time.
@end-summary -->

# Turn-Loop Unification Blueprint

**Status:** design, approved-in-direction by user 2026-07-03. Execution not started.
**Lands on:** `feat/agentic-conversation-loop` (the only branch where turn_loop exists).

## The one insight that de-risks everything

**Agentic HyDE is ALREADY an action.** On the branch, the turn_loop `RETRIEVE`
action composes `generate_hyde` + `judge_chunks` as a library (the controller
authors the HyDE inside its decision; `retrieve.py:150`, lazy import `:84-85`).
So the biggest "mode" is already demoted — the migration is finishing a pattern
that's proven in-tree, not inventing one.

## End state

One orchestrator = `run_turn_loop` (`turn_loop/orchestrator.py:209`), the path
for **every** query. Modes become the action space:

| Action | Replaces | Status |
| --- | --- | --- |
| `RETRIEVE (flat)` | `agentic_retrieval` | **already an action** — keep |
| `RETRIEVE (tree)` | `tree_retrieval` | new `strategy='tree'` flag + a `retrieve_tree` dep |
| `DECOMPOSE` | `deep_research` *cheap half* (fan-out) | **new action** — 1 split LLM call + N sub-query rounds into the flat pool |
| `DEEP_STUDY` | — | exists (windowed full-doc read) |
| `CLARIFY` / `ANSWER` | — | exist; `ANSWER` gate subsumes `compute_composite_confidence` |
| *(future, separate)* `REPORT` | `deep_research` *heavy half* (Gemini-style multi-hundred-turn report) | **deferred, NOT folded in** |

User-facing params collapse: 5 booleans + `fast_path` + `max_agentic_rounds` +
4 duplicated validators → an **enum waypoint** (`retrieval_strategy`, kills the
O(n²) mutual-exclusion validators structurally) → finally an **`effort` dial**
(`fast|balanced|thorough`) resolved once in `TurnBudget`. Old booleans stay as
deprecation aliases.

## Confidence router (the "route itself" piece)

A cheap **pre-flight** router at the `turn_loop_runner` seam, unifying signals
that **already exist** — no new inference infra:
- `query_confidence` / `_heuristic_confidence` (`query_processor.py:412`)
- `_has_compound_marker` (`query_shape.py:52`) → biases `DECOMPOSE`
- `_has_backward_reference` / `suppress_memory` → memory-aware vs fresh first action

Output = an `(initial_action, effort)` **hint**, never a hard override (fail-open
preserved → degrades to today's single-round baseline). **Fast-lane:** high-confidence
+ short + non-compound → deterministic `RETRIEVE(flat)+ANSWER` with NO first
controller LLM call; the controller only re-engages if that answer fails the gate.
This keeps the common case at ~linear latency.

> Note: `query_confidence` is hardcoded to `1.0` on the retrieval/hard sub-mode
> path (`query_processor.py:766`) — the router is blind there until that's restored.

## Phases (each independently shippable)

0. **Merge develop→branch + enum waypoint.** Land 488d17e (HyDE panel, follow-ups,
   quality fixes); replace the 4–5 exclusion validators with one enum-membership
   check + boolean deprecation shims. *Risk:* the `schemas.py` + `rag_chain.py`
   hand-merge (both trees edited the same validator/field block). *Validate:* every
   legacy boolean combo maps to the same dispatch; re-run develop's tests on the merge.
1. **`DECOMPOSE` action** (port only the single `_decompose` call + prompt; fan out
   over existing `retrieve_ranked`; **delete** the heavy report loop `_recurse_topic`
   /`_sufficiency_check`/`_early_stop_ok`/`_apply_per_topic_rerank` — all duplicate
   what the turn loop already owns). *Gate:* latency-test before shipping.
2. **tree as a `RETRIEVE` strategy variant** (new `retrieve_tree` dep). *Gate behind
   evidence it helps* — tree is often dormant in this corpus; defer if it doesn't win.
3. **Pre-flight router + fast-lane.** — **CORE DONE (Phase 3, `turn_loop/router.py`).**
   Pure `route(RouteSignals, RouteConfig) → RouteHint` at the runner seam
   (`build_route_signals` → `resolve_route_hint`), reusing the existing
   classifiers (`has_compound_marker` / `heuristic_confidence` /
   `has_backward_reference` / `detect_suppress_memory`). Compound → advisory
   DECOMPOSE seed (makes Phase 1's action *get chosen*); short/confident/
   self-contained factoid → fast lane (skip the first controller LLM call,
   deterministic RETRIEVE→ANSWER, re-engage on gate fail) — opt-in
   (`RAG_TURN_LOOP_FAST_LANE_ENABLED`, default off). `effort`
   (`fast`/`balanced`/`thorough`) scales `TurnBudget` (`balanced` == today).
   Advisory + fail-open throughout (a `None`/neutral hint == pre-router loop).
   Surfaced on `metadata.turn_loop.router` + `turn_action.source`. *Still to
   do before default-on:* the labeled routing eval set + fast-lane p50/p95 ==
   linear measurement (the gate for flipping `FAST_LANE_ENABLED` on).
4. **turn_loop becomes THE path; collapse enum→effort; retire dead orchestrators**
   (delete `_alt_active` dr/agentic branches + ~15 guards + `_run_deep_research`
   heavy loop + Stage-7.5 composite gate). *Risk: the big one* — keep retired code
   feature-flagged one release as rollback; full eval-harness pass required.

## Load-bearing risks (from the audit)

- **Latency:** a naive turn loop is 3+ LLM calls/query (decide→judge→answer→gate)
  vs linear's one-retrieve-one-generate. The fast-lane is the mitigation and must
  be *proven* to match linear p50/p95 on shared-GPU ai01.
- **Full-distribution regression at Phase 4:** MEMORY records classes where existing
  behavior wins (reranker helps CHI-vs-AXI comparison; linear retrieval itself is
  good). turn_loop must not lose those — feature-flag, don't delete, until proven.
- **Score-calibration mismatch (real blocker, not plumbing):** the loop's answer
  gate uses RAW server-scale scores; the develop gate uses CALIBRATED reranker
  scores. Unifying them needs score normalization first — no signal does this today.
- **Router quality is unmeasured** — needs a labeled routing eval set before it's
  trusted beyond hint-only.
- **`graph_context`/KG has no home** in the turn_loop pool (`EvidenceChunk` lacks
  the field) — decide before deleting the linear KG-expand stage.

## Latency experiment #2 — RESULT (2026-07-03, live ai03 branch, warm)

**Verdict: GO for DECOMPOSE, conditional on PARALLEL fan-out.**

Measured on the healthy agentic path (the DR-as-proxy path is broken — see below):
- Normal agentic compound query: **6.6–7.9s** to the retrieval event, 2 LLM calls,
  server loop 3.3–4.5s, ~3s fixed overhead (query_processing 2.7s + Temporal/SSE).
- Per-LLM-call ≈ 1.5–2.2s; hybrid search ≈ 0–1s (cheap). **Latency is dominated by
  the count of SEQUENTIAL LLM round-trips**, not by sub-query or search count.

DECOMPOSE cost model (constructed): 1 decompose call + N sub-query rounds + merge.
- **Parallel fan-out** (asyncio.gather): ≈ decompose (~2s) + one concurrent judge
  wave (~2s) + final (~1.5s) ≈ **~8–11s** end-to-end — **~1.3–1.5× a normal query**
  (only ONE extra sequential round-trip). VIABLE.
- Serial fan-out: ~N× blowup (~14–25s @ N=3). The failure mode to avoid.

→ **"Parallel fan-out over sub-queries" is a HARD REQUIREMENT of the DECOMPOSE
action, not an optimization.** Router fast-lane keeps factoids at ~6s (never
invoke DECOMPOSE). The headline "controller = N× latency" risk does NOT
materialize under parallel fan-out.

**Bonus finding — deep_research is LIVE-BROKEN on the branch:** it calls
`_rerank_deep_research` → dev reranker 404 → hangs 40–157s → returns 0 results.
The develop `_safe_rerank` fix (commit 488d17e) repairs exactly this → concrete
motivation for Phase 0, and design validation that the blueprint's deletion of
`_rerank_deep_research` removes this failure mode entirely.

## Latency experiments to run (before committing to Phase 1/3)

1. Fast-lane vs full loop vs linear, trivial factoid (p50/p95).
2. **turn_loop invoking DECOMPOSE fan-out** (the user's explicit ask) — isolate cost:
   split call vs N searches vs N judge rounds; compare to develop deep_research.
3. Per-query LLM-call-count histogram across a representative query set.
4. Judge latency under shared-GPU contention (`_judge_round` on the critical path).
5. Escalation cost when the fast-lane's ANSWER fails the gate.

## Branch reconciliation

Work happens **on the branch** (turn_loop lives only there). Merge order:
**develop→branch** now (Phase 0); **branch→develop only after Phase 4** proves
turn_loop ≥ every retired mode. ai03 dev (:8102) already runs this branch and is
the live probe surface — deploy by *adapting onto* branch files (see
`[[ai03-dev-runs-turn-loop-branch]]`), never overwriting.
