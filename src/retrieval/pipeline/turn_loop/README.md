<!-- @summary
Turn-level agentic conversation loop: one controller LLM per turn iterates
RETRIEVE / DEEP_STUDY / CLARIFY / ANSWER actions in the API process until a
final-answer confidence gate passes, streaming every decision as typed events.
LOOP CORE phase: schemas, orchestrator, all four action stages, event emitter,
and TurnContext assembly are implemented and DI-pure (no Temporal/MinIO/Redis/
Weaviate/server imports); endpoint wiring injects real seams via TurnLoopDeps.
@end-summary -->

# `turn_loop/` — Turn-Level Agentic Conversation Loop

One controller LLM per turn operates in a loop, choosing per iteration among
**RETRIEVE** (new/refined/HyDE hybrid search), **DEEP_STUDY** (windowed read of
one full MinIO source document), **CLARIFY** (terminal: LLM-authored question
back to the user with clickable hints), and **ANSWER** (terminal attempt:
draft gated on a weighted confidence score; failure feeds back into the next
decision). The loop runs in the **API process** (so every decision streams as
a typed SSE event); the worker contributes exactly one durable activity,
`retrieve_ranked`. Activated per request via `turn_loop`
(`server/schemas.py QueryRequest`) or the `RAG_TURN_LOOP_ENABLED` config gate;
mutually exclusive with `deep_research` / `agentic_retrieval` /
`tree_retrieval`.

The existing `pipeline/agentic/` package is consumed as a **library** (the
round judge composes `judge_chunks`; the HyDE hypothetical answer is authored
by the controller inside its decision, so no separate `generate_hyde` call is
made). All LLM calls — controller, judge, deep-study reads, clarify, draft,
self-score — charge ONE `TurnBudget.max_llm_calls` ledger.

**Design (single source of truth): `docs/retrieval/TURN_LOOP_DESIGN.md`.**

## Files

| File | Role |
| --- | --- |
| `__init__.py` | Public facade: `run_turn_loop`, `build_turn_context`, all schema re-exports. Callers import from the package only. |
| `schemas.py` | The full typed contract surface (see schema ownership below). |
| `common.py` | Shared side-effect-light helpers used by multiple stages (CLAUDE.md §2): `one_line()` (whitespace-normalizing flattener), `preview_chars()` (per-chunk preview cap reader), `TOP_PREVIEW_COUNT` (event preview count). Centralized so stage files don't each carry a copy. |
| `orchestrator.py` | The loop: standalone-query resolution up front, budget checks (actions / LLM ledger / wall clock), controller dispatch, terminal handling, the router fast-lane seed (`_seed_decision`), best-effort exhaustion exit, never-raise containment. |
| `standalone.py` | **Follow-up standalone-query resolution** (multi-turn): reuses the shared `retrieval_query_rewriter` prompt to resolve a follow-up's back-references (pronouns/demonstratives) into a self-contained query that seeds RETRIEVE/DECOMPOSE + the controller's query authoring. Generation keeps the verbatim query + full `TurnContext`, so memory grounds the answer without poisoning the retrieval seed. Fresh first turn → no LLM call; fail-open to the literal query (`RAG_TURN_LOOP_STANDALONE_QUERY_ENABLED`). |
| `router.py` | **Pre-flight confidence router** (pure): `route(RouteSignals, RouteConfig) → RouteHint` — seeds the first action + `effort` as an ADVISORY hint (short/confident/self-contained factoid → fast lane; else no seed). A compound query is **not** seeded here — `is_compound` only excludes it from the fast lane; the positive compound → DECOMPOSE decision moved to the controller's `query_shape` classification. Zero heavy imports; the runner gathers the signals. |
| `controller.py` | Per-iteration action selection (controller LLM, salvage-parsed JSON, §6 fail-open ladder); renders the router hint into the first prompt; emits a top-level `query_shape` (`single_facet`/`compound`/`vague`) and coerces an opening single-RETRIEVE to DECOMPOSE when the query is `compound` (`_coerce_shape_decompose`, the LLM-driven replacement for the router's compound regex — §0); home of the deterministic evidence digest and the model-alias getters. |
| `retrieve.py` | RETRIEVE action — `retrieve_ranked` seam call + chunk-id dedup + agentic `judge_chunks` composition; kept chunks pooled in judge-rank order. |
| `decompose.py` | DECOMPOSE action — one split LLM call fans a compound question into focused sub-queries retrieved **in parallel** (`asyncio.gather`) into the same flat pool, judged once as a round. |
| `deep_study.py` | DEEP_STUDY action — `fetch_document` seam + anchored overlapping-window walk (`refactored_char_start >= 0` guard, `-1` sentinel → window 0); window findings enter the pool as `deep_study`-provenance chunks. |
| `clarify.py` | CLARIFY action (terminal) — LLM-authored question + hint/scoping chips, hints capped at `clarify_max_hints`, deterministic fallback. |
| `answer.py` | ANSWER action (terminal attempt) — streamed draft (live `draft` events, reasoning + content kinds) + self-score + weighted gate (judge / self / citation coverage). |
| `events.py` | `TurnEventEmitter`: trace append + gated SSE forward (`RAG_TURN_LOOP_STREAM_EVENTS`), sink errors swallowed; the charged LLM-call wrapper (single ledger + `llm_call` event). |
| `context.py` | `build_turn_context`: tolerant assembly of the typed `TurnContext` from the memory layer's structured dict. |

## Pre-flight router + fast lane + effort dial (Phase 3)

Before the loop, a cheap LLM-free **router** (`router.py`) runs once at the
runner seam (`server/turn_loop_runner.py` — `build_route_signals` →
`resolve_route_hint`) and returns a `RouteHint` that the loop treats as
**advice**, never a hard override (so the whole path fails open to the
pre-router baseline). Signals are the classifiers the codebase already owns —
`query_shape.has_compound_marker`, `query_processor.heuristic_confidence` /
`has_backward_reference` / `detect_suppress_memory` — so there is no new
inference on the critical path.

- **Compound query** → **not** seeded by the router. Compound → DECOMPOSE is now
  the controller's job: it emits a top-level `query_shape`, and
  `controller._coerce_shape_decompose` rewrites an opening (iteration 0)
  single-RETRIEVE into DECOMPOSE when `query_shape == compound`
  (`RAG_TURN_LOOP_SHAPE_DECOMPOSE_ENABLED`). Classifying by the reasoning LLM
  instead of a keyword regex is the §0 generic fix (a comparison phrased without
  "vs"/"and" is still caught). Additive-safe — it self-disables unless
  `decompose_anchor_raw` makes the fan-out a superset of the RETRIEVE it replaces
  — and fail-open (an absent/unknown shape skips the coercion). The router's
  `is_compound` signal now only holds a possibly-compound query **out** of the
  fast lane (never skip the controller for one).
- **Fast lane** (short, high-confidence, self-contained, single-facet) → the
  loop skips the first controller LLM call and runs a deterministic
  RETRIEVE→ANSWER (`_seed_decision`), re-engaging the controller only if that
  answer fails the gate. This is the router's one non-advisory move — hence
  opt-in (`RAG_TURN_LOOP_FAST_LANE_ENABLED`, default off until the routing
  eval confirms p50/p95 matches linear).
- **Effort** (`fast` / `balanced` / `thorough`) selects the `TurnBudget` scale
  via `TurnBudget.from_settings(effort=...)` (scales `max_actions` /
  `max_llm_calls`; the wall clock is a fixed ceiling, never scaled). `balanced`
  is byte-for-byte today's budget.

The routing decision is surfaced verbatim on `metadata.turn_loop.router` and
each `turn_action` event carries `source: "router" | "facet_guard" |
"loop_guard" | "controller"` (the decision ladder in `_select_decision`).

## Control flow (design §5)

Per iteration: budget check → router seed OR controller decision
(`turn_action` event) → dispatch. RETRIEVE / DECOMPOSE / DEEP_STUDY grow the
evidence pool and loop. CLARIFY ends
the turn as `ask_user`. ANSWER drafts, self-scores, and evaluates
`gate = w_judge * judge_pool_confidence + w_self * self_score +
w_citation * citation_coverage` against
`RAG_TURN_LOOP_ANSWER_CONFIDENCE_THRESHOLD`; pass ends the turn, fail records
`GateFeedback` (weakest component, unsupported claims, judge gaps) for the
next controller prompt. Budget exhaustion exits best-effort: the best failed
draft, else one final draft if the LLM ledger allows, else an explicit
cannot-answer-confidently message over the evidence digest — never an empty
response.

**Commit guards (deterministic, no LLM).** Two guards break the "controller
won't commit" spirals a prompt nudge alone can't (a small controller keeps
gathering while the judge still names *something* missing). Both sit in the
decision ladder ahead of the controller and force an ANSWER:

- **facet-commit** (`facet_guard`, `RAG_TURN_LOOP_FACET_COMMIT_ENABLED`): once a
  multi-way DECOMPOSE has covered every facet — each decomposed sub-question has
  ≥1 judge-kept chunk (`TurnState.facets`) — the pool can synthesize the
  comparison, so the loop answers instead of exploring further. Fixes the
  DECOMPOSE spiral where a comparison decomposes perfectly but never commits.
- **no-progress** (`loop_guard`, `RAG_TURN_LOOP_MAX_NO_PROGRESS_ROUNDS`): the
  opposite signal — after N consecutive gather rounds add zero new evidence, the
  loop answers from the pool (or the `fallback_chunks` floor) gathered so far.

When a *forced* ANSWER then fails the gate, more retrieval is futile (the
comparison is complete, or the corpus keeps yielding nothing), so the loop
commits the best grounded draft best-effort — `stop_reason` `facets_covered` or
`no_progress_stall` — rather than burning the rest of the action/wall-clock
budget. Both guards fail-open (a config flag disables each) and never fire
before the evidence to justify them exists.

Fail-open ladder (design §6): every LLM output is `</think>`-stripped and
salvage-parsed; an unusable controller decision becomes RETRIEVE with the
verbatim user query on iteration 1 (else an ANSWER attempt); an unusable judge
keeps the whole round; an unusable clarification degrades to a deterministic
question built from the named gaps.

## Boundaries (what this package does NOT do)

- **Input safety**: sanitize + PII gating run ONCE in the server layer before
  `run_turn_loop` is called (design §3) — the loop assumes a safe query.
- **Answer replay**: the accepted answer is returned in `TurnLoopResult`;
  replaying it as standard `token` SSE events is the endpoint's job.
- **Infrastructure**: no Temporal/MinIO/Redis/Weaviate/server imports — the
  retrieval activity, document fetch (conversation-cached), LLM provider, and
  SSE bridge are all injected through `TurnLoopDeps`.
- **Doc suppression**: loop requests never inject `ignored_doc_ids`; dedup is
  by stable chunk id only, so deepening into a prior document is a
  first-class move.

## Schema ownership

`schemas.py` is the **single canonical contract module** for this package
(CLAUDE.md §2): every cross-module type of the loop — actions, decision,
budget, evidence pool unit, gate feedback, events, state, results, the
`TurnLoopDeps` DI seam, and the cross-turn `TurnContext` — is defined there
and re-exported by `__init__.py`. Stage modules import contracts from
`schemas.py` and MUST NOT define their own. The module is config-free at
import time; `TurnBudget.from_settings()` is the one (lazy) bridge to the
`RAG_TURN_LOOP_*` / `RAG_TURN_CONTEXT_*` block in `config/settings.py`
(validated by `validate_turn_loop_config()`).

## Prompts

`prompts/turn_controller_decide.md` (carries the advisory `{{ router_hint }}`),
`turn_decompose.md`, `turn_deep_study_read.md`, `turn_clarify_generate.md`,
`turn_answer_selfscore.md` — `{{ var }}` convention, loaded/rendered via the
shared `src/common/prompts.py` helpers, strict-JSON output salvage-parsed with
fail-open defaults (design §6).

## Tests

`tests/retrieval/turn_loop/` — DI-driven unit tests over a fake provider and
fake deps (no infrastructure): controller fail-open, dispatch/ordering, dedup,
deep-study sentinel/window walk, clarify caps, gate pass/fail loop-back,
budget-exhaustion best-effort, wall-clock stop, and single-ledger accounting.
