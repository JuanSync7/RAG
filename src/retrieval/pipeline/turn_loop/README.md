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
| `orchestrator.py` | The loop: budget checks (actions / LLM ledger / wall clock), controller dispatch, terminal handling, best-effort exhaustion exit, never-raise containment. |
| `controller.py` | Per-iteration action selection (controller LLM, salvage-parsed JSON, §6 fail-open ladder); home of the deterministic evidence digest and the model-alias getters. |
| `retrieve.py` | RETRIEVE action — `retrieve_ranked` seam call + chunk-id dedup + agentic `judge_chunks` composition; kept chunks pooled in judge-rank order. |
| `deep_study.py` | DEEP_STUDY action — `fetch_document` seam + anchored overlapping-window walk (`refactored_char_start >= 0` guard, `-1` sentinel → window 0); window findings enter the pool as `deep_study`-provenance chunks. |
| `clarify.py` | CLARIFY action (terminal) — LLM-authored question + hint/scoping chips, hints capped at `clarify_max_hints`, deterministic fallback. |
| `answer.py` | ANSWER action (terminal attempt) — streamed draft (live `draft` events, reasoning + content kinds) + self-score + weighted gate (judge / self / citation coverage). |
| `events.py` | `TurnEventEmitter`: trace append + gated SSE forward (`RAG_TURN_LOOP_STREAM_EVENTS`), sink errors swallowed; the charged LLM-call wrapper (single ledger + `llm_call` event). |
| `context.py` | `build_turn_context`: tolerant assembly of the typed `TurnContext` from the memory layer's structured dict. |

## Control flow (design §5)

Per iteration: budget check → controller decision (`turn_action` event) →
dispatch. RETRIEVE / DEEP_STUDY grow the evidence pool and loop. CLARIFY ends
the turn as `ask_user`. ANSWER drafts, self-scores, and evaluates
`gate = w_judge * judge_pool_confidence + w_self * self_score +
w_citation * citation_coverage` against
`RAG_TURN_LOOP_ANSWER_CONFIDENCE_THRESHOLD`; pass ends the turn, fail records
`GateFeedback` (weakest component, unsupported claims, judge gaps) for the
next controller prompt. Budget exhaustion exits best-effort: the best failed
draft, else one final draft if the LLM ledger allows, else an explicit
cannot-answer-confidently message over the evidence digest — never an empty
response.

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

`prompts/turn_controller_decide.md`, `turn_deep_study_read.md`,
`turn_clarify_generate.md`, `turn_answer_selfscore.md` — `{{ var }}`
convention, loaded/rendered via the shared `src/common/prompts.py` helpers,
strict-JSON output salvage-parsed with fail-open defaults (design §6).

## Tests

`tests/retrieval/turn_loop/` — DI-driven unit tests over a fake provider and
fake deps (no infrastructure): controller fail-open, dispatch/ordering, dedup,
deep-study sentinel/window walk, clarify caps, gate pass/fail loop-back,
budget-exhaustion best-effort, wall-clock stop, and single-ledger accounting.
