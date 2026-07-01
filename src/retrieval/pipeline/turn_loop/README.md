<!-- @summary
Turn-level agentic conversation loop: one controller LLM per turn iterates
RETRIEVE / DEEP_STUDY / CLARIFY / ANSWER actions in the API process until a
final-answer confidence gate passes, streaming every decision as typed events.
CONTRACTS phase: the typed schema surface and public facade are in place; the
orchestrator and per-action modules land in subsequent phases.
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

The existing `pipeline/agentic/` package is consumed as a **library**
(`generate_hyde`, `judge_chunks`, chunk-id dedup) — not extended or nested.
All LLM calls charge one `TurnBudget.max_llm_calls` ledger.

**Design (single source of truth): `docs/retrieval/TURN_LOOP_DESIGN.md`.**

## Files

| File | Role | Status |
| --- | --- | --- |
| `__init__.py` | Public facade: `run_turn_loop` (lazy, PEP 562) + all schema re-exports. Callers import from the package only. | present |
| `schemas.py` | The full typed contract surface (see schema ownership below). | present |
| `controller.py` | Per-iteration action selection (controller LLM, salvage-parsed JSON). | later phase |
| `retrieve.py` | RETRIEVE action — HyDE/rewrite + `retrieve_ranked` activity + judge/dedup composition. | later phase |
| `deep_study.py` | MinIO fetch + windowed study loop. | later phase |
| `clarify.py` | LLM-authored clarification + hint/scoping questions. | later phase |
| `answer.py` | Draft generation + confidence gate. | later phase |
| `events.py` | Typed event emitter → injected async callback (SSE bridge). | later phase |
| `context.py` | TurnContext assembly from conversation memory. | later phase |
| `orchestrator.py` | The loop: budgets, dispatch, stop conditions; exports `run_turn_loop`. | later phase |

## Schema ownership

`schemas.py` is the **single canonical contract module** for this package
(CLAUDE.md §2): every cross-module type of the loop — actions, decision,
budget, evidence pool unit, gate feedback, events, state, results, the
`TurnLoopDeps` DI seam, and the cross-turn `TurnContext` — is defined there
and re-exported by `__init__.py`. Later-phase modules import contracts from
`schemas.py` and MUST NOT define their own. The module is config-free at
import time; `TurnBudget.from_settings()` is the one (lazy) bridge to the
`RAG_TURN_LOOP_*` / `RAG_TURN_CONTEXT_*` block in `config/settings.py`
(validated by `validate_turn_loop_config()`).

## Prompts

`prompts/turn_controller_decide.md`, `turn_deep_study_read.md`,
`turn_clarify_generate.md`, `turn_answer_selfscore.md` — `{{ var }}`
convention, loaded/rendered via the shared `src/common/prompts.py` helpers,
strict-JSON output salvage-parsed with fail-open defaults (design §6).
