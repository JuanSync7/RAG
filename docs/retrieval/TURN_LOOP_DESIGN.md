<!-- @summary
Design document for the turn-level agentic conversation loop: an API-process
controller that iterates RETRIEVE / DEEP_STUDY / CLARIFY / ANSWER actions until
a final-answer confidence gate passes, with full stream observability and a
typed turn-context memory transfer contract.
@end-summary -->

# Turn-Level Agentic Conversation Loop — Design

Status: approved for implementation on `feat/agentic-conversation-loop`
(based on `origin/develop` @ f1cea68). Grounded in a verified code map of this
worktree (2026-07-02); all file:line references are to this tree.

## 1. Problem

Multi-turn conversational RAG on develop today splits turn handling across
three mutually-exclusive loops, none of which operates at turn level:

| Loop | Scope | Limitation |
|---|---|---|
| LangGraph pre-retrieval (`query_processor.py:649-670`) | query hygiene | clarifies only pre-evidence, with canned templates |
| Agentic HyDE/judge INC loop (`src/retrieval/pipeline/agentic/`) | retrieval rounds | only action is retrieve-more; confidence gates evidence sufficiency, not answer quality |
| REQ-706 confidence re-retrieval (`rag_chain.py:3079-3174`) | one post-gen retry | never mutates the query; alpha/limit nudges only |

Empirically (live 2-turn runs against dev, 2026-06-30): "tell me more" drifts
off-topic (query reformulation loses the anchor), refinements lose the carried
concept, meta-questions trigger pointless fresh searches, and the memory
layer's unconditional `ignored_doc_ids` hard-suppression
(`server/routes/query.py:516` → `not_in` filters in every retrieval branch,
`rag_chain.py:1696-1699`) makes it impossible to deepen into a document the
previous turn already used.

## 2. Goal

One controller LLM per turn operating in a loop until a final-answer
confidence gate passes, choosing per iteration among:

- **RETRIEVE** — new/refined/HyDE query against hybrid search (composing the
  existing agentic primitives: `generate_hyde`, judge, dedup),
- **DEEP_STUDY** — fetch the full source document from MinIO and read it
  window-by-window to extract the answer,
- **CLARIFY** *(terminal)* — throw an LLM-authored clarification question back
  to the user, with clickable hint/scoping questions,
- **ANSWER** *(terminal attempt)* — draft, gate on confidence; below threshold
  the controller loops with the gate's feedback.

Plus: a typed **TurnContext** contract for what transfers between turns, a
**multi-turn eval harness** with labeled traces, and **100% stream
observability** — every HyDE query, routing decision, and LLM call surfaced as
typed SSE events rendered before/alongside the thinking box.

Non-goals (v1): replacing the existing single-shot pipeline (the loop is a
gated alternative path); cross-conversation memory; UI redesign beyond the
activity log and clarify chips.

## 3. Placement: the loop runs in the API process

Verified constraint: retrieval today is one blocking
`await execute_workflow(RAGQueryWorkflow)` (`server/routes/query.py:527-532`);
nothing escapes the Temporal activity mid-run, so a worker-side loop cannot
stream its decisions. Verified capabilities: the API process **already** runs
LLM generation in-process for streaming (`_stream_llm`, query.py:191-240, via
`LLMProvider.generate_stream(include_reasoning=True)`), holds a MinIO client
(`server/api.py:80-84`), and owns Redis conversation memory (query.py). LLM
calls are HTTP to inference endpoints — no GPU models needed in-process.

Therefore:

- **API process** hosts the turn loop: controller decisions, HyDE generation,
  judge calls, deep-study reads, clarification authoring, answer drafting and
  streaming, memory read/write, SSE event emission at every step.
- **Worker** gains exactly one new activity, `retrieve_ranked`: embed (query
  and/or HyDE text) → hybrid search (`_do_search` choke point,
  rag_chain.py:717) → cross-encoder rerank → thin/nav filter → ranked chunks.
  No Stage 1, no LangGraph, **no LLM calls inside the activity** — it is
  idempotent and safe under Temporal retries (`maximum_attempts` stays
  small; the per-activity timeout must be < `RAG_TURN_LOOP_WALL_CLOCK_MS`,
  enforced by `validate_turn_loop_config()`).
- Input safety runs once at turn start in the API process: reuse the
  sanitize/PII-gate modules directly if importable without worker-only deps
  (verify at implementation; if not importable, add them to a small
  `prepare_turn` activity). The pre-retrieval LangGraph reformulate loop is
  NOT run for loop turns — the controller owns query design.

Durability trade-off (explicit): the loop itself is not durable — an API crash
mid-loop loses the turn, same as streamed generation today. Retrieval
primitives remain durable Temporal activities. Accepted for interactive chat.

Concurrency note: a loop turn holds its API request slot longer than a
single-shot query. `RAG_TURN_LOOP_WALL_CLOCK_MS` bounds it; the overload
semaphore already applies. Revisit a dedicated loop semaphore if saturation
appears.

## 4. Package layout

```
src/retrieval/pipeline/turn_loop/
  __init__.py        # thin public facade: run_turn_loop, TurnLoopResult, schemas re-exports
  schemas.py         # TurnBudget, TurnAction, TurnDecision, TurnState, TurnEvent, TurnLoopResult, TurnContext types
  controller.py      # per-iteration action selection (controller LLM, salvage-parsed JSON)
  retrieve.py        # RETRIEVE action — HyDE/rewrite + retrieve_ranked activity call + judge/dedup composition
  deep_study.py      # MinIO fetch + windowed study loop
  clarify.py         # LLM-authored clarification + hint/scoping questions
  answer.py          # draft generation + confidence gate
  events.py          # typed event emitter → injected async callback (SSE bridge)
  context.py         # TurnContext assembly from conversation memory
  orchestrator.py    # the loop: budgets, dispatch, stop conditions
  README.md
```

The existing `pipeline/agentic/` package is consumed as a library
(`generate_hyde` hyde.py:75-144, `judge_chunks` judge.py:144-211, dedup by
stable chunk id) — **not** extended or nested. Its `AgenticRetrieval`
orchestrator is not invoked; the turn loop composes the same primitives with
its own budget so LLM calls are charged to one ledger (no double-budgeting).

## 5. Action space and control flow

```
turn start
  ├─ sanitize + PII gate (once)
  ├─ TurnContext ← conversation memory (prior Q&A, chunk refs, docs studied,
  │                clarifications pending, rolling summary)
  └─ loop (while budgets hold):
       controller LLM ← TurnContext + evidence pool + gate feedback
         → TurnDecision {action, reason, confidence, args}
       ├─ RETRIEVE(query_text, hypothetical_answer?, target_aspect?)
       │     retrieve_ranked activity → dedup vs pool by chunk uuid → judge
       │     keep/rank → pool grows; judge missing_information → controller
       ├─ DEEP_STUDY(document_ref, question)
       │     resolve doc (document_id → build_document_id(source_key) → source;
       │     guard refactored_char_start >= 0, else DocCard heading match or
       │     window 0) → fetch full markdown from MinIO (asyncio.to_thread,
       │     cached per conversation) → read windows via turn_deep_study_read
       │     prompt, accumulate notes, cap RAG_TURN_LOOP_DEEP_STUDY_MAX_WINDOWS
       ├─ CLARIFY (terminal)
       │     turn_clarify_generate prompt ← missing_information, tried
       │     queries, conflicts → {question, hints[], scoping_questions[]}
       │     → RAGResponse action='ask_user' + clarification_message +
       │     metadata.turn_loop.hints; persisted to memory
       └─ ANSWER (terminal attempt)
             draft via generator build_messages (stream as draft/reasoning
             events) → gate: answer confidence = weighted(judge pool
             confidence, LLM self-score, citation coverage) vs
             RAG_TURN_LOOP_ANSWER_CONFIDENCE_THRESHOLD
             ├─ pass → accepted: replay draft as token events → done
             └─ fail → gate feedback (weakest component + judge
                       missing_information) returned to controller; loop
  └─ budget exhausted → best-effort ANSWER with low-confidence warning
     (never an unexplained empty response; reuse the existing
     low-confidence warning append precedent)
```

Loop-level rules:

- **No hard doc suppression.** Loop requests do not inject
  `ignored_doc_ids`. Dedup-by-chunk-uuid keeps served chunks from re-entering
  the pool redundantly, and the controller sees served-chunk refs in
  TurnContext — deepening into a prior document is a first-class move, not a
  filtered-out one. (Behavior change vs the memory suppression REQs — scoped
  to loop requests only.)
- **Confidence semantics.** The Stage 7.5 composite weights assume calibrated
  reranker scores; TEI/vLLM rerank paths pass raw server scores and agentic
  pools carry synthetic rank-stamped scores. The loop therefore uses its own
  gate (weights configurable) fed by judge pool confidence, not
  `compute_composite_confidence` verbatim. REQ-706 and Stage 7.5 do not run
  for loop turns (generation happens in-loop).
- **Mutual exclusion, config-default bypass included.** Request-schema
  validator forbids `turn_loop` with `deep_research`/`agentic_retrieval`/
  `tree_retrieval`; additionally any chain-side resolution must treat
  `_turn_active` with priority (`_agentic_active = ... and not _turn_active`)
  so an env default `RAG_AGENTIC_RETRIEVAL_ENABLED=true` cannot smuggle the
  INC loop under a loop turn.
- **No regex/hardcoded routing.** Controller decisions are LLM judgments over
  typed state (CLAUDE.md §0). The routing package's `regex_decompose`/glossary
  are not used. Document targeting uses the DocCard catalog (embedding-based)
  when no chunk anchors exist.

## 6. Prompts (`prompts/`, `{{ var }}` convention)

| File | In | Out (salvage-parsed JSON) |
|---|---|---|
| `turn_controller_decide.md` | turn context, evidence pool digest, budgets left, gate feedback | `{action, reason, confidence, args{...}}` |
| `turn_deep_study_read.md` | question, window text, notes so far | `{notes, answer_found, next_window_hint}` |
| `turn_clarify_generate.md` | question, gaps, tried queries, conflicts | `{question, hints[], scoping_questions[]}` |
| `turn_answer_selfscore.md` | question, draft, evidence digest | `{self_score, unsupported_claims[]}` |

All calls: alias `controller` (or `judge` where noted), guided-JSON off by
default (`RAG_AGENTIC_LLM_JSON_MODE` rationale, settings.py:1397-1406),
`</think>`-strip + `parse_json_object` salvage, fail-open to safe defaults
(controller parse failure → RETRIEVE with the verbatim user query on
iteration 1, else ANSWER attempt). Promote the duplicated `_render`/prompt
loader into a shared helper (`src/common/prompts.py`) instead of a 5th copy.

## 7. TurnContext memory transfer (extend, don't fork, `platform/memory`)

- `ConversationTurn` += `actions: list[TurnActionRecord]` (action, reason, ms,
  llm_calls), `chunk_refs: list[ChunkRef]` — `{chunk_id (weaviate uuid),
  document_id, source_key, heading, score, refactored_char_start/end,
  preview}` (uuid sourced at the store layer, `weaviate/store.py:698`, plumbed
  through `_source_refs`), `answer_confidence: float | None`,
  `clarification: {question, hints[]} | None`.
- `ConversationMeta` += `docs_studied: list[{document_id, windows_read,
  sections, conclusion, ts}]` (same JSON-list meta pattern and tolerant
  `_decode_id_list`-style migration as relevant/ignored ids).
- `MemoryContext` += structured turn context alongside the prose
  `context_text`; the rolling summary (`_llm_summarize`) receives the
  structured fields so compaction preserves grounding.
- Growth control: `RAG_TURN_CONTEXT_STORE_FULL_TEXT=false` — store
  `preview` capped at `RAG_TURN_CONTEXT_PREVIEW_CHARS` instead of the current
  uncapped full chunk text (fixes the 8dfb367 regression; stale comment at
  memory `schemas.py:24` corrected). Full text is recoverable by chunk uuid.
- A pending clarification is stored on the turn; the next turn's TurnContext
  surfaces it so "the second one" resolves against the offered hints without
  any string matching.

## 8. Stream observability (additive SSE + console activity log)

New typed events (server/schemas.py models; `_sse` envelope unchanged;
`StreamEventData` in `user-types.ts:61-70` extended):

| Event | Payload (abridged) |
|---|---|
| `turn_action` | `{index, action, reason, confidence}` |
| `hyde_query` | `{round, hypothetical_answer, search_terms, target_aspect}` |
| `retrieve_result` | `{round, added, dup, pool_size, top: [{doc, heading, score}]}` |
| `judge_verdict` | `{round, kept, sufficient, confidence, missing_information}` |
| `deep_study` | `{document_id, title, window, of_windows, notes_preview}` |
| `llm_call` | `{alias, purpose, ms, prompt_tokens, completion_tokens}` |
| `draft` | `{attempt, text_delta}` (draft tokens, live) |
| `gate` | `{attempt, score, threshold, passed, weakest}` |
| `clarify` | `{question, hints[], scoping_questions[]}` |

Ordering: activity events stream live throughout the loop; the accepted
answer is then replayed as standard `token` events (existing console
contract), with `reasoning` events emitted as captured. The user watches
drafts being made and discarded — that is the point.

Console: a lazy collapsible **activity log** block inserted into
`.bubble-wrap` **before** `.reasoning-block` (exact template: the reasoning
block itself, `streaming.ts:210-238`; textContent-only). Clarify hints render
as clickable chips that resubmit — reusing the DR-suggestion chip mechanism
(`chatMode.ts:94-125`, `registerDrSuggestionResubmit`). Parity: admin console
(`admin-query.ts:87-124`) and `server/cli_client.py` render the same events;
`buildQueryBody` gains the `turn_loop` flag (also closing the acknowledged
INC-3 parity gap pattern). Non-streaming `/query` returns the same records in
`metadata.turn_loop.trace`. Event names stay 1:1 with OTel span vocabulary
(docs/observability.md:120-170).

## 9. Config (settings.py, docstring-per-key; lazy `validate_turn_loop_config()`)

`RAG_TURN_LOOP_ENABLED=false` (master gate; per-request override like
agentic), `RAG_TURN_LOOP_MAX_ACTIONS`, `RAG_TURN_LOOP_MAX_LLM_CALLS`,
`RAG_TURN_LOOP_WALL_CLOCK_MS`, `RAG_TURN_LOOP_ANSWER_CONFIDENCE_THRESHOLD`,
`RAG_TURN_LOOP_ANSWER_GATE_WEIGHTS` (judge/self/citation),
`RAG_TURN_LOOP_MAX_ANSWER_ATTEMPTS`, `RAG_TURN_LOOP_CONTROLLER_MODEL_ALIAS`
(default `controller`), `RAG_TURN_LOOP_JUDGE_MODEL_ALIAS` (default `judge`),
`RAG_TURN_LOOP_DEEP_STUDY_MAX_DOCS`, `RAG_TURN_LOOP_DEEP_STUDY_WINDOW_CHARS`,
`RAG_TURN_LOOP_DEEP_STUDY_WINDOW_OVERLAP_CHARS`,
`RAG_TURN_LOOP_DEEP_STUDY_MAX_WINDOWS`, `RAG_TURN_LOOP_CLARIFY_MAX_HINTS`,
`RAG_TURN_LOOP_STREAM_EVENTS=true`, `RAG_TURN_LOOP_RETRIEVE_TOP_K`,
`RAG_TURN_CONTEXT_MAX_CHUNK_REFS`, `RAG_TURN_CONTEXT_PREVIEW_CHARS`,
`RAG_TURN_CONTEXT_STORE_FULL_TEXT=false`.

Validator checks: wall clock < workflow/activity timeout; threshold in (0,1];
gate weights sum to 1; deep-study window > overlap; contradictory flags.

## 10. Eval harness (`evals/conversation/`)

Fixture format (versioned JSON, DR-harness conventions):

```jsonc
{ "version": 1, "domain": "...", "conversations": [
  { "id": "verif-env-deepen", "turns": [
    { "query": "How do I create a verification environment for a new project?",
      "expect": { "actions_allowed": ["RETRIEVE", "ANSWER"],
                  "terminal": "answer",
                  "chunks": [{ "source": "...", "heading_contains": "..." }] } },
    { "query": "tell me more",
      "expect": { "terminal": "answer",
                  "anchor_docs_retained": true,
                  "actions_allowed": ["RETRIEVE", "DEEP_STUDY", "ANSWER"] } }
  ] } ] }
```

Harness: threads one `conversation_id` through the turns, captures the SSE
event trace (reuse `_parse_sse`, tests/server/test_query_endpoints.py:775-790)
or drives `run_turn_loop` directly through DI seams (fake provider / fake
retrieve — the `src/eval` smoke pattern). Metrics: per-turn action accuracy,
terminal accuracy, anchor-doc retention (drift rate), clarify quality
(judge-scored), gate calibration. Pytest marker `eval_conversation`
registered in pyproject. Trace #1 is the verification-environment example
from the live 2026-06-30 session. FakeMemory doubles from
tests/server/test_query_endpoints.py:63-128 are the reuse path for endpoint
tests.

## 11. Risks carried into implementation

1. `retrieve_ranked` is the largest new backend surface (worker activity +
   registration + chain primitive method) — budgeted as its own track, not
   "almost nothing".
2. Temporal: `maximum_attempts=1..3` acceptable only because the activity has
   no LLM calls; document the budget hierarchy TurnBudget > per-activity
   timeout.
3. Console TS has a build step — both `web/src` and built assets must ship
   consistently.
4. Memory growth: preview-capped refs, migration-tolerant decode, note in
   docs.
5. `refactored_char_start == -1` sentinel guard in deep-study anchoring.
6. The card index is not tenant-scoped — the loop must apply its own filters
   on any routed doc ids rather than trusting cards.
