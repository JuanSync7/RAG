<!-- @summary
Multi-turn conversation eval suite for the turn-level agentic loop: golden
scripted conversations driven through the REAL run_turn_loop orchestrator
offline (scripted fakes via the TurnLoopDeps DI seam), or against a live API
via /query/stream SSE. Measures action accuracy, terminal accuracy, anchor-doc
retention/drift, expected-chunk recall and clarify-hint quality.
@end-summary -->

# Conversation Eval Suite (turn loop)

Evaluates the turn-level agentic conversation loop
(`src/retrieval/pipeline/turn_loop`, design:
`docs/retrieval/TURN_LOOP_DESIGN.md` section 10) on **golden multi-turn
conversations** with labeled expectations. Two drive modes:

1. **OFFLINE (default — zero infrastructure).** Drives the *real*
   `run_turn_loop` through its `TurnLoopDeps` DI seam with scripted fakes: a
   fake LLM provider returning deterministic canned responses per call purpose
   (controller / judge / hyde / deep-study read / clarify / self-score /
   draft) and a fake `retrieve_ranked` returning fixture-defined chunks. A
   real `TurnContext` is carried across turns, so cross-turn context transfer
   is genuinely exercised — the orchestrator's control flow, budgets, gate and
   context consumption are all under test.
2. **LIVE (opt-in).** Threads one `conversation_id` through
   `POST /query/stream` with `turn_loop: true` and rebuilds the same per-turn
   trace from the typed SSE events (python `urllib` in-process — never
   curl/wget on these hosts).

## Layout

```
conversation/
├── README.md                       # this file
├── __init__.py
├── conftest.py                     # fixture loaders + skip-if-unpopulated / skip-if-unimportable
├── harness.py                      # both drive modes, fakes, TraceResult + metrics, CLI
├── test_conversation_traces.py    # pytest entry (marker: eval_conversation, offline)
└── fixtures/
    └── golden_conversations.json  # versioned golden conversations (see schema)
```

## Fixture schema (version 1)

```jsonc
{
  "version": 1,
  "domain": "asic",
  "budget": { "max_actions": 6 },            // optional TurnBudget field overrides
  "conversations": [
    {
      "id": "verif-env-deepen",
      "turns": [
        {
          "query": "How do I create a verification environment for a new project?",
          "expect": {
            "actions_allowed": ["RETRIEVE", "ANSWER"],   // containment check
            "terminal": "answer",                        // "answer" | "clarify"
            "chunks": [                                  // ChunkMatcher entries
              { "source": "verification/verif_env_setup.md",
                "heading_contains": "Verification Environment" }
            ],
            "anchor_docs_retained": true,                // optional (turn >= 2)
            "anchor_docs": ["doc-verif-env-setup"],      // optional explicit anchors
            "min_hints": 1                               // optional (clarify turns)
          },
          "script": {                                    // offline determinism
            "controller_decisions": [ { "action": "...", "reason": "...", "confidence": 0.8, "args": {} } ],
            "judge_verdicts":       [ { "ranking": [0], "sufficient": true, "confidence": 0.9, "missing_information": "" } ],
            "retrieve_chunks":      [ [ { "chunk_id": "...", "document_id": "...", "source": "...", "heading": "...", "text": "..." } ] ],
            "answer_draft":         "…final draft text…",
            "self_score":           { "self_score": 0.9, "unsupported_claims": [] },
            "clarify_response":     { "question": "...", "hints": ["..."], "scoping_questions": ["..."] },
            "hyde_responses":       [ { "hypothetical_answer": "...", "search_terms": [], "target_aspect": "..." } ],
            "deep_study_reads":     [ { "notes": "...", "answer_found": true, "next_window_hint": "" } ],
            "documents":            { "doc-id": { "title": "...", "text": "…full markdown…" } }
          }
        }
      ]
    }
  ]
}
```

Script semantics: `controller_decisions` and `judge_verdicts` are consumed in
order per turn (the last judge verdict repeats if the loop judges extra
rounds); `retrieve_chunks` is one chunk list per `retrieve_ranked` call
(exhausted → `[]`); the remaining keys are optional with fail-open defaults.
Judge verdicts may be written in the concise shape — the harness expands them
to satisfy both the verbose and concise judge parsers. Unscripted calls are
recorded on the trace (`unscripted_calls`) for debugging, answered with safe
defaults so the loop always terminates.

## Metrics

| Metric | Description |
| --- | --- |
| `action_accuracy` | Fraction of expecting turns whose taken actions are a subset of `actions_allowed` |
| `terminal_accuracy` | Fraction of expecting turns ending in the expected terminal (`answer`/`clarify`) |
| `anchor_retention_rate` / `anchor_drift_rate` | Turns declaring `anchor_docs_retained`: explicit `anchor_docs` must ALL reappear; otherwise ANY overlap with the previous turn's documents counts as retained. Drift = 1 − retention |
| `avg_chunk_recall` | Fraction of `expect.chunks` matchers hit in the turn's final evidence pool |
| `clarify_hint_accuracy` | Turns declaring `min_hints`: clarification carries at least that many hints |
| `clarify_quality` | **Placeholder hook** — judge-scored later (live); `N/A` offline |

## Running

```bash
# Offline (default; no infra — marker keeps it out of normal CI runs):
uv run --extra dev python -m pytest evals/conversation -m eval_conversation -q

# CLI report, offline:
uv run --extra dev python -m evals.conversation.harness \
    --fixtures evals/conversation/fixtures/golden_conversations.json \
    --output /tmp/conversation_eval_report.json

# CLI report, LIVE against a running API (or set RAG_EVAL_API_BASE;
# optional bearer token via RAG_EVAL_API_TOKEN):
uv run --extra dev python -m evals.conversation.harness \
    --fixtures evals/conversation/fixtures/golden_conversations.json \
    --output /tmp/conversation_eval_report.json \
    --api-base http://127.0.0.1:8000
```

Offline tests **skip** (with the exact import error) while the orchestrator
module (`turn_loop/orchestrator.py`, `run_turn_loop`) has not landed, and skip
if the fixture file is unpopulated — the suite never fails for missing
prerequisites, only for behavioral regressions.

## Golden conversations

| id | Exercises |
| --- | --- |
| `verif-env-deepen` | The verification-environment example (live 2026-06-30 session): answer, then "tell me more" deepening with anchor-doc retention |
| `clarify-underspecified` | Underspecified query → terminal clarification with hints; next turn picks a hint and answers (pending-clarification transfer) |
| `refine-with-anchor` | Broad answer, then a "that is not what I want" narrowing turn: FRESH retrieve while retaining the named anchor document |
