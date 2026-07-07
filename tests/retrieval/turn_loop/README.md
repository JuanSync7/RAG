<!-- @summary
Tests for the turn-level agentic conversation loop package
(`src/retrieval/pipeline/turn_loop/`): per-stage unit tests (controller
decide, retrieve round, deep-study window walk, clarify authoring, answer
draft + confidence gate), the event emitter / charged-call ledger, TurnContext
assembly, and end-to-end orchestrator runs against scripted fakes — no
infrastructure required.
@end-summary -->

# tests/retrieval/turn_loop/

Tests for `src/retrieval/pipeline/turn_loop/` — everything runs through the
`TurnLoopDeps` DI seam with the scripted fakes in `conftest.py` (fake
provider, fake `retrieve_ranked`, fake `fetch_document`), so the whole loop is
exercised with zero Temporal/MinIO/LLM infrastructure.

## Files

| File | Purpose |
| --- | --- |
| `conftest.py` | Shared fakes/factories: scripted `FakeProvider` (agenerate + generate_stream), `make_deps` / `make_budget` / `make_chunk`, JSON response builders, event helpers |
| `test_controller.py` | Controller decide: prompt-var substitution, digest determinism, fail-open ladder (parse failure, provider error, exhausted ledger), query normalization |
| `test_events.py` | `TurnEventEmitter`: sink-error swallow, stream-flag gating, `latest_event_payload`, charge-before-call, `llm_call` payloads, exhausted-ledger skip |
| `test_retrieve.py` | RETRIEVE stage: dedup + dup counting, judge-rank pool order, judge fail-open keep-all, hyde event, seam-failure degradation |
| `test_deep_study.py` | DEEP_STUDY stage: `-1` anchor sentinel, anchored start window, window budget, `answer_found` stop, resume, pool findings provenance |
| `test_clarify.py` | CLARIFY stage: hint cap, event payload, gaps-into-prompt, deterministic fallback |
| `test_answer.py` | ANSWER stage: citation-coverage scanning, gate math, draft-precedes-gate ordering, reasoning-vs-content draft kinds, feedback assembly |
| `test_orchestrator.py` | End-to-end loop runs: dispatch + §8 event ordering, gate loop-back, budget exhaustion best-effort exits, wall-clock stop, never-raise containment |
| `test_context.py` | `build_turn_context` tolerance, chunk-ref cap, pending clarification, digest rendering |

## Running

```bash
uv run --extra dev python -m pytest tests/retrieval/turn_loop -q
```
