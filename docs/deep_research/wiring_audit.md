# Deep Research × Query-Processing Wiring Audit

Status: 2026-05-08
Scope: confirm gate ordering between `query_processing` and `deep_research`
in `RAGChain.run`, document edge cases, record any wiring fix made.

## TL;DR

- The query-processing **confidence gate fires before deep research**. With
  `mode="query"` and `deep_research=True`, a low-confidence query result
  (action=`ASK_USER`) returns immediately at `rag_chain.py:797` without ever
  invoking `_run_deep_research`.
- With `mode="retrieval"` (used by the eval harness), `process_query`
  short-circuits to `action=SEARCH, confidence=1.0` (`query_processor.py:767-797`),
  so the ask_user gate cannot fire regardless of query messiness. Routing tests
  therefore must run in `mode="query"` to exercise the gate.
- Stages 2–5 (kg_expansion, embedding, hybrid_search, reranking) are correctly
  skipped when `_dr_active=True` via the `not _dr_active` guards.
- The four budget-trip-to-`ask_user` sites added by the recent fix
  (lines 905–921, 950–963, 1001–1014, 1052–1065) all gate on
  `not _dr_active`, so post-DR stage budget exhaustion no longer trips DR
  to ask_user. This part of the claim is verified.

## Call ordering inside `RAGChain.run`

1. PII gate (synchronous, optional).
2. **Stage 1: `process_query` + input rails** (`rag_chain.py:719-738`).
3. Guardrails merge gate — may return `ask_user`/`canned`
   (`rag_chain.py:782-795`).
4. **`query_result.action == QueryAction.ASK_USER` gate**
   (`rag_chain.py:797-810`). Returns `ask_user` to the caller. Does NOT
   inspect `deep_research`.
5. Stage-1 budget-exhausted gate (`rag_chain.py:811-825`). Same shape.
6. State hoisting + `_dr_active = bool(deep_research)`
   (`rag_chain.py:841-852`).
7. If `_dr_active`: `_run_deep_research` + `_rerank_deep_research`
   (`rag_chain.py:854-889`).
8. Stages 2–5 (each guarded by `not _dr_active`).
9. Generation + confidence routing (unchanged).

## Gate priority — what wins when both `deep_research=True` AND query is messy?

| `mode`        | confidence path                         | wins  |
|---------------|------------------------------------------|-------|
| `retrieval`   | bypassed; SEARCH at confidence=1.0       | DR runs |
| `query` (LLM up)   | LLM confidence loop; messy → ASK_USER   | **ask_user wins** (DR skipped) |
| `query` (LLM down) | heuristic: word-count → low confidence  | **ask_user wins** (DR skipped) |
| `query`, fast_path=True | confidence forced to 1.0           | DR runs |

The middle two rows are the edge cases. With the user-facing chat surface
(which uses `mode="query"`), `deep_research=True` does **not** override the
ask_user gate even though the user has explicitly opted into the latency cost
of DR. DR's own decomposer is designed to recover from messy/terse queries via
sub-question expansion; rejecting the query before DR ever runs defeats the
opt-in.

## Decision: fix vs. document

We fix this. Rationale:

- The user explicitly toggled `deep_research=True` and accepted DR's latency
  cost. Routing them to ask_user re-imposes the friction DR was meant to absorb.
- DR is more robust to messy input than the linear pipeline (LLM-driven
  topic decomposition, multi-round sufficiency checks). Letting DR see the
  raw query is strictly better than refusing to retrieve.
- We still respect *truly* empty/injection-flagged queries: those are caught
  in `sanitize_node` and return ASK_USER with confidence=0.0. The fix only
  bypasses the LLM-confidence loop's "this is too vague" verdict, not the
  hard sanitizer rejections.

### Fix shape

In `rag_chain.py` around line 797:

- When `deep_research=True`, bypass the
  `query_result.action == QueryAction.ASK_USER` early-return UNLESS the
  ask_user came from the sanitizer (matched against the sanitizer's two known
  clarification strings: "Your query appears to be empty." or "Your query
  could not be processed."). The sanitizer represents a request-level reject
  (empty/injection), which we honour even when DR is on.
- Skip the budget-exhausted-after-stage-1 ask_user return when
  `deep_research=True` for the same reason (DR has its own budget envelope;
  query_processing's overrun should not cancel DR before it starts).

Why the message-match instead of `confidence > 0.0`: empirically the LLM
evaluator can return `confidence=0.0` with `iterations=max` (parser failure or
genuinely vague queries the loop never lifted), which is a "don't know" not a
"reject". With DR on, "don't know" should fall through to DR's decomposer,
not refuse retrieval.

## Confirmed: stage-2 through stage-5 skip when DR is active

- `kg_expansion`: `if not _dr_active and self._kg_expander` (`rag_chain.py:894`)
- `embedding`: `if not _dr_active` block at `rag_chain.py:926-948`
- `hybrid_search`: `if not _dr_active` block at `rag_chain.py:971-999`
- `reranking`: `if not _dr_active` rerank-the-search-results path; the
  `_dr_active` branch reranks the DR pool instead (`rag_chain.py:1037-1050`)

All four stage-budget-trip sites also gate on `not _dr_active` so that
post-DR latency does not flip the chain into ask_user.

## Edge cases worth tracking

- **PII redaction + DR**: PII gate runs before query_processing and rewrites
  the query; DR receives the redacted form. Behaviour preserved.
- **Guardrails merge gate**: still fires before DR. A guardrails reject (input
  rail violation) routes to ask_user/canned regardless of DR. This is correct —
  guardrails are a security boundary, not a retrieval-quality signal.
- **Sanitizer rejections** (empty/injection): `confidence=0.0` short-circuits
  before the LangGraph loop runs. The fix above preserves these rejections.
- **`mode="retrieval"` callers** (eval harness, retrieval API): unaffected —
  query_processing already returns SEARCH there.

## Test coverage added

- Fixture `evals/retrieval/deep_research/fixtures/asic/ask_user_routing.json`
  with messy-but-answerable queries plus one genuinely-vague control.
- Harness extension: `ModeResult.action`, `expected_action` support,
  `action_match_rate` aggregate.
- New pytest `evals/retrieval/deep_research/test_routing.py` (marker
  `eval_deep_research`) asserts per-query expected_action matches.
