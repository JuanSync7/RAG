---
slice_id: P5-faithfulness-scoring
validable_outcome: |
  `uv run pytest tests/eval/test_pack_faithfulness.py -q` reports all FAST tests green, and the live test correctly skips when infra is down (or passes when infra is up). Specifically:
    1. `test_score_goldens_calls_judge_per_query` (FAST): `score_goldens(results, goldens, judge_client)`
       iterates each golden in `results.per_query` and calls `judge_client.score(JudgeQuestion(...))`
       EXACTLY ONCE per non-skipped golden. Uses a fake `JudgeClient` that records every call.
       Asserts:
       - Number of judge calls == number of goldens with non-empty `retrieved_chunks`.
       - Each `JudgeQuestion.qid`, `qtype`, `query`, `expected_answer_span`, and `chunk_texts`
         match the golden + retrieval result.
       - Goldens with empty `retrieved_chunks` are SKIPPED (not scored, counted in
         `total_queries_skipped`).
    2. `test_score_goldens_returns_frozen_faithfulness_results` (FAST):
       returned `FaithfulnessResults` is `@dataclass(frozen=True)`, holds
       `collection_name`, `per_query: Mapping[str, QueryFaithfulnessResult]`,
       `total_queries_scored: int`, `total_queries_skipped: int`. Mutation raises
       `FrozenInstanceError`. Each `QueryFaithfulnessResult` has `qid`, `qtype`,
       `score: float`, `reasoning: str`.
    3. `test_aggregate_faithfulness_by_qtype` (FAST):
       `aggregate_faithfulness_by_qtype(faithfulness_results, goldens)` averages
       per-query scores into per-qtype mean scores. Mirrors P4's
       `aggregate_recall_by_qtype` contract:
       - Skipped goldens (no retrieved chunks) are EXCLUDED from the denominator.
       - Qtypes with zero scored goldens are OMITTED from the result dict (no nan, no 0.0).
       - Uses `statistics.mean`.
    4. `test_retrieve_for_goldens_now_includes_chunk_texts` (FAST):
       `QueryRetrievalResult` gains an additive `retrieved_chunks: tuple[str, ...]`
       field parallel to `retrieved_sources`. Monkeypatches
       `src.eval.runner.retrieve.search` to return synthetic `SearchResult`s with
       `text="chunk-A"`, `text="chunk-B"`. Asserts `result.retrieved_chunks == ("chunk-A", "chunk-B")`
       AND `result.retrieved_sources` still works (P4 contract preserved — no regression).
    5. `test_eval_report_extended_with_faithfulness` (FAST):
       `EvalReport` gains `faithfulness_by_qtype: Mapping[str, float]` and
       `total_queries_judged: int` fields. Existing fields (`per_query_recall`,
       `recall_by_qtype`, etc.) remain unchanged. A defaulted construction with
       only the existing P4 fields still works (additive). The mutation
       `report.faithfulness_by_qtype = {}` raises `FrozenInstanceError`.
    6. `test_build_eval_report_wires_recall_and_faithfulness` (FAST):
       `build_eval_report(retrieval_results, faithfulness_results, recall_by_qtype, per_query_recall) -> EvalReport`
       returns an `EvalReport` whose `recall_by_qtype` matches the input AND
       `faithfulness_by_qtype` matches the aggregation of `faithfulness_results`.
       Asserts the back-references are correct and `total_queries_judged` ==
       `faithfulness_results.total_queries_scored`.
    7. `test_judge_question_for_adversarial_carries_none_span` (FAST):
       Adversarial goldens have `expected_answer_span = None`. `score_goldens` must
       still build a `JudgeQuestion` (with `expected_answer_span=None`) and call
       the judge. The judge's qtype-aware rubric handles the None case (it scores
       refusal correctness — P5.0 prompt already specifies). Asserts the
       `JudgeQuestion.expected_answer_span` passed to the fake judge is None and
       the resulting `QueryFaithfulnessResult` is recorded.
    8. `test_score_goldens_opentitan_live` (SLOW + INTEGRATION):
       `pytestmark = [pytest.mark.slow, pytest.mark.integration]` decorators at
       function level (NOT module level — fast tests share the file). Skip block:
       Weaviate via `create_persistent_client()` try/except, then MinIO socket
       probe via `MINIO_ENDPOINT`, then LLM-API-key probe via
       `os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or
       os.environ.get("ANTHROPIC_API_KEY")`. When all three are up:
       executes the full chain: load pack → execute_plan → retrieve_for_goldens →
       score_goldens against the opentitan_riscv pack's TWO factoid goldens (limit
       to 2 to keep cost bounded). Asserts:
       - At least 1 factoid was scored (no infra-mute).
       - Mean factoid `faithfulness_by_qtype["factoid"] >= 0.3` (low floor —
         this is a smoke test, not a quality gate; quality gates live in
         thresholds.yaml in a later slice).
       - Best-effort collection cleanup in `finally`.
  PLUS: `uv run pytest tests/eval/ tests/vector_db/ tests/retrieval/ -q -m "not slow and not integration"` stays fully green (regression).
touches:
  - src/eval/__init__.py
  - src/eval/runner/__init__.py
  - src/eval/runner/retrieve.py
  - src/eval/runner/report.py
  - src/eval/runner/faithfulness.py
  - tests/eval/test_pack_faithfulness.py
depends_on:
  - P5.0-judge-client
  - P4-retrieval-metrics
  - P3-pack-ingest-runner
---

# P5 — Faithfulness scoring executor

## End-state (validable, machine-checkable)

A new `score_goldens(retrieval_results: RetrievalResults, goldens: Mapping[str, list[Golden]], judge_client: JudgeClient) -> FaithfulnessResults` function at `src/eval/runner/faithfulness.py` consumes P4's retrieval output and P5.0's judge client to score each golden's faithfulness on a 0.0–1.0 scale. Plus `aggregate_faithfulness_by_qtype` parallel to P4's recall aggregation. `EvalReport` is additively extended with `faithfulness_by_qtype` and `total_queries_judged`. `QueryRetrievalResult` is additively extended with `retrieved_chunks: tuple[str, ...]` so the judge can see the chunk text. 7 fast tests + 1 dual-marker live test.

## Current state (surveyed)

- `src/eval/runner/retrieve.py` (P4) returns `QueryRetrievalResult(qid, qtype, query, retrieved_sources: tuple[str, ...], k)`. Does NOT carry chunk text. P5 adds `retrieved_chunks: tuple[str, ...]` additively.
- `src/vector_db/common/schemas.py:39` defines `SearchResult` with `text: str`, `score: float`, `metadata: dict`. Source of chunk text: `result.text`.
- `src/eval/runner/judge.py` (P5.0) exposes `JudgeClient.score(JudgeQuestion) -> JudgmentScore` and `JudgeQuestion(qid, qtype, query, expected_answer_span, chunk_texts)`. Ready to consume.
- `src/eval/runner/report.py` defines `EvalReport(collection_name, k, per_query_recall, recall_by_qtype, total_queries_scored, total_queries_skipped)`. Extend with `faithfulness_by_qtype: Mapping[str, float] = field(default_factory=dict)` and `total_queries_judged: int = 0` so existing P4 construction still works.
- `src/eval/runner/metrics.py` (P4) has `aggregate_recall_by_qtype` — the canonical aggregation pattern (omit-empty-qtype, `statistics.mean`). P5 mirrors this contract.

## Plan-spec alignment

P5 is the executor companion to P5.0's contract. It bridges P4's retrieval output into the judge surface. Outputs a `FaithfulnessResults` parallel to P4's `RetrievalResults`, and an extended `EvalReport` that carries both retrieval AND faithfulness aggregates. The live test exercises the FULL chain (plan → ingest → retrieve → judge) on the opentitan_riscv pack.

Limited to TWO factoid goldens in the live test to bound LLM cost. Real quality gating against `thresholds.yaml` is a separate later slice (P5.5 or P6).

## Required deliverables

### Code

1. **`src/eval/runner/retrieve.py`** — additive extension:
   - Add `retrieved_chunks: tuple[str, ...]` to `QueryRetrievalResult` (after `retrieved_sources`).
   - In `retrieve_for_goldens`, after `hits = search(...)`, populate `chunks = tuple(hit.text for hit in hits)` and pass to `QueryRetrievalResult(retrieved_chunks=chunks, ...)`.
   - Empty-query branch: `retrieved_chunks=()`.
   - `@summary` block updated.

2. **`src/eval/runner/faithfulness.py`** — new module:
   - `@dataclass(frozen=True)` `QueryFaithfulnessResult(qid, qtype, score, reasoning)`.
   - `@dataclass(frozen=True)` `FaithfulnessResults(collection_name, per_query: Mapping[str, QueryFaithfulnessResult], total_queries_scored: int, total_queries_skipped: int)`.
   - `score_goldens(retrieval_results, goldens, judge_client) -> FaithfulnessResults`:
     - Iterate `retrieval_results.per_query`. For each `QueryRetrievalResult`:
       - If `len(retrieved_chunks) == 0` → SKIP (increment `total_queries_skipped`).
       - Else: find the matching `Golden` (search by qid through `goldens` qtype map — small N, linear scan is fine). Build `JudgeQuestion(qid, qtype=result.qtype, query=result.query, expected_answer_span=golden.expected_answer_span, chunk_texts=result.retrieved_chunks)`. Call `judge_client.score(question)`. Record as `QueryFaithfulnessResult(qid, qtype, score, reasoning)`.
     - Return `FaithfulnessResults`.
   - `aggregate_faithfulness_by_qtype(faithfulness_results, goldens) -> dict[str, float]`:
     - Mirror `aggregate_recall_by_qtype`: per qtype, collect scored qids, average, omit empty qtypes.

3. **`src/eval/runner/report.py`** — additively extend `EvalReport`:
   - Add `faithfulness_by_qtype: Mapping[str, float] = field(default_factory=dict)`.
   - Add `total_queries_judged: int = 0`.
   - Add a new top-level helper `build_eval_report(retrieval_results, faithfulness_results, recall_by_qtype, per_query_recall) -> EvalReport`. (If you prefer, place this helper in a separate module — but `report.py` is fine for a builder colocated with the dataclass.)
   - Keep all existing P4 fields and their semantics unchanged.

4. **`src/eval/runner/__init__.py`** — re-export `QueryFaithfulnessResult`, `FaithfulnessResults`, `score_goldens`, `aggregate_faithfulness_by_qtype`, `build_eval_report`. Add to `__all__`. Sorted.

5. **`src/eval/__init__.py`** — surface the new exports.

### Tests (`tests/eval/test_pack_faithfulness.py`)

7 fast tests + 1 dual-marker live test enumerated above. Patterns:

- Fast tests construct synthetic `RetrievalResults` directly (no Weaviate). Build a fake `JudgeClient` with a `.score(question)` method that records calls and returns `JudgmentScore(score=0.7, reasoning="ok")`.
- Test 4 uses `monkeypatch.setattr("src.eval.runner.retrieve.search", fake_search)` and a fake `get_embedding_provider` + fake `create_persistent_client`/`close_client` per the P4 test pattern.
- Test 8 (live) follows the P3/P4 dual-marker pattern. Use `@pytest.mark.slow` + `@pytest.mark.integration` decorators on the test function. Probe ALL THREE: Weaviate, MinIO, LLM API key. Best-effort cleanup in `finally`. Use the helper `_skip_if_no_judge_llm()` that checks for any of `LLM_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` in env.

## Constraints

- Boundary discipline: ONLY the 6 files in `touches`.
- No new runtime deps.
- The `EvalReport` extension MUST be additive — existing P4 construction with only `(collection_name, k, per_query_recall, recall_by_qtype, total_queries_scored, total_queries_skipped)` MUST still work. Use `field(default_factory=dict)` and `int = 0` defaults.
- The `QueryRetrievalResult` extension MUST be additive — P4 test (`test_retrieve_for_goldens_wires_search_per_query`) MUST still pass without modification. Verify by running it post-implementation.
- No marker softening, no test xfailed, no skip outside the live-infra-probe path.
- `score_goldens` MUST NOT call any LLM directly — only through `judge_client.score`. The judge_client is the seam.
- No retry / backoff logic in P5. If the judge raises, let it propagate (the live test will catch infra issues at the skip layer).
- No `samples_per_claim > 1` averaging in P5 — single-sample only. Multi-sample is a later slice.

## Anti-gaming guards

- Test 1 must assert EXACT JudgeQuestion field values, not just "judge was called". Capture and inspect.
- Test 4 must assert BOTH `retrieved_sources` AND `retrieved_chunks` are populated — single-field assertion lets the additive contract drift.
- Test 7 (None expected_answer_span) ensures adversarial/out_of_corpus goldens flow through without `KeyError` or `AttributeError`. If `score_goldens` blindly does `golden.expected_answer_span.lower()` it will crash on None — this test catches it.
- Mutation probe A (SA2 post-green): in `score_goldens`, change the skip condition from `len(retrieved_chunks) == 0` to `len(retrieved_sources) == 0`. Test 1 should still pass if all results have BOTH populated, but test 4 will catch the contract drift if a result has sources but no chunks (the assertion is on chunks). Actually a cleaner mutation: REMOVE the skip branch entirely. Test 1 MUST RED — the call count would exceed the goldens-with-chunks count if a chunkless result is passed in.
- Mutation probe B: change `judge_client.score(question).score` to `0.5` (hardcoded). Test 1 still passes (judge call recorded), but test 6 fails because the aggregated score differs from the fake's return value. To make this fully discriminating, test 6's fake judge MUST return varied scores (e.g. {q1: 0.9, q2: 0.3}) so the aggregation has signal.
- Red-reason proof: before any implementation in `src/eval/runner/faithfulness.py`, run `uv run pytest tests/eval/test_pack_faithfulness.py -q`. Expected: `ImportError: cannot import name 'score_goldens' from 'src.eval.runner'`. NOT a `FileNotFoundError` on pack files. Capture into `.delivery/plan/wp-P5-faithfulness-scoring-red-proof.txt`.

## Out of scope (defer)

- Quality gating against `thresholds.yaml` for faithfulness floors — separate later slice.
- `samples_per_claim > 1` multi-sample judge calls + averaging.
- A CLI entrypoint to run the full pack-eval loop (`eval run <pack>`).
- Caching judge results across runs.
- Token-cost telemetry / budget enforcement.
- Differentiating fixture-style judge prompts per qtype (the P5.0 prompt already handles this in-prompt).
- Pre-recorded cassette replay for fast tests of the live path — fast tests use the fake JudgeClient, that's sufficient.

## Agent green-gate vs. human-review checkpoint

- **Agent green-gate (this slice ends here):** 7 fast tests green, live test passes-or-skips depending on infra, regression sweep green, commit on `feat/p5-faithfulness-scoring`.
- **Human-review checkpoint (separate):** reviewer runs the live test against real infra + real judge LLM, inspects per-query reasoning, confirms scores are sensible. May iterate on the P5.0 prompt body based on observed scores.
