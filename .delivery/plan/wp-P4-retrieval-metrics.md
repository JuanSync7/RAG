---
slice_id: P4-retrieval-metrics
validable_outcome: |
  `uv run pytest tests/eval/test_pack_retrieve_metrics.py -q` reports all FAST tests green, and the live test passes-or-skips depending on infra:
    1. `test_retrieve_for_goldens_wires_search_per_query` (FAST): monkeypatches `src.eval.runner.retrieve.search` and `src.eval.runner.retrieve.get_embedding_provider`. Asserts `search` is called EXACTLY once per golden query (across all qtypes), with `collection=<plan.collection_name>`, `limit=k`, and `query=<golden.query>`. Order: same as iteration order of `pack.goldens[qtype]` for each qtype.
    2. `test_recall_at_k_pure_function` (FAST): table-driven on `recall_at_k(retrieved_sources, expected_sources)`:
       - `(["docs/a.md","docs/b.md","docs/c.md"], ["docs/a.md"])` → `1.0` (single-gold hit)
       - `(["docs/x.md","docs/y.md"], ["docs/a.md"])` → `0.0` (miss)
       - `(["docs/a.md","docs/y.md"], ["docs/a.md","docs/b.md"])` → `0.5` (partial: 1 of 2 golds)
       - `(["docs/a.md","docs/b.md"], ["docs/a.md","docs/b.md"])` → `1.0` (full)
       - `([], ["docs/a.md"])` → `0.0` (empty retrieved)
    3. `test_aggregate_recall_by_qtype` (FAST): given synthetic `per_query` results across 3 qtypes where:
       - `factoid` has 2 queries, both with non-empty `expected_source_docs`, recalls = [1.0, 0.0]
       - `qfs` has 1 query with non-empty `expected_source_docs`, recall = 0.5
       - `adversarial` has 1 query with `expected_source_docs == []`
       Expected output: `{"factoid": 0.5, "qfs": 0.5}`. The `adversarial` qtype is EXCLUDED from the result dict (no key) because it has no scoreable queries.
    4. `test_eval_report_is_frozen` (FAST): `EvalReport` is `@dataclass(frozen=True)`. Mutation raises `FrozenInstanceError`. Field shape: `collection_name`, `k`, `per_query_recall: Mapping[str, float]`, `recall_by_qtype: Mapping[str, float]`, `total_queries_scored: int`, `total_queries_skipped: int`.
    5. `test_opentitan_recall_at_5_meets_floor` (SLOW + INTEGRATION): live test with `@pytest.mark.slow` + `@pytest.mark.integration` decorators. Probes Weaviate/MinIO (skip on infra-down). On infra-up: load OpenTitan pack → `execute_plan(plan)` → `retrieve_for_goldens(plan.collection_name, pack.goldens, k=5)` → `aggregate_recall_by_qtype(...)`. Assert `recall_by_qtype["factoid"] >= 0.5`. Print all per-qtype recalls to stdout for human-review visibility. Best-effort collection cleanup in `finally`.
  PLUS: `uv run pytest tests/eval/ tests/vector_db/ tests/retrieval/ -q -m "not slow and not integration"` stays fully green.
touches:
  - src/eval/__init__.py
  - src/eval/runner/__init__.py
  - src/eval/runner/retrieve.py
  - src/eval/runner/metrics.py
  - src/eval/runner/report.py
  - tests/eval/test_pack_retrieve_metrics.py
depends_on:
  - P3-pack-ingest-runner
---

# P4 — Retrieval + recall@k metrics over the ingested pack

## End-state (validable, machine-checkable)

The runner gains two new capabilities: (a) `retrieve_for_goldens(collection_name, goldens, k=5) -> RetrievalResults` that runs each golden's query through the existing vector search against the named collection, and (b) `recall_at_k(retrieved_sources, expected_sources) -> float` + `aggregate_recall_by_qtype(results, goldens) -> dict[str, float]` for the metric. The new `EvalReport` frozen dataclass holds the aggregated outcome. A live test asserts `recall@5 >= 0.5` on factoid against the real OpenTitan ingest.

## Current state (surveyed by SA1)

- `src.vector_db.search(client, query, query_embedding, alpha, limit, filters=None, collection=None) -> list[SearchResult]` is the canonical retrieval entry point. `SearchResult.metadata["source"]` holds the chunk source path (e.g. `"docs/foo.md"`) — EXACTLY matching golden `expected_source_docs` format.
- `src.core.embeddings.get_embedding_provider().embed_documents([query])[0]` produces a query vector (BGE-M3 local model, lazy-loaded).
- `src.vector_db.create_persistent_client()` / `close_client()` for client lifecycle (same as P3 live test).
- `Golden` schema: `qid`, `qtype`, `query`, `expected_source_docs: list[str]`. Non-empty for factoid/qfs/multi_topic/multi_aspect; empty for adversarial/out_of_corpus.
- `pack.goldens: dict[str, list[Golden]]`.
- No existing `recall_at_k` or `EvalReport` anywhere in `src/eval/`. New ground.
- P3's `IngestReport` lives in `src/eval/runner/report.py`. P4 extends this file by adding `EvalReport` alongside.

## Plan-spec alignment

P4 is the first slice that READS what P3 wrote. It validates the pipeline end-to-end: pack content (P1/P2) → ingest (P3) → retrieval → metric. This is the first time the goldens authored in P2 are actually exercised against real retrieval; if the OpenTitan factoid recall fails, the regression is either in the goldens (bad grounding) or retrieval (wrong collection routing). Either failure is informative.

## Required deliverables

### Code

1. **`src/eval/runner/retrieve.py`** — new module:
   - Imports at module top: `from src.vector_db import search, create_persistent_client, close_client`, `from src.core.embeddings import get_embedding_provider`, `from src.eval.pack import Golden`, plus stdlib `dataclasses`, `typing`.
   - `@dataclass(frozen=True)` `QueryRetrievalResult`:
     - `qid: str`, `qtype: str`, `query: str`, `retrieved_sources: tuple[str, ...]`, `k: int`.
   - `@dataclass(frozen=True)` `RetrievalResults`:
     - `collection_name: str`, `k: int`, `per_query: Mapping[str, QueryRetrievalResult]` (key = qid).
   - `retrieve_for_goldens(collection_name: str, goldens: Mapping[str, list[Golden]], k: int = 5) -> RetrievalResults`:
     - Opens Weaviate client via `create_persistent_client()` (try/finally close).
     - Loads embed provider once via `get_embedding_provider()`.
     - For each qtype, for each golden in iteration order, embed query → call `search(client, query, query_embedding, alpha=0.5, limit=k, collection=collection_name)` → extract `result.metadata["source"]` per chunk → build `QueryRetrievalResult` with `retrieved_sources` in retrieval rank order.
     - Aggregates into `RetrievalResults`.
   - `@summary` block.

2. **`src/eval/runner/metrics.py`** — new module:
   - `def recall_at_k(retrieved_sources: Sequence[str], expected_sources: Sequence[str]) -> float`:
     - Returns `len(set(expected_sources) & set(retrieved_sources)) / len(expected_sources)` if expected_sources is non-empty.
     - If `expected_sources == []`, raise `ValueError("recall_at_k requires non-empty expected_sources")` — callers must filter empty-gold goldens BEFORE calling.
   - `def aggregate_recall_by_qtype(results: RetrievalResults, goldens: Mapping[str, list[Golden]]) -> dict[str, float]`:
     - For each qtype, collect per-query recalls ONLY for goldens with non-empty `expected_source_docs`.
     - If a qtype has zero scoreable goldens (all have empty `expected_source_docs`), OMIT it from the result dict entirely. No `nan`, no zero — just omit.
     - Return `dict[qtype, mean]`. Sorted iteration order is fine but not required.
   - `@summary` block.

3. **`src/eval/runner/report.py`** — extend with `EvalReport`:
   - `@dataclass(frozen=True)` `EvalReport`:
     - `collection_name: str`
     - `k: int`
     - `per_query_recall: Mapping[str, float]` (key = qid)
     - `recall_by_qtype: Mapping[str, float]`
     - `total_queries_scored: int`
     - `total_queries_skipped: int`
   - Do NOT modify existing `IngestReport` shape.

4. **`src/eval/runner/__init__.py`** — re-export `retrieve_for_goldens`, `recall_at_k`, `aggregate_recall_by_qtype`, `EvalReport`, `RetrievalResults`, `QueryRetrievalResult`. Add to `__all__`.

5. **`src/eval/__init__.py`** — add new symbols to the public surface alongside existing P0.5/P3.0/P3 exports. Keep `__all__` sorted.

### Tests (`tests/eval/test_pack_retrieve_metrics.py`)

5 tests per `validable_outcome`.

Patterns:
- Fast tests monkeypatch `src.eval.runner.retrieve.search` (the LOCAL name) AND `src.eval.runner.retrieve.get_embedding_provider` (replace with a `lambda: FakeEmbedder()` that has `embed_documents` returning `[[0.0] * 1024]`). Also patch `create_persistent_client` and `close_client` with no-op stubs.
- Test 1's spy captures every `search` call's kwargs; assertions check call count == sum of len(goldens[qtype]) for all qtypes, plus per-call kwarg shape.
- Test 2 is a pure-function table test. Use `@pytest.mark.parametrize`.
- Test 3 builds synthetic `RetrievalResults` and `goldens` dicts in-test (Golden objects via `Golden(qid=..., qtype=..., query=..., expected_source_docs=...)` — no pack load needed). Asserts dict equality with `pytest.approx` for floats.
- Test 5 (live): copy the dual-marker + skip pattern from P3's `test_execute_plan_opentitan_live`. After execute_plan + retrieve + aggregate, assert `recall_by_qtype["factoid"] >= 0.5`. Use `pytest.fail(...)` with the full recall dict as message on assertion failure for visibility. `print()` the recall_by_qtype dict so `pytest -s` shows actual values.

## Constraints

- Boundary discipline: ONLY the 6 files in `touches`.
- No new runtime deps.
- Dual-marker decorators on the live test (`@pytest.mark.slow` + `@pytest.mark.integration`).
- No tests skipped/xfailed outside the live-infra-probe path.
- DO NOT modify `plan.py` or `execute.py` (P3.0 / P3 code stays untouched).
- `retrieve_for_goldens` MUST NOT call `execute_plan` — they are independent steps. The live test composes them.
- `recall_at_k([], expected)` → `0.0`. `recall_at_k(retrieved, [])` → `ValueError` (caller must filter). This asymmetry is intentional: empty retrieval is a real (bad) outcome; empty expected is a categorization mismatch.

## Anti-gaming guards

- Test 1 monkeypatches the LOCAL name (`src.eval.runner.retrieve.search`), not `src.vector_db.search`. If SA2 lazy-imports `search` inside the function, the patch won't bind — verify module-top import in `retrieve.py`.
- Test 5 (live) MUST NOT mock anything. Real retrieval against the freshly-ingested collection.
- Mutation probes for SA2 to run after green:
  - Mutation A: change `set(expected) & set(retrieved)` to `set(expected) ^ set(retrieved)` (xor instead of intersection). Test 2 partial case MUST RED-FAIL.
  - Mutation B: change `aggregate_recall_by_qtype` to INCLUDE empty-gold qtypes with `recall=0.0`. Test 3 MUST RED-FAIL on the `"adversarial" not in result` assertion.
  - Mutation C: change `alpha=0.5` to `alpha=0.0` (keyword-only) in `retrieve_for_goldens`. Test 1 MAY pass (if not asserting alpha), but live test 5 likely changes recall. Soft probe — not required for slice closure, but worth running on the live test if infra is available.
- Red-reason proof: before any implementation, write the test file and run `uv run python -m pytest tests/eval/test_pack_retrieve_metrics.py -q -m "not slow and not integration"`. Expected: `ImportError: cannot import name 'retrieve_for_goldens' from 'src.eval.runner'`. NOT a Weaviate connection error. Capture verbatim into `.delivery/plan/wp-P4-retrieval-metrics-red-proof.txt`.

## Out of scope (defer)

- Judge / answer-faithfulness metrics — P5.
- nDCG, MRR, or any rank-aware metric — could be a P4.x follow-up; recall@k is the minimal-viable metric to validate the loop.
- Threshold pinning per qtype in `thresholds.yaml` enforcement against actual measurement — could be P4.5 or folded into P5.
- Caching query embeddings between runs — P-future.
- Retrieval against PDF / multi-modal corpora — out of pack scope.

## Agent green-gate vs. human-review checkpoint

- **Agent green-gate (this slice ends here):** 4 fast tests green, live test passes-or-skips, regression sweep green, commit on `feat/p4-retrieval-metrics`.
- **Human-review checkpoint (separate, unblocks P5):** human runs the live test against real infra and inspects the printed per-qtype recall dict. If factoid recall@5 is anywhere near 1.0 the goldens + retrieval are tightly aligned; if it's 0.5–0.7 there's room for improvement but the loop works; if it's < 0.3 there's a real problem (wrong collection, broken embedding, paraphrased grounding).
