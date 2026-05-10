# Deep Research — Efficiency Improvements

This guide documents the tightened **early-stop gate** and **adaptive-depth**
heuristics added to the deep-research orchestrator
(`src/retrieval/pipeline/deep_research.py`). The two changes are coupled and
their goal is the same: reduce LLM-call cost on simple/single-aspect queries
without regressing recall on disjoint or multi-aspect queries.

Companion docs: [`observability.md`](observability.md) covers Prometheus,
Langfuse, and Temporal surfaces; this doc focuses on the loop heuristics.

## TL;DR

Before, the loop terminated as soon as a sufficiency check returned
`is_sufficient: true`. A single noisy "yes" from the auditor was enough to
stop, which over-fired for terse queries where the seed retrieval happens
to look adequate after one pass — but it also under-fired (the orchestrator
kept recursing on coherent single-topic queries that the auditor flagged as
insufficient out of an abundance of caution).

Two coupled improvements:

1. **Tightened early-stop gate** — `is_sufficient: true` is necessary but
   not sufficient. The pool must also clear a chunk floor, a source-diversity
   floor, and a non-decreasing confidence trend.
2. **Adaptive depth** — after the first decompose, the LLM's
   `split_confidence` and topic count cap `max_depth` so soft splits don't
   recurse as deeply as confident multi-topic ones.

## Early-stop gate

The new gate evaluates `(suff, pool)` and returns one of:

- `"iter0_sufficient"` — fired on the very first sufficiency check (round 0).
- `"sufficient+floor+diversity"` — fired on a downstream sufficiency check
  when the gate's full criteria are satisfied.
- `None` — keep going.

Gate criteria (all required when `early_stop_enabled=True`):

| Condition | Default | Why |
|---|---|---|
| `suff.is_sufficient == True` | — | the auditor must have positively voted to stop |
| `len(pool.chunks_by_id) >= early_stop_min_chunks` | `3` | one-chunk pools are not credible evidence |
| distinct `metadata['source']` count `>= early_stop_min_sources` | `2` | a single-source pool may be a corpus blind spot |
| confidence trend non-decreasing on this pool | — | reject oscillation: a confident "yes" that later drops in confidence keeps the loop alive |

The trend check uses `suff["confidence"]` when present (a float in `[0,1]`),
falling back to `1.0` for `is_sufficient=true`. Single-iteration pools always
pass the trend check trivially. Confidence values are appended to
`TopicPool.confidence_history`.

**Hard floor**: round 0 (root retrieve + first sufficiency check) always
runs, even when the gate would later fire. The gate is consulted *after*
the iteration completes.

### Disabling the gate

Tests and back-compat callers can revert to legacy single-bool stop
semantics by setting `early_stop_enabled=False`:

```python
budget = DeepResearchBudget(early_stop_enabled=False)
```

Under that flag, any `is_sufficient: true` stops the loop immediately
(reason `"iter0_sufficient"` for round 0, `"sufficient_legacy"` otherwise).

## Adaptive depth

After the first decompose call returns, the orchestrator inspects:

- `n_topics = len(topics)` — clipped to `budget.max_topics`
- `split_confidence = float(decomp.get("split_confidence", 0.0))`

and rewrites `self._budget.max_depth` according to:

| `n_topics` | `split_confidence` | Action |
|---|---|---|
| `1` | any | cap `max_depth = min(max_depth, 2)` — coherent query, don't recurse widely |
| `>= 2` | `>= 0.7` | preserve `max_depth` — confident split, keep generous depth |
| `>= 2` | `< 0.7` | cap `max_depth = min(max_depth, 2)` — soft split, don't waste recursion |

Missing `split_confidence` defaults to `0.0`, i.e. the low-confidence path —
this is the backward-compatible behavior when the LLM returns the legacy
prompt's response shape.

### Prompt update

`prompts/deep_research_multi_queries_gen.md` was extended to ask the model
for `split_confidence` in its JSON output. The orchestrator parses it
defensively: a missing or non-numeric field becomes `0.0`.

`prompts/deep_research_sufficiency_check.md` was likewise extended to ask
for an optional `confidence` field used by the early-stop trend check.

## Configuration

New fields on `DeepResearchBudget`:

```python
early_stop_enabled: bool = True
early_stop_min_chunks: int = 3
early_stop_min_sources: int = 2
```

New fields on `DeepResearchResult`:

```python
early_stop_reason: Optional[str] = None  # "iter0_sufficient" | "sufficient+floor+diversity" | "sufficient_legacy" | None
split_confidence: float = 0.0            # value parsed from the first decompose
```

New field on `TopicPool`:

```python
confidence_history: list[float] = field(default_factory=list)
```

## Observability

The Langfuse parent trace receives two new attributes:

- `split_confidence: float` — set after the first decompose returns.
- `early_stop_reason: str` — set when the early-stop gate fires.

The `rag_dr_runs_total{outcome="early_stop"}` counter still tracks
early-stop runs (semantics: any run that did not reach decompose). Tighter
gating only affects when the counter increments, not its label set.

## Expected impact

The efficiency fixture (`evals/retrieval/deep_research/fixtures/asic/efficiency_queries.json`)
targets 4 single-aspect queries that previously consumed 8 LLM calls each.
Target: `<= 3` calls on single-aspect, no recall regression on the
multi-aspect golden set.

End-to-end measurement on the worktree was **infra-gated**: the local
Weaviate collection was empty (`ObjectsCount: 0`) and the embedded harness
exceeded the smoke-test timeout. Pre-merge eval runs against a populated
collection should compare:

- `aggregate.avg_llm_calls.deep_research` on `efficiency_queries.json` —
  expect a drop versus the pre-change baseline.
- `aggregate.avg_recall_at_k.deep_research` on `golden_queries.json` —
  expect parity with the pre-change baseline (no regression on dr-001,
  dr-002, dr-004).

Test coverage is in `tests/retrieval/test_dr_efficiency.py` (10 tests
across early-stop, adaptive-depth, and result-field groups).

## Per-topic reranking (with split-confidence gate)

When DR decomposes a query into multiple topics, simply concatenating
per-topic chunks and reranking against the original question leaves
breadth on the table for genuinely disjoint queries (e.g. "compare lint
runtime overhead with formal proof memory footprint"). The original
question is a poor anchor for either sub-topic, and the global rerank
ends up dominated by whichever topic has more chunks.

### When it fires

The orchestrator applies per-topic rerank in `_apply_per_topic_rerank()`
(called inside `research()` just before returning) when **all** of the
following hold:

1. `result.is_unified is False` (LLM emitted a true multi-topic split)
2. `len(result.topic_pools) >= 2`
3. `result.split_confidence >= budget.per_topic_rerank_min_confidence`
   (default `0.6`)
4. `budget.per_topic_rerank_enabled is True` (default `True`)
5. A reranker was wired into `DeepResearch(reranker=...)`
   (RAGChain plumbs `self.reranker` automatically)

### Round-robin contract

For each `TopicPool`, the orchestrator calls
`reranker.rerank(query=pool.rerank_anchor, documents=pool.chunks,
top_k=per_topic_top_k)` (anchored on the pool's `rerank_anchor`,
**not** the original query). The N per-topic ranked lists are merged in
strict round-robin order:

    t1[0], t2[0], ..., tN[0], t1[1], t2[1], ..., tN[1], ...

Every topic contributes its top-1 before any topic contributes its
second. The caller (`RAGChain._rerank_deep_research`) applies a final
`rerank_top_k` cap on the merged list.

### Config knobs (`DeepResearchBudget`)

| Field | Default | Purpose |
| --- | --- | --- |
| `per_topic_rerank_enabled` | `True` | Master switch |
| `per_topic_rerank_min_confidence` | `0.6` | Below this, fall through to legacy behaviour (low-confidence splits often *are* one topic) |
| `per_topic_top_k` | `3` | Per-pool slice taken before round-robin merge |

### Surfacing on the result

`DeepResearchResult` carries:

- `per_topic_rerank_applied: bool`
- `per_topic_rerank_skipped_reason: str | None` — one of `"unified"`,
  `"low_confidence"`, `"single_topic"`, `"disabled"`, `"no_reranker"`
- `per_topic_rerank_merged: list[RankedResult] | None`

### Metrics & tracing

- Counter `rag_dr_per_topic_rerank_total{mode=on|off|skipped_unified}`
  is incremented exactly once per `research()` call.
- A Langfuse span `dr_per_topic_rerank` is opened on the parent trace
  when the rerank actually fires.

### Expected impact

Disjoint multi-topic queries: improved recall by guaranteeing each
topic gets representation in the top-K, at the cost of one additional
rerank call per topic. Coherent or unified queries: zero impact
(skipped via `is_unified` or `single_topic`).

### A/B harness

```
RAGWEAVE_AB_REPORT_ONLY=1 \
    uv run pytest evals/retrieval/deep_research/test_per_topic_rerank_ab.py \
        -m eval_deep_research -v -s
```

The harness runs each query in
`evals/retrieval/deep_research/fixtures/asic/disjoint_queries.json`
twice (treatment off / on), prints `recall@k` and p50/p95 latency
deltas, and — without `RAGWEAVE_AB_REPORT_ONLY=1` — fails the suite if
treatment regresses recall by >0.05 or p95 latency by >50%.

Test coverage:

- Unit: `tests/retrieval/test_dr_per_topic_rerank.py` (10 tests)
- Smoke: `evals/retrieval/deep_research/test_per_topic_rerank_smoke.py`
- A/B (ASIC fixture, recall+latency): `evals/retrieval/deep_research/test_per_topic_rerank_ab.py::test_per_topic_rerank_ab`
- A/B (live RagWeave corpus, latency-only): `…::test_per_topic_rerank_ab_ragweave_corpus` — uses
  `evals/retrieval/deep_research/fixtures/ragweave/disjoint_queries.json`. Recall
  is best-effort prefix match on the small engineering-doc corpus; the gate is
  p95 latency only (>50% regression fails). To run:
  `RAG_WEAVIATE_MODE=networked RAGWEAVE_AB_REPORT_ONLY=1 uv run --no-sync pytest …`.

### First measured A/B run (RagWeave corpus, 5 disjoint queries, 2026-05-09)

Live containerized Weaviate (611 chunks, 5 distinct engineering docs):

| Metric | Baseline | Treatment | Δ |
| --- | --- | --- | --- |
| p50 latency | 55.7s | 61.2s | +5.6s (+10%) |
| p95 latency | 93.8s | 79.3s | **-14.5s (-15%)** |
| Fire rate | — | 4/5 | gate working |

The slowest baseline query (`litellm-vs-ops`, 93.8s) finished 14.5s faster
under treatment — round-robin merge gave the topic pools enough source
diversity to avoid a deeper DR iteration. p50 cost is bounded at ~5s/query
overhead from per-pool rerank calls. Sample size is too small for
statistical confidence; full recall+latency signal requires the OpenTitan
corpus to be ingested (path-prefix recall on the RagWeave corpus reads as
0/0 across both arms — irrelevant for the latency-only gate).
