# Agentic Retrieval Loop — Engineering Guide

> Status: **INC-1 (single-round MVP) and INC-2 (multi-round loop + judge-as-ranker
> + QFS routing) shipped.** INC-3 (CLI/UI parity + telemetry surfacing) is
> planned. This guide documents the architecture and the current behaviour.

## 1. What it is

A controller LLM drives retrieval as a **tool**, the way an agent uses tools
until it is satisfied. Each round it writes a **HyDE** (hypothetical-answer)
query, the host retrieves with it, a separate LLM **judge** **gates** each chunk
(relevance + faithfulness) and emits a **listwise ranking** (best→worst), plus a
**set-sufficiency** verdict; judge-approved chunks **accumulate**, and the loop
retries a *different* HyDE (targeting the judge's named gap) until the kept set is
sufficient or a budget caps it. Single-document queries converge in round 1;
multi-document (QFS) queries iterate.

This subsumes the earlier ad-hoc HyDE + navigational-filter work: instead of
fixing one bad-chunk class at a time, a generic, model-driven judge decides what
reaches generation — an information-content judgment, never a pattern/vendor
match (CLAUDE.md §0).

### Why the judge ranks the raw pool (the INC-2 reframe)

A measured study (22 real queries, blind LLM judges) found the cross-encoder
reranker is **net-negative**: mean nDCG@12 of the raw hybrid order is **0.671**
vs **0.408** reranked (the CE was worse on 15/22). An LLM **ranking the raw
hybrid pool** scored **0.958** (beat the CE 22/22 and raw-hybrid 22/22). So INC-2
**bypasses the cross-encoder by default** (`RAG_AGENTIC_RANKER="judge"`) and lets
the judge order the raw pool. The pointwise relevance/faithfulness scores only
**gate** keep/drop; the **order** comes from the judge's explicit listwise
ranking (pointwise scores tie heavily and an order built from them collapses back
to hybrid order — so ordering must be listwise to realise the 0.958 win).

The cross-encoder remains available (`RAG_AGENTIC_RANKER="cross_encoder"`) for the
one query class the study found it helped (broad cross-protocol comparison).

## 2. Where it lives

Like deep-research, the loop **replaces the linear stages 2–5**. It forks
`RAGChain.run` at the deep-research seam:

```
_dr_active      = bool(deep_research)
_agentic_active = (per-request override else RAG_AGENTIC_RETRIEVAL_ENABLED) and not _dr_active
_alt_active     = _dr_active or _agentic_active     # linear stages 2-5 skip when set
```

`if _agentic_active:` calls `RAGChain._run_agentic_retrieval(...)`, which runs
the orchestrator in **one `asyncio.run` island inside the single Temporal
activity** (the proven deep-research pattern — not a cyclic Temporal workflow),
populating `reranked` + `graph_context` directly. Stage-6 generation,
guardrails, and the `/query/stream` worker/api split all run unchanged. The
confidence-routing RE_RETRIEVE (Stage 7.5) is suppressed on the agentic path
(its kept pool is already judge-curated). The activity result-cache is
**bypassed** when the loop is active (it is nondeterministic by design).

Mutually exclusive with `deep_research` / `tree_retrieval` (rejected at the
request schema and defensively at the seam).

## 3. The per-round loop

1. **HyDE** (`agentic/hyde.py`) — controller writes a short hypothetical answer
   (embedded, answer-space) + literal `search_terms` (BM25 anchor), conditioned on
   `tried_hyde` + the previous round's named gap. Fail-open: on failure → fall
   back to embedding the processed query (never worse than baseline).
2. **Retrieve-as-tool** — embed the HyDE answer **keyed on the HyDE text** (never
   `processed_query`, so the LRU is not poisoned) + BM25 = processed query +
   search terms; hybrid search via the existing idempotent `_do_search`.
3. **Dedup CHECK** on `SearchResult.object_id`. Chunks are **not** marked seen
   yet — only chunks actually shown to the judge are (step 5), so a deep-pool
   tail truncated below `judge_pool_max` is **not burned** and can resurface in a
   later round.
4. **Thin/nav filter** — reuses `_filter_thin_candidates`.
5. **Rank source**:
   - `ranker="judge"` (default): **no cross-encoder** — take the raw hybrid
     pool `[:RAG_AGENTIC_JUDGE_POOL_MAX]` as candidates (hybrid score retained).
   - `ranker="cross_encoder"`: rerank to the ORIGINAL question, narrowing to
     `RAG_AGENTIC_KEEP_TOP_K_PER_ROUND` (INC-1 behaviour).
   Only these candidates are marked **seen**.
6. **Judge** (`agentic/judge.py`) — generic per-`(question, chunk)` relevance +
   faithfulness (**the keep gate**), a listwise `ranking` (**the order**), and a
   pool sufficiency verdict. **Fail-open keep-all** on unparseable/empty JSON,
   with `rank = -1` (no order signal) so the caller falls back to hybrid order —
   a flaky model emitting `{}` never drops the round *and* never collapses to a
   flat order.
7. **Accumulate** chunks with `keep` and `relevance ≥ threshold` and
   `faithfulness ≥ threshold`; **reserve** every judged candidate into the
   cross-round `best_seen` pool (anti-refusal reservoir).
8. **Stop gates** (`_pre_round_stop` / `_post_round_stop`): `sufficient`
   (affirmative pool verdict + confidence + kept floor + source floor),
   `no_progress` (a round retrieved nothing fresh), `judge_unavailable`
   (fail-open pool → no signal to continue), `round_cap`, `llm_budget`,
   `wall_clock`. `max_rounds ≤ 1` is single-round mode (`stop_reason="single_round"`).
9. **Finalize** — **order** the kept set (single round → the round's listwise
   rank; multi-round → ONE authoritative re-rank of the kept union, budget
   permitting; fail-open → raw-hybrid order). **Anti-refusal floor**: if fewer
   than `RAG_AGENTIC_MIN_KEPT_CHUNKS` were kept, backfill from the **cross-round**
   `best_seen` reservoir (never just the last round — the terminating round is
   often the empty `no_progress` one). Stamp strictly-decreasing ordinal scores
   so the **kept block always outranks the backfill block**, then per-doc
   diversity cap + truncate to `min(rerank_top_k, RAG_AGENTIC_FINAL_MAX_CHUNKS)`.

### QFS routing (no surface classifier)

The effective round ceiling is: explicit per-request `max_agentic_rounds`
(clamped) → else `RAG_AGENTIC_QFS_AUTO ? RAG_AGENTIC_MAX_ROUNDS : 1`. The
**judge's sufficiency verdict is the real router** — a single-document query is
called sufficient after round 1 and stops; a compound/QFS query keeps surfacing a
named gap and iterates. No corpus/vendor/linguistic surface pattern classifies
QFS (CLAUDE.md §0). To reproduce INC-1 exactly, set `RAG_AGENTIC_QFS_AUTO=false`
**and** `RAG_AGENTIC_RANKER="cross_encoder"` (the two switches are independent —
the ceiling controls rounds, the ranker controls order).

### Deferred to a later increment

The **embedding-cosine HyDE diversity re-prompt** (`RAG_AGENTIC_HYDE_DIVERSITY_MAX_COSINE`)
is intentionally **dormant**: HyDE repetition is already handled by the controller
prompt (it diversifies on `tried_hyde` + the named gap) and by the `no_progress`
stop (a recycled HyDE retrieves only seen chunks and ends the loop cleanly). The
key is reserved for re-introduction once telemetry shows frequent
high-cosine-yet-still-fresh recycling.

## 4. Judge model & the ranker

The judge/controller use the `judge` / `controller` LiteLLM router aliases,
which **fall back to the default model** when no dedicated model is configured —
so the loop works with zero new GPU deployment. Set `RAG_LLM_JUDGE_MODEL` /
`RAG_LLM_CONTROLLER_MODEL` to a dedicated (e.g. JSON-reliable instruct) model to
upgrade quality **with no code change**.

Because the default model (a reasoning model) can emit `{}` under guided JSON,
the judge is **fail-open / keep-all**: it never silently drops a round. This is
the load-bearing reliability decision.

**Coupling to be aware of:** the measured 0.958 ranking is only realised when the
`judge` alias points at a **JSON-reliable** backend (a small instruct model, or
the headless Claude-CLI judge). With the default reasoning model the judge fails
open often and the order degrades to **raw-hybrid (~0.671)** — still better than
the cross-encoder (0.408), so enabling the loop never silently underperforms
hybrid, but the headline win needs a reliable judge.

> **The `controller` (HyDE) alias MUST be an instruct model, not a reasoning
> model.** HyDE is short structured generation under a small token budget
> (`RAG_AGENTIC_HYDE_MAX_TOKENS`, default 256). A reasoning model spends that
> entire budget inside its hidden `<think>` block and returns **empty content**,
> so HyDE yields `None` and the loop silently degrades to embedding the *literal*
> query — discarding the answer-space HyDE that lifts answer-bearing chunks into
> the keep window. This failure is now counted as **`hyde_failures`** in the
> agentic telemetry; a persistently nonzero value means the controller is
> mis-provisioned (point its alias at the same instruct model as the judge). On
> uk-ai03 dev, both run on the qwen2.5-7b instruct model via the `:18005` tunnel
> (`docker-compose.dev.yml`), with `RAG_AGENTIC_LLM_JSON_MODE=true` for guided
> JSON on both calls.

## 5. Configuration (`config/settings.py`, `RAG_AGENTIC_*`)

| Key | Default | Effect |
| --- | --- | --- |
| `RAG_AGENTIC_RETRIEVAL_ENABLED` | `false` | Master gate. Per-request `agentic_retrieval` overrides it. |
| `RAG_AGENTIC_RANKER` | `judge` | Who orders the pool: `judge` (bypass CE, listwise rank the raw hybrid pool) or `cross_encoder` (INC-1). |
| `RAG_AGENTIC_JUDGE_POOL_MAX` | `40` | Max raw-hybrid candidates judged per round in `judge` mode (only the judged slice is marked seen). |
| `RAG_AGENTIC_MAX_ROUNDS` | `3` | Max HyDE-retry rounds (the judge's sufficiency verdict usually stops earlier). |
| `RAG_AGENTIC_MAX_LLM_CALLS` | `10` | Controller + judge + finalize-rerank call budget. |
| `RAG_AGENTIC_WALL_CLOCK_MS` | `45000` | Loop wall-clock (kept under the activity timeout). |
| `RAG_AGENTIC_KEEP_TOP_K_PER_ROUND` | `8` | Candidates shown to the judge in `cross_encoder` mode. |
| `RAG_AGENTIC_FINAL_MAX_CHUNKS` | `8` | Hard cap on chunks fed to generation. |
| `RAG_AGENTIC_MIN_KEPT_CHUNKS` | `3` | Kept floor + anti-refusal fallback floor. |
| `RAG_AGENTIC_MIN_SOURCES` | `1` | Distinct-source floor for a 'sufficient' stop (auto-clamped to 1 under a source/heading filter). |
| `RAG_AGENTIC_RELEVANCE_THRESHOLD` | `0.5` | Min judge relevance to keep (gate, not order). |
| `RAG_AGENTIC_FAITHFULNESS_THRESHOLD` | `0.5` | Min judge faithfulness to keep (gate, not order). |
| `RAG_AGENTIC_SUFFICIENCY_TARGET` | `0.7` | Min pool confidence to stop satisfied. |
| `RAG_AGENTIC_HYDE_DIVERSITY_MAX_COSINE` | `0.92` | HyDE-variant redundancy guard — **reserved/dormant** (see §3). |
| `RAG_AGENTIC_HYDE_MAX_TOKENS` | `256` | HyDE answer length cap. |
| `RAG_AGENTIC_HYDE_TEMPERATURE` | `0.4` | Controller sampling temperature. |
| `RAG_AGENTIC_CONTROLLER_MODEL_ALIAS` | `controller` | Router alias for the controller. |
| `RAG_AGENTIC_JUDGE_MODEL_ALIAS` | `judge` | Router alias for the judge (point at a JSON-reliable model for the ranking win). |
| `RAG_AGENTIC_QFS_AUTO` | `true` | Allow multi-round when no explicit round override; the judge's sufficiency verdict routes. |
| `RAG_STAGE_BUDGET_AGENTIC_RETRIEVAL_MS` | `45000` | TimingPool stage budget. |

Per-request (on `QueryRequest` / `ConsoleQueryRequest` / `RAGRequest`):
`agentic_retrieval: Optional[bool]`, `max_agentic_rounds: Optional[int]`.

`validate_agentic_retrieval_config()` fail-fasts on out-of-range thresholds or a
kept floor exceeding the final cap.

## 6. Code map

- `config/settings.py` — `RAG_AGENTIC_*` + `validate_agentic_retrieval_config`.
- `server/schemas.py` — `agentic_retrieval` / `max_agentic_rounds` + contradiction validators (both request models).
- `server/activities.py` — unpack + cache-key + cache-bypass-when-active + `run()` kwargs.
- `src/retrieval/pipeline/rag_chain.py` — `_agentic_active`/`_alt_active` seam, the `if _agentic_active:` branch, `_run_agentic_retrieval`, telemetry, Stage-7.5 suppression.
- `src/retrieval/pipeline/agentic/` — the loop package (`state` / `hyde` / `judge` / `orchestrator`).
- `src/platform/llm/provider.py` + `schemas.py` — `controller` / `judge` router aliases.
- `prompts/agentic_hyde_generate.md`, `prompts/agentic_chunk_judge.md` — generic prompts (§0-compliant).
- `tests/retrieval/test_agentic_retrieval.py` — judge keep/drop + fail-open, HyDE fail-open, single-round flow, anti-refusal, HyDE-keyed retrieval.

## 7. Troubleshooting

- **Empty answers** — check the judge isn't dropping everything: with the default
  model, fail-open keep-all should prevent this; the anti-refusal floor backfills
  from reranked candidates. Confirm `RAG_AGENTIC_MIN_KEPT_CHUNKS` ≤ retrieval pool.
- **No effect** — `RAG_AGENTIC_RETRIEVAL_ENABLED=false` and no per-request
  override → the branch never runs. Set `agentic_retrieval=true` on the request.
- **Slow** — each round adds 1 controller + 1 judge LLM call (multi-round adds 1
  finalize re-rank); with a reasoning model as the fallback judge this dominates
  latency. Point `RAG_LLM_JUDGE_MODEL` at a faster instruct model.
- **Ranking looks like plain hybrid order** — expected when the judge alias is the
  default reasoning model (it fails JSON and falls open to hybrid order). Point
  `RAG_LLM_JUDGE_MODEL` at a JSON-reliable backend to get the listwise ranking.
- **Loop always runs to `round_cap`** — usually `RAG_AGENTIC_SUFFICIENCY_TARGET`
  too high for the judge's confidence, or `RAG_AGENTIC_MIN_SOURCES > 1` on a
  single-document query (auto-clamped to 1 only when a source/heading filter is
  set). Check `stop_reason`.
- **Telemetry** — `response.metadata["agentic_retrieval"]`
  (`rounds_run`, `kept_count`, `llm_calls`, `judge_calls`, `ranker`,
  `ranker_calls`, `backfilled`, `stop_reason`, `elapsed_ms`).
