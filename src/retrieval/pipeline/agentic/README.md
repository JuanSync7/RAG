<!-- @summary
The agentic HyDE/controller/judge retrieval loop: a controller LLM drives
retrieval as a tool (generating HyDE queries), an LLM judge GATES each chunk
(relevance/faithfulness) and RANKS the kept set (listwise), judge-approved chunks
accumulate, and the loop retries a different HyDE (targeting the named gap) until
the kept set is sufficient or a budget caps it. Replaces the linear stages 2-5
(like deep_research). INC-2: multi-round loop + judge-as-ranker over the raw
hybrid pool (cross-encoder bypassed by default) + QFS-via-sufficiency routing.
@end-summary -->

# `agentic/` — Agentic Retrieval Loop

A controller LLM turns retrieval into a **tool** it drives, mirroring how an
agent uses tools until satisfied. Activated per request via `agentic_retrieval`
(or the `RAG_AGENTIC_RETRIEVAL_ENABLED` config), it forks `RAGChain.run` at the
same seam as deep-research (`_alt_active`) and populates `reranked` +
`graph_context` directly, so generation, guardrails and the `/query/stream`
path run unchanged.

## Flow (per round)

1. **HyDE** (`hyde.py`) — the controller writes a short hypothetical *answer*
   (embedded, answer-space) + literal `search_terms` (BM25 anchor), targeting the
   previous round's named gap.
2. **Retrieve-as-tool** — the chain's injected `retrieve` closure embeds the
   HyDE answer (keyed on the HyDE text, never `processed_query`) and runs hybrid
   search via the existing `_do_search` primitive (idempotent, retry-wrapped).
3. **Dedup CHECK** — on `SearchResult.object_id`; chunks are marked *seen* only
   once judged (a deep tail truncated by `judge_pool_max` is never burned).
4. **Thin/nav filter** — reuses the chain's `_filter_thin_candidates`.
5. **Rank source** — `ranker="judge"` (default): take the **raw hybrid pool**
   (cross-encoder bypassed; the measured-best path); `ranker="cross_encoder"`:
   rerank to the ORIGINAL question (INC-1).
6. **Judge** (`judge.py`) — generic, model-driven per-chunk relevance +
   faithfulness (**keep gate**), a listwise `ranking` (**the order**), and set
   sufficiency. **Fail-open**: unparseable/empty output keeps all candidates with
   `rank=-1` (never drops a round; order falls back to raw-hybrid, never flat).
7. **Accumulate** judge-approved chunks above both thresholds + **reserve** every
   judged candidate cross-round.
8. **Stop / Finalize** — sufficiency / no-progress / budget gates; finalize
   orders the kept set (single round → round rank; multi-round → one re-rank),
   and the **anti-refusal floor** backfills from the **cross-round** reservoir so
   a terminal empty round never yields empty context.

## Files

| File | Role |
| --- | --- |
| `__init__.py` | Public facade (stable import surface). |
| `state.py` | Typed contracts: `AgenticBudget`, `HydeVariant`, `ChunkVerdict`, `PoolVerdict`, `AgenticState`, `AgenticResult`. |
| `hyde.py` | `generate_hyde` — controller HyDE generation (fail-open → None → baseline search). |
| `judge.py` | `judge_chunks` — generic chunk judge (fail-open keep-all). |
| `orchestrator.py` | `AgenticRetrieval` — composes the chain's retrieve/rerank/filter helpers around HyDE + judge. |

## Status

**INC-2 (this version): multi-round loop + judge-as-ranker + QFS routing.** The
loop iterates HyDE→retrieve→judge until the judge calls the kept set sufficient or
a budget caps it; the cross-encoder is bypassed by default and the judge's
listwise ranking orders the raw hybrid pool. `RAG_AGENTIC_QFS_AUTO=false` +
`RAG_AGENTIC_RANKER="cross_encoder"` reproduce INC-1. The cosine HyDE-diversity
re-prompt is reserved/dormant (see the design guide §3).

Config: the `RAG_AGENTIC_*` family in `config/settings.py`
(`validate_agentic_retrieval_config()`). Judge/controller use the `judge` /
`controller` router aliases (fall back to the default model — set
`RAG_LLM_JUDGE_MODEL` to a JSON-reliable model to realise the ranking win with no
code change). See `docs/retrieval/AGENTIC_RETRIEVAL_DESIGN.md`.
