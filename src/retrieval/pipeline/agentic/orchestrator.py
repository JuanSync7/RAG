# @summary
# AgenticRetrieval — the controller<->retrieve<->judge orchestrator. INC-2 runs a
# multi-round, gap-driven, accumulating loop: each round the controller writes a
# HyDE, the host retrieves-as-a-tool, the LLM judge GATES chunks (relevance +
# faithfulness threshold) and emits a listwise ORDER (best->worst rank), approved
# chunks accumulate, and the loop retries a different HyDE until the judge calls the
# kept set sufficient or a budget caps it (round/llm-call/wall-clock). Single-doc
# queries converge in round 1; multi-doc (QFS) iterate. Ordering comes from the
# judge's listwise rank (the measured near-ideal signal), NOT pointwise scores;
# with a flaky judge it falls open to raw-hybrid order (still > the cross-encoder).
# The cross-encoder is bypassed by default (ranker="judge"); ranker="cross_encoder"
# restores the INC-1 ordering. Anti-refusal backfill draws from a CROSS-ROUND
# reservoir so a terminal empty round never yields empty context. Instantiated per
# request (no shared state); reuses the chain's retrieve/rerank/filter helpers via
# injected callables (no rag_chain import — avoids a cycle).
# Exports: AgenticRetrieval
# Deps: asyncio, time, src.retrieval.pipeline.agentic.{state,hyde,judge}, deep_research (_chunk_id)
# @end-summary
"""Agentic HyDE/controller/judge retrieval orchestrator (INC-2: multi-round)."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, Callable, Optional

from src.retrieval.common.schemas import RankedResult
from src.retrieval.pipeline.agentic.hyde import generate_hyde
from src.retrieval.pipeline.agentic.judge import judge_chunks
from src.retrieval.pipeline.agentic.state import (
    AgenticBudget,
    AgenticResult,
    AgenticState,
    HydeVariant,
)
# One-way reuse of the stable chunk-id (SearchResult.object_id-based) — the only
# dedup needed, done on SearchResults BEFORE conversion to RankedResult.
from src.retrieval.pipeline.deep_research import _chunk_id

logger = logging.getLogger(__name__)

# Type aliases for the injected chain helpers.
RetrieveFn = Callable[[str, list], Awaitable[list]]      # (hyde_answer, search_terms) -> [SearchResult]
ThinFilterFn = Callable[[list], list]                    # [SearchResult] -> [SearchResult]
DocDiversityFn = Callable[[list, int], list]             # ([RankedResult], top_k) -> [RankedResult]

# Upper bound on the cross-round anti-refusal reservoir (memory guard). Far above
# any realistic kept/backfill width; capped by hybrid score when exceeded.
_RESERVOIR_CAP = 64


class AgenticRetrieval:
    """Per-request agentic retrieval loop.

    The orchestrator owns NO retrieval/rerank logic of its own — it composes the
    chain's existing primitives (passed as callables) around the two model-driven
    steps (HyDE generation and chunk judging), exactly as ``DeepResearch`` is
    handed ``kb_retrieve``/``reranker``.
    """

    def __init__(
        self,
        *,
        provider,
        retrieve: RetrieveFn,
        reranker,
        thin_filter: ThinFilterFn,
        doc_diversity: DocDiversityFn,
        original_question: str,
        processed_query: str,
        budget: AgenticBudget,
        controller_alias: str,
        judge_alias: str,
        hyde_max_tokens: int,
        hyde_temperature: float,
        llm_timeout_s: int,
        final_top_k: int,
        json_mode: bool = False,
        judge_concise: bool = False,
        fill_mode: str = "hybrid",
        graph_context: str = "",
    ) -> None:
        self._provider = provider
        self._retrieve = retrieve
        self._reranker = reranker
        self._thin_filter = thin_filter
        self._doc_diversity = doc_diversity
        self._budget = budget
        self._ranker = budget.ranker
        self._controller_alias = controller_alias
        self._judge_alias = judge_alias
        self._hyde_max_tokens = hyde_max_tokens
        self._hyde_temperature = hyde_temperature
        self._llm_timeout_s = llm_timeout_s
        self._final_top_k = max(1, min(final_top_k, budget.final_max_chunks))
        self._json_mode = json_mode
        self._judge_concise = judge_concise
        self._fill_mode = fill_mode
        self._graph_context = graph_context
        self._state = AgenticState(
            original_question=original_question,
            processed_query=processed_query,
        )
        self._started_at = 0.0

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------

    async def run(self) -> AgenticResult:
        """Run the multi-round loop and return curated ranked chunks + telemetry.

        Single-round mode (``budget.max_rounds <= 1``) reproduces INC-1: exactly
        one round, ``stop_reason="single_round"``. Otherwise the loop iterates
        until a stop gate fires.
        """
        self._started_at = time.perf_counter()
        s = self._state
        b = self._budget
        single_mode = b.max_rounds <= 1

        while True:
            pre = self._pre_round_stop()
            if pre:
                s.stop_reason = s.stop_reason or pre
                break
            await self._run_round()
            post = self._post_round_stop()
            if post:
                s.stop_reason = post
                break

        if single_mode:
            # INC-1 semantics: one round, regardless of which gate technically
            # fired (the test contract + the docs call this "single_round").
            s.stop_reason = "single_round"
        elif not s.stop_reason:
            s.stop_reason = "round_cap"

        reranked = await self._finalize()
        elapsed_ms = (time.perf_counter() - self._started_at) * 1000.0
        return AgenticResult(
            reranked=reranked,
            graph_context=self._graph_context,
            rounds_run=s.round_index,
            hyde_variants_tried=len(s.tried_hyde),
            kept_count=len(s.kept),
            llm_calls=s.llm_calls,
            judge_calls=s.judge_calls,
            ranker=self._ranker,
            ranker_calls=s.ranker_calls,
            backfilled=s.backfilled,
            stop_reason=s.stop_reason or "single_round",
            elapsed_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------
    # Stop gates
    # ------------------------------------------------------------------

    def _elapsed_ms(self) -> float:
        return (time.perf_counter() - self._started_at) * 1000.0

    def _pre_round_stop(self) -> Optional[str]:
        """Budget gate checked BEFORE entering a round. Reserves room for the two
        model calls a round needs (HyDE + judge)."""
        s, b = self._state, self._budget
        if s.round_index >= b.max_rounds:
            return "round_cap"
        if s.llm_calls + 2 > b.max_llm_calls:
            return "llm_budget"
        if self._elapsed_ms() >= b.wall_clock_ms:
            return "wall_clock"
        return None

    def _post_round_stop(self) -> Optional[str]:
        """Convergence gate checked AFTER a round. Only an AFFIRMATIVE judge
        verdict (a present pool verdict) can continue past round 1; on fail-open
        (no pool verdict) we stop and let finalize fall back to hybrid order, so a
        flaky judge degrades to single-round behaviour rather than burning rounds.
        """
        s, b = self._state, self._budget
        if self._elapsed_ms() >= b.wall_clock_ms:
            return "wall_clock"
        # A round that retrieved nothing new (HyDE recycling into already-seen
        # chunks) reports the accurate reason — checked before the fail-open
        # judge_unavailable branch, since on such a round the judge never ran.
        if s.last_fresh_count == 0 and s.round_index >= 1:
            return "no_progress"
        pv = s.last_pool
        if pv is not None:
            if (
                pv.sufficient
                and pv.confidence >= b.sufficiency_target
                and len(s.kept) >= b.min_kept_chunks
                and self._distinct_sources() >= b.min_sources
            ):
                return "sufficient"
        elif s.round_index >= 1:
            # Judge gave no affirmative verdict (fail-open). No signal to continue.
            return "judge_unavailable"
        if s.round_index >= b.max_rounds:
            return "round_cap"
        return None

    def _distinct_sources(self) -> int:
        seen: set[str] = set()
        for r in self._state.kept:
            md = r.metadata or {}
            seen.add(str(md.get("source") or md.get("document_id") or ""))
        return len(seen)

    # ------------------------------------------------------------------
    # One round
    # ------------------------------------------------------------------

    async def _run_round(self) -> None:
        """Execute one HyDE -> retrieve -> (rank) -> judge -> keep round.

        Updates loop state in place (kept, reservoir, per-round signals). Does NOT
        order or score for generation — that is finalize's job, so the ORDER is a
        single decision over the full cross-round kept set.
        """
        s = self._state
        s.round_index += 1
        b = self._budget

        # 1. HyDE generation (controller). Fail-open -> embed the processed query
        #    so retrieval is never worse than a plain hybrid search.
        variant = await self._next_hyde()
        s.tried_hyde.append(variant.hypothetical_answer)

        # 2. Retrieve-as-tool (embed HyDE answer + BM25 anchor + hybrid search).
        try:
            search_results = await self._retrieve(
                variant.hypothetical_answer, variant.search_terms
            )
        except Exception as exc:  # noqa: BLE001 — a retrieval failure is empty, not fatal
            logger.warning("agentic retrieve failed in round %d: %s", s.round_index, exc)
            search_results = []

        # 3. Dedup CHECK vs everything seen so far. Do NOT mark seen yet — only
        #    chunks actually shown to the judge are marked, so a deep-pool tail
        #    truncated below judge_pool_max is never permanently burned (it can
        #    resurface and be judged in a later round).
        fresh = []
        for sr in search_results:
            if _chunk_id(sr) in s.seen_chunk_ids:
                continue
            fresh.append(sr)
        s.last_fresh_count = len(fresh)
        if not fresh:
            self._mark_unproductive()
            return

        # 4. Drop thin/heading-only + navigational stubs.
        filtered = self._thin_filter(fresh)
        if not filtered:
            self._mark_unproductive()
            return

        # 5. Build the round's candidate RankedResults.
        if self._ranker == "cross_encoder":
            # INC-1 path: the cross-encoder narrows + scores against the ORIGINAL
            # question (not the HyDE text).
            reranked = await asyncio.to_thread(
                self._reranker.rerank,
                query=s.original_question,
                documents=filtered,
                top_k=b.keep_top_k_per_round,
            )
            # The reranker drops the tail beyond top_k; mark seen ONLY the chunks
            # it kept (symmetric with the judge path), so a chunk the CE ranked
            # low can resurface in a later round. Match reranked outputs back to
            # their source SearchResults by text (always preserved by a reranker,
            # unlike object_id/metadata identity) — a duplicate-text collision can
            # only over-mark, never lose a judged chunk.
            kept_texts = {r.text for r in reranked}
            judged_sources = [sr for sr in filtered if sr.text in kept_texts]
            # Copy metadata so per-chunk stamping never mutates the shared
            # SearchResult dict (the rerankers alias it by reference).
            candidates = [
                RankedResult(text=r.text, score=float(r.score), metadata=dict(r.metadata))
                for r in reranked
            ]
        else:
            # "judge": bypass the cross-encoder; judge the RAW hybrid pool. The
            # hybrid score is preserved on .score as the fail-open ordering key.
            slice_ = filtered[: max(1, b.judge_pool_max)]
            candidates = [
                RankedResult(text=sr.text, score=float(sr.score), metadata=dict(sr.metadata))
                for sr in slice_
            ]
            judged_sources = slice_

        # Mark seen ONLY the chunks carried into the ranker/judge.
        for sr in judged_sources:
            s.seen_chunk_ids.add(_chunk_id(sr))

        if not candidates:
            self._mark_unproductive()
            return

        # 6. Judge: GATE (relevance/faithfulness) + ORDER (listwise rank) + pool.
        verdicts, pool = await judge_chunks(
            self._provider,
            model_alias=self._judge_alias,
            original_question=s.original_question,
            candidates=candidates,
            timeout_s=self._llm_timeout_s,
            json_mode=self._json_mode,
            concise=self._judge_concise,
            max_keep=self._final_top_k,
        )
        s.llm_calls += 1
        s.judge_calls += 1
        s.last_pool = pool

        ranked_any = False
        newly_kept = 0
        for v in verdicts:
            if not (0 <= v.index < len(candidates)):
                continue
            c = candidates[v.index]
            md = c.metadata
            md["agentic_relevance"] = v.relevance
            md["agentic_faithfulness"] = v.faithfulness
            md["agentic_round"] = s.round_index
            md["agentic_rank"] = v.rank
            if v.rank >= 0:
                ranked_any = True
            if (
                v.keep
                and v.relevance >= b.relevance_threshold
                and v.faithfulness >= b.faithfulness_threshold
            ):
                s.kept.append(c)
                newly_kept += 1
        s.last_newly_kept = newly_kept
        s.last_ranked = ranked_any

        # 7. Cross-round anti-refusal reservoir: every judged candidate (deduped).
        self._reserve(candidates)

        # 8. Record the named gap for the next round's HyDE.
        if pool is not None:
            s.coverage_aspects = (
                [pool.missing_information] if pool.missing_information else []
            )

    def _mark_unproductive(self) -> None:
        """Reset the per-round signals for a round that judged nothing."""
        s = self._state
        s.last_pool = None
        s.last_newly_kept = 0
        s.last_ranked = False

    async def _next_hyde(self) -> HydeVariant:
        """Produce the next HyDE variant, with a baseline fallback on failure.

        Diversification across rounds is driven entirely by the controller prompt
        (it sees ``tried_hyde`` + the named gap); a recycled HyDE that retrieves
        only already-seen chunks ends the loop via ``no_progress``. The
        embedding-cosine re-prompt guard is deferred (see ``RAG_AGENTIC_HYDE_*``).
        """
        s = self._state
        variant = await generate_hyde(
            self._provider,
            model_alias=self._controller_alias,
            original_question=s.original_question,
            tried_hyde=s.tried_hyde,
            coverage_aspects=s.coverage_aspects,
            max_tokens=self._hyde_max_tokens,
            temperature=self._hyde_temperature,
            timeout_s=self._llm_timeout_s,
            json_mode=self._json_mode,
        )
        s.llm_calls += 1
        if variant is None:
            # Fail-open: embed the processed query itself → one standard hybrid
            # search. Never worse than the baseline linear path.
            return HydeVariant(
                hypothetical_answer=s.processed_query,
                search_terms=[],
                target_aspect="",
            )
        return variant

    # ------------------------------------------------------------------
    # Reservoir
    # ------------------------------------------------------------------

    @staticmethod
    def _rr_id(c: RankedResult) -> str:
        """Stable id for a RankedResult (which carries no object_id) — the same
        (source|heading|chunk_index) fallback ``deep_research._chunk_id`` uses."""
        md = getattr(c, "metadata", None) or {}
        src = md.get("source") or md.get("document_id") or "unknown"
        heading = md.get("heading", "")
        idx = md.get("chunk_index", "")
        if heading or idx != "":
            return f"meta:{src}|{heading}|{idx}"
        return f"hash:{hash(c.text or '') & 0xFFFFFFFFFFFFFFFF:x}"

    def _reserve(self, candidates: list[RankedResult]) -> None:
        s = self._state
        for c in candidates:
            cid = self._rr_id(c)
            if cid in s.best_seen_ids:
                continue
            s.best_seen_ids.add(cid)
            s.best_seen.append(c)
        if len(s.best_seen) > _RESERVOIR_CAP:
            s.best_seen.sort(key=lambda r: r.score, reverse=True)
            s.best_seen = s.best_seen[:_RESERVOIR_CAP]
            s.best_seen_ids = {self._rr_id(c) for c in s.best_seen}

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------

    async def _finalize(self) -> list[RankedResult]:
        """Order the kept set for generation, backfill an anti-refusal floor from
        the cross-round reservoir, and stamp a strict ordering for the diversity
        cap. The kept block ALWAYS outranks the backfill block."""
        s = self._state
        b = self._budget

        kept_ordered = await self._order_kept(list(s.kept))

        # Fill the top-K from the CROSS-ROUND hybrid reservoir, with the judge's
        # picks promoted on top. How many slots to fill is the promote-vs-drop
        # decision (RAG_AGENTIC_FILL_MODE): always to final_top_k ("hybrid", the
        # recall safety net for a cautious judge), only the anti-refusal floor
        # ("none", trust a strong judge), or per-query by the judge's sufficiency
        # ("adaptive"). The reservoir is cross-round — never just the last round,
        # which is often the empty no_progress one.
        ordered = list(kept_ordered)
        target = self._fill_target(len(ordered))
        if len(ordered) < target and s.best_seen:
            have = {self._rr_id(c) for c in ordered}
            backfill = [c for c in s.best_seen if self._rr_id(c) not in have]
            backfill.sort(key=lambda r: r.score, reverse=True)  # hybrid/CE score
            added = backfill[: max(target - len(ordered), 0)]
            s.backfilled = len(added)
            ordered.extend(added)

        if not ordered:
            return []

        # Stamp strictly-decreasing ordinal scores so (a) the kept block outranks
        # the backfill block and (b) _apply_doc_diversity's score-sort preserves
        # THIS order rather than collapsing ties back to hybrid order. The real
        # judge relevance stays in metadata['agentic_relevance'].
        total = len(ordered)
        for pos, c in enumerate(ordered):
            c.score = float(total - pos)

        return self._doc_diversity(ordered, self._final_top_k)

    def _fill_target(self, kept_n: int) -> int:
        """How many top-K slots to fill from the hybrid reservoir. Always at least
        the anti-refusal floor (so a non-empty retrieval never yields empty
        context). The cross-encoder path already returns an ordered top-K, so it
        only gets the floor; the judge path applies the promote-vs-drop policy."""
        b = self._budget
        floor = b.min_kept_chunks
        if self._ranker != "judge":
            return floor
        mode = self._fill_mode
        if mode == "none":
            return floor
        if mode == "adaptive":
            pv = self._state.last_pool
            confident = (
                pv is not None
                and pv.sufficient
                and pv.confidence >= b.sufficiency_target
                and kept_n >= b.min_kept_chunks
            )
            return floor if confident else max(floor, self._final_top_k)
        # "hybrid" (default): always fill to the generation width (recall net).
        return max(floor, self._final_top_k)

    async def _order_kept(self, kept: list[RankedResult]) -> list[RankedResult]:
        """Return the kept chunks in final generation order.

        ranker="cross_encoder": order by the cross-encoder score (INC-1).
        ranker="judge", single round: order by the round's listwise rank (the
            judge-as-ranker signal); unranked/fail-open chunks fall to hybrid order
            — never a flat collapse.
        ranker="judge", multi round: per-round ranks are not cross-comparable, so
            re-rank the kept union ONCE (budget permitting) for a single
            authoritative order; on failure fall back to hybrid-score order.
        """
        if len(kept) <= 1:
            return kept
        if self._ranker == "cross_encoder":
            return sorted(kept, key=lambda r: float(getattr(r, "score", 0.0)), reverse=True)

        s, b = self._state, self._budget
        if s.round_index <= 1:
            return self._by_round_rank(kept)

        # Multi-round: one authoritative listwise re-rank over the kept union.
        if (
            len(kept) > 1
            and s.llm_calls + 1 <= b.max_llm_calls
            and self._elapsed_ms() < b.wall_clock_ms
        ):
            try:
                verdicts, _ = await judge_chunks(
                    self._provider,
                    model_alias=self._judge_alias,
                    original_question=s.original_question,
                    candidates=kept,
                    timeout_s=self._llm_timeout_s,
                    json_mode=self._json_mode,
                    concise=self._judge_concise,
                    max_keep=self._final_top_k,
                )
                s.llm_calls += 1
                s.ranker_calls += 1
                ranks = {
                    v.index: v.rank for v in verdicts if 0 <= v.index < len(kept)
                }
                if any(r >= 0 for r in ranks.values()):
                    def key(i: int):
                        r = ranks.get(i, -1)
                        if r >= 0:
                            return (0, r, 0.0)
                        return (1, 0, -float(getattr(kept[i], "score", 0.0)))
                    return [kept[i] for i in sorted(range(len(kept)), key=key)]
            except Exception as exc:  # noqa: BLE001 — fall back to hybrid order
                logger.warning("agentic finalize re-rank failed: %s", exc)

        # Robust fallback: raw-hybrid order (beats the cross-encoder in aggregate).
        return sorted(kept, key=lambda r: float(getattr(r, "score", 0.0)), reverse=True)

    @staticmethod
    def _by_round_rank(kept: list[RankedResult]) -> list[RankedResult]:
        """Single-round order: judge-ranked chunks by rank asc, then any unranked
        (omitted / fail-open) chunks by hybrid score desc. If the judge ranked
        nothing this is pure hybrid-score order — never a flat 1.0 collapse."""
        def key(c: RankedResult):
            md = c.metadata or {}
            rank = md.get("agentic_rank", -1)
            if isinstance(rank, int) and rank >= 0:
                return (0, rank, 0.0)
            return (1, 0, -float(getattr(c, "score", 0.0)))
        return sorted(kept, key=key)
