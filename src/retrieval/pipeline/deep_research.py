# @summary
# Deep-research retrieval orchestrator: an opt-in alternative to the linear
# kg_expand → embed → hybrid_search → rerank pipeline. Builds a recursive
# tree of topic-grouped sub-questions, anchored to the user's pinned
# original question, then returns either a unified pool (Option B) or
# per-topic pools (hybrid) for the caller to rerank.
#
# Replaces stages 2–5 of RAGChain.run when retrieval_sub_mode == "deep_research".
# Generation downstream is unchanged.
#
# Exports: DeepResearch, DeepResearchBudget, DeepResearchResult, TopicPool
# Deps: asyncio, dataclasses, src.platform.llm.provider, kgweave.client
# @end-summary
"""Deep-research orchestrator (RAGflow tree-decomposition adapted for RagWeave).

Design notes:

- The user's processed query is pinned as ``original_question``. Sufficiency
  checks are anchored to it at every depth, preventing topic drift across
  recursion levels.
- The first decomposition (depth-0 → depth-1) is the topic split. The LLM
  emits ``topics`` groups; if ``len(topics) > 1`` the caller reranks per-topic
  (hybrid). Otherwise the caller reranks against ``original_question`` (B).
- Below depth 1, sub-questions are gap-fillers within an existing topic; their
  retrievals merge into the parent topic's pool, not into new topic splits.
- All blocking calls (Weaviate search, reranker, KG expand, embeddings) are
  wrapped via ``asyncio.to_thread`` in the partials supplied by the caller.
  LLM calls use ``LLMProvider.agenerate`` natively.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from src.platform.llm.provider import LLMProvider
from src.platform.observability import get_tracer
from src.retrieval.deep_research.metrics import (
    DR_ITERATION_DURATION,
    DR_ITERATIONS,
    DR_LLM_CALLS_TOTAL,
    DR_PER_TOPIC_RERANK_TOTAL,
    DR_RUN_DURATION,
    DR_RUNS_TOTAL,
    DR_TOPIC_COUNT,
)
from src.vector_db.common.schemas import SearchResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public contracts
# ---------------------------------------------------------------------------


@dataclass
class DeepResearchBudget:
    """Hard caps on the recursive loop. All three apply simultaneously."""

    max_nodes: int = 12               # max number of retrieval calls
    max_llm_calls: int = 24           # max sufficiency + decomposition calls combined
    wall_clock_ms: int = 30_000       # max total elapsed
    max_depth: int = 3                # max recursion depth (root = 0)
    max_topics: int = 3               # cap on depth-1 topic count
    per_topic_questions: int = 3      # cap on sub-questions per node
    graph_context_max_chars: int = 4000
    # Tightened early-stop gate: requires confident sufficiency PLUS a chunk
    # floor and source-diversity floor. Disable to revert to single-bool stop.
    early_stop_enabled: bool = True
    early_stop_min_chunks: int = 3
    early_stop_min_sources: int = 2
    # Per-topic reranking with split-confidence gate. When the orchestrator
    # decomposes into N>=2 topics AND the LLM's split_confidence clears
    # ``per_topic_rerank_min_confidence``, each pool is reranked against its
    # own ``rerank_anchor`` (not the original query) and merged round-robin.
    per_topic_rerank_enabled: bool = True
    per_topic_rerank_min_confidence: float = 0.6
    per_topic_top_k: int = 3


@dataclass
class TopicPool:
    """Per-topic chunk pool. The orchestrator returns one pool per depth-1
    topic (or one pool named ``"<root>"`` when the tree never split).

    ``rerank_anchor`` is the string the caller should use when reranking this
    pool: the topic name (hybrid mode) or the original question (B mode).
    """

    name: str
    rerank_anchor: str
    chunks_by_id: dict[str, SearchResult] = field(default_factory=dict)
    executed_queries: set[str] = field(default_factory=set)
    # Confidence values across iterations on this pool. Used by the early-stop
    # trend check (no oscillation: confidence is monotonically non-decreasing).
    confidence_history: list[float] = field(default_factory=list)

    def merge_chunks(self, new_chunks: list[SearchResult]) -> None:
        for c in new_chunks:
            cid = _chunk_id(c)
            # Keep the first occurrence (preserves earliest score reference).
            self.chunks_by_id.setdefault(cid, c)

    def evidence_text(self, max_chars: int) -> str:
        if not self.chunks_by_id:
            return "(no evidence retrieved yet)"
        parts: list[str] = []
        used = 0
        for i, c in enumerate(self.chunks_by_id.values()):
            snippet = (c.text or "").strip()
            if not snippet:
                continue
            entry = f"[{i + 1}] {snippet}"
            if used + len(entry) > max_chars:
                parts.append(f"... ({len(self.chunks_by_id) - i} more chunks omitted)")
                break
            parts.append(entry)
            used += len(entry) + 2
        return "\n\n".join(parts)


@dataclass
class DeepResearchResult:
    """Output of a deep-research run.

    The caller decides rerank strategy by inspecting ``len(topic_pools)``.
    """

    topic_pools: list[TopicPool]
    graph_context: str = ""
    node_count: int = 0
    llm_call_count: int = 0
    elapsed_ms: float = 0.0
    budget_exhausted: bool = False
    budget_exhausted_reason: Optional[str] = None
    is_unified: bool = True   # True when single topic / B mode applies
    iteration_count: int = 0  # Number of recursion iterations executed.
    decomposed: bool = False  # True if the orchestrator ran a multi-topic decomposition.
    topic_count: int = 0      # Number of topic pools in the result (>=1).
    # When the run terminated via the early-stop gate, names which condition
    # fired (e.g. "iter0_sufficient", "sufficient+floor+diversity"). None when
    # the run did not early-stop.
    early_stop_reason: Optional[str] = None
    # Confidence the LLM expressed in the topic split during the first
    # decompose call (0.0 if the field was missing — backward compat).
    split_confidence: float = 0.0
    # Per-topic rerank surfacing. ``per_topic_rerank_applied`` is True iff
    # the per-topic rerank actually fired this run; otherwise
    # ``per_topic_rerank_skipped_reason`` carries one of:
    # "unified" | "low_confidence" | "single_topic" | "disabled".
    # ``per_topic_rerank_merged`` is the round-robin-merged ranked list when
    # applied (the caller may further trim to a global top_k cap).
    per_topic_rerank_applied: bool = False
    per_topic_rerank_skipped_reason: Optional[str] = None
    per_topic_rerank_merged: Optional[list[Any]] = None


# Type aliases for the partials the caller supplies.
KBRetrieveFn = Callable[[str], Awaitable[list[SearchResult]]]
KGRetrieveFn = Callable[[str], Awaitable[Optional["KGSnippet"]]]


@dataclass
class KGSnippet:
    """KG retrieval payload — keep small. ``terms`` may be merged into the
    BM25 query upstream of ``kb_retrieve``; ``graph_context`` is accumulated
    and prepended to the generator prompt."""

    terms: list[str] = field(default_factory=list)
    graph_context: str = ""


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------


_PROMPT_DIR = Path(__file__).resolve().parents[3] / "prompts"
_SUFFICIENCY_PROMPT_FILE = _PROMPT_DIR / "deep_research_sufficiency_check.md"
_DECOMP_PROMPT_FILE = _PROMPT_DIR / "deep_research_multi_queries_gen.md"


def _load_prompt(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Deep-research prompt not found at {path}: {exc}") from exc


def _render(template: str, **vars_: str) -> str:
    rendered = template
    for k, v in vars_.items():
        rendered = rendered.replace("{{ " + k + " }}", v)
    return rendered


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk_id(chunk: SearchResult) -> str:
    """Stable id for dedupe. Prefer object_id; fall back to (source, heading,
    chunk_index) tuple from metadata; last resort is a text hash."""
    if chunk.object_id:
        return f"oid:{chunk.object_id}"
    md = chunk.metadata or {}
    src = md.get("source") or md.get("document_id") or "unknown"
    heading = md.get("heading", "")
    idx = md.get("chunk_index", "")
    if heading or idx != "":
        return f"meta:{src}|{heading}|{idx}"
    text = chunk.text or ""
    return f"hash:{hash(text) & 0xFFFFFFFFFFFFFFFF:x}"


def _truncate(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[: n - 3] + "..."


def _parse_json_object(content: str) -> Optional[dict[str, Any]]:
    """Best-effort JSON-object parse. Tolerates code-fenced output."""
    text = (content or "").strip()
    if text.startswith("```"):
        # Strip ```json … ``` fences.
        text = text.lstrip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    # Try direct parse first.
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    # Salvage: extract first {...} block.
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class DeepResearch:
    """Recursive, topic-grouped retrieval loop.

    The orchestrator is single-use: instantiate per query. State (counters,
    pools, executed-queries set) lives on the instance, so concurrent queries
    must use separate instances.
    """

    # Side channel for evaluation harnesses: the most recent ``DeepResearchResult``
    # produced by any instance in this process. RAGResponse does not carry a
    # generic ``metadata`` field yet (owned by another workstream); evals read
    # this attribute to surface ``llm_call_count`` etc. without a schema change.
    last_result: Optional["DeepResearchResult"] = None

    def __init__(
        self,
        provider: LLMProvider,
        kb_retrieve: KBRetrieveFn,
        kg_retrieve: Optional[KGRetrieveFn],
        original_question: str,
        processed_query: str,
        budget: Optional[DeepResearchBudget] = None,
        model_alias: str = "default",
        llm_timeout_s: int = 30,
        reranker: Optional[Any] = None,
    ) -> None:
        self._provider = provider
        self._kb_retrieve = kb_retrieve
        self._kg_retrieve = kg_retrieve
        self._original_question = original_question
        self._processed_query = processed_query
        self._budget = budget or DeepResearchBudget()
        self._model_alias = model_alias
        self._llm_timeout_s = llm_timeout_s
        self._reranker = reranker
        # Per-topic rerank decision bookkeeping (set by _apply_per_topic_rerank).
        self._per_topic_rerank_applied: bool = False
        self._per_topic_rerank_skipped_reason: Optional[str] = None
        self._per_topic_rerank_merged: Optional[list[Any]] = None

        self._sufficiency_template = _load_prompt(_SUFFICIENCY_PROMPT_FILE)
        self._decomp_template = _load_prompt(_DECOMP_PROMPT_FILE)

        self._start_ms: float = 0.0
        self._node_count: int = 0
        self._llm_call_count: int = 0
        self._iteration_count: int = 0
        self._decomposed: bool = False
        self._stop_reason: Optional[str] = None
        # Tighter early-stop bookkeeping.
        self._early_stop_reason: Optional[str] = None
        self._split_confidence: float = 0.0
        # Global executed-queries set — prevents the same query string being
        # issued twice anywhere in the tree.
        self._executed_queries: set[str] = set()
        # Accumulated graph_context, deduped, length-capped.
        self._graph_context_chunks: list[str] = []
        self._graph_context_seen: set[str] = set()
        self._graph_context_len: int = 0
        # Observability — set in research() before _research_impl runs.
        self._trace: Any = None

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def research(self) -> DeepResearchResult:
        run_start = time.perf_counter()
        # Open the parent trace (no-op when OBSERVABILITY_PROVIDER is unset).
        try:
            tracer = get_tracer()
            self._trace = tracer.trace(
                "deep_research",
                metadata={
                    "original_question": self._original_question,
                    "processed_query": self._processed_query,
                    "max_topics": self._budget.max_topics,
                    "max_depth": self._budget.max_depth,
                },
            )
        except Exception:  # noqa: BLE001 — fail open
            self._trace = None
        try:
            result = await self._research_impl()
            result = self._apply_per_topic_rerank(result)
        except BaseException as exc:
            DR_RUNS_TOTAL.labels(outcome="error").inc()
            DR_RUN_DURATION.observe(time.perf_counter() - run_start)
            DR_ITERATIONS.observe(self._iteration_count)
            self._end_trace(status="error", error=exc if isinstance(exc, Exception) else None)
            raise
        if result.budget_exhausted:
            outcome = "budget_exhausted"
        elif not self._decomposed:
            outcome = "early_stop"
        else:
            outcome = "ok"
        DR_RUNS_TOTAL.labels(outcome=outcome).inc()
        DR_RUN_DURATION.observe(time.perf_counter() - run_start)
        DR_ITERATIONS.observe(self._iteration_count)
        self._end_trace(status="ok")
        return result

    def _end_trace(self, *, status: str, error: Optional[Exception] = None) -> None:
        """Best-effort terminate the parent trace. Fails open."""
        t = self._trace
        if t is None:
            return
        end = getattr(t, "end", None)
        if not callable(end):
            return
        try:
            end(status=status, error=error)
        except Exception:  # noqa: BLE001
            pass

    def _trace_set_attribute(self, key: str, value: object) -> None:
        t = self._trace
        if t is None:
            return
        fn = getattr(t, "set_attribute", None)
        if not callable(fn):
            return
        try:
            fn(key, value)
        except Exception:  # noqa: BLE001
            pass

    async def _research_impl(self) -> DeepResearchResult:
        self._start_ms = time.perf_counter() * 1000
        # Root pool: searches against the processed query.
        root_pool = TopicPool(
            name="<root>",
            rerank_anchor=self._original_question,
        )

        # Round 0 — root retrieval against the processed query.
        await self._retrieve_into(root_pool, self._processed_query)

        # First sufficiency check (always runs — hard floor).
        suff = await self._sufficiency_check(root_pool)
        if suff is None or self._budget_check():
            return self._build_result(topic_pools=[root_pool], is_unified=True)
        early = self._early_stop_ok(root_pool, suff, iter0=True)
        if early is not None:
            self._early_stop_reason = early
            return self._build_result(topic_pools=[root_pool], is_unified=True)

        # First decomposition — drives the topic split.
        decomp = await self._decompose(
            current_query=self._processed_query,
            missing=str(suff.get("missing_information", "")),
            evidence_pool=root_pool,
        )
        topics = (decomp or {}).get("topics") or []
        topics = topics[: self._budget.max_topics]
        self._trace_set_attribute("topic_count", len(topics))

        # Adaptive depth: use the LLM's confidence in the split to decide
        # how deeply to recurse. Missing field defaults to 0.0 (low conf).
        try:
            self._split_confidence = float((decomp or {}).get("split_confidence") or 0.0)
        except (TypeError, ValueError):
            self._split_confidence = 0.0
        self._trace_set_attribute("split_confidence", self._split_confidence)
        n_topics = len(topics)
        if n_topics == 1:
            # Coherent query — cap remaining recursion tightly.
            self._budget.max_depth = min(self._budget.max_depth, 2)
        elif n_topics >= 2 and self._split_confidence < 0.7:
            # Unsure split — cap to avoid wasted recursion.
            self._budget.max_depth = min(self._budget.max_depth, 2)
        # else: confident multi-topic split — keep generous depth.

        if not topics:
            return self._build_result(topic_pools=[root_pool], is_unified=True)

        if len(topics) == 1:
            # Single topic with refinements — recurse, but keep one pool.
            single = topics[0]
            qs = (single.get("questions") or [])[: self._budget.per_topic_questions]
            await self._recurse_topic(root_pool, qs, depth=self._budget.max_depth - 1)
            return self._build_result(topic_pools=[root_pool], is_unified=True)

        # Multi-topic — fan out. Each topic owns its own pool, seeded with the
        # root pool's chunks (they were broad and may be relevant to any topic;
        # final per-topic rerank against the topic anchor will filter).
        topic_pools: list[TopicPool] = []
        recur_tasks: list[Awaitable[None]] = []
        for t in topics:
            name = str(t.get("name") or "topic").strip() or "topic"
            qs = (t.get("questions") or [])[: self._budget.per_topic_questions]
            anchor = qs[0]["question"] if qs and isinstance(qs[0], dict) and "question" in qs[0] else name
            tp = TopicPool(name=name, rerank_anchor=str(anchor))
            tp.merge_chunks(list(root_pool.chunks_by_id.values()))
            tp.executed_queries.update(root_pool.executed_queries)
            topic_pools.append(tp)
            recur_tasks.append(self._recurse_topic(tp, qs, depth=self._budget.max_depth - 1))

        await asyncio.gather(*recur_tasks, return_exceptions=True)
        return self._build_result(topic_pools=topic_pools, is_unified=False)

    # ------------------------------------------------------------------
    # Recursion
    # ------------------------------------------------------------------

    async def _recurse_topic(
        self,
        pool: TopicPool,
        questions: list[Any],
        depth: int,
    ) -> None:
        if depth <= 0 or self._budget_check():
            return
        # Open per-recursion-level span on the parent trace. Fail open.
        iter_span = None
        try:
            if self._trace is not None:
                iter_span = self._trace.span(
                    f"dr_iteration_depth_{depth}",
                    attributes={
                        "depth": depth,
                        "topic": pool.name,
                        "question_count": len(questions),
                    },
                )
        except Exception:  # noqa: BLE001
            iter_span = None
        try:
            tasks: list[Awaitable[None]] = []
            for q in questions:
                if not isinstance(q, dict):
                    continue
                query = str(q.get("query") or q.get("question") or "").strip()
                if not query:
                    continue
                if query in self._executed_queries:
                    continue
                tasks.append(self._execute_question(pool, q, depth))
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            if iter_span is not None:
                try:
                    iter_span.end(status="ok")
                except Exception:  # noqa: BLE001
                    pass

    async def _execute_question(
        self,
        pool: TopicPool,
        q: dict[str, Any],
        depth: int,
    ) -> None:
        if self._budget_check():
            return
        query = str(q.get("query") or "").strip()
        nl_question = str(q.get("question") or query).strip()
        if not query:
            return
        await self._retrieve_into(pool, query)

        # Sufficiency check anchored to ORIGINAL question.
        if self._budget_check():
            return
        suff = await self._sufficiency_check(pool)
        if suff is None:
            return
        if self._early_stop_ok(pool, suff) is not None:
            return
        if depth <= 1:
            return

        # Recurse: emit gap-filler queries within this topic.
        decomp = await self._decompose(
            current_query=query,
            missing=str(suff.get("missing_information", "")),
            evidence_pool=pool,
        )
        if not decomp:
            return
        deeper_topics = decomp.get("topics") or []
        # Inside an existing topic we don't re-split — flatten all returned
        # questions into this pool's recursion.
        deeper_questions: list[dict[str, Any]] = []
        for t in deeper_topics:
            for qq in (t.get("questions") or [])[: self._budget.per_topic_questions]:
                deeper_questions.append(qq)
                if len(deeper_questions) >= self._budget.per_topic_questions:
                    break
            if len(deeper_questions) >= self._budget.per_topic_questions:
                break
        if not deeper_questions:
            return
        await self._recurse_topic(pool, deeper_questions, depth - 1)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    async def _retrieve_into(self, pool: TopicPool, query: str) -> None:
        if not query or query in self._executed_queries:
            return
        if self._budget_check():
            return
        self._executed_queries.add(query)
        pool.executed_queries.add(query)
        self._node_count += 1

        # Run KG and KB retrieval in parallel — independent network calls.
        kg_task: Optional[Awaitable[Optional[KGSnippet]]] = None
        if self._kg_retrieve is not None:
            kg_task = asyncio.create_task(self._safe_kg(query))
        try:
            kb_results = await self._safe_kb(query)
        except Exception as exc:  # noqa: BLE001 — log and continue
            logger.warning("deep_research kb_retrieve failed for query=%r: %s", query, exc)
            kb_results = []
        kg_snippet: Optional[KGSnippet] = None
        if kg_task is not None:
            try:
                kg_snippet = await kg_task
            except Exception as exc:  # noqa: BLE001
                logger.warning("deep_research kg_retrieve failed for query=%r: %s", query, exc)
                kg_snippet = None
        if kb_results:
            pool.merge_chunks(kb_results)
        if kg_snippet and kg_snippet.graph_context:
            self._absorb_graph_context(kg_snippet.graph_context)

    async def _safe_kb(self, query: str) -> list[SearchResult]:
        return await self._kb_retrieve(query)

    async def _safe_kg(self, query: str) -> Optional[KGSnippet]:
        if self._kg_retrieve is None:
            return None
        return await self._kg_retrieve(query)

    def _absorb_graph_context(self, snippet: str) -> None:
        s = (snippet or "").strip()
        if not s or s in self._graph_context_seen:
            return
        if self._graph_context_len + len(s) > self._budget.graph_context_max_chars:
            return
        self._graph_context_seen.add(s)
        self._graph_context_chunks.append(s)
        self._graph_context_len += len(s) + 2

    # ------------------------------------------------------------------
    # LLM calls
    # ------------------------------------------------------------------

    async def _sufficiency_check(self, pool: TopicPool) -> Optional[dict[str, Any]]:
        if self._budget_check():
            return None
        iter_start = time.perf_counter()
        self._iteration_count += 1
        evidence = pool.evidence_text(max_chars=8000)
        prompt = _render(
            self._sufficiency_template,
            original_question=self._original_question,
            evidence_pool=evidence,
        )
        try:
            return await self._call_json([{"role": "user", "content": prompt}], purpose="sufficiency")
        finally:
            DR_ITERATION_DURATION.observe(time.perf_counter() - iter_start)

    async def _decompose(
        self,
        current_query: str,
        missing: str,
        evidence_pool: TopicPool,
    ) -> Optional[dict[str, Any]]:
        if self._budget_check():
            return None
        evidence = evidence_pool.evidence_text(max_chars=8000)
        prompt = _render(
            self._decomp_template,
            original_question=self._original_question,
            current_query=current_query,
            evidence_pool=evidence,
            missing_information=missing or "(no specific gap was named)",
        )
        result = await self._call_json([{"role": "user", "content": prompt}], purpose="decompose")
        if result is not None:
            self._decomposed = True
            topics = result.get("topics") or []
            DR_TOPIC_COUNT.observe(len(topics))
        return result

    async def _call_json(
        self,
        messages: list[dict[str, Any]],
        *,
        purpose: str,
    ) -> Optional[dict[str, Any]]:
        if self._budget_check():
            return None
        self._llm_call_count += 1
        stage = "sufficiency" if purpose == "sufficiency" else "multi_queries"
        DR_LLM_CALLS_TOTAL.labels(stage=stage).inc()

        # Open a Generation span on the parent trace. Fail open.
        gen = None
        try:
            if self._trace is not None:
                # Render input as the last message content (a single user prompt).
                prompt_input = ""
                if messages and isinstance(messages[-1], dict):
                    prompt_input = str(messages[-1].get("content") or "")
                gen = self._trace.generation(
                    name=f"dr_{purpose}",
                    model=self._model_alias,
                    input=prompt_input,
                    metadata={
                        "iteration": self._iteration_count,
                        "node_count": self._node_count,
                        "purpose": purpose,
                    },
                )
        except Exception:  # noqa: BLE001
            gen = None

        try:
            resp = await self._provider.agenerate(
                messages,
                model_alias=self._model_alias,
                response_format={"type": "json_object"},
                timeout=self._llm_timeout_s,
                temperature=0.0,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("deep_research %s LLM call failed: %s", purpose, exc)
            if gen is not None:
                try:
                    gen.end(status="error", error=exc)
                except Exception:  # noqa: BLE001
                    pass
            return None

        if gen is not None:
            try:
                gen.set_output(resp.content or "")
                gen.set_token_counts(
                    int(getattr(resp, "prompt_tokens", 0) or 0),
                    int(getattr(resp, "completion_tokens", 0) or 0),
                )
                gen.end(status="ok")
            except Exception:  # noqa: BLE001
                pass
        return _parse_json_object(resp.content or "")

    # ------------------------------------------------------------------
    # Budget / result
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Early-stop gate
    # ------------------------------------------------------------------

    def _early_stop_ok(
        self,
        pool: TopicPool,
        suff: dict[str, Any],
        *,
        iter0: bool = False,
    ) -> Optional[str]:
        """Return a non-None reason string when the tightened early-stop
        condition is satisfied; otherwise None.

        Conditions (ALL required):
        1. ``suff.is_sufficient == True``
        2. Pool has at least ``early_stop_min_chunks`` chunks (chunk floor)
        3. Pool spans at least ``early_stop_min_sources`` distinct
           ``metadata['source']`` values (source diversity)
        4. Confidence trend is monotonically non-decreasing across this
           pool's iterations (no oscillation). A single iteration always
           passes the trend check.
        """
        if not self._budget.early_stop_enabled:
            # Legacy behaviour: any True is_sufficient stops.
            if suff.get("is_sufficient", False):
                return "iter0_sufficient" if iter0 else "sufficient_legacy"
            return None
        if not suff.get("is_sufficient", False):
            return None
        # Track confidence trend on the pool. Use explicit "confidence" if the
        # LLM emits one; fall back to bool→{0.0,1.0}.
        raw_conf = suff.get("confidence")
        try:
            conf = float(raw_conf) if raw_conf is not None else 1.0
        except (TypeError, ValueError):
            conf = 1.0
        history = pool.confidence_history
        # Non-decreasing trend: latest >= previous max.
        if history and conf < max(history):
            # Oscillation — bail.
            history.append(conf)
            return None
        history.append(conf)
        # Chunk floor.
        if len(pool.chunks_by_id) < self._budget.early_stop_min_chunks:
            return None
        # Source diversity.
        sources: set[str] = set()
        for c in pool.chunks_by_id.values():
            md = c.metadata or {}
            src = md.get("source") or md.get("document_id")
            if src:
                sources.add(str(src))
        if len(sources) < self._budget.early_stop_min_sources:
            return None
        return "iter0_sufficient" if iter0 else "sufficient+floor+diversity"

    def _budget_check(self) -> bool:
        if self._stop_reason is not None:
            return True
        if self._node_count >= self._budget.max_nodes:
            self._stop_reason = "max_nodes"
            return True
        if self._llm_call_count >= self._budget.max_llm_calls:
            self._stop_reason = "max_llm_calls"
            return True
        elapsed = time.perf_counter() * 1000 - self._start_ms
        if elapsed >= self._budget.wall_clock_ms:
            self._stop_reason = "wall_clock"
            return True
        return False

    # ------------------------------------------------------------------
    # Per-topic rerank
    # ------------------------------------------------------------------

    def _apply_per_topic_rerank(self, result: DeepResearchResult) -> DeepResearchResult:
        """Decide and (optionally) apply per-topic reranking on the topic
        pools. Always emits exactly one ``DR_PER_TOPIC_RERANK_TOTAL`` increment
        and surfaces the decision on the result.
        """
        budget = self._budget
        applied = False
        reason: Optional[str] = None
        merged: Optional[list[Any]] = None
        mode: str = "off"

        if not budget.per_topic_rerank_enabled:
            reason = "disabled"
            mode = "off"
        elif result.is_unified:
            reason = "unified"
            mode = "skipped_unified"
        elif len(result.topic_pools) < 2:
            reason = "single_topic"
            mode = "off"
        elif (result.split_confidence or 0.0) < budget.per_topic_rerank_min_confidence:
            reason = "low_confidence"
            mode = "off"
        elif self._reranker is None:
            # No reranker plumbed in (caller will rerank). Skip silently.
            reason = "no_reranker"
            mode = "off"
        else:
            # Open span on the parent trace; fail open.
            span = None
            try:
                if self._trace is not None:
                    span = self._trace.span(
                        "dr_per_topic_rerank",
                        attributes={
                            "topic_count": len(result.topic_pools),
                            "split_confidence": result.split_confidence,
                            "per_topic_top_k": budget.per_topic_top_k,
                        },
                    )
            except Exception:  # noqa: BLE001
                span = None
            try:
                per_topic_ranked: list[list[Any]] = []
                for pool in result.topic_pools:
                    chunks = list(pool.chunks_by_id.values())
                    if not chunks:
                        per_topic_ranked.append([])
                        continue
                    try:
                        ranked = self._reranker.rerank(
                            query=pool.rerank_anchor or self._original_question,
                            documents=chunks,
                            top_k=budget.per_topic_top_k,
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("per-topic rerank failed for pool=%r: %s",
                                       pool.name, exc)
                        ranked = []
                    per_topic_ranked.append(list(ranked))
                # Round-robin merge: t1[0], t2[0], ..., tN[0], t1[1], ...
                merged = []
                if per_topic_ranked:
                    max_len = max((len(r) for r in per_topic_ranked), default=0)
                    for i in range(max_len):
                        for r in per_topic_ranked:
                            if i < len(r):
                                merged.append(r[i])
                applied = True
                mode = "on"
            finally:
                if span is not None:
                    try:
                        span.end(status="ok")
                    except Exception:  # noqa: BLE001
                        pass

        # Emit metric exactly once per run.
        try:
            DR_PER_TOPIC_RERANK_TOTAL.labels(mode=mode).inc()
        except Exception:  # noqa: BLE001
            pass

        # Surface on the result and stash on the orchestrator for inspection.
        result.per_topic_rerank_applied = applied
        result.per_topic_rerank_skipped_reason = None if applied else reason
        result.per_topic_rerank_merged = merged
        self._per_topic_rerank_applied = applied
        self._per_topic_rerank_skipped_reason = result.per_topic_rerank_skipped_reason
        self._per_topic_rerank_merged = merged
        # Refresh the side-channel snapshot so callers (and tests) read the
        # post-decision state.
        DeepResearch.last_result = result
        return result

    def _build_result(
        self,
        *,
        topic_pools: list[TopicPool],
        is_unified: bool,
    ) -> DeepResearchResult:
        elapsed = time.perf_counter() * 1000 - self._start_ms
        result = DeepResearchResult(
            topic_pools=topic_pools,
            graph_context=_truncate(
                "\n\n".join(self._graph_context_chunks),
                self._budget.graph_context_max_chars,
            ),
            node_count=self._node_count,
            llm_call_count=self._llm_call_count,
            elapsed_ms=elapsed,
            budget_exhausted=self._stop_reason is not None,
            budget_exhausted_reason=self._stop_reason,
            is_unified=is_unified,
            iteration_count=self._iteration_count,
            decomposed=self._decomposed,
            topic_count=len(topic_pools),
            early_stop_reason=self._early_stop_reason,
            split_confidence=self._split_confidence,
        )
        if self._early_stop_reason is not None:
            self._trace_set_attribute("early_stop_reason", self._early_stop_reason)
        DeepResearch.last_result = result
        return result
