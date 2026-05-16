# @summary
# End-to-end RAG pipeline for query processing, KG expansion, hybrid search, reranking,
# visual retrieval, and LLM generation.  KG graph_context (REQ-KG-794, REQ-KG-796) is
# threaded from expansion to the generator prompt, placed before document chunks when
# non-empty and omitted entirely when empty.
# Stage 7.5 implements a bounded internal re-retrieval loop (REQ-706 option B): when
# composite confidence is below the high threshold and retry budget remains, the chain
# re-runs search+rerank+generate+score with broader params (alpha-=0.15, limit+=5) and
# returns whichever attempt scores higher. Surfaces both scores in RAGResponse.
# Main classes: RAGChain, RAGResponse. Deps: src.vector_db, src.guardrails, src.retrieval.generation.nodes.generator, src.retrieval.query.nodes.query_processor, src.retrieval.common.schemas, src.core, src.platform
# @end-summary
"""Main RAG chain that orchestrates the full retrieval pipeline."""
from __future__ import annotations


from typing import Any, Optional
import logging
import statistics
import time

from src.core.embeddings import get_embedding_provider
from src.retrieval.query.nodes import get_reranker_provider
from src.retrieval.query.nodes import process_query
from kgweave.client import get_client as _get_kg_client, KGQueryService
from src.vector_db import (
    create_persistent_client,
    get_client,
    close_client,
    ensure_collection,
    search,
    SearchFilter,
)
from src.retrieval.generation.nodes import OllamaGenerator
from src.platform.observability import get_tracer
from src.platform.reliability import get_retry_provider
from src.platform.schemas import RetryPolicy
from src.platform import (
    validate_alpha,
    validate_filter_value,
    validate_positive_int,
)
from src.retrieval.query import (
    QueryAction,
    QueryResult,
)
from src.retrieval.common import (
    RAGRequest,
    RAGResponse,
    RankedResult,
    VisualPageResult,
    AskUserReason,
    StageOutcome,
    StageStatus,
    TerminalDecision,
    decide_terminal,
)
from src.platform.token_budget import calculate_budget, get_capabilities, TokenBudgetSnapshot
from src.platform.llm.provider import get_llm_provider
from src.retrieval.generation.nodes import get_system_prompt
from src.retrieval.pipeline.query_shape import should_suggest_deep_research
from src.retrieval.pipeline.deep_research import (
    DeepResearch,
    DeepResearchBudget,
    DeepResearchResult,
    KGSnippet,
    TopicPool,
)
from config.settings import (
    HYBRID_SEARCH_ALPHA, SEARCH_LIMIT, RERANK_TOP_K,
    KG_ENABLED, GENERATION_ENABLED,
    RETRY_BACKOFF_MULTIPLIER,
    RETRY_INITIAL_BACKOFF_SECONDS,
    RETRY_MAX_ATTEMPTS,
    RETRY_MAX_BACKOFF_SECONDS,
    MAX_SANITIZATION_ITERATIONS,
    QUERY_CONFIDENCE_THRESHOLD,
    RAG_DEFAULT_FAST_PATH,
    RAG_RETRIEVAL_TIMEOUT_MS,
    RAG_STAGE_BUDGET_QUERY_PROCESSING_MS,
    RAG_STAGE_BUDGET_KG_EXPANSION_MS,
    RAG_STAGE_BUDGET_EMBEDDING_MS,
    RAG_STAGE_BUDGET_HYBRID_SEARCH_MS,
    RAG_STAGE_BUDGET_RERANKING_MS,
    RAG_STAGE_BUDGET_GENERATION_MS,
    RAG_RETRIEVAL_QUALITY_STRONG_THRESHOLD,
    RAG_RETRIEVAL_QUALITY_MODERATE_THRESHOLD,
    RAG_RETRIEVAL_QUALITY_WEAK_THRESHOLD,
    RAG_STAGE_BUDGET_DEEP_RESEARCH_MS,
    RAG_DEEP_RESEARCH_MAX_NODES,
    RAG_DEEP_RESEARCH_MAX_LLM_CALLS,
    RAG_DEEP_RESEARCH_MAX_DEPTH,
    RAG_DEEP_RESEARCH_MAX_TOPICS,
    RAG_DEEP_RESEARCH_PER_TOPIC_QUESTIONS,
    RAG_DEEP_RESEARCH_GRAPH_CONTEXT_MAX_CHARS,
    RAG_DEEP_RESEARCH_KB_TOP_PER_NODE,
    RAG_RETRIEVAL_INIT_POOL_MAX_WORKERS,
    RAG_RETRIEVAL_STAGE1_POOL_MAX_WORKERS,
    RAG_RETRIEVAL_EMBEDDING_CACHE_MAX_SIZE,
)
from src.platform import TimingPool
from config.settings import GUARDRAIL_BACKEND
from src.guardrails import (
    run_input_rails,
    run_output_rails,
    register_rag_chain,
    redact_pii,
    RailMergeGate,
)
from src.guardrails.common import GuardrailsMetadata
from config.settings import (
    RAG_CONFIDENCE_ROUTING_ENABLED,
    RAG_CONFIDENCE_HIGH_THRESHOLD,
    RAG_CONFIDENCE_LOW_THRESHOLD,
    RAG_CONFIDENCE_RETRIEVAL_WEIGHT,
    RAG_CONFIDENCE_LLM_WEIGHT,
    RAG_CONFIDENCE_CITATION_WEIGHT,
    RAG_CONFIDENCE_RE_RETRIEVE_MAX_RETRIES,
    RAG_DOCUMENT_FORMATTING_ENABLED,
    RAG_VISUAL_RETRIEVAL_ENABLED,
    RAG_STAGE_BUDGET_VISUAL_RETRIEVAL_MS,
)
from config.settings import (
    RAG_TREE_RETRIEVAL_ENABLED,
    RAG_TREE_DESCENT_TOP_K,
    RAG_TREE_DESCENT_LEAVES_PER_SECTION,
    RAG_TREE_DESCENT_DOC_DIVERSITY_TOP_PER_DOC,
    RAG_TREE_LIFT_SEED_K,
    RAG_TREE_LIFT_SIBLINGS,
    RAG_STAGE_BUDGET_TREE_DESCENT_MS,
    RAG_STAGE_BUDGET_TREE_LIFT_MS,
)
from config.settings import (
    RAG_RERANK_FUSION_ENABLED,
    RAG_RERANK_RRF_K,
    RAG_RERANK_RRF_LAMBDA,
    RAG_RERANK_HEADING_LAMBDA,
    RAG_RERANK_ANCHOR_LAMBDA,
    RAG_RERANK_ANCHOR_K,
)
from src.retrieval.query.nodes.rerank_fusion import (
    anchor_confidence,
    fuse_scores,
    heading_match_score,
)
import os as _os
RAG_TREE_SCHEMA_PRESENT: bool = _os.environ.get(
    "RAG_TREE_SCHEMA_PRESENT", "true"
).lower() in ("true", "1", "yes")

logger = logging.getLogger("rag.rag_chain")


class RAGChain:
    """End-to-end RAG pipeline: query processing -> KG expansion -> hybrid search -> reranking."""

    def __init__(self, persistent_weaviate: bool = True):
        from concurrent.futures import ThreadPoolExecutor, Future

        self.tracer = get_tracer()
        self.retry_provider = get_retry_provider()
        self.retry_policy = RetryPolicy(
            max_attempts=RETRY_MAX_ATTEMPTS,
            initial_backoff_seconds=RETRY_INITIAL_BACKOFF_SECONDS,
            max_backoff_seconds=RETRY_MAX_BACKOFF_SECONDS,
            backoff_multiplier=RETRY_BACKOFF_MULTIPLIER,
        )

        # Persistent Weaviate connection avoids per-query startup cost
        self._persistent_weaviate = persistent_weaviate
        self._weaviate_client = None
        if persistent_weaviate:
            logger.info("Opening persistent Weaviate connection...")
            self._weaviate_client = create_persistent_client()
            ensure_collection(self._weaviate_client)
            logger.info("Weaviate connected (persistent mode).")

        # GPU models must load sequentially (parallel .to(cuda) causes meta
        # tensor errors), but KG + generator can load concurrently with them.
        def _load_kg() -> Optional[KGQueryService]:
            # Returns the KG read-side service. In-process by default; flips
            # to HTTP transparently when KGWEAVE_API_URL is set.
            _t0 = time.perf_counter()
            result: Optional[KGQueryService] = None
            try:
                if not KG_ENABLED:
                    logger.info("Knowledge graph disabled (set RAG_KG_ENABLED=true to enable).")
                    return None
                try:
                    service = _get_kg_client()
                    health = service.health()
                    stats = health.stats or {}
                    logger.info(
                        "KG service ready (%s): %s nodes, %s edges",
                        health.backend,
                        stats.get("nodes", "?"),
                        stats.get("edges", "?"),
                    )
                    result = service
                except Exception as e:
                    logger.warning("Could not initialize KG service: %s", e)
                return result
            finally:
                logger.info("_load_kg elapsed: %.1fms", (time.perf_counter() - _t0) * 1000)

        def _load_generator() -> Optional[OllamaGenerator]:
            _t0 = time.perf_counter()
            result: Optional[OllamaGenerator] = None
            try:
                if GENERATION_ENABLED:
                    try:
                        gen = OllamaGenerator()
                        if gen.is_available():
                            logger.info("Ollama generator ready (model: %s).", gen.model)
                            result = gen
                        else:
                            logger.warning("Ollama not available. Generation disabled.")
                    except Exception as e:
                        logger.warning("Failed to initialize Ollama generator: %s", e)
                else:
                    logger.info("Generation disabled (set RAG_GENERATION_ENABLED=true to enable).")
                return result
            finally:
                logger.info("_load_generator elapsed: %.1fms", (time.perf_counter() - _t0) * 1000)

        def _load_models_sequential():
            """Load GPU models one at a time to avoid meta tensor conflicts."""
            _t0 = time.perf_counter()
            try:
                logger.info("Loading embedding model...")
                _t_emb = time.perf_counter()
                emb = get_embedding_provider()
                logger.info("Embedding provider ready in %.1fms.", (time.perf_counter() - _t_emb) * 1000)

                logger.info("Loading reranker...")
                _t_rer = time.perf_counter()
                rer = get_reranker_provider()
                logger.info("Reranker model loaded in %.1fms.", (time.perf_counter() - _t_rer) * 1000)
                return emb, rer
            finally:
                logger.info("_load_models_sequential elapsed: %.1fms", (time.perf_counter() - _t0) * 1000)

        with ThreadPoolExecutor(max_workers=RAG_RETRIEVAL_INIT_POOL_MAX_WORKERS) as pool:
            fut_models = pool.submit(_load_models_sequential)
            fut_kg = pool.submit(_load_kg)
            fut_gen = pool.submit(_load_generator)

            self.embeddings, self.reranker = fut_models.result()
            self._kg_expander: Optional[KGQueryService] = fut_kg.result()
            self._generator: Optional[OllamaGenerator] = fut_gen.result()

        # Embedding LRU cache — avoids re-computing embeddings for repeated
        # exact queries (REQ-306). Keyed on the exact query string.
        from collections import OrderedDict
        self._embedding_cache: OrderedDict = OrderedDict()
        self._embedding_cache_max = RAG_RETRIEVAL_EMBEDDING_CACHE_MAX_SIZE

        # Initialize guardrail backend (REQ-701: once at startup, not per-query)
        self._guardrails_merge_gate = None
        if GUARDRAIL_BACKEND:
            self._init_guardrails()

        # FR-603, FR-615: Visual retrieval lazy-loading attributes
        self._visual_retrieval_enabled = RAG_VISUAL_RETRIEVAL_ENABLED
        self._visual_model = None
        self._visual_processor = None
        if self._visual_retrieval_enabled:
            from config.settings import validate_visual_retrieval_config
            validate_visual_retrieval_config()  # FR-111: fail fast on bad config
            logger.info("Visual retrieval enabled — model will be loaded on first visual query.")

        logger.info("RAG chain ready.")

    def close(self) -> None:
        """Release persistent resources (database connection, visual model)."""
        if self._weaviate_client is not None:
            try:
                close_client(self._weaviate_client)
            except Exception as e:
                logger.warning("Error closing database client: %s", e)
            self._weaviate_client = None
            logger.info("Database connection closed.")

        # FR-613: unload visual model if loaded
        if self._visual_model is not None:
            try:
                from src.ingest.support import unload_colqwen_model
                unload_colqwen_model(self._visual_model)
                logger.info("ColQwen2 visual model unloaded.")
            except Exception as e:
                logger.warning("Error unloading visual model: %s", e)
            self._visual_model = None
            self._visual_processor = None

    def _init_guardrails(self) -> None:
        """Initialize the guardrail backend and merge gate."""
        logger.info("Initializing guardrails backend (backend=%r)...", GUARDRAIL_BACKEND)
        self._guardrails_merge_gate = RailMergeGate()
        register_rag_chain(self)
        logger.info("Guardrails backend initialized.")

    def _ensure_visual_model(self) -> None:
        """Lazy-load ColQwen2 model on first visual query. FR-603

        Imports are deferred to keep visual dependencies out of the text-only path.
        """
        if self._visual_model is not None:
            return  # warm path

        with self.tracer.span("visual_retrieval.model_load"):
            from src.ingest.support import (
                ensure_colqwen_ready,
                load_colqwen_model,
            )
            from config.settings import RAG_INGESTION_COLQWEN_MODEL

            logger.info("Loading ColQwen2 model for visual retrieval (cold start)...")
            ensure_colqwen_ready()
            self._visual_model, self._visual_processor = load_colqwen_model(
                RAG_INGESTION_COLQWEN_MODEL
            )
            logger.info("ColQwen2 model loaded for visual retrieval.")

    def _run_visual_retrieval(
        self, processed_query: str, tenant_id: Optional[str]
    ) -> list[VisualPageResult]:
        """Execute the visual retrieval track. FR-601, FR-605, FR-607, FR-609

        Encodes the processed query via ColQwen2, searches the visual collection,
        generates presigned URLs for matched pages, and returns visual results.

        Args:
            processed_query: The processed (reformulated) query text.
            tenant_id: Optional tenant filter.

        Returns:
            List of VisualPageResult objects.
        """
        from config.settings import (
            RAG_VISUAL_RETRIEVAL_LIMIT,
            RAG_VISUAL_RETRIEVAL_MIN_SCORE,
        )

        # FR-603: ensure model is loaded
        self._ensure_visual_model()

        # FR-605: encode text query (uses processed query, not raw)
        from src.ingest.support import embed_text_query

        with self.tracer.span("visual_retrieval.text_encode"):
            query_vector = embed_text_query(
                self._visual_model, self._visual_processor, processed_query
            )
        logger.debug(
            "Visual query vector: %d dimensions", len(query_vector)
        )

        # FR-609: search visual collection
        from src.vector_db import search_visual

        with self.tracer.span("visual_retrieval.search") as vs_span:
            page_records = search_visual(
                client=self._weaviate_client,
                query_vector=query_vector,
                limit=RAG_VISUAL_RETRIEVAL_LIMIT,
                score_threshold=RAG_VISUAL_RETRIEVAL_MIN_SCORE,
                tenant_id=tenant_id,
            )
            vs_span.set_attribute("result_count", len(page_records))

        # FR-607: generate presigned URLs per result (per-page isolation — NFR-905)
        from src.db.minio import (
            create_client,
            get_page_image_url,
        )

        results: list[VisualPageResult] = []
        with self.tracer.span("visual_retrieval.presigned_urls"):
            minio_client = create_client()
            for record in page_records:
                try:
                    url = get_page_image_url(
                        minio_client,
                        minio_key=record["minio_key"],
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to generate presigned URL for page %s/%d: %s — skipping page.",
                        record.get("document_id", "?"),
                        record.get("page_number", 0),
                        exc,
                    )
                    continue

                results.append(VisualPageResult(
                    document_id=record["document_id"],
                    page_number=record["page_number"],
                    source_key=record["source_key"],
                    source_name=record["source_name"],
                    score=record["score"],
                    page_image_url=url,
                    total_pages=record["total_pages"],
                    page_width_px=record["page_width_px"],
                    page_height_px=record["page_height_px"],
                ))

        if results:
            score_range = f"{results[-1].score:.3f}-{results[0].score:.3f}"
            logger.debug("Visual results score range: %s", score_range)
        logger.info("Visual retrieval returned %d results.", len(results))
        return results

    def _do_search(self, bm25_query, query_embedding, alpha, search_limit, filters):
        """Run hybrid search against the database layer (persistent or transient client)."""
        _t0 = time.perf_counter()
        try:
            if self._weaviate_client is not None:
                results = search(
                    client=self._weaviate_client,
                    query=bm25_query,
                    query_embedding=query_embedding,
                    alpha=alpha,
                    limit=search_limit,
                    filters=filters,
                )
            else:
                with get_client() as client:
                    ensure_collection(client)
                    results = search(
                        client=client,
                        query=bm25_query,
                        query_embedding=query_embedding,
                        alpha=alpha,
                        limit=search_limit,
                        filters=filters,
                    )
            logger.debug(
                "_do_search returned %d hits in %.1fms (alpha=%.2f, limit=%d)",
                len(results), (time.perf_counter() - _t0) * 1000, alpha, search_limit,
            )
            return results
        except Exception as exc:
            logger.error(
                "_do_search failed after %.1fms: %s",
                (time.perf_counter() - _t0) * 1000, exc,
            )
            raise

    # ─── Tree retrieval helpers (TREE_RETRIEVAL_DESIGN.md §4) ────────────
    @staticmethod
    def _build_leaf_only_filter_clauses(schema_present: bool) -> list:
        """Return the default Stage-4 filter clauses that pin retrieval to leaves.

        When ``schema_present`` is True (post-1.2.0 collections), section nodes
        live in the same collection as leaf chunks. Without an explicit
        ``node_kind="chunk"`` filter, section text would surface as low-quality
        evidence whenever the retrieval flag is off — see the design doc's
        section-leak analysis. Operators on legacy collections that haven't
        run the 1.2.0 migration set ``schema_present=False`` to skip the
        filter (which would otherwise fail with "no such property").
        """
        if not schema_present:
            return []
        return [SearchFilter(property="node_kind", operator="eq", value="chunk")]

    def _expand_sections_to_leaves(
        self,
        section_parent_ids: list[str],
        per_section_limit: int,
        base_filters,
    ) -> list:
        """Default expansion: fetch leaf chunks under each section's id.

        Tests stub this method; the real implementation reads from the vector
        store via ``fetch_chunks_by_parent_section``.
        """
        from src.vector_db.weaviate.store import fetch_chunks_by_parent_section
        if self._weaviate_client is None:
            return []
        raw = fetch_chunks_by_parent_section(
            self._weaviate_client,
            section_parent_ids,
            limit_per_section=per_section_limit,
        )
        return [
            type("E", (), {"text": r["text"], "score": 0.0,
                           "metadata": r["metadata"], "collection": "default"})()
            for r in raw
        ]

    def _fetch_siblings(
        self,
        parent_ids: list[str],
        per_group_limit: int,
        base_filters,
    ) -> list:
        """Fetch sibling chunks under each parent_section_id (Stage 4c lift).

        Tests stub this method to avoid hitting Weaviate; the real
        implementation routes through the same store helper as descent
        expansion.
        """
        return self._expand_sections_to_leaves(
            parent_ids, per_group_limit, base_filters
        )

    def _run_tree_descent(
        self,
        bm25_query: str,
        query_embedding,
        alpha: float,
        descent_top_k: int,
        leaves_per_section: int,
        base_filters,
    ) -> list:
        """Stage 4b — search section nodes, then expand to their leaves.

        Each expanded leaf is tagged with ``_anchor_rank`` (private metadata
        key, leading underscore) — the 0-indexed rank of the section that
        produced it within ``section_hits``. The rerank-fusion module
        consumes this signal when its anchor-confidence component is
        enabled (R1.3).
        """
        section_filter = SearchFilter(
            property="node_kind", operator="eq", value="section"
        )
        filters = list(base_filters or []) + [section_filter]
        section_hits = self._do_search(
            bm25_query, query_embedding, alpha, descent_top_k, filters
        )
        # Per TREE_RETRIEVAL_DESIGN.md §4.1: leaves under section S have
        # parent_section_id == S.chunk_id. So descent expansion uses each
        # section's own chunk_id as the lookup key.
        parent_ids = [
            (h.metadata.get("chunk_id") or "")
            for h in section_hits
        ]
        # Map each parent_section_id → its 0-indexed rank in the section
        # search. The leaves returned by ``_expand_sections_to_leaves`` will
        # be tagged with the rank of the section that produced them, looked
        # up by their ``parent_section_id`` metadata (which production
        # ingest writes for every leaf — see store.py prepare_object).
        anchor_rank_by_pid: dict[str, int] = {}
        for rank, pid in enumerate(parent_ids):
            if pid and pid not in anchor_rank_by_pid:
                anchor_rank_by_pid[pid] = rank

        leaves = self._expand_sections_to_leaves(
            parent_ids, leaves_per_section, base_filters
        )
        for leaf in leaves:
            meta = getattr(leaf, "metadata", None)
            if meta is None:
                meta = {}
                try:
                    leaf.metadata = meta
                except AttributeError:  # pragma: no cover
                    continue
            pid = meta.get("parent_section_id") or ""
            ar = anchor_rank_by_pid.get(pid)
            if ar is not None:
                meta["_anchor_rank"] = ar
        return leaves

    def _run_tree_lift(
        self,
        seed_results: list,
        seed_top_k: int,
        siblings_per_group: int,
        base_filters,
    ) -> list:
        """Stage 4c — for the top seeds, fetch siblings under their parent.

        Each sibling is tagged with ``_anchor_rank`` — the 0-indexed rank
        of the seed that supplied its parent section. Siblings whose
        parent_section comes from the strongest seed score the highest
        anchor-confidence boost in the rerank-fusion stage (R1.3).
        """
        seeds = list(seed_results)[:seed_top_k]
        parent_ids: list[str] = []
        seen: set[str] = set()
        for seed in seeds:
            psid = (seed.metadata or {}).get("parent_section_id") or ""
            if not psid or psid in seen:
                continue
            seen.add(psid)
            parent_ids.append(psid)
        if not parent_ids:
            return []
        anchor_rank_by_pid: dict[str, int] = {
            pid: rank for rank, pid in enumerate(parent_ids)
        }
        siblings = self._fetch_siblings(parent_ids, siblings_per_group, base_filters)
        for sib in siblings:
            meta = getattr(sib, "metadata", None)
            if meta is None:
                meta = {}
                try:
                    sib.metadata = meta
                except AttributeError:  # pragma: no cover
                    continue
            pid = meta.get("parent_section_id") or ""
            ar = anchor_rank_by_pid.get(pid)
            if ar is not None:
                meta["_anchor_rank"] = ar
        return siblings

    def _collect_candidates(
        self,
        *,
        bm25_query: str,
        query_embedding,
        alpha: float,
        search_limit: int,
        base_filters: list,
        tree_enabled: bool,
        schema_present: bool,
        descent_top_k: int,
        leaves_per_section: int,
        lift_seed_k: int,
        siblings_per_group: int,
        doc_diversity_top_per_doc: int,
    ) -> list:
        """Run Stage 4 + (optionally) 4b descent + 4c lift; return merged list.

        Always applies the leaf-only filter to Stage 4 when ``schema_present``.
        Tree sub-stages run only when ``tree_enabled`` is True. Descent results
        are doc-diversity capped before merge so one verbose document cannot
        dominate the final candidate pool.
        """
        leaf_filters = list(base_filters or []) + self._build_leaf_only_filter_clauses(
            schema_present=schema_present
        )
        leaf_results = self._do_search(
            bm25_query, query_embedding, alpha, search_limit, leaf_filters or None
        )

        if not tree_enabled:
            return list(leaf_results)

        descent = self._run_tree_descent(
            bm25_query=bm25_query,
            query_embedding=query_embedding,
            alpha=alpha,
            descent_top_k=descent_top_k,
            leaves_per_section=leaves_per_section,
            base_filters=base_filters,
        )
        descent_capped = self._apply_doc_diversity_cap(
            descent, top_per_doc=doc_diversity_top_per_doc
        )

        lift = self._run_tree_lift(
            seed_results=list(leaf_results),
            seed_top_k=lift_seed_k,
            siblings_per_group=siblings_per_group,
            base_filters=base_filters,
        )

        # Merge with uuid (preferred) / text fallback dedup, preserving the
        # first occurrence's score so Stage 4 hits keep their hybrid scores.
        merged: list = []
        seen: set[str] = set()
        for src in (leaf_results, descent_capped, lift):
            for item in src:
                key = ""
                meta = getattr(item, "metadata", None) or {}
                key = (
                    str(getattr(item, "uuid", "") or "")
                    or str(meta.get("uuid", "") or "")
                    or str(meta.get("chunk_id", "") or "")
                    or getattr(item, "text", "")
                )
                if key in seen:
                    continue
                seen.add(key)
                merged.append(item)
        return merged

    @staticmethod
    def _apply_doc_diversity_cap(hits: list, top_per_doc: int) -> list:
        """Keep at most ``top_per_doc`` hits per document_id, in score order.

        Stable: input order is preserved across survivors, so callers that
        already sorted by score get a top-per-doc-capped list still sorted by
        score.
        """
        kept: list = []
        per_doc: dict[str, int] = {}
        for h in hits:
            doc_id = (h.metadata or {}).get("document_id") or ""
            count = per_doc.get(doc_id, 0)
            if count >= top_per_doc:
                continue
            per_doc[doc_id] = count + 1
            kept.append(h)
        return kept

    # ─── Rerank fusion helpers (R1 — BM25-RRF + heading + anchor) ────────
    @staticmethod
    def _candidate_key(item) -> str:
        """Stable identifier for a candidate across BM25 + CE pools.

        Uses the same precedence as ``_collect_candidates`` dedup so a
        candidate seen in both pools matches by key.
        """
        meta = getattr(item, "metadata", None) or {}
        return (
            str(getattr(item, "uuid", "") or "")
            or str(meta.get("uuid", "") or "")
            or str(meta.get("chunk_id", "") or "")
            or getattr(item, "text", "")
        )

    def _collect_bm25_ranks(
        self,
        bm25_query: str,
        query_embedding,
        search_limit: int,
        base_filters: list,
        candidates: list,
    ) -> list[Optional[int]]:
        """Run a BM25-only Weaviate search and assign 0-indexed ranks to ``candidates``.

        Implementation note: Weaviate v4's hybrid query exposes only the
        fused score (and a free-text ``explain_score``). Rather than parse
        explain-score we run a parallel ``alpha=0.0`` (BM25-only) hybrid
        search with the same query and filters. The cost is low because
        BM25 is the fast path on the inverted index.

        Returns:
            Same-length list as ``candidates``: a 0-indexed BM25 rank, or
            ``None`` for candidates that did not appear in the BM25 pool
            (e.g., descent/lift leaves never seen by BM25).
        """
        try:
            bm25_hits = self._do_search(
                bm25_query, query_embedding, 0.0, search_limit, base_filters
            )
        except Exception as exc:  # pragma: no cover — defensive only
            logger.warning(
                "_collect_bm25_ranks: BM25-only search failed (%s); "
                "ranks will be None for all candidates", exc,
            )
            return [None] * len(candidates)
        rank_by_key: dict[str, int] = {}
        for rank, hit in enumerate(bm25_hits):
            key = self._candidate_key(hit)
            if key and key not in rank_by_key:
                rank_by_key[key] = rank
        return [
            rank_by_key.get(self._candidate_key(c)) for c in candidates
        ]

    @staticmethod
    def _heading_scores_for(query: str, candidates: list) -> list[float]:
        return [
            heading_match_score(
                query,
                ((getattr(c, "metadata", None) or {}).get("heading_path") or []),
            )
            for c in candidates
        ]

    @staticmethod
    def _anchor_confs_for(candidates: list, anchor_k: int) -> list[float]:
        out: list[float] = []
        for c in candidates:
            meta = getattr(c, "metadata", None) or {}
            out.append(anchor_confidence(meta.get("_anchor_rank"), anchor_k=anchor_k))
        return out

    def _apply_rerank_fusion(
        self,
        *,
        query: str,
        candidates: list,
        reranked: list[RankedResult],
        bm25_ranks: list[Optional[int]],
        rerank_top_k: int,
    ) -> list[RankedResult]:
        """Refine reranker output using the fusion module.

        ``reranked`` is the cross-encoder's top-K (already CE-sorted). We
        rebuild a positional alignment with ``candidates``/``bm25_ranks``
        by candidate key, compute fused scores, re-sort, and slice. When
        fusion is disabled, returns ``reranked`` unchanged so behavior is
        bit-identical to the legacy path.
        """
        if not RAG_RERANK_FUSION_ENABLED or not reranked:
            return reranked

        # Map original candidates → BM25 rank via key.
        cand_key_to_bm25: dict[str, Optional[int]] = {}
        for c, br in zip(candidates, bm25_ranks):
            cand_key_to_bm25[self._candidate_key(c)] = br

        # Build the parallel arrays in CE order (the order of ``reranked``).
        ce_scores: list[float] = []
        bm25_aligned: list[Optional[int]] = []
        heading_aligned: list[float] = []
        anchor_aligned: list[float] = []
        for r in reranked:
            key = self._candidate_key(r)
            ce_scores.append(float(r.score))
            bm25_aligned.append(cand_key_to_bm25.get(key))
            heading_aligned.append(
                heading_match_score(query, (r.metadata or {}).get("heading_path") or [])
            )
            anchor_aligned.append(
                anchor_confidence(
                    (r.metadata or {}).get("_anchor_rank"),
                    anchor_k=RAG_RERANK_ANCHOR_K,
                )
            )

        fused = fuse_scores(
            ce_scores=ce_scores,
            bm25_ranks=bm25_aligned,
            heading_scores=heading_aligned,
            anchor_confs=anchor_aligned,
            weights={
                "rrf": RAG_RERANK_RRF_LAMBDA,
                "heading": RAG_RERANK_HEADING_LAMBDA,
                "anchor": RAG_RERANK_ANCHOR_LAMBDA,
            },
            k_rrf=RAG_RERANK_RRF_K,
        )

        # Apply fused scores back onto the existing RankedResults and resort.
        rescored = [
            RankedResult(text=r.text, score=fused[i], metadata=r.metadata)
            for i, r in enumerate(reranked)
        ]
        rescored.sort(key=lambda r: r.score, reverse=True)
        return rescored[:rerank_top_k]
    # ------------------------------------------------------------------
    # Deep-research branch helpers
    # ------------------------------------------------------------------

    def _run_deep_research(
        self,
        *,
        original_question: str,
        processed_query: str,
        ignored_doc_ids: Optional[list[str]],
        source_filter: Optional[str],
        heading_filter: Optional[str],
        tenant_id: Optional[str],
        alpha: float,
    ) -> DeepResearchResult:
        """Run the recursive topic-grouped retrieval loop synchronously
        (one ``asyncio.run`` island contained inside the activity thread)."""
        import asyncio as _asyncio

        # Build SearchFilter list once — reused for every kb_retrieve call.
        filters: list[SearchFilter] = []
        if ignored_doc_ids:
            filters.append(
                SearchFilter(
                    property="document_id",
                    operator="not_in",
                    value=list(ignored_doc_ids),
                )
            )
        if source_filter:
            filters.append(SearchFilter(property="source", operator="eq", value=source_filter))
        if heading_filter:
            filters.append(SearchFilter(property="heading", operator="eq", value=heading_filter))
        if tenant_id and tenant_id != "default":
            filters.append(SearchFilter(property="tenant_id", operator="eq", value=tenant_id))
        filters_arg = filters or None

        # Capture self in the closures via local refs so we don't hold method
        # bindings across the async boundary.
        embeddings = self.embeddings
        retry_provider = self.retry_provider
        retry_policy = self.retry_policy
        do_search = self._do_search
        kg_expander = self._kg_expander
        kb_top_per_node = max(1, RAG_DEEP_RESEARCH_KB_TOP_PER_NODE)

        async def kb_retrieve(query: str):
            def _blocking():
                emb = embeddings.embed_query(query)
                # Mirror linear-path BM25 fusion: KG-expanded terms aren't
                # available per sub-query inside the tree; the KG snippet
                # comes back via kg_retrieve and feeds graph_context only.
                results = retry_provider.execute(
                    operation_name="weaviate_hybrid_search_dr",
                    fn=lambda: do_search(query, emb, alpha, kb_top_per_node, filters_arg),
                    policy=retry_policy,
                    idempotency_key=(
                        f"dr_search:{query}:{source_filter}:{heading_filter}:{kb_top_per_node}"
                    ),
                )
                return list(results or [])

            return await _asyncio.to_thread(_blocking)

        async def kg_retrieve(query: str):
            if kg_expander is None:
                return None

            def _blocking():
                er = kg_expander.expand(query, depth=1)
                terms = list(getattr(er, "terms", []) or [])
                gc = getattr(er, "graph_context", "") or ""
                return KGSnippet(terms=terms, graph_context=gc)

            try:
                return await _asyncio.to_thread(_blocking)
            except Exception as exc:  # noqa: BLE001
                logger.debug("dr kg_retrieve failed for query=%r: %s", query, exc)
                return None

        budget = DeepResearchBudget(
            max_nodes=RAG_DEEP_RESEARCH_MAX_NODES,
            max_llm_calls=RAG_DEEP_RESEARCH_MAX_LLM_CALLS,
            wall_clock_ms=RAG_STAGE_BUDGET_DEEP_RESEARCH_MS,
            max_depth=RAG_DEEP_RESEARCH_MAX_DEPTH,
            max_topics=RAG_DEEP_RESEARCH_MAX_TOPICS,
            per_topic_questions=RAG_DEEP_RESEARCH_PER_TOPIC_QUESTIONS,
            graph_context_max_chars=RAG_DEEP_RESEARCH_GRAPH_CONTEXT_MAX_CHARS,
        )

        orchestrator = DeepResearch(
            provider=get_llm_provider(),
            kb_retrieve=kb_retrieve,
            kg_retrieve=kg_retrieve if kg_expander is not None else None,
            original_question=original_question,
            processed_query=processed_query,
            budget=budget,
            reranker=self.reranker,
        )

        return _asyncio.run(orchestrator.research())

    def _rerank_deep_research(
        self,
        dr_result: DeepResearchResult,
        processed_query: str,
        rerank_top_k: int,
    ) -> list[RankedResult]:
        """Convert deep-research topic pools into a final ``list[RankedResult]``.

        Routing rule:

        - Single topic (``is_unified=True``): rerank the merged pool against
          ``processed_query`` (Option B).
        - Multiple topics: rerank each topic's pool against its own
          ``rerank_anchor`` (the topic name / first sub-question), take a
          per-topic share of ``rerank_top_k``, dedupe, concatenate.
        """
        pools = dr_result.topic_pools
        if not pools:
            return []

        # If the orchestrator already applied per-topic rerank with the
        # confidence gate, consume its round-robin merge directly (capped to
        # rerank_top_k). This is the fast path for confident multi-topic splits.
        if dr_result.per_topic_rerank_applied and dr_result.per_topic_rerank_merged is not None:
            return list(dr_result.per_topic_rerank_merged)[:rerank_top_k]

        if dr_result.is_unified or len(pools) == 1:
            pool = pools[0]
            chunks = list(pool.chunks_by_id.values())
            if not chunks:
                return []
            return self.reranker.rerank(
                query=processed_query,
                documents=chunks,
                top_k=rerank_top_k,
            )

        # Hybrid: per-topic rerank.
        per_topic_k = max(1, rerank_top_k // len(pools))
        seen_ids: set[str] = set()
        merged: list[RankedResult] = []
        for pool in pools:
            chunks = list(pool.chunks_by_id.values())
            if not chunks:
                continue
            ranked = self.reranker.rerank(
                query=pool.rerank_anchor or processed_query,
                documents=chunks,
                top_k=per_topic_k,
            )
            for r in ranked:
                # Stable id for dedup across topic pools.
                md = r.metadata or {}
                key = (
                    md.get("object_id")
                    or md.get("chunk_id")
                    or f"{md.get('source','?')}|{md.get('heading','')}|{md.get('chunk_index','')}"
                )
                if key in seen_ids:
                    continue
                seen_ids.add(key)
                merged.append(r)
        return merged

    @staticmethod
    def _ranked_from_search_results(search_results, top_k: int) -> list[RankedResult]:
        """Convert search results into a sorted RankedResult list."""
        _t0 = time.perf_counter()
        ranked = [
            RankedResult(text=r.text, score=r.score, metadata=r.metadata)
            for r in search_results
        ]
        ranked.sort(key=lambda result: result.score, reverse=True)
        top = ranked[:top_k]
        logger.debug(
            "_ranked_from_search_results: kept %d of %d hits in %.2fms",
            len(top), len(ranked), (time.perf_counter() - _t0) * 1000,
        )
        return top

    def run(
        self,
        query: str,
        alpha: float = HYBRID_SEARCH_ALPHA,
        search_limit: int = SEARCH_LIMIT,
        rerank_top_k: int = RERANK_TOP_K,
        source_filter: Optional[str] = None,
        heading_filter: Optional[str] = None,
        skip_generation: bool = False,
        tenant_id: Optional[str] = None,
        max_query_iterations: int = MAX_SANITIZATION_ITERATIONS,
        fast_path: Optional[bool] = None,
        overall_timeout_ms: int = RAG_RETRIEVAL_TIMEOUT_MS,
        stage_budget_overrides: Optional[dict[str, int]] = None,
        memory_context: Optional[str] = None,
        memory_recent_turns: Optional[list[dict[str, str]]] = None,
        conversation_id: Optional[str] = None,
        retry_count: int = 0,
        ignored_doc_ids: Optional[list[str]] = None,
        mode: str = "query",
        retrieval_sub_mode: str = "auto",
        extra_processing: bool = False,
        tree_retrieval: Optional[bool] = None,
        deep_research: bool = False,
    ) -> RAGResponse:
        """Execute the full RAG pipeline.

        Args:
            query: Raw user query.
            alpha: Hybrid search balance (0=BM25, 1=vector).
            search_limit: Number of results from hybrid search.
            rerank_top_k: Number of top results after reranking.
            source_filter: Optional source filename to filter results by.
            heading_filter: Optional section heading to filter results by.
            skip_generation: If True, skip LLM generation (stages 1-5 only).
                Useful when the caller will stream generation separately.
            tenant_id: Optional tenant identifier for multi-tenant deployments.
            max_query_iterations: Max LLM reformulation attempts before
                asking the user for clarification.
            fast_path: Skip LLM query reformulation when True. Defaults to
                ``RAG_DEFAULT_FAST_PATH`` when None.
            overall_timeout_ms: Wall-clock budget for the full pipeline in ms.
            stage_budget_overrides: Per-stage budget overrides in ms.
            memory_context: Summarised conversation history to prepend.
            memory_recent_turns: Recent turn list for multi-turn context.
            conversation_id: Opaque ID propagated into the response for
                session tracking.
            retry_count: Number of re-retrieval attempts already performed
                by the caller. Used by confidence routing to decide whether
                another RE_RETRIEVE cycle is available.

        Returns:
            RAGResponse with results or clarification message.
        """
        stage_budget_overrides = stage_budget_overrides or {}
        stage_budgets = {
            "query_processing": int(stage_budget_overrides.get("query_processing", RAG_STAGE_BUDGET_QUERY_PROCESSING_MS)),
            "kg_expansion": int(stage_budget_overrides.get("kg_expansion", RAG_STAGE_BUDGET_KG_EXPANSION_MS)),
            "embedding": int(stage_budget_overrides.get("embedding", RAG_STAGE_BUDGET_EMBEDDING_MS)),
            "hybrid_search": int(stage_budget_overrides.get("hybrid_search", RAG_STAGE_BUDGET_HYBRID_SEARCH_MS)),
            "reranking": int(stage_budget_overrides.get("reranking", RAG_STAGE_BUDGET_RERANKING_MS)),
            "generation": int(stage_budget_overrides.get("generation", RAG_STAGE_BUDGET_GENERATION_MS)),
            "visual_retrieval": int(stage_budget_overrides.get("visual_retrieval", RAG_STAGE_BUDGET_VISUAL_RETRIEVAL_MS)),
            "deep_research": int(stage_budget_overrides.get("deep_research", RAG_STAGE_BUDGET_DEEP_RESEARCH_MS)),
        }
        tp = TimingPool(
            overall_budget_ms=float(overall_timeout_ms),
            stage_budgets={k: float(v) for k, v in stage_budgets.items()},
        )
        pipeline_start = tp.pipeline_start  # for final span attribute

        def _budget_clarification(stage: str) -> str:
            return (
                "This request reached the retrieval timeout budget during "
                f"{stage}. Please narrow the query or try again."
            )

        # Per-stage outcomes ledger threaded through to ``decide_terminal``.
        # Stages append to this as they run (or are skipped). Bail-out sites
        # consult ``decide_terminal`` rather than authoring action strings.
        outcomes: list[StageOutcome] = []

        def _decision_to_action_fields(d: TerminalDecision) -> dict:
            """Translate a TerminalDecision into RAGResponse kwargs.

            Centralises enum -> string serialisation so the RAGResponse
            wire shape stays string-based (back-compat with the
            response.action equality check on the consumer side).
            """
            return {
                "action": d.action,
                "ask_user_reason": (
                    d.ask_user_reason.value if d.ask_user_reason is not None else None
                ),
                "degraded": d.degraded,
            }

        with self.tracer.span("rag_chain.run", {"query_length": len(query)}) as root_span:
            alpha = validate_alpha(alpha)
            search_limit = validate_positive_int("search_limit", search_limit)
            rerank_top_k = validate_positive_int("rerank_top_k", rerank_top_k)
            source_filter = validate_filter_value("source_filter", source_filter)
            heading_filter = validate_filter_value("heading_filter", heading_filter)
            # Retrieval mode never generates an answer regardless of caller intent.
            if mode == "retrieval":
                skip_generation = True

            # Stage 1: Query processing (+ input rails in parallel if NeMo enabled)
            t0 = time.perf_counter()
            with self.tracer.span("rag_chain.process_query", parent=root_span):
                processing_query = query
                if memory_context:
                    processing_query = (
                        "Conversation context:\n"
                        f"{memory_context}\n\n"
                        "Current user question:\n"
                        f"{query}"
                    )

                # Run query processing and input rails in parallel (REQ-702)
                # PII gate: if PII detection is enabled, scan the query FIRST
                # before sending it to the LLM for reformulation. This prevents
                # PII from being sent to the LLM in the parallel execution window.
                guardrails_metadata = None
                input_rail_result = None
                merge_decision = None
                pii_gated_query = processing_query

                if GUARDRAIL_BACKEND:
                    from concurrent.futures import ThreadPoolExecutor as _TP, Future as _Fut

                    # PII gate: run PII detection synchronously before parallel stage
                    with self.tracer.span("rag_chain.pii_gate", parent=root_span):
                        try:
                            redacted_text, pii_detections = redact_pii(query)
                            if pii_detections:
                                # Use redacted query for LLM processing
                                pii_gated_query = redacted_text
                                if memory_context:
                                    pii_gated_query = (
                                        "Conversation context:\n"
                                        f"{memory_context}\n\n"
                                        "Current user question:\n"
                                        f"{redacted_text}"
                                    )
                                logger.info(
                                    "PII gate: %d detections redacted before LLM processing",
                                    len(pii_detections),
                                )
                        except Exception as e:
                            logger.warning("PII gate failed: %s — continuing with original query", e)
                            pii_gated_query = processing_query

                    with _TP(max_workers=RAG_RETRIEVAL_STAGE1_POOL_MAX_WORKERS, thread_name_prefix="stage1") as stage1_pool:
                        qp_future: _Fut = stage1_pool.submit(
                            process_query,
                            pii_gated_query,
                            QUERY_CONFIDENCE_THRESHOLD,
                            max_query_iterations,
                            RAG_DEFAULT_FAST_PATH if fast_path is None else bool(fast_path),
                            memory_context,
                            query,
                            mode,
                            retrieval_sub_mode,
                        )
                        rail_future: _Fut = stage1_pool.submit(
                            run_input_rails,
                            query,
                            tenant_id or "",
                        )

                        query_result = qp_future.result()
                        input_rail_result = rail_future.result()
                else:
                    query_result: QueryResult = process_query(
                        processing_query,
                        max_iterations=max_query_iterations,
                        fast_path=RAG_DEFAULT_FAST_PATH if fast_path is None else bool(fast_path),
                        memory_context=memory_context,
                        user_query=query,
                        mode=mode,
                        retrieval_sub_mode=retrieval_sub_mode,
                    )
            tp.record("query_processing", "retrieval", started_at=t0)
            if tp.check_stage_budget("query_processing"):
                tp.mark_budget_exhausted("query_processing")

            # Apply merge gate if input rails ran (REQ-707)
            if input_rail_result is not None and self._guardrails_merge_gate is not None:
                merge_decision = self._guardrails_merge_gate.merge(
                    query_result, input_rail_result
                )

                # Record per-rail timings into the timing pool
                for r in input_rail_result.rail_executions:
                    tp.record(f"input_rail_{r.rail_name}", "guardrails", ms=r.execution_ms)

                guardrails_metadata = {
                    "enabled": True,
                    "input_rails": [
                        {
                            "rail_name": r.rail_name,
                            "verdict": r.verdict.value,
                            "execution_ms": r.execution_ms,
                            "details": r.details,
                        }
                        for r in input_rail_result.rail_executions
                    ],
                    "intent": input_rail_result.intent,
                    "intent_confidence": input_rail_result.intent_confidence,
                    "total_rail_ms": sum(
                        r.execution_ms for r in input_rail_result.rail_executions
                    ),
                }

                # Handle reject/canned responses from merge gate.
                # Routed through ``decide_terminal`` so the action choice
                # (and ask_user_reason on reject) is single-sourced.
                if merge_decision["action"] in ("reject", "canned"):
                    # Detect injection vs empty-text by looking at input-rail
                    # verdicts; sanitiser empty-rejects produce no rail error.
                    if merge_decision["action"] == "reject":
                        injection_detected = any(
                            r.verdict.value not in ("allow", "pass")
                            for r in (input_rail_result.rail_executions if input_rail_result else [])
                        )
                        if injection_detected:
                            outcomes.append(
                                StageOutcome("input_rails", StageStatus.ERROR, "injection")
                            )
                    decision = decide_terminal(
                        outcomes=outcomes,
                        deep_research=bool(deep_research),
                        has_results=False,
                        sanitizer_verdict=merge_decision["action"],
                    )
                    return RAGResponse(
                        query=query,
                        processed_query=query_result.processed_query,
                        query_confidence=query_result.confidence,
                        clarification_message=merge_decision["message"],
                        stage_timings=tp.entries(),
                        timing_totals=tp.totals(),
                        budget_exhausted=tp.budget_exhausted,
                        budget_exhausted_stage=tp.budget_exhausted_stage,
                        conversation_id=conversation_id,
                        guardrails=guardrails_metadata,
                        **_decision_to_action_fields(decision),
                    )

            # Bypass the LLM-confidence-loop ask_user gate when the caller has
            # explicitly opted into deep_research. Rationale: the user has
            # already accepted DR's latency cost, and DR's recursive decomposer
            # is designed to recover from messy/terse queries. Without this
            # bypass, the LLM-confidence loop's "this seems too vague" verdict
            # (or its exhausted-with-confidence=0.0 path) refuses to retrieve
            # even though DR could have answered.
            #
            # Hard sanitizer rejections (empty/injection_detected) author
            # specific clarification strings. We still honour those by NOT
            # bypassing when the message matches one of the sanitizer's
            # signature strings — the sanitizer fires before the LangGraph loop
            # and represents a request-level reject (security/empty), not a
            # retrieval-quality concern.
            _SANITIZER_REJECT_MESSAGES = (
                "Your query appears to be empty.",
                "Your query could not be processed.",
            )
            _is_sanitizer_reject = (
                query_result.clarification_message is not None
                and any(
                    s in query_result.clarification_message
                    for s in _SANITIZER_REJECT_MESSAGES
                )
            )
            _dr_bypass_ask_user = bool(deep_research) and not _is_sanitizer_reject
            # The LLM-confidence loop's "vague query" verdict is bypassed
            # under DR (DR's recursive decomposer recovers from vague
            # queries on its own); ``decide_terminal`` enforces that policy.
            if query_result.action == QueryAction.ASK_USER:
                # Hard sanitizer rejects (empty/injection) can also surface
                # via QueryAction.ASK_USER; keep their stronger reason.
                sanitizer_verdict = "reject" if _is_sanitizer_reject else None
                decision = decide_terminal(
                    outcomes=outcomes,
                    deep_research=bool(deep_research),
                    has_results=False,
                    sanitizer_verdict=sanitizer_verdict,
                    confidence_ask_user=not _is_sanitizer_reject,
                )
                # Under DR the decision may come back as "answer"; only return
                # early when the centralized brain agrees we should give up.
                if decision.action != "answer":
                    return RAGResponse(
                        query=query,
                        processed_query=query_result.processed_query,
                        query_confidence=query_result.confidence,
                        clarification_message=query_result.clarification_message,
                        stage_timings=tp.entries(),
                        timing_totals=tp.totals(),
                        budget_exhausted=tp.budget_exhausted,
                        budget_exhausted_stage=tp.budget_exhausted_stage,
                        conversation_id=conversation_id,
                        guardrails=guardrails_metadata,
                        **_decision_to_action_fields(decision),
                    )
            if tp.budget_exhausted:
                outcomes.append(
                    StageOutcome("query_processing", StageStatus.BUDGET_EXHAUSTED)
                )
                decision = decide_terminal(
                    outcomes=outcomes,
                    deep_research=bool(deep_research),
                    has_results=False,
                )
                if decision.action != "answer":
                    tp.log_summary()
                    return RAGResponse(
                        query=query,
                        processed_query=query_result.processed_query,
                        query_confidence=query_result.confidence,
                        clarification_message=_budget_clarification("query processing"),
                        stage_timings=tp.entries(),
                        timing_totals=tp.totals(),
                        budget_exhausted=True,
                        budget_exhausted_stage=tp.budget_exhausted_stage,
                        conversation_id=conversation_id,
                        guardrails=guardrails_metadata,
                        **_decision_to_action_fields(decision),
                    )

            # Use PII-redacted query if available from merge gate
            processed_query = (
                merge_decision.get("query", query_result.processed_query)
                if merge_decision
                else query_result.processed_query
            )

            # REQ-1205: Suppress-memory routing — context reset detected
            if query_result.suppress_memory:
                # Use standalone_query for retrieval, strip all memory
                processed_query = query_result.standalone_query
                memory_context = None
                memory_recent_turns = None

            # Hoisted state used by stages 2–5 OR by the deep-research branch.
            # When deep_research=True, the recursive loop populates `reranked`
            # + `graph_context` + `kg_expanded_terms` directly and stages 2–5
            # are skipped via the `_dr_active` guard.
            kg_expanded_terms: list[str] = []
            graph_context: str = ""
            bm25_query: str = processed_query
            query_embedding = None
            search_results: list = []
            reranked: list[RankedResult] = []
            dr_result: Optional[DeepResearchResult] = None
            _dr_active = bool(deep_research)

            if _dr_active:
                t0 = time.perf_counter()
                with self.tracer.span("rag_chain.deep_research", parent=root_span) as dr_span:
                    try:
                        dr_result = self._run_deep_research(
                            original_question=query,
                            processed_query=processed_query,
                            ignored_doc_ids=ignored_doc_ids,
                            source_filter=source_filter,
                            heading_filter=heading_filter,
                            tenant_id=tenant_id,
                            alpha=alpha,
                        )
                        reranked = self._rerank_deep_research(
                            dr_result, processed_query, rerank_top_k,
                        )
                        graph_context = dr_result.graph_context or ""
                        dr_span.set_attribute("dr_node_count", dr_result.node_count)
                        dr_span.set_attribute("dr_llm_calls", dr_result.llm_call_count)
                        dr_span.set_attribute("dr_topic_count", len(dr_result.topic_pools))
                        dr_span.set_attribute("dr_is_unified", dr_result.is_unified)
                        dr_span.set_attribute("dr_budget_exhausted", dr_result.budget_exhausted)
                        if dr_result.budget_exhausted_reason:
                            dr_span.set_attribute(
                                "dr_budget_reason", dr_result.budget_exhausted_reason
                            )
                    except Exception as exc:  # noqa: BLE001
                        logger.exception("deep_research branch failed: %s", exc)
                        dr_span.set_attribute("dr_error", str(exc))
                        # Fall back to empty pool — generation will see no context
                        # and emit the standard "no evidence" response.
                        reranked = []
                        graph_context = ""
                        dr_result = None
                tp.record("deep_research", "retrieval", started_at=t0)
                if tp.check_stage_budget("deep_research"):
                    tp.mark_budget_exhausted("deep_research")

            # Stage 2: KG expansion
            t0 = time.perf_counter()
            with self.tracer.span("rag_chain.kg_expand", parent=root_span) as kg_span:
                if not _dr_active and self._kg_expander:
                    expansion_result = self._kg_expander.expand(processed_query, depth=1)
                    kg_expanded_terms = (
                        expansion_result.terms
                        if hasattr(expansion_result, "terms")
                        else list(expansion_result)
                    )
                    graph_context = getattr(expansion_result, "graph_context", "")
                kg_span.set_attribute("kg_expanded_terms_count", len(kg_expanded_terms))
                kg_span.set_attribute("kg_graph_context_len", len(graph_context))
            tp.record("kg_expansion", "retrieval", started_at=t0)
            if not _dr_active and tp.check_stage_budget("kg_expansion"):
                tp.mark_budget_exhausted("kg_expansion")
            # When deep research is active it already consumed the retrieval
            # budget by design; stages 2-5 are skipped, so don't let their
            # passive budget checks trip the chain to ask_user.
            if tp.budget_exhausted and not _dr_active:
                outcomes.append(
                    StageOutcome("kg_expansion", StageStatus.BUDGET_EXHAUSTED)
                )
                decision = decide_terminal(
                    outcomes=outcomes,
                    deep_research=bool(deep_research),
                    has_results=False,
                )
                if decision.action != "answer":
                    tp.log_summary()
                    return RAGResponse(
                        query=query,
                        processed_query=processed_query,
                        query_confidence=query_result.confidence,
                        clarification_message=_budget_clarification("KG expansion"),
                        kg_expanded_terms=kg_expanded_terms or None,
                        stage_timings=tp.entries(),
                        timing_totals=tp.totals(),
                        budget_exhausted=True,
                        budget_exhausted_stage=tp.budget_exhausted_stage,
                        conversation_id=conversation_id,
                        **_decision_to_action_fields(decision),
                    )

            if not _dr_active:
                if kg_expanded_terms:
                    bm25_query = processed_query + " " + " ".join(kg_expanded_terms[:3])
                else:
                    bm25_query = processed_query

            # Stage 3: Query embedding (with LRU cache for exact repeats)
            t0 = time.perf_counter()
            with self.tracer.span("rag_chain.embed_query", parent=root_span) as embed_span:
                if not _dr_active:
                    cache_hit = processed_query in self._embedding_cache
                    if cache_hit:
                        query_embedding = self._embedding_cache[processed_query]
                        self._embedding_cache.move_to_end(processed_query)
                        embed_span.set_attribute("cache_hit", True)
                    else:
                        query_embedding = self.embeddings.embed_query(processed_query)
                        self._embedding_cache[processed_query] = query_embedding
                        if len(self._embedding_cache) > self._embedding_cache_max:
                            self._embedding_cache.popitem(last=False)
                        embed_span.set_attribute("cache_hit", False)
                else:
                    embed_span.set_attribute("skipped_for_deep_research", True)
            tp.record("embedding", "retrieval", started_at=t0)
            if not _dr_active and tp.check_stage_budget("embedding"):
                tp.mark_budget_exhausted("embedding")
            if tp.budget_exhausted and not _dr_active:
                outcomes.append(
                    StageOutcome("embedding", StageStatus.BUDGET_EXHAUSTED)
                )
                decision = decide_terminal(
                    outcomes=outcomes,
                    deep_research=bool(deep_research),
                    has_results=False,
                )
                if decision.action != "answer":
                    tp.log_summary()
                    return RAGResponse(
                        query=query,
                        processed_query=processed_query,
                        query_confidence=query_result.confidence,
                        clarification_message=_budget_clarification("embedding"),
                        kg_expanded_terms=kg_expanded_terms or None,
                        stage_timings=tp.entries(),
                        timing_totals=tp.totals(),
                        budget_exhausted=True,
                        budget_exhausted_stage=tp.budget_exhausted_stage,
                        conversation_id=conversation_id,
                        **_decision_to_action_fields(decision),
                    )

            # Stage 4: Hybrid search
            t0 = time.perf_counter()
            filters = []
            if not _dr_active:
                if ignored_doc_ids:
                    filters.append(
                        SearchFilter(
                            property="document_id",
                            operator="not_in",
                            value=list(ignored_doc_ids),
                        )
                    )
                if source_filter:
                    filters.append(SearchFilter(property="source", operator="eq", value=source_filter))
                if heading_filter:
                    filters.append(SearchFilter(property="heading", operator="eq", value=heading_filter))
                if tenant_id and tenant_id != "default":
                    filters.append(SearchFilter(property="tenant_id", operator="eq", value=tenant_id))

            # Per-request override beats config default (P4 §6).
            tree_enabled_effective = (
                RAG_TREE_RETRIEVAL_ENABLED
                if tree_retrieval is None
                else bool(tree_retrieval)
            )
            with self.tracer.span("rag_chain.hybrid_search", parent=root_span) as search_span:
                if not _dr_active:
                    search_results = self.retry_provider.execute(
                        operation_name="weaviate_hybrid_search",
                        fn=lambda: self._collect_candidates(
                            bm25_query=bm25_query,
                            query_embedding=query_embedding,
                            alpha=alpha,
                            search_limit=search_limit,
                            base_filters=filters,
                            tree_enabled=tree_enabled_effective,
                            schema_present=RAG_TREE_SCHEMA_PRESENT,
                            descent_top_k=RAG_TREE_DESCENT_TOP_K,
                            leaves_per_section=RAG_TREE_DESCENT_LEAVES_PER_SECTION,
                            lift_seed_k=RAG_TREE_LIFT_SEED_K,
                            siblings_per_group=RAG_TREE_LIFT_SIBLINGS,
                            doc_diversity_top_per_doc=RAG_TREE_DESCENT_DOC_DIVERSITY_TOP_PER_DOC,
                        ),
                        policy=self.retry_policy,
                        idempotency_key=f"search:{processed_query}:{source_filter}:{heading_filter}:{search_limit}:tree={tree_enabled_effective}",
                    )
                    search_span.set_attribute("search_result_count", len(search_results))
                    search_span.set_attribute("tree_retrieval_enabled", tree_enabled_effective)
                    search_span.set_attribute(
                        "tree_retrieval_source",
                        "request" if tree_retrieval is not None else "config",
                    )
                else:
                    search_span.set_attribute("skipped_for_deep_research", True)
            tp.record("hybrid_search", "retrieval", started_at=t0)
            if not _dr_active and tp.check_stage_budget("hybrid_search"):
                tp.mark_budget_exhausted("hybrid_search")
            if tp.budget_exhausted and not _dr_active:
                outcomes.append(
                    StageOutcome("hybrid_search", StageStatus.BUDGET_EXHAUSTED)
                )
                partial = self._ranked_from_search_results(search_results, rerank_top_k)
                decision = decide_terminal(
                    outcomes=outcomes,
                    deep_research=bool(deep_research),
                    has_results=bool(partial),
                )
                tp.log_summary()
                # Partial results -> action=search; no results -> ask_user.
                clarif = (
                    None
                    if decision.action == "search"
                    else _budget_clarification("hybrid search")
                )
                return RAGResponse(
                    query=query,
                    processed_query=processed_query,
                    query_confidence=query_result.confidence,
                    results=partial,
                    clarification_message=clarif,
                    kg_expanded_terms=kg_expanded_terms or None,
                    stage_timings=tp.entries(),
                    timing_totals=tp.totals(),
                    budget_exhausted=True,
                    budget_exhausted_stage=tp.budget_exhausted_stage,
                    conversation_id=conversation_id,
                    **_decision_to_action_fields(decision),
                )

            # Degraded fallback: if hybrid_search returned 0 results AND we're
            # not on the DR path, retry once with BM25-only (alpha=1.0) and
            # looser top-k. Only if THAT also returns 0 do we route to
            # ``decide_terminal`` for the NO_RESULTS ask_user.
            degraded_attempted = False
            if not _dr_active and not search_results:
                degraded_attempted = True
                try:
                    degraded_results = self._do_search(
                        bm25_query,
                        query_embedding,
                        1.0,
                        max(search_limit * 2, search_limit),
                        filters or None,
                    )
                    if degraded_results:
                        search_results = degraded_results
                        outcomes.append(
                            StageOutcome("hybrid_search_degraded", StageStatus.OK)
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("degraded BM25-only fallback failed: %s", exc)

            if not _dr_active and not search_results:
                outcomes.append(StageOutcome("hybrid_search", StageStatus.EMPTY))
                decision = decide_terminal(
                    outcomes=outcomes,
                    deep_research=bool(deep_research),
                    has_results=False,
                    degraded=degraded_attempted,
                )
                tp.log_summary()
                clarif = (
                    "Your query did not match any documents in the corpus. "
                    "Please rephrase or broaden the query."
                    if decision.ask_user_reason is not None
                    else None
                )
                return RAGResponse(
                    query=query,
                    processed_query=processed_query,
                    query_confidence=query_result.confidence,
                    results=[],
                    clarification_message=clarif,
                    kg_expanded_terms=kg_expanded_terms or None,
                    stage_timings=tp.entries(),
                    timing_totals=tp.totals(),
                    budget_exhausted=tp.budget_exhausted,
                    budget_exhausted_stage=tp.budget_exhausted_stage,
                    conversation_id=conversation_id,
                    **_decision_to_action_fields(decision),
                )

            # Stage 5: Reranking (cross-encoder + fusion R1)
            t0 = time.perf_counter()
            with self.tracer.span("rag_chain.rerank", parent=root_span) as rerank_span:
                if not _dr_active:
                    # 5a. BM25-only rank lookup, used by the RRF fusion component.
                    #     Run in parallel-style alongside CE scoring; if fusion is
                    #     disabled at config level, skip the extra round-trip.
                    bm25_ranks: list[Optional[int]] = []
                    if RAG_RERANK_FUSION_ENABLED:
                        bm25_ranks = self._collect_bm25_ranks(
                            bm25_query=bm25_query,
                            query_embedding=query_embedding,
                            search_limit=search_limit,
                            base_filters=filters,
                            candidates=list(search_results),
                        )

                    # 5b. Cross-encoder rerank — the heavy lifter.
                    reranked = self.reranker.rerank(
                        query=processed_query,
                        documents=search_results,
                        top_k=max(rerank_top_k, len(search_results)) if RAG_RERANK_FUSION_ENABLED else rerank_top_k,
                    )

                    # 5c. Apply fusion (BM25-RRF + heading + anchor) on top of CE.
                    if RAG_RERANK_FUSION_ENABLED:
                        reranked = self._apply_rerank_fusion(
                            query=processed_query,
                            candidates=list(search_results),
                            reranked=reranked,
                            bm25_ranks=bm25_ranks,
                            rerank_top_k=rerank_top_k,
                        )
                        rerank_span.set_attribute("fusion_enabled", True)

                scores = [r.score for r in reranked]
                if scores:
                    rerank_span.set_attribute("rerank_score_min", min(scores))
                    rerank_span.set_attribute("rerank_score_max", max(scores))
                    rerank_span.set_attribute("rerank_score_mean", statistics.mean(scores))
                if _dr_active:
                    rerank_span.set_attribute("from_deep_research", True)
            tp.record("reranking", "retrieval", started_at=t0)
            if not _dr_active and tp.check_stage_budget("reranking"):
                tp.mark_budget_exhausted("reranking")
            if tp.budget_exhausted and not _dr_active:
                outcomes.append(
                    StageOutcome("reranking", StageStatus.BUDGET_EXHAUSTED)
                )
                decision = decide_terminal(
                    outcomes=outcomes,
                    deep_research=bool(deep_research),
                    has_results=bool(reranked),
                )
                tp.log_summary()
                clarif = (
                    None
                    if decision.action == "search"
                    else _budget_clarification("reranking")
                )
                return RAGResponse(
                    query=query,
                    processed_query=processed_query,
                    query_confidence=query_result.confidence,
                    results=reranked,
                    clarification_message=clarif,
                    kg_expanded_terms=kg_expanded_terms or None,
                    stage_timings=tp.entries(),
                    timing_totals=tp.totals(),
                    budget_exhausted=True,
                    budget_exhausted_stage=tp.budget_exhausted_stage,
                    conversation_id=conversation_id,
                    **_decision_to_action_fields(decision),
                )

            # Classify retrieval quality based on reranker scores (REQ-403)
            retrieval_quality = "insufficient"
            retrieval_quality_note = None
            if reranked:
                best_score = max(r.score for r in reranked)
                if best_score >= RAG_RETRIEVAL_QUALITY_STRONG_THRESHOLD:
                    retrieval_quality = "strong"
                elif best_score >= RAG_RETRIEVAL_QUALITY_MODERATE_THRESHOLD:
                    retrieval_quality = "moderate"
                elif best_score >= RAG_RETRIEVAL_QUALITY_WEAK_THRESHOLD:
                    retrieval_quality = "weak"
                    retrieval_quality_note = (
                        "Retrieved documents have limited relevance to your query. "
                        "The answer below is based on the best available evidence."
                    )
                else:
                    retrieval_quality = "insufficient"
                    retrieval_quality_note = (
                        "No highly relevant documents were found. The answer below "
                        "is generated from loosely related content and may not be reliable."
                    )

            # REQ-1201: Fallback retrieval on standalone_query when primary is weak.
            # Skipped under deep_research — the recursive loop already explored
            # alternative formulations exhaustively, so a single-shot fallback
            # adds no signal and burns budget.
            if (
                not _dr_active
                and retrieval_quality in ("weak", "insufficient")
                and not query_result.suppress_memory
                and query_result.standalone_query
                and query_result.standalone_query != processed_query
            ):
                t0 = time.perf_counter()
                with self.tracer.span("rag_chain.fallback_retrieval", parent=root_span) as fb_span:
                    # Embed standalone_query (check cache first)
                    fb_query = query_result.standalone_query
                    fb_cache_hit = fb_query in self._embedding_cache
                    if fb_cache_hit:
                        fb_embedding = self._embedding_cache[fb_query]
                        self._embedding_cache.move_to_end(fb_query)
                    else:
                        fb_embedding = self.embeddings.embed_query(fb_query)
                        self._embedding_cache[fb_query] = fb_embedding
                        if len(self._embedding_cache) > self._embedding_cache_max:
                            self._embedding_cache.popitem(last=False)

                    # Build BM25 query with KG terms
                    fb_bm25_query = (
                        fb_query + " " + " ".join(kg_expanded_terms[:3])
                        if kg_expanded_terms
                        else fb_query
                    )

                    # Hybrid search with same parameters
                    fb_search_results = self.retry_provider.execute(
                        operation_name="weaviate_hybrid_search_fallback",
                        fn=lambda: self._do_search(
                            fb_bm25_query, fb_embedding, alpha, search_limit, filters or None,
                        ),
                        policy=self.retry_policy,
                        idempotency_key=f"search_fb:{fb_query}:{source_filter}:{heading_filter}:{search_limit}",
                    )

                    # Rerank fallback results
                    if fb_search_results:
                        fb_reranked = self.reranker.rerank(
                            query=fb_query,
                            documents=fb_search_results,
                            top_k=rerank_top_k,
                        )

                        # Compare best reranker scores: primary vs fallback
                        primary_best = max(r.score for r in reranked) if reranked else 0.0
                        fallback_best = max(r.score for r in fb_reranked) if fb_reranked else 0.0

                        fb_span.set_attribute("primary_best_score", primary_best)
                        fb_span.set_attribute("fallback_best_score", fallback_best)
                        fb_span.set_attribute("fallback_wins", fallback_best > primary_best)

                        if fallback_best > primary_best:
                            logger.info(
                                "Fallback retrieval improved best score: %.3f -> %.3f",
                                primary_best, fallback_best,
                            )
                            reranked = fb_reranked
                            scores = [r.score for r in reranked]
                            # Re-classify retrieval quality with new scores
                            if fallback_best >= RAG_RETRIEVAL_QUALITY_STRONG_THRESHOLD:
                                retrieval_quality = "strong"
                                retrieval_quality_note = None
                            elif fallback_best >= RAG_RETRIEVAL_QUALITY_MODERATE_THRESHOLD:
                                retrieval_quality = "moderate"
                                retrieval_quality_note = None
                            elif fallback_best >= RAG_RETRIEVAL_QUALITY_WEAK_THRESHOLD:
                                retrieval_quality = "weak"
                                retrieval_quality_note = (
                                    "Retrieved documents have limited relevance to your query. "
                                    "The answer below is based on the best available evidence."
                                )
                tp.record("fallback_retrieval", "retrieval", started_at=t0)

            # Stage 5.4: Visual retrieval (FR-601, FR-615, NFR-905)
            visual_results = None  # None = disabled semantics
            if self._visual_retrieval_enabled:
                t0 = time.perf_counter()
                with self.tracer.span("rag_chain.visual_retrieval", parent=root_span) as vr_span:
                    try:
                        visual_results = self._run_visual_retrieval(
                            processed_query, tenant_id
                        )
                        vr_span.set_attribute("visual_result_count", len(visual_results))
                    except Exception as exc:
                        logger.warning(
                            "Visual retrieval failed (non-fatal): %s — continuing without visual results.",
                            exc,
                        )
                        visual_results = []  # enabled-but-failed semantics
                        vr_span.set_attribute("visual_error", str(exc))
                tp.record("visual_retrieval", "retrieval", started_at=t0)

            # Stage 5.5: Document formatting (REQ-501–REQ-503)
            version_conflicts = []
            formatted_context_str = None
            if RAG_DOCUMENT_FORMATTING_ENABLED and reranked:
                t0 = time.perf_counter()
                from src.retrieval.generation.nodes import format_context
                with self.tracer.span("rag_chain.format_context", parent=root_span) as fmt_span:
                    formatted = format_context(reranked)
                    formatted_context_str = formatted.context_string
                    version_conflicts = formatted.version_conflicts
                    fmt_span.set_attribute("chunk_count", formatted.chunk_count)
                    fmt_span.set_attribute("version_conflicts", len(version_conflicts))
                tp.record("document_formatting", "retrieval", started_at=t0)

            # Stage 6: Generation (skippable for streaming callers)
            generated_answer = None
            gen_result = None  # type: ignore[assignment]
            generation_source = None
            if not skip_generation and self._generator and not tp.budget_exhausted and (
                reranked or (query_result.has_backward_reference and (memory_context or memory_recent_turns))
            ):
                t0 = time.perf_counter()
                with self.tracer.span("rag_chain.generate", parent=root_span) as generate_span:
                    context_chunks = [r.text for r in reranked]
                    scores = [r.score for r in reranked]

                    # REQ-1203, REQ-1204, REQ-1205: Memory-aware generation routing
                    generation_source = None
                    if query_result.suppress_memory:
                        # REQ-1205: Context reset — no memory in generation
                        effective_turns = None
                        effective_memory = None
                        generation_source = "retrieval"
                    elif retrieval_quality in ("strong", "moderate"):
                        if query_result.has_backward_reference:
                            # REQ-1204: Hybrid — retrieval succeeded + backward ref
                            effective_turns = memory_recent_turns
                            effective_memory = memory_context
                            generation_source = "retrieval+memory"
                        else:
                            # Standard retrieval path
                            effective_turns = memory_recent_turns
                            effective_memory = memory_context
                            generation_source = (
                                "retrieval+memory"
                                if (memory_context or memory_recent_turns)
                                else "retrieval"
                            )
                    elif query_result.has_backward_reference and not memory_context and not memory_recent_turns:
                        # REQ-1203 guard: backward ref on fresh conversation — deterministic BLOCK
                        effective_turns = None
                        effective_memory = None
                        context_chunks = []
                        scores = []
                        formatted_context_str = None
                        reranked = []
                        generation_source = None
                        skip_generation = True
                        generated_answer = (
                            "Insufficient conversation history to answer this follow-up. "
                            "Please provide more context or start with a specific question."
                        )
                    elif query_result.has_backward_reference and (memory_context or memory_recent_turns):
                        # REQ-1203: Memory-generation path — both retrievals weak + backward ref + non-empty memory
                        effective_turns = memory_recent_turns
                        effective_memory = memory_context
                        generation_source = "memory"
                        # Clear document context — generate from memory only
                        context_chunks = []
                        scores = []
                        formatted_context_str = None
                        reranked = []  # No docs for memory-only generation
                    else:
                        # Weak retrieval, no backward ref — suppress recent_turns only (spec line 91)
                        # memory_context (rolling summary) is still passed per spec requirement
                        effective_turns = None
                        effective_memory = memory_context
                        generation_source = "retrieval"

                    # Use formatted context if document formatting is enabled
                    # (skip_generation may be set by fresh-convo guard above)
                    if not skip_generation:
                        if formatted_context_str:
                            gen_result = self._generator.generate(
                                query=processed_query,
                                context_chunks=[formatted_context_str],
                                scores=None,
                                memory_context=effective_memory,
                                recent_turns=effective_turns,
                                graph_context=graph_context,
                            )
                        else:
                            gen_result = self._generator.generate(
                                query=processed_query,
                                context_chunks=context_chunks,
                                scores=scores,
                                memory_context=effective_memory,
                                recent_turns=effective_turns,
                                graph_context=graph_context,
                            )
                        generated_answer = gen_result.answer or None
                        if gen_result.error is not None:
                            generate_span.set_attribute(
                                "generation_error_kind", gen_result.error.kind.value
                            )
                    generate_span.set_attribute("generated_answer_present", bool(generated_answer))
                tp.record("generation", "generation", started_at=t0)
                if tp.check_stage_budget("generation"):
                    tp.mark_budget_exhausted("generation")

            # Token budget snapshot (post-retrieval, post-generation)
            token_budget = None
            try:
                context_texts = [r.text for r in reranked]
                snapshot = calculate_budget(
                    system_prompt=get_system_prompt(),
                    memory_context=memory_context,
                    chunks=context_texts,
                    query=processed_query,
                    model=self._generator.model if self._generator else None,
                )
                # Enrich with actual token usage from the LLM response
                actual_resp = gen_result.raw_response if gen_result is not None else None
                if actual_resp and actual_resp.prompt_tokens:
                    snapshot = TokenBudgetSnapshot(
                        input_tokens=snapshot.input_tokens,
                        context_length=snapshot.context_length,
                        output_reservation=snapshot.output_reservation,
                        usage_percent=snapshot.usage_percent,
                        model_name=snapshot.model_name,
                        breakdown=snapshot.breakdown,
                        actual_prompt_tokens=actual_resp.prompt_tokens,
                        actual_completion_tokens=actual_resp.completion_tokens,
                        actual_total_tokens=actual_resp.total_tokens,
                        cost_usd=actual_resp.cost_usd,
                    )
                token_budget = snapshot
            except Exception as exc:
                logger.debug("Token budget calculation failed: %s", exc)

            # Stage 7: Output rails (REQ-703: parallel with consensus gate)
            if (
                generated_answer
                and GUARDRAIL_BACKEND
                and not tp.budget_exhausted
            ):
                t0 = time.perf_counter()
                with self.tracer.span("rag_chain.output_rails", parent=root_span):
                    context_chunks = [r.text for r in reranked]
                    output_rail_result = run_output_rails(
                        answer=generated_answer,
                        context_chunks=context_chunks,
                    )
                tp.record("output_rails", "guardrails", started_at=t0)

                # Record per-rail timings into the timing pool
                for r in output_rail_result.rail_executions:
                    tp.record(f"output_rail_{r.rail_name}", "guardrails", ms=r.execution_ms)

                # Apply output rail results
                generated_answer = output_rail_result.final_answer or generated_answer

                # Add output rail metadata to guardrails
                if guardrails_metadata is None:
                    guardrails_metadata = {"enabled": True, "total_rail_ms": 0.0}
                guardrails_metadata["output_rails"] = [
                    {
                        "rail_name": r.rail_name,
                        "verdict": r.verdict.value,
                        "execution_ms": r.execution_ms,
                        "details": r.details,
                    }
                    for r in output_rail_result.rail_executions
                ]
                guardrails_metadata["faithfulness_score"] = output_rail_result.faithfulness_score
                guardrails_metadata["faithfulness_warning"] = output_rail_result.faithfulness_warning
                guardrails_metadata["total_rail_ms"] = guardrails_metadata.get(
                    "total_rail_ms", 0.0
                ) + sum(
                    r.execution_ms for r in output_rail_result.rail_executions
                )

            # Stage 7.25: Output sanitization (REQ-704)
            if generated_answer:
                from src.retrieval.generation.nodes import sanitize_answer
                generated_answer = sanitize_answer(
                    generated_answer,
                    system_prompt=get_system_prompt(),
                )

            # Stage 7.5: Composite confidence scoring + routing (REQ-701, REQ-706)
            composite_confidence = None
            confidence_breakdown_dict = None
            post_guardrail_action = None
            verification_warning = None
            re_retrieval_suggested = False
            re_retrieval_params = None
            first_composite = None

            if (
                RAG_CONFIDENCE_ROUTING_ENABLED
                and generated_answer
                and (reranked or generation_source == "memory")
                and not tp.budget_exhausted
            ):
                t0 = time.perf_counter()
                with self.tracer.span("rag_chain.confidence_routing", parent=root_span) as conf_span:
                    from src.retrieval.generation.confidence import compute_composite_confidence
                    from src.retrieval.generation.confidence import route_by_confidence
                    from src.retrieval.generation.confidence import PostGuardrailAction

                    # #8: Memory path has no retrieval signal — reranker_scores=[] gives
                    # retrieval_score=0.0 by design.  Composite still provides meaningful
                    # signal via LLM self-report and citation marker presence/absence.
                    reranker_scores = [r.score for r in reranked]
                    llm_confidence_text = (
                        gen_result.confidence if gen_result is not None else "medium"
                    )
                    context_texts = [r.text for r in reranked]

                    breakdown = compute_composite_confidence(
                        reranker_scores=reranker_scores,
                        llm_confidence_text=llm_confidence_text,
                        answer=generated_answer,
                        retrieved_texts=context_texts,
                        retrieval_weight=RAG_CONFIDENCE_RETRIEVAL_WEIGHT,
                        llm_weight=RAG_CONFIDENCE_LLM_WEIGHT,
                        citation_weight=RAG_CONFIDENCE_CITATION_WEIGHT,
                    )
                    composite_confidence = breakdown.composite
                    confidence_breakdown_dict = {
                        "retrieval_score": breakdown.retrieval_score,
                        "llm_score": breakdown.llm_score,
                        "citation_score": breakdown.citation_score,
                        "composite": breakdown.composite,
                        "retrieval_weight": breakdown.retrieval_weight,
                        "llm_weight": breakdown.llm_weight,
                        "citation_weight": breakdown.citation_weight,
                    }

                    action = route_by_confidence(
                        composite=breakdown.composite,
                        retry_count=retry_count,
                        high_threshold=RAG_CONFIDENCE_HIGH_THRESHOLD,
                        low_threshold=RAG_CONFIDENCE_LOW_THRESHOLD,
                        max_retries=RAG_CONFIDENCE_RE_RETRIEVE_MAX_RETRIES,
                    )
                    post_guardrail_action = action.value

                    conf_span.set_attribute("composite_confidence", breakdown.composite)
                    conf_span.set_attribute("routing_action", action.value)
                    conf_span.set_attribute("retry_count", retry_count)
                tp.record("confidence_routing", "retrieval", started_at=t0)

                logger.info(
                    "Confidence routing: composite=%.2f action=%s retry=%d "
                    "(retrieval=%.2f llm=%.2f citation=%.2f)",
                    breakdown.composite,
                    action.value,
                    retry_count,
                    breakdown.retrieval_score,
                    breakdown.llm_score,
                    breakdown.citation_score,
                )

                # Act on routing decision.
                re_retrieval_suggested = False
                re_retrieval_params = None

                if action == PostGuardrailAction.RE_RETRIEVE and generation_source != "memory":
                    # Bounded internal re-retrieval loop (REQ-706, option B).
                    # Re-run search + rerank + generate + score with broader params.
                    # Reuses already-embedded query to avoid redundant model calls.
                    # Input-rails and PII gate are NOT re-run (query is already clean).
                    first_composite = breakdown.composite
                    retry_alpha = max(0.0, alpha - 0.15)
                    retry_search_limit = search_limit + 5
                    retry_bm25_query = bm25_query

                    with self.tracer.span("rag_chain.re_retrieval", parent=root_span) as rr_span:
                        rr_span.set_attribute("retry_alpha", retry_alpha)
                        rr_span.set_attribute("retry_search_limit", retry_search_limit)

                        retry_search_results = self.retry_provider.execute(
                            operation_name="weaviate_hybrid_search_re_retrieval",
                            fn=lambda: self._do_search(
                                retry_bm25_query,
                                query_embedding,
                                retry_alpha,
                                retry_search_limit,
                                filters or None,
                            ),
                            policy=self.retry_policy,
                            idempotency_key=(
                                f"re_retrieve:{processed_query}:{source_filter}:"
                                f"{heading_filter}:{retry_search_limit}"
                            ),
                        )

                        if retry_search_results:
                            retry_reranked = self.reranker.rerank(
                                query=processed_query,
                                documents=retry_search_results,
                                top_k=rerank_top_k,
                            )
                        else:
                            retry_reranked = []

                        retry_answer = generated_answer
                        if retry_reranked and self._generator:
                            retry_gen_result = self._generator.generate(
                                query=processed_query,
                                context_chunks=[r.text for r in retry_reranked],
                                scores=[r.score for r in retry_reranked],
                                memory_context=None,
                                recent_turns=None,
                                graph_context=graph_context,
                            )
                            retry_answer = retry_gen_result.answer or None
                            retry_confidence = retry_gen_result.confidence
                        else:
                            retry_confidence = "medium"

                        retry_breakdown = None
                        if retry_reranked and retry_answer:
                            retry_breakdown = compute_composite_confidence(
                                reranker_scores=[r.score for r in retry_reranked],
                                llm_confidence_text=retry_confidence,
                                answer=retry_answer,
                                retrieved_texts=[r.text for r in retry_reranked],
                                retrieval_weight=RAG_CONFIDENCE_RETRIEVAL_WEIGHT,
                                llm_weight=RAG_CONFIDENCE_LLM_WEIGHT,
                                citation_weight=RAG_CONFIDENCE_CITATION_WEIGHT,
                            )

                    if retry_breakdown is not None and retry_breakdown.composite >= breakdown.composite:
                        reranked = retry_reranked
                        generated_answer = retry_answer
                        breakdown = retry_breakdown
                        composite_confidence = retry_breakdown.composite
                        confidence_breakdown_dict = {
                            "retrieval_score": retry_breakdown.retrieval_score,
                            "llm_score": retry_breakdown.llm_score,
                            "citation_score": retry_breakdown.citation_score,
                            "composite": retry_breakdown.composite,
                            "retrieval_weight": retry_breakdown.retrieval_weight,
                            "llm_weight": retry_breakdown.llm_weight,
                            "citation_weight": retry_breakdown.citation_weight,
                        }

                    action = route_by_confidence(
                        composite=breakdown.composite,
                        retry_count=retry_count + 1,
                        high_threshold=RAG_CONFIDENCE_HIGH_THRESHOLD,
                        low_threshold=RAG_CONFIDENCE_LOW_THRESHOLD,
                        max_retries=RAG_CONFIDENCE_RE_RETRIEVE_MAX_RETRIES,
                    )
                    post_guardrail_action = action.value

                    logger.info(
                        "Re-retrieval loop: first=%.2f second=%.2f final_action=%s",
                        first_composite,
                        breakdown.composite,
                        action.value,
                    )

                if action == PostGuardrailAction.RE_RETRIEVE and generation_source == "memory":
                    # REQ-1203: Re-retrieval not applicable on memory path — re-route to FLAG
                    verification_warning = (
                        "This answer was generated from conversation history and has limited confidence. "
                        "Please verify against source documents."
                    )
                    generated_answer = (
                        f"{generated_answer}\n\n"
                        f"---\n"
                        f"⚠️ {verification_warning}"
                    )
                    post_guardrail_action = PostGuardrailAction.FLAG.value
                elif action == PostGuardrailAction.RE_RETRIEVE:
                    # Retries exhausted — advisory flag for caller-driven broader pass.
                    re_retrieval_suggested = True
                    re_retrieval_params = {
                        "alpha": max(0.0, alpha - 0.15),
                        "search_limit": search_limit + 5,
                        "rerank_top_k": rerank_top_k,
                        "fast_path": True,
                    }
                    verification_warning = (
                        "This answer has moderate confidence. A broader search "
                        "may yield better results — re-retrieval is available."
                    )
                elif action == PostGuardrailAction.BLOCK:
                    generated_answer = (
                        "Insufficient documentation found to provide a reliable answer. "
                        "Please try a more specific query."
                    )
                elif action == PostGuardrailAction.FLAG:
                    verification_warning = (
                        "This answer has limited confidence. "
                        "Please verify against source documents before relying on it."
                    )
                    generated_answer = (
                        f"{generated_answer}\n\n"
                        f"---\n"
                        f"⚠️ {verification_warning}"
                    )
                    if first_composite is not None:
                        re_retrieval_suggested = True
                        re_retrieval_params = {
                            "alpha": max(0.0, alpha - 0.15),
                            "search_limit": search_limit + 5,
                            "rerank_top_k": rerank_top_k,
                            "fast_path": True,
                        }

            # Extract LLM self-reported confidence as structured data.
            # Display formatting is the UI/console layer's responsibility.
            llm_confidence = gen_result.confidence if gen_result is not None else None
            generation_error_payload = (
                gen_result.error.to_dict()
                if (gen_result is not None and gen_result.error is not None)
                else None
            )

            # Advisory DR-suggestion chip — baseline path only. When DR was
            # already active the user opted in; suggesting it again would just
            # be noise (and risks a re-run loop on the UI side).
            dr_suggestion_payload: Optional[dict[str, Any]] = None
            if not _dr_active:
                try:
                    suggest, reason = should_suggest_deep_research(query, reranked)
                    dr_suggestion_payload = {"suggest": bool(suggest), "reason": reason}
                except Exception:  # noqa: BLE001 — never let an advisory hint break the response
                    dr_suggestion_payload = None

            tp.log_summary()
            root_span.set_attribute("duration_ms", int((time.perf_counter() - pipeline_start) * 1000))

            response_metadata: dict = {}
            if _dr_active:
                if dr_result is not None:
                    response_metadata["deep_research"] = {
                        "iteration_count": int(dr_result.iteration_count),
                        "llm_call_count": int(dr_result.llm_call_count),
                        "node_count": int(dr_result.node_count),
                        "topic_count": int(dr_result.topic_count),
                        "decomposed": bool(dr_result.decomposed),
                        "is_unified": bool(dr_result.is_unified),
                        "budget_exhausted": bool(dr_result.budget_exhausted),
                        "budget_exhausted_reason": dr_result.budget_exhausted_reason,
                        "elapsed_ms": float(dr_result.elapsed_ms),
                    }
                else:
                    # DR was requested but the orchestrator failed/early-exited
                    # before producing a result. Surface zeros so downstream
                    # observability (Temporal search attributes) still gets
                    # consistent shape.
                    response_metadata["deep_research"] = {
                        "iteration_count": 0,
                        "llm_call_count": 0,
                        "node_count": 0,
                        "topic_count": 0,
                        "decomposed": False,
                        "is_unified": True,
                        "budget_exhausted": False,
                        "budget_exhausted_reason": None,
                        "elapsed_ms": 0.0,
                    }

            return RAGResponse(
                query=query,
                processed_query=processed_query,
                query_confidence=query_result.confidence,
                action="search",
                results=reranked,
                kg_expanded_terms=kg_expanded_terms or None,
                generated_answer=generated_answer,
                stage_timings=tp.entries(),
                timing_totals=tp.totals(),
                budget_exhausted=tp.budget_exhausted,
                budget_exhausted_stage=tp.budget_exhausted_stage,
                conversation_id=conversation_id,
                guardrails=guardrails_metadata,
                token_budget=token_budget,
                composite_confidence=composite_confidence,
                confidence_breakdown=confidence_breakdown_dict,
                post_guardrail_action=post_guardrail_action,
                version_conflicts=(
                    [{"spec_stem": c.spec_stem, "versions": c.versions} for c in version_conflicts]
                    if version_conflicts
                    else None
                ),
                retry_count=retry_count,
                verification_warning=verification_warning,
                retrieval_quality=retrieval_quality,
                retrieval_quality_note=retrieval_quality_note,
                re_retrieval_suggested=re_retrieval_suggested if RAG_CONFIDENCE_ROUTING_ENABLED else False,
                re_retrieval_params=re_retrieval_params,
                first_composite=first_composite if RAG_CONFIDENCE_ROUTING_ENABLED else None,
                visual_results=visual_results,
                generation_source=generation_source,
                llm_confidence=llm_confidence,
                generation_error=generation_error_payload,
                history_decision=query_result.history_decision,
                history_turns_used=query_result.history_turns_used,
                dr_suggestion=dr_suggestion_payload,
                metadata=response_metadata,
            )


__all__ = ["RAGChain", "RAGResponse"]
