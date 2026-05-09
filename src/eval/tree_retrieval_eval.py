"""Tree-retrieval evaluation harness.

Provides:
- ``EvalReport`` — aggregated metrics (mean recall@k, mean nDCG@k, n_queries, per-query detail).
- ``run_eval`` — drive a retriever over (queries, gold) and aggregate metrics.
- ``build_simulated_retriever`` — wire an in-memory fixture into the live
  ``RAGChain._collect_candidates`` pipeline by stubbing only the vector-store
  boundary methods (``_do_search``, ``_expand_sections_to_leaves``,
  ``_fetch_siblings``). The same Stage-4 / 4b / 4c logic the production
  pipeline runs is exercised, including the leaf-only filter, dedup, and
  doc-diversity cap.

After the live pipeline returns merged candidates, the harness applies a
deterministic rerank (``text_score + 0.5 * heading_score``, where ``score``
is token-overlap Jaccard) — a stand-in for the cross-encoder reranker that
production uses. This is the design's "reranker is the great equalizer"
contract from TREE_RETRIEVAL_DESIGN.md §4.3.
"""
# @summary
# Eval harness for tree retrieval — runs queries through the real
# _collect_candidates with stubbed vector-store IO, then reranks with a
# deterministic lexical+heading score. Reports recall@k and nDCG@k.
# Exports: EvalReport, run_eval, build_simulated_retriever
# Deps: src.retrieval.pipeline.rag_chain.RAGChain, src.eval.metrics, src.eval.fixtures
# @end-summary
from __future__ import annotations

import re
import types
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from src.eval.fixtures import Fixture
from src.eval.metrics import mean_metric, ndcg_at_k, recall_at_k


# ─── Lexical helpers ───────────────────────────────────────────────────────

_TOKEN = re.compile(r"\w+")


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN.findall(text or "") if len(t) > 1}


def _jaccard(query: str, text: str) -> float:
    qt = _tokens(query)
    if not qt:
        return 0.0
    tt = _tokens(text)
    if not tt:
        return 0.0
    return len(qt & tt) / len(qt)


# ─── Test-double scaffolding for RAGChain ──────────────────────────────────


class _DummySpan:
    def set_attribute(self, *_a, **_kw): return None
    def end(self, *_a, **_kw): return None
    def __enter__(self): return self
    def __exit__(self, *_): return False


class _DummyTracer:
    def span(self, *_a, **_kw): return _DummySpan()
    def start_span(self, *_a, **_kw): return _DummySpan()


class _DummyRetryProvider:
    def execute(self, operation_name, fn, policy, idempotency_key):
        return fn()


def _make_chain():
    """Construct a bare ``RAGChain`` with the minimal attributes needed by
    ``_collect_candidates``. No real Weaviate, no real tracer, no embeddings.
    """
    from src.retrieval.pipeline.rag_chain import RAGChain  # local import

    chain = object.__new__(RAGChain)
    chain.tracer = _DummyTracer()
    chain.retry_provider = _DummyRetryProvider()
    chain.retry_policy = object()
    chain._persistent_weaviate = False
    chain._weaviate_client = None
    chain._embedding_cache = OrderedDict()
    chain._embedding_cache_max = 128
    return chain


def _hit_from_chunk(chunk: dict, score: float):
    """SearchResult-shaped namespace from a fixture chunk dict."""
    return types.SimpleNamespace(
        text=chunk["text"],
        score=score,
        metadata=dict(chunk),
        collection="default",
        uuid=chunk["chunk_id"],
    )


# ─── Eval report and runner ────────────────────────────────────────────────


@dataclass
class EvalReport:
    recall_at_k: float
    ndcg_at_k: float
    n_queries: int
    per_query: dict = field(default_factory=dict)
    # Pool metrics — measure the candidate pool BEFORE rerank, so they
    # attribute tree-retrieval's contribution independently of how well
    # the cross-encoder later ranks formula/code chunks. Zero when not
    # computed.
    pool_recall_at_k: float = 0.0
    pool_ndcg_at_k: float = 0.0


def run_eval(
    *,
    retriever: Callable[..., Sequence[str]],
    queries: Sequence[Mapping | object],
    gold: Mapping[str, set],
    tree_enabled: bool,
    k_recall: int,
    k_ndcg: int,
    compute_pool: bool = False,
) -> EvalReport:
    """Drive ``retriever`` over each query and aggregate metrics.

    ``retriever`` is called as ``retriever(query_text, tree_enabled=...)``
    and must return an ordered ``Sequence[chunk_id]``.

    When ``compute_pool=True``, the retriever is also called with
    ``return_pool=True`` to fetch the pre-rerank candidate-pool ids; pool
    metrics are aggregated alongside the post-rerank ones.
    """
    recalls: list[float] = []
    ndcgs: list[float] = []
    pool_recalls: list[float] = []
    pool_ndcgs: list[float] = []
    per_query: dict = {}

    for q in queries:
        qid = q["qid"] if isinstance(q, Mapping) else q.qid
        text = q["text"] if isinstance(q, Mapping) else q.text
        retrieved = list(retriever(text, tree_enabled=tree_enabled))
        gold_ids = gold.get(qid, set())
        r = recall_at_k(retrieved, gold_ids, k=k_recall)
        n = ndcg_at_k(retrieved, gold_ids, k=k_ndcg)
        recalls.append(r)
        ndcgs.append(n)
        entry: dict = {
            "recall": r,
            "ndcg": n,
            "retrieved": retrieved[:k_ndcg],
        }
        if compute_pool:
            pool = list(retriever(text, tree_enabled=tree_enabled, return_pool=True))
            # Pool recall uses ``k_ndcg`` not ``k_recall``: the pool is not yet
            # ranked by relevance (first items are Stage-4 hits, then descent,
            # then lift), so "top 5" of the pool just measures Stage 4. The
            # meaningful pool question is "did tree retrieval bring gold into
            # the candidate window the reranker will see?" — which is k_ndcg.
            pr = recall_at_k(pool, gold_ids, k=k_ndcg)
            pn = ndcg_at_k(pool, gold_ids, k=k_ndcg)
            pool_recalls.append(pr)
            pool_ndcgs.append(pn)
            entry["pool_recall"] = pr
            entry["pool_ndcg"] = pn
            entry["pool"] = pool[:k_ndcg]
        per_query[qid] = entry

    return EvalReport(
        recall_at_k=mean_metric(recalls),
        ndcg_at_k=mean_metric(ndcgs),
        n_queries=len(list(queries)),
        per_query=per_query,
        pool_recall_at_k=mean_metric(pool_recalls),
        pool_ndcg_at_k=mean_metric(pool_ndcgs),
    )


# ─── Simulated retriever ───────────────────────────────────────────────────


# Default eval config — knobs are eval-specific (wider than production
# defaults so tree-retrieval signal isn't capped before the rerank can sort).
_EVAL_SEARCH_LIMIT = 6
_EVAL_DESCENT_TOP_K = 5
_EVAL_LEAVES_PER_SECTION = 3
_EVAL_LIFT_SEED_K = 5
_EVAL_SIBLINGS_PER_GROUP = 4
_EVAL_DOC_DIVERSITY_TOP_PER_DOC = 10


def build_simulated_retriever(
    fixture: Fixture,
    *,
    reranker: Callable[[str, list], list] | None = None,
    apply_fusion: bool = False,
    fusion_weights: dict | None = None,
):
    """Return ``retrieve(query_text, *, tree_enabled) -> [chunk_id, ...]``.

    The simulator stubs ``_do_search`` (lexical scoring + filter application),
    ``_expand_sections_to_leaves`` and ``_fetch_siblings`` (in-memory
    parent-section index lookup), then calls the live
    ``RAGChain._collect_candidates``.

    Args:
        fixture: hydrated ``Fixture`` (corpus + queries + gold).
        reranker: optional ``(query_text, items) -> sorted_items`` callable.
            When None (default), candidates are sorted by a lexical fallback
            (text_jaccard + 0.5 * heading_jaccard) — fast, dependency-free,
            adequate for synthetic gates. Pass ``TEIRerankerAdapter`` (or any
            callable) to score with a real cross-encoder for real-data evals.
        apply_fusion: When True, the production rerank-fusion module is
            applied on top of the reranker's output (BM25-RRF + heading
            + anchor). Lets the harness measure fusion deltas in the
            same lexical sandbox.
        fusion_weights: Override fusion weights, e.g. ``{"rrf": 1.0,
            "heading": 0.0, "anchor": 0.0}`` to isolate a single signal.
    """
    chunks: list[dict] = list(fixture.all_chunks())

    by_parent: dict[str, list[dict]] = {}
    for chunk in chunks:
        if chunk.get("node_kind") == "chunk":
            psid = chunk.get("parent_section_id", "")
            if psid:
                by_parent.setdefault(psid, []).append(chunk)

    def _matches_filter(chunk: dict, filt) -> bool:
        prop = getattr(filt, "property", None)
        op = getattr(filt, "operator", None)
        val = getattr(filt, "value", None)
        if op != "eq":
            return True
        return chunk.get(prop) == val

    def _apply_filters(items: list[dict], filters) -> list[dict]:
        if not filters:
            return items
        return [c for c in items if all(_matches_filter(c, f) for f in filters)]

    def retrieve(query_text: str, *, tree_enabled: bool,
                 return_pool: bool = False) -> list[str]:
        chain = _make_chain()

        def _do_search(bm25_query, query_emb, alpha, search_limit, filters):
            scope = _apply_filters(chunks, filters or [])
            scored = sorted(
                scope,
                key=lambda c: _jaccard(bm25_query, c["text"]),
                reverse=True,
            )
            scored = [c for c in scored if _jaccard(bm25_query, c["text"]) > 0.0]
            return [
                _hit_from_chunk(c, _jaccard(bm25_query, c["text"]))
                for c in scored[: max(0, search_limit or 0)]
            ]

        def _expand(parent_ids, per_section_limit, base_filters):
            out: list = []
            for psid in parent_ids:
                if not psid:
                    continue
                bucket = by_parent.get(psid, [])
                # Apply any base_filters (e.g., document_id restrictions); the
                # leaf-only constraint is implicit because by_parent only
                # contains node_kind=chunk rows.
                bucket = _apply_filters(bucket, base_filters or [])
                for c in bucket[: max(0, per_section_limit or 0)]:
                    out.append(_hit_from_chunk(c, 0.0))
            return out

        chain._do_search = _do_search
        chain._expand_sections_to_leaves = _expand
        chain._fetch_siblings = _expand

        merged = chain._collect_candidates(
            bm25_query=query_text,
            query_embedding=None,
            alpha=0.5,
            search_limit=_EVAL_SEARCH_LIMIT,
            base_filters=[],
            tree_enabled=tree_enabled,
            schema_present=True,
            descent_top_k=_EVAL_DESCENT_TOP_K,
            leaves_per_section=_EVAL_LEAVES_PER_SECTION,
            lift_seed_k=_EVAL_LIFT_SEED_K,
            siblings_per_group=_EVAL_SIBLINGS_PER_GROUP,
            doc_diversity_top_per_doc=_EVAL_DOC_DIVERSITY_TOP_PER_DOC,
        )

        if return_pool:
            # Pre-rerank pool — preserves whatever order ``_collect_candidates``
            # returned (Stage 4 → descent → lift, with first-occurrence dedup).
            return [
                (getattr(m, "metadata", None) or {}).get("chunk_id", "")
                for m in merged
            ]

        # Compute per-candidate CE-stand-in scores in the same order as the
        # candidate list. We want them deterministic so fusion math is exact.
        def _ce_score(item) -> float:
            text_score = _jaccard(query_text, getattr(item, "text", ""))
            meta = getattr(item, "metadata", None) or {}
            heading_text = " ".join(meta.get("heading_path") or [])
            head_score = _jaccard(query_text, heading_text)
            return text_score + 0.5 * head_score

        if reranker is not None:
            # Trust an external reranker (e.g. TEI cross-encoder); we still
            # carry its output into fusion when ``apply_fusion`` is True.
            merged_sorted = list(reranker(query_text, merged))
        else:
            merged_sorted = sorted(merged, key=_ce_score, reverse=True)

        if apply_fusion:
            # Local import keeps eval module otherwise-pure.
            from src.retrieval.query.nodes.rerank_fusion import (  # noqa: PLC0415
                anchor_confidence,
                fuse_scores,
                heading_match_score,
            )

            # BM25-only ranks: re-sort the merged candidates by raw BM25
            # (text-only jaccard) to mimic Weaviate's BM25-only side-call.
            bm25_sorted = sorted(
                merged,
                key=lambda c: _jaccard(query_text, getattr(c, "text", "")),
                reverse=True,
            )
            bm25_rank_by_key: dict[str, int] = {}
            for r, h in enumerate(bm25_sorted):
                meta = getattr(h, "metadata", None) or {}
                key = str(meta.get("chunk_id", "")) or getattr(h, "text", "")
                if key and key not in bm25_rank_by_key:
                    bm25_rank_by_key[key] = r

            ce_scores: list[float] = []
            bm25_aligned: list[int | None] = []
            heading_aligned: list[float] = []
            anchor_aligned: list[float] = []
            for item in merged_sorted:
                meta = getattr(item, "metadata", None) or {}
                key = str(meta.get("chunk_id", "")) or getattr(item, "text", "")
                ce_scores.append(_ce_score(item))
                bm25_aligned.append(bm25_rank_by_key.get(key))
                heading_aligned.append(
                    heading_match_score(query_text, meta.get("heading_path") or [])
                )
                anchor_aligned.append(
                    anchor_confidence(meta.get("_anchor_rank"), anchor_k=10)
                )

            weights = fusion_weights or {
                "rrf": 1.0, "heading": 0.15, "anchor": 0.10,
            }
            fused = fuse_scores(
                ce_scores=ce_scores,
                bm25_ranks=bm25_aligned,
                heading_scores=heading_aligned,
                anchor_confs=anchor_aligned,
                weights=weights,
                k_rrf=60,
            )
            paired = sorted(
                zip(merged_sorted, fused), key=lambda kv: kv[1], reverse=True
            )
            merged_sorted = [item for item, _ in paired]

        return [
            (getattr(m, "metadata", None) or {}).get("chunk_id", "")
            for m in merged_sorted
        ]

    return retrieve


# ─── TEI HTTP reranker adapter ─────────────────────────────────────────────


class TEIRerankerAdapter:
    """Reranker callable backed by TEI's ``/rerank`` HTTP endpoint.

    Matches the production ``TEIReranker`` wire protocol (POST /rerank with
    ``{query, texts, truncate}``; response is a list of ``{index, score}``
    sorted descending). Stays in this module so the eval harness has no
    runtime dependency on the production retrieval modules — keeps the eval
    layer cleanly importable from anywhere.

    Usage::

        from src.eval.tree_retrieval_eval import (
            TEIRerankerAdapter, build_simulated_retriever,
        )
        rer = TEIRerankerAdapter(base_url="http://localhost:8082")
        retriever = build_simulated_retriever(fx, reranker=rer)
    """

    def __init__(self, base_url: str = "http://localhost:8082", timeout: float = 30.0):
        import httpx  # local import — eval is otherwise pure-python

        self._client = httpx.Client(timeout=timeout)
        self._base_url = base_url.rstrip("/")

    def __call__(self, query: str, items: list) -> list:
        if not items:
            return []
        texts = [getattr(it, "text", "") for it in items]
        resp = self._client.post(
            f"{self._base_url}/rerank",
            json={"query": query, "texts": texts, "truncate": True},
        )
        resp.raise_for_status()
        ranked = resp.json()  # [{"index": i, "score": s}, ...] desc by score
        return [items[entry["index"]] for entry in ranked]

    def close(self) -> None:
        self._client.close()
