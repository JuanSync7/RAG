"""Integration test for the deep_research branch in RAGChain.run.

Verifies the wire-up: when ``deep_research=True``, the linear stages 2–5
(KG expand, embed, hybrid search, rerank) are skipped, the orchestrator's
output flows into ``reranked``, and ``graph_context`` is preserved through
to generation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.retrieval.common.schemas import RankedResult
from src.retrieval.pipeline.deep_research import DeepResearchResult, TopicPool
from src.vector_db.common.schemas import SearchResult


def _mk_chain():
    """Construct a RAGChain with all heavy dependencies mocked."""
    with patch("src.retrieval.pipeline.rag_chain.create_persistent_client"), \
         patch("src.retrieval.pipeline.rag_chain.ensure_collection"), \
         patch("src.retrieval.pipeline.rag_chain.get_embedding_provider") as gep, \
         patch("src.retrieval.pipeline.rag_chain.get_reranker_provider") as grp, \
         patch("src.retrieval.pipeline.rag_chain.OllamaGenerator") as gen_cls, \
         patch("src.retrieval.pipeline.rag_chain._get_kg_client") as gkc, \
         patch("src.retrieval.pipeline.rag_chain.KG_ENABLED", False), \
         patch("src.retrieval.pipeline.rag_chain.GENERATION_ENABLED", False):
        gep.return_value = MagicMock()
        grp.return_value = MagicMock()
        gen_cls.return_value = MagicMock()
        gkc.return_value = None

        from src.retrieval.pipeline.rag_chain import RAGChain  # noqa: PLC0415
        chain = RAGChain(persistent_weaviate=False)

    # Disable generation explicitly so .run() returns at the retrieval boundary.
    chain._generator = None
    return chain


def _canned_dr_result() -> DeepResearchResult:
    """A unified-pool result with two chunks and graph context."""
    pool = TopicPool(name="<root>", rerank_anchor="anchor")
    pool.merge_chunks([
        SearchResult(
            text="alpha-text",
            score=0.9,
            metadata={"source": "doc.md", "heading": "h", "chunk_index": 0},
            object_id="oid-1",
        ),
        SearchResult(
            text="beta-text",
            score=0.7,
            metadata={"source": "doc.md", "heading": "h", "chunk_index": 1},
            object_id="oid-2",
        ),
    ])
    return DeepResearchResult(
        topic_pools=[pool],
        graph_context="GRAPH-CONTEXT-PINNED",
        node_count=3,
        llm_call_count=4,
        elapsed_ms=120.0,
        budget_exhausted=False,
        is_unified=True,
    )


def test_deep_research_flag_skips_linear_stages_and_uses_branch():
    chain = _mk_chain()

    # The reranker should be called once (final rerank against processed_query)
    # but the hybrid_search and embed_query paths must NOT fire under DR.
    chain.embeddings.embed_query = MagicMock(side_effect=AssertionError(
        "embed_query must NOT be called when deep_research=True"
    ))
    chain.reranker.rerank = MagicMock(return_value=[
        RankedResult(text="alpha-text", score=0.9, metadata={"source": "doc.md"}),
        RankedResult(text="beta-text", score=0.7, metadata={"source": "doc.md"}),
    ])
    chain._do_search = MagicMock(side_effect=AssertionError(
        "_do_search must NOT be called from the linear path under deep_research=True"
    ))

    canned = _canned_dr_result()
    with patch.object(chain, "_run_deep_research", return_value=canned) as run_dr, \
         patch("src.retrieval.pipeline.rag_chain.process_query") as proc:
        proc.return_value = MagicMock(
            processed_query="X Y Z",
            standalone_query="X Y Z",
            confidence=0.9,
            action=MagicMock(value="search"),
            has_backward_reference=False,
            suppress_memory=False,
        )

        response = chain.run(
            query="how does X work and how does Y work",
            deep_research=True,
            skip_generation=True,
        )

    run_dr.assert_called_once()
    # Branch should have triggered a final rerank against processed_query.
    chain.reranker.rerank.assert_called_once()
    rerank_call = chain.reranker.rerank.call_args
    assert rerank_call.kwargs["query"] == "X Y Z"

    # Graph context from DR must flow into the response (via being attached to
    # the kg span attribute path — visible by virtue of having been threaded).
    # The RAGResponse.results contain the reranked chunks.
    assert len(response.results) == 2
    assert response.results[0].text == "alpha-text"


def test_deep_research_multi_topic_uses_per_topic_rerank():
    """When DR returns multiple topic pools, the chain must rerank each pool
    against its own anchor, NOT against the original processed_query."""
    chain = _mk_chain()

    pool_a = TopicPool(name="verification", rerank_anchor="verification methodology?")
    pool_a.merge_chunks([SearchResult(text="ver-a", score=0.5, metadata={"source": "v.md"}, object_id="va")])
    pool_b = TopicPool(name="lint flow", rerank_anchor="lint flow?")
    pool_b.merge_chunks([SearchResult(text="lint-a", score=0.5, metadata={"source": "l.md"}, object_id="la")])

    multi = DeepResearchResult(
        topic_pools=[pool_a, pool_b],
        graph_context="",
        node_count=4,
        llm_call_count=6,
        elapsed_ms=200.0,
        is_unified=False,
    )

    seen_queries: list[str] = []

    def _fake_rerank(*, query, documents, top_k):
        seen_queries.append(query)
        return [RankedResult(text=d.text, score=0.9, metadata=d.metadata) for d in documents]

    chain.reranker.rerank = MagicMock(side_effect=_fake_rerank)

    with patch.object(chain, "_run_deep_research", return_value=multi), \
         patch("src.retrieval.pipeline.rag_chain.process_query") as proc:
        proc.return_value = MagicMock(
            processed_query="verification lint flow",
            standalone_query="verification lint flow",
            confidence=0.9,
            action=MagicMock(value="search"),
            has_backward_reference=False,
            suppress_memory=False,
        )
        response = chain.run(
            query="how does verification work and how does lint flow work",
            deep_research=True,
            skip_generation=True,
        )

    # One rerank per topic pool, each against its own anchor.
    assert seen_queries == ["verification methodology?", "lint flow?"]
    # Both chunks present in final results.
    texts = {r.text for r in response.results}
    assert texts == {"ver-a", "lint-a"}


def test_deep_research_metadata_surfaces_on_response():
    """``response.metadata['deep_research']`` must carry iteration/topic/llm
    counters so Temporal search attributes can index a DR run."""
    chain = _mk_chain()

    pool = TopicPool(name="<root>", rerank_anchor="anchor")
    pool.merge_chunks([
        SearchResult(
            text="alpha-text", score=0.9,
            metadata={"source": "doc.md", "heading": "h", "chunk_index": 0},
            object_id="oid-1",
        ),
    ])
    canned = DeepResearchResult(
        topic_pools=[pool],
        graph_context="g",
        node_count=5,
        llm_call_count=7,
        elapsed_ms=321.0,
        budget_exhausted=False,
        is_unified=True,
        iteration_count=3,
        decomposed=True,
        topic_count=1,
    )

    chain.reranker.rerank = MagicMock(return_value=[
        RankedResult(text="alpha-text", score=0.9, metadata={"source": "doc.md"}),
    ])

    with patch.object(chain, "_run_deep_research", return_value=canned), \
         patch("src.retrieval.pipeline.rag_chain.process_query") as proc:
        proc.return_value = MagicMock(
            processed_query="X Y Z",
            standalone_query="X Y Z",
            confidence=0.9,
            action=MagicMock(value="search"),
            has_backward_reference=False,
            suppress_memory=False,
        )
        response = chain.run(
            query="how does X work",
            deep_research=True,
            skip_generation=True,
        )

    dr_meta = response.metadata.get("deep_research")
    assert dr_meta is not None
    assert dr_meta["iteration_count"] == 3
    assert dr_meta["llm_call_count"] == 7
    assert dr_meta["node_count"] == 5
    assert dr_meta["topic_count"] == 1
    assert dr_meta["decomposed"] is True
    assert dr_meta["is_unified"] is True
    assert dr_meta["budget_exhausted"] is False


def test_baseline_query_has_empty_metadata():
    """Without deep_research, response.metadata is an empty dict (no DR key)."""
    chain = _mk_chain()
    chain.embeddings.embed_query = MagicMock(return_value=[0.1, 0.2])
    chain._do_search = MagicMock(return_value=[])
    chain.reranker.rerank = MagicMock(return_value=[])

    with patch("src.retrieval.pipeline.rag_chain.process_query") as proc:
        proc.return_value = MagicMock(
            processed_query="x",
            standalone_query="x",
            confidence=0.9,
            action=MagicMock(value="search"),
            has_backward_reference=False,
            suppress_memory=False,
        )
        response = chain.run(query="x", deep_research=False, skip_generation=True)

    assert "deep_research" not in response.metadata
