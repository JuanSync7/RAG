# @summary
# Integration tests for the dr_suggestion response wiring (query_shape module) inside RAGChain.run.
# Verifies the suggestion is populated on the baseline path and suppressed
# when deep_research=True.
# Deps: pytest, unittest.mock, src.retrieval.pipeline.rag_chain
# @end-summary
"""Integration tests for ``dr_suggestion`` wiring in ``RAGChain.run``.

Exercises the same construction pattern as
``test_rag_chain_deep_research.py`` — heavy deps mocked at the module
boundary so the test can run without Weaviate / Ollama.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.retrieval.common.schemas import RankedResult
from src.retrieval.pipeline.deep_research import DeepResearchResult, TopicPool
from src.vector_db.common.schemas import SearchResult


def _mk_chain():
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
    chain._generator = None
    return chain


def _proc_query_mock(processed: str = "X Y Z"):
    return MagicMock(
        processed_query=processed,
        standalone_query=processed,
        confidence=0.9,
        action=MagicMock(value="search"),
        has_backward_reference=False,
        suppress_memory=False,
        history_decision=None,
        history_turns_used=0,
    )


def _baseline_search_results(sources: list[str]) -> list[SearchResult]:
    return [
        SearchResult(
            text=f"chunk-{i}",
            score=0.9 - 0.05 * i,
            metadata={"source": s, "heading": "h", "chunk_index": i},
            object_id=f"oid-{i}",
        )
        for i, s in enumerate(sources)
    ]


def _ranked_from(sources: list[str]) -> list[RankedResult]:
    return [
        RankedResult(text=f"chunk-{i}", score=0.9 - 0.05 * i, metadata={"source": s})
        for i, s in enumerate(sources)
    ]


def test_baseline_compound_query_low_diversity_emits_suggestion():
    chain = _mk_chain()
    chain.embeddings.embed_query = MagicMock(return_value=[0.1, 0.2, 0.3])
    chain._do_search = MagicMock(return_value=_baseline_search_results(["only.md"] * 5))
    chain.reranker.rerank = MagicMock(return_value=_ranked_from(["only.md"] * 5))

    with patch("src.retrieval.pipeline.rag_chain.process_query") as proc:
        proc.return_value = _proc_query_mock("verification methodology and lint flow")
        response = chain.run(
            query="verification methodology and lint flow",
            deep_research=False,
            skip_generation=True,
        )

    assert response.dr_suggestion is not None
    assert response.dr_suggestion["suggest"] is True
    assert response.dr_suggestion["reason"]


def test_baseline_simple_query_no_suggestion():
    chain = _mk_chain()
    chain.embeddings.embed_query = MagicMock(return_value=[0.1, 0.2, 0.3])
    chain._do_search = MagicMock(return_value=_baseline_search_results(["only.md"] * 5))
    chain.reranker.rerank = MagicMock(return_value=_ranked_from(["only.md"] * 5))

    with patch("src.retrieval.pipeline.rag_chain.process_query") as proc:
        proc.return_value = _proc_query_mock("what is gpio")
        response = chain.run(
            query="what is gpio",
            deep_research=False,
            skip_generation=True,
        )

    assert response.dr_suggestion is not None
    assert response.dr_suggestion["suggest"] is False


def test_baseline_compound_query_high_diversity_no_suggestion():
    chain = _mk_chain()
    chain.embeddings.embed_query = MagicMock(return_value=[0.1, 0.2, 0.3])
    chain._do_search = MagicMock(
        return_value=_baseline_search_results([f"d{i}.md" for i in range(5)])
    )
    chain.reranker.rerank = MagicMock(
        return_value=_ranked_from([f"d{i}.md" for i in range(5)])
    )

    with patch("src.retrieval.pipeline.rag_chain.process_query") as proc:
        proc.return_value = _proc_query_mock("a and b")
        response = chain.run(
            query="a and b",
            deep_research=False,
            skip_generation=True,
        )

    assert response.dr_suggestion["suggest"] is False


def test_deep_research_path_suppresses_suggestion():
    chain = _mk_chain()
    chain.reranker.rerank = MagicMock(return_value=_ranked_from(["only.md"] * 2))
    pool = TopicPool(name="<root>", rerank_anchor="anchor")
    pool.merge_chunks(_baseline_search_results(["only.md", "only.md"]))
    canned = DeepResearchResult(
        topic_pools=[pool],
        graph_context="",
        node_count=1,
        llm_call_count=1,
        elapsed_ms=10.0,
        budget_exhausted=False,
        is_unified=True,
    )
    with patch.object(chain, "_run_deep_research", return_value=canned), \
         patch("src.retrieval.pipeline.rag_chain.process_query") as proc:
        proc.return_value = _proc_query_mock("a and b")
        response = chain.run(
            query="a and b",
            deep_research=True,
            skip_generation=True,
        )

    # Baseline-only signal — must not fire when DR was active.
    assert response.dr_suggestion is None or response.dr_suggestion.get("suggest") is False
    assert response.dr_suggestion is None
