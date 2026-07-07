"""Span-tree integration test for RAGChain.run.

Mocks the deepest IO boundaries (vector store, LLM provider, reranker)
and lets the real RAGChain logic flow through to verify the expected
``retrieval.*`` span tree shape and parent-child relationships.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.retrieval.common.schemas import RankedResult
from src.vector_db.common.schemas import SearchResult


def _mk_chain():
    """Construct a RAGChain with all heavy deps mocked, generation off."""
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

        from src.retrieval.pipeline.rag_chain import RAGChain
        chain = RAGChain(persistent_weaviate=False)

    chain._generator = None
    return chain


def _spans_named(exporter, name):
    return [s for s in exporter.get_finished_spans() if s.name == name]


def _ancestor_names(span, by_id):
    """Walk the parent chain and return the chain of ancestor names."""
    names = []
    cur = span
    while cur is not None and cur.parent is not None:
        parent = by_id.get(cur.parent.span_id)
        if parent is None:
            break
        names.append(parent.name)
        cur = parent
    return names


def test_rag_run_emits_retrieval_rag_root_with_required_attrs(otel_capture):
    chain = _mk_chain()
    chain.embeddings.embed_query = MagicMock(return_value=[0.1] * 4)
    chain._do_search = MagicMock(return_value=[
        SearchResult(
            text="alpha",
            score=0.9,
            metadata={"source": "d.md", "chunk_id": "a"},
            object_id="oid-1",
        ),
    ])
    chain.reranker.rerank = MagicMock(return_value=[
        RankedResult(text="alpha", score=0.9, metadata={"source": "d.md"}),
    ])

    with patch("src.retrieval.pipeline.rag_chain.process_query") as proc:
        proc.return_value = MagicMock(
            processed_query="alpha",
            standalone_query="alpha",
            confidence=0.9,
            action=MagicMock(value="search"),
            has_backward_reference=False,
            suppress_memory=False,
        )
        chain.run(query="hello world", skip_generation=True)

    roots = _spans_named(otel_capture, "retrieval.rag")
    assert len(roots) == 1
    attrs = roots[0].attributes
    assert attrs["query_len"] == len("hello world")
    assert attrs["retrieval_mode"] == "query"
    assert attrs["generation_enabled"] is False
    assert "top_k" in attrs


def test_rag_run_span_tree_shape_nests_search_and_rerank(otel_capture):
    chain = _mk_chain()
    chain.embeddings.embed_query = MagicMock(return_value=[0.1] * 4)
    chain._do_search = MagicMock(return_value=[
        SearchResult(
            text="alpha",
            score=0.9,
            metadata={"source": "d.md", "chunk_id": "a"},
            object_id="oid-1",
        ),
    ])
    chain.reranker.rerank = MagicMock(return_value=[
        RankedResult(text="alpha", score=0.9, metadata={"source": "d.md"}),
    ])

    with patch("src.retrieval.pipeline.rag_chain.process_query") as proc:
        proc.return_value = MagicMock(
            processed_query="alpha",
            standalone_query="alpha",
            confidence=0.9,
            action=MagicMock(value="search"),
            has_backward_reference=False,
            suppress_memory=False,
        )
        chain.run(query="alpha?", skip_generation=True)

    spans = otel_capture.get_finished_spans()
    by_id = {s.context.span_id: s for s in spans}
    rag_root = next(s for s in spans if s.name == "retrieval.rag")

    # Expected child spans of retrieval.rag (direct or transitive).
    expected_descendants = {
        "retrieval.rag.process_query",
        "retrieval.search.hybrid",
        "retrieval.rerank",
        "retrieval.embed_query",
        "retrieval.collect_candidates",
    }
    descendant_names: set[str] = set()
    for s in spans:
        if s is rag_root:
            continue
        ancestors = _ancestor_names(s, by_id)
        if "retrieval.rag" in ancestors:
            descendant_names.add(s.name)

    missing = expected_descendants - descendant_names
    assert not missing, f"Missing expected descendants under retrieval.rag: {missing}"


def test_rag_run_propagates_error_status(otel_capture):
    chain = _mk_chain()
    chain.embeddings.embed_query = MagicMock(side_effect=RuntimeError("boom"))

    with patch("src.retrieval.pipeline.rag_chain.process_query") as proc:
        proc.return_value = MagicMock(
            processed_query="alpha",
            standalone_query="alpha",
            confidence=0.9,
            action=MagicMock(value="search"),
            has_backward_reference=False,
            suppress_memory=False,
        )
        try:
            chain.run(query="alpha?", skip_generation=True)
        except Exception:
            pass

    roots = _spans_named(otel_capture, "retrieval.rag")
    assert len(roots) == 1
    # OTel status code: ERROR=2, OK=1, UNSET=0
    from opentelemetry.trace import StatusCode
    assert roots[0].status.status_code == StatusCode.ERROR
