# @summary
# Wiring tests for the B3 slice — Stage-1 document routing inside RAGChain.run
# (DOCUMENT_ROUTING_DESIGN.md §3, §6.2, §6.3, §7). Drives the implementation that,
# when RAG_DOCUMENT_ROUTING_ENABLED is on (and NOT deep_research), runs
# decompose_query -> route_documents per sub-query -> unions routed doc_ids ->
# feeds routed_doc_ids into _collect_candidates (B2) while the single rerank stays
# the final authority. CRITICAL no-regression: disabled (default) -> neither
# decompose_query nor route_documents is called and _collect_candidates receives
# routed_doc_ids=None (today's exact path).
# @end-summary
"""TDD scaffold for B3 (wire Stage-1 routing into run()).

Mirrors the full-``run()`` harness in
``tests/retrieval/test_rag_chain_deep_research.py``: a real :class:`RAGChain`
with every heavy dependency mocked, driven via ``run(query, skip_generation=True)``.

We spy on ``_collect_candidates`` (on the instance) to assert the exact
``routed_doc_ids`` it is called with, and we patch ``decompose_query`` /
``route_documents`` *as imported into the rag_chain module namespace* so we can
assert whether — and how — Stage-1 fired.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.retrieval.common.schemas import RankedResult
from src.retrieval.routing import DecompositionResult, RoutingResult


def _mk_chain():
    """Construct a RAGChain with all heavy dependencies mocked (run-drivable)."""
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


def _stub_process_query(processed: str = "x"):
    """Return a patch context for process_query yielding a simple QueryResult."""
    proc = patch("src.retrieval.pipeline.rag_chain.process_query")
    return proc, MagicMock(
        processed_query=processed,
        standalone_query=processed,
        confidence=0.9,
        action=MagicMock(value="search"),
        has_backward_reference=False,
        suppress_memory=False,
    )


def _wire_minimal_retrieval(chain):
    """Stub the heavy retrieval primitives so run() reaches Stage-4 cleanly."""
    chain.embeddings.embed_query = MagicMock(return_value=[0.1, 0.2, 0.3])
    chain._do_search = MagicMock(return_value=[])
    chain.reranker.rerank = MagicMock(return_value=[
        RankedResult(text="t", score=0.9, metadata={"source": "doc.md"}),
    ])


def _spy_collect_candidates(chain):
    """Wrap _collect_candidates so we can read the routed_doc_ids it was given.

    Returns a list that records the ``routed_doc_ids`` kwarg of each call.
    The wrapper returns an empty candidate pool (run() tolerates this).
    """
    seen: list = []
    real_kwargs: list = []

    def _spy(*args, **kwargs):
        seen.append(kwargs.get("routed_doc_ids"))
        real_kwargs.append(kwargs)
        return []

    chain._collect_candidates = MagicMock(side_effect=_spy)
    chain._collect_candidates.seen_routed = seen
    chain._collect_candidates.seen_kwargs = real_kwargs
    return chain._collect_candidates


# ─── DISABLED (default): zero behaviour change ──────────────────────────────


def test_disabled_does_not_call_routing_and_passes_none():
    """RAG_DOCUMENT_ROUTING_ENABLED=False (default): neither decompose_query nor
    route_documents is invoked, and _collect_candidates gets routed_doc_ids=None."""
    chain = _mk_chain()
    _wire_minimal_retrieval(chain)
    cc = _spy_collect_candidates(chain)

    proc_patch, proc_val = _stub_process_query("x")
    with proc_patch as proc, \
         patch("src.retrieval.pipeline.rag_chain.decompose_query") as dq, \
         patch("src.retrieval.pipeline.rag_chain.route_documents") as rd:
        proc.return_value = proc_val
        chain.run(query="x", skip_generation=True)

    dq.assert_not_called()
    rd.assert_not_called()
    assert cc.seen_routed == [None]


# ─── ENABLED + identity decomposition ───────────────────────────────────────


def test_enabled_identity_routes_and_feeds_doc_ids():
    """Enabled + identity decomposition ([q]): route_documents -> ["dX","dY"]
    flows into _collect_candidates as routed_doc_ids; the identity sub-query
    (== processed_query) reuses the already-computed embedding (no re-embed)."""
    chain = _mk_chain()
    _wire_minimal_retrieval(chain)
    cc = _spy_collect_candidates(chain)

    identity = DecompositionResult(["x"], method="identity", decomposed=False)
    routed = RoutingResult(doc_ids=["dX", "dY"], scores={"dX": 0.9, "dY": 0.8}, used=True)

    proc_patch, proc_val = _stub_process_query("x")
    with proc_patch as proc, \
         patch("src.retrieval.pipeline.rag_chain.RAG_DOCUMENT_ROUTING_ENABLED", True), \
         patch("src.retrieval.pipeline.rag_chain.decompose_query", return_value=identity), \
         patch("src.retrieval.pipeline.rag_chain.route_documents", return_value=routed) as rd:
        proc.return_value = proc_val
        chain.run(query="x", skip_generation=True)

    assert cc.seen_routed == [["dX", "dY"]]
    # Identity sub-query equals processed_query -> embedded exactly once.
    chain.embeddings.embed_query.assert_called_once()
    # route_documents called once for the single identity sub-query, with the
    # already-computed query embedding and the sub-query text.
    rd.assert_called_once()
    assert rd.call_args.kwargs["query_text"] == "x"
    assert rd.call_args.args[0] == [0.1, 0.2, 0.3]


# ─── ENABLED + decomposed (per-sub-query routing -> union) ───────────────────


def test_enabled_decomposed_unions_deduped_doc_ids():
    """Decomposed into [AXI4, CHI]: each sub-query routes independently; the
    routed doc_ids are unioned + de-duped (first-seen order); route_documents
    is called once per sub-query."""
    chain = _mk_chain()
    _wire_minimal_retrieval(chain)
    cc = _spy_collect_candidates(chain)

    decomp = DecompositionResult(["AXI4", "CHI"], method="llm", decomposed=True)

    def _route(embedding, **kwargs):
        if kwargs.get("query_text") == "AXI4":
            return RoutingResult(doc_ids=["axi-spec", "shared"], used=True)
        return RoutingResult(doc_ids=["chi-spec", "shared"], used=True)

    proc_patch, proc_val = _stub_process_query("AXI4 vs CHI")
    with proc_patch as proc, \
         patch("src.retrieval.pipeline.rag_chain.RAG_DOCUMENT_ROUTING_ENABLED", True), \
         patch("src.retrieval.pipeline.rag_chain.decompose_query", return_value=decomp), \
         patch("src.retrieval.pipeline.rag_chain.route_documents", side_effect=_route) as rd:
        proc.return_value = proc_val
        chain.run(query="AXI4 vs CHI", skip_generation=True)

    # Union de-duped, first-seen order: AXI4's docs then CHI's new doc.
    assert cc.seen_routed == [["axi-spec", "shared", "chi-spec"]]
    assert rd.call_count == 2
    seen_texts = {c.kwargs["query_text"] for c in rd.call_args_list}
    assert seen_texts == {"AXI4", "CHI"}


# ─── ENABLED + low-confidence routing -> flat fallback ───────────────────────


def test_enabled_low_confidence_falls_back_to_flat():
    """route_documents used=False -> routed_doc_ids is None (the flat path is
    byte-identical to today's no-regression behaviour)."""
    chain = _mk_chain()
    _wire_minimal_retrieval(chain)
    cc = _spy_collect_candidates(chain)

    identity = DecompositionResult(["x"], method="identity", decomposed=False)
    unused = RoutingResult(doc_ids=[], scores={}, used=False)

    proc_patch, proc_val = _stub_process_query("x")
    with proc_patch as proc, \
         patch("src.retrieval.pipeline.rag_chain.RAG_DOCUMENT_ROUTING_ENABLED", True), \
         patch("src.retrieval.pipeline.rag_chain.decompose_query", return_value=identity), \
         patch("src.retrieval.pipeline.rag_chain.route_documents", return_value=unused):
        proc.return_value = proc_val
        chain.run(query="x", skip_generation=True)

    assert cc.seen_routed == [None]


# ─── ENABLED + deep_research -> routing NOT invoked ──────────────────────────


def test_deep_research_bypasses_routing():
    """deep_research=True bypasses the flat path entirely; routing must not run,
    so decompose_query / route_documents are never called."""
    chain = _mk_chain()
    chain.reranker.rerank = MagicMock(return_value=[])

    from src.retrieval.pipeline.deep_research import DeepResearchResult, TopicPool
    pool = TopicPool(name="<root>", rerank_anchor="anchor")
    canned = DeepResearchResult(
        topic_pools=[pool], graph_context="", node_count=1,
        llm_call_count=1, elapsed_ms=1.0, is_unified=True,
    )

    proc_patch, proc_val = _stub_process_query("x")
    with proc_patch as proc, \
         patch("src.retrieval.pipeline.rag_chain.RAG_DOCUMENT_ROUTING_ENABLED", True), \
         patch.object(chain, "_run_deep_research", return_value=canned), \
         patch("src.retrieval.pipeline.rag_chain.decompose_query") as dq, \
         patch("src.retrieval.pipeline.rag_chain.route_documents") as rd:
        proc.return_value = proc_val
        chain.run(query="x", deep_research=True, skip_generation=True)

    dq.assert_not_called()
    rd.assert_not_called()


# ─── Robustness: a raising router degrades to flat ───────────────────────────


def test_routing_exception_degrades_to_flat():
    """If route_documents raises (it shouldn't, but be defensive), run() still
    completes with routed_doc_ids=None (Stage-1 must never break run())."""
    chain = _mk_chain()
    _wire_minimal_retrieval(chain)
    cc = _spy_collect_candidates(chain)

    identity = DecompositionResult(["x"], method="identity", decomposed=False)

    proc_patch, proc_val = _stub_process_query("x")
    with proc_patch as proc, \
         patch("src.retrieval.pipeline.rag_chain.RAG_DOCUMENT_ROUTING_ENABLED", True), \
         patch("src.retrieval.pipeline.rag_chain.decompose_query", return_value=identity), \
         patch("src.retrieval.pipeline.rag_chain.route_documents",
               side_effect=RuntimeError("boom")):
        proc.return_value = proc_val
        response = chain.run(query="x", skip_generation=True)

    assert response is not None
    assert cc.seen_routed == [None]


def test_decompose_exception_degrades_to_flat():
    """If decompose_query raises, run() still completes with routed_doc_ids=None."""
    chain = _mk_chain()
    _wire_minimal_retrieval(chain)
    cc = _spy_collect_candidates(chain)

    proc_patch, proc_val = _stub_process_query("x")
    with proc_patch as proc, \
         patch("src.retrieval.pipeline.rag_chain.RAG_DOCUMENT_ROUTING_ENABLED", True), \
         patch("src.retrieval.pipeline.rag_chain.decompose_query",
               side_effect=RuntimeError("boom")), \
         patch("src.retrieval.pipeline.rag_chain.route_documents") as rd:
        proc.return_value = proc_val
        response = chain.run(query="x", skip_generation=True)

    assert response is not None
    assert cc.seen_routed == [None]
    rd.assert_not_called()
