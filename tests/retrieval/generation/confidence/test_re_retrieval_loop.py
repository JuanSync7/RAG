"""Tests for re-retrieval loop semantics (option B, REQ-706).

Covers:
- High composite → no re-retrieval, immediate return.
- Medium composite with retry budget → second attempt runs, higher-scoring result returned.
- Medium composite with retry budget exhausted → FLAG with verification_warning.
- Low composite with retry budget exhausted → BLOCK.
- NaN composite → BLOCK (regression).

These tests drive the real bounded internal loop added to rag_chain.run() in
the fix/gen-rerouting-loop branch.
"""
from __future__ import annotations

import math
import unittest
from unittest.mock import MagicMock, patch

from src.retrieval.generation.confidence.routing import route_by_confidence
from src.retrieval.generation.confidence.schemas import PostGuardrailAction


class TestRouteByConfidenceLoopContract:
    """Unit tests for route_by_confidence re-retrieval loop semantics."""

    def test_high_composite_returns_immediately(self):
        """composite >= high_threshold → RETURN, no retry budget consumed."""
        action = route_by_confidence(0.80, retry_count=0, max_retries=1)
        assert action == PostGuardrailAction.RETURN

    def test_medium_composite_with_budget_triggers_re_retrieve(self):
        """composite in medium band with budget → RE_RETRIEVE."""
        action = route_by_confidence(0.60, retry_count=0, max_retries=1)
        assert action == PostGuardrailAction.RE_RETRIEVE

    def test_medium_composite_budget_exhausted_flags(self):
        """composite in medium band after retry exhausted → FLAG (not BLOCK)."""
        action = route_by_confidence(0.60, retry_count=1, max_retries=1)
        assert action == PostGuardrailAction.FLAG

    def test_low_composite_budget_exhausted_blocks(self):
        """composite below low_threshold after retry exhausted → BLOCK."""
        action = route_by_confidence(0.30, retry_count=1, max_retries=1)
        assert action == PostGuardrailAction.BLOCK

    def test_nan_composite_blocks(self):
        """NaN composite must always BLOCK — regression guard."""
        action = route_by_confidence(math.nan, retry_count=0)
        assert action == PostGuardrailAction.BLOCK

    def test_nan_composite_ignores_retry_budget(self):
        """NaN composite BLOCK even when retry budget is available."""
        action = route_by_confidence(math.nan, retry_count=0, max_retries=5)
        assert action == PostGuardrailAction.BLOCK

    def test_medium_composite_multi_retry_budget(self):
        """With max_retries=2, retry_count=1 should still re-retrieve."""
        action = route_by_confidence(0.60, retry_count=1, max_retries=2)
        assert action == PostGuardrailAction.RE_RETRIEVE

    def test_medium_composite_multi_retry_exhausted(self):
        """With max_retries=2, retry_count=2 should FLAG (medium composite)."""
        action = route_by_confidence(0.60, retry_count=2, max_retries=2)
        assert action == PostGuardrailAction.FLAG


def _make_ranked_result(score: float, text: str = "sample text"):
    from src.retrieval.common import RankedResult
    return RankedResult(text=text, score=score, metadata={"source": "doc.pdf"})


def _make_chain():
    """Build a RAGChain with all external deps stubbed out and a mock generator."""
    from src.retrieval.pipeline.rag_chain import RAGChain

    with patch("src.retrieval.pipeline.rag_chain.create_persistent_client"), \
         patch("src.retrieval.pipeline.rag_chain.ensure_collection"), \
         patch("src.retrieval.pipeline.rag_chain.get_embedding_provider"), \
         patch("src.retrieval.pipeline.rag_chain.get_reranker_provider"), \
         patch("src.retrieval.pipeline.rag_chain.OllamaGenerator"), \
         patch("src.retrieval.pipeline.rag_chain._get_kg_client"):
        chain = RAGChain(persistent_weaviate=False)

    chain.embeddings = MagicMock()
    chain.embeddings.embed_query.return_value = [0.1] * 768

    chain.reranker = MagicMock()

    span_cm = MagicMock()
    span_cm.__enter__ = MagicMock(return_value=MagicMock())
    span_cm.__exit__ = MagicMock(return_value=False)
    chain.tracer = MagicMock()
    chain.tracer.span.return_value = span_cm

    chain.retry_provider = MagicMock()
    chain._kg_expander = None
    chain._weaviate_client = None
    chain._persistent_weaviate = False

    from src.retrieval.generation.nodes import GenerationResult

    mock_gen = MagicMock()
    mock_gen.is_available.return_value = True
    mock_gen.generate.return_value = GenerationResult(
        answer="Generated answer.",
        confidence="medium",
        raw_response=None,
        error=None,
    )
    chain._generator = mock_gen

    return chain


def _make_query_result():
    from src.retrieval.query import QueryAction, QueryResult
    mock_qr = MagicMock(spec=QueryResult)
    mock_qr.action = QueryAction.SEARCH
    mock_qr.processed_query = "test query"
    mock_qr.confidence = 0.9
    mock_qr.clarification_message = None
    mock_qr.has_backward_reference = False
    mock_qr.suppress_memory = False
    mock_qr.standalone_query = None
    mock_qr.history_decision = None
    mock_qr.history_turns_used = 0
    return mock_qr


_BASE_PATCHES = [
    ("src.retrieval.pipeline.rag_chain.RAG_CONFIDENCE_ROUTING_ENABLED", True),
    ("src.retrieval.pipeline.rag_chain.RAG_CONFIDENCE_HIGH_THRESHOLD", 0.70),
    ("src.retrieval.pipeline.rag_chain.RAG_CONFIDENCE_LOW_THRESHOLD", 0.50),
    ("src.retrieval.pipeline.rag_chain.RAG_CONFIDENCE_RE_RETRIEVE_MAX_RETRIES", 1),
    ("src.retrieval.pipeline.rag_chain.GUARDRAIL_BACKEND", None),
    ("src.retrieval.pipeline.rag_chain.KG_ENABLED", False),
    ("src.retrieval.pipeline.rag_chain.GENERATION_ENABLED", True),
    ("src.retrieval.pipeline.rag_chain.RAG_DOCUMENT_FORMATTING_ENABLED", False),
    ("src.retrieval.pipeline.rag_chain.RAG_VISUAL_RETRIEVAL_ENABLED", False),
]


class TestRagChainReRetrievalLoop(unittest.TestCase):
    """Integration-style tests for the re-retrieval loop inside RAGChain.run().

    These tests mock the external I/O (search, reranker, generator) to verify
    that the loop logic inside rag_chain is correct without requiring real
    infrastructure.
    """

    def _run(self, chain, *, composite_sequence, search_results_sequence):
        """Run chain.run() with patched confidence routing and search."""
        composite_iter = iter(composite_sequence)
        search_iter = iter(search_results_sequence)

        def fake_composite(**kwargs):
            from src.retrieval.generation.confidence.schemas import ConfidenceBreakdown
            score = next(composite_iter)
            return ConfidenceBreakdown(
                retrieval_score=score, llm_score=score, citation_score=score, composite=score
            )

        def fake_search(*args, **kwargs):
            try:
                return next(search_iter)
            except StopIteration:
                return []

        chain.retry_provider.execute.side_effect = fake_search
        chain.reranker.rerank.side_effect = lambda query, documents, top_k: documents

        patches = {name: val for name, val in _BASE_PATCHES}

        with patch("src.retrieval.pipeline.rag_chain.RAG_CONFIDENCE_ROUTING_ENABLED", True), \
             patch("src.retrieval.pipeline.rag_chain.RAG_CONFIDENCE_HIGH_THRESHOLD", 0.70), \
             patch("src.retrieval.pipeline.rag_chain.RAG_CONFIDENCE_LOW_THRESHOLD", 0.50), \
             patch("src.retrieval.pipeline.rag_chain.RAG_CONFIDENCE_RE_RETRIEVE_MAX_RETRIES", 1), \
             patch("src.retrieval.pipeline.rag_chain.GUARDRAIL_BACKEND", None), \
             patch("src.retrieval.pipeline.rag_chain.KG_ENABLED", False), \
             patch("src.retrieval.pipeline.rag_chain.GENERATION_ENABLED", True), \
             patch("src.retrieval.pipeline.rag_chain.RAG_DOCUMENT_FORMATTING_ENABLED", False), \
             patch("src.retrieval.pipeline.rag_chain.RAG_VISUAL_RETRIEVAL_ENABLED", False), \
             patch("src.retrieval.pipeline.rag_chain.process_query", return_value=_make_query_result()), \
             patch("src.retrieval.generation.confidence.compute_composite_confidence", side_effect=fake_composite):
            response = chain.run("test query")

        return response

    def test_high_composite_no_reretrieval(self):
        """composite=0.80 → RETURN immediately, search called exactly once."""
        chain = _make_chain()
        result1 = [_make_ranked_result(0.9, "good doc")]

        response = self._run(
            chain,
            composite_sequence=[0.85],
            search_results_sequence=[result1],
        )

        assert response.post_guardrail_action == "return", (
            f"Expected 'return', got {response.post_guardrail_action}"
        )
        assert response.re_retrieval_suggested is False
        assert chain.retry_provider.execute.call_count == 1, (
            f"Search should be called once for high-confidence answers, "
            f"got {chain.retry_provider.execute.call_count}"
        )

    def test_medium_composite_with_budget_triggers_loop(self):
        """composite=0.60 (medium) with retry budget → loop fires, search called twice.

        The second attempt scores 0.80 (above threshold) → returned as RETURN.
        Both composite scores must be surfaced: first_composite=0.60, composite_confidence=0.80.
        """
        chain = _make_chain()
        result1 = [_make_ranked_result(0.5, "weak doc")]
        result2 = [_make_ranked_result(0.8, "stronger doc")]

        response = self._run(
            chain,
            composite_sequence=[0.60, 0.80],
            search_results_sequence=[result1, result2],
        )

        assert chain.retry_provider.execute.call_count == 2, (
            f"Search should be called twice (initial + retry), "
            f"got {chain.retry_provider.execute.call_count}"
        )
        assert response.post_guardrail_action == "return", (
            f"Second attempt scored 0.80 (above threshold), expected RETURN, "
            f"got {response.post_guardrail_action}"
        )
        assert response.first_composite == 0.60, (
            f"first_composite should be 0.60, got {response.first_composite}"
        )
        assert response.composite_confidence == 0.80, (
            f"composite_confidence should be the higher second score 0.80, "
            f"got {response.composite_confidence}"
        )

    def test_medium_composite_retry_exhausted_flags(self):
        """composite=0.60 on both attempts with max_retries=1 → FLAG with verification_warning.

        When retries are exhausted and composite is still in the medium band,
        re_retrieval_suggested must be True so the caller can request a broader pass.
        """
        chain = _make_chain()
        result1 = [_make_ranked_result(0.5, "weak doc")]
        result2 = [_make_ranked_result(0.5, "still weak doc")]

        response = self._run(
            chain,
            composite_sequence=[0.60, 0.60],
            search_results_sequence=[result1, result2],
        )

        assert response.post_guardrail_action == "flag", (
            f"Exhausted retries with medium composite → FLAG, "
            f"got {response.post_guardrail_action}"
        )
        assert response.re_retrieval_suggested is True, (
            "re_retrieval_suggested must be True when retries exhausted "
            "but composite is in medium band (caller can request broader pass)"
        )
        assert response.verification_warning is not None, (
            "verification_warning must be set when action is FLAG"
        )

    def test_low_composite_retry_exhausted_blocks(self):
        """composite=0.30 on both attempts with max_retries=1 → BLOCK."""
        chain = _make_chain()
        result1 = [_make_ranked_result(0.1, "irrelevant doc")]
        result2 = [_make_ranked_result(0.1, "still irrelevant doc")]

        response = self._run(
            chain,
            composite_sequence=[0.30, 0.30],
            search_results_sequence=[result1, result2],
        )

        assert response.post_guardrail_action == "block", (
            f"Low composite exhausted retries → BLOCK, got {response.post_guardrail_action}"
        )

    def test_nan_composite_blocks_without_retry(self):
        """NaN composite → BLOCK immediately, search called exactly once (no retry)."""
        chain = _make_chain()
        result1 = [_make_ranked_result(0.5, "some doc")]

        response = self._run(
            chain,
            composite_sequence=[float("nan")],
            search_results_sequence=[result1],
        )

        assert response.post_guardrail_action == "block", (
            f"NaN composite → BLOCK, got {response.post_guardrail_action}"
        )
        assert chain.retry_provider.execute.call_count == 1, (
            f"NaN should BLOCK without retrying — search must be called once, "
            f"got {chain.retry_provider.execute.call_count}"
        )
