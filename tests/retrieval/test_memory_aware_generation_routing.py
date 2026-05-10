"""Tests for memory-aware generation routing: fallback retrieval, memory-generation path,
suppress-memory routing, BLOCK/FLAG filtering, and generation source tracking."""

import pytest
from collections import OrderedDict
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from src.retrieval.common.schemas import RAGResponse, RankedResult
from src.retrieval.query.schemas import QueryResult, QueryAction


# ---------------------------------------------------------------------------
# Test infrastructure helpers
# ---------------------------------------------------------------------------

class _DummySpan:
    """No-op span that satisfies both context-manager and attribute-setter contracts."""

    def set_attribute(self, key, value):
        return None

    def end(self, status="ok", error=None):
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class _DummyTracer:
    """No-op tracer whose span() acts as a context manager, matching the real backend API."""

    @contextmanager
    def span(self, name, attributes=None, parent=None):
        yield _DummySpan()


class _DummyRetryProvider:
    """Pass-through retry provider: executes the callable once without back-off."""

    def execute(self, operation_name, fn, policy, idempotency_key):
        return fn()


def _build_chain():
    """Construct a RAGChain whose heavy dependencies are all stubbed out.

    Uses object.__new__ to skip __init__ entirely, then manually populates only
    the attributes that chain.run() reads during the routing tests below.
    """
    from src.retrieval.pipeline.rag_chain import RAGChain

    chain = object.__new__(RAGChain)
    chain.tracer = _DummyTracer()
    chain.retry_provider = _DummyRetryProvider()
    chain.retry_policy = object()
    chain._persistent_weaviate = False
    chain._weaviate_client = None
    chain._kg_expander = None
    chain._generator = None
    chain.embeddings = None
    chain.reranker = None
    chain._guardrails_merge_gate = None
    chain._embedding_cache = OrderedDict()
    chain._embedding_cache_max = 128
    chain._visual_retrieval_enabled = False
    chain._visual_model = None
    chain._visual_processor = None
    return chain


def _strong_query_result(**overrides) -> QueryResult:
    """Return a QueryResult that represents a strong, self-contained retrieval query."""
    defaults = dict(
        processed_query="What is the SPI clock frequency?",
        confidence=0.92,
        action=QueryAction.SEARCH,
        standalone_query="What is the SPI clock frequency?",
        suppress_memory=False,
        has_backward_reference=False,
        iterations=1,
    )
    defaults.update(overrides)
    return QueryResult(**defaults)


def _make_strong_reranked() -> list:
    """Return a list of RankedResult objects that classify as 'strong' retrieval quality."""
    # RAG_RETRIEVAL_QUALITY_STRONG_THRESHOLD is 0.7 by default; score 0.85 exceeds it.
    return [RankedResult(text="SPI runs at 50 MHz.", score=0.85, metadata={"source": "ds.pdf"})]


# ---------------------------------------------------------------------------
# Shared process_query and search mock targets
# ---------------------------------------------------------------------------

_PROCESS_QUERY_TARGET = "src.retrieval.pipeline.rag_chain.process_query"
_SEARCH_TARGET = "src.retrieval.pipeline.rag_chain.search"
_ENSURE_COLLECTION_TARGET = "src.retrieval.pipeline.rag_chain.ensure_collection"
_GET_CLIENT_TARGET = "src.retrieval.pipeline.rag_chain.get_client"


# ---------------------------------------------------------------------------
# Class 1: RAGResponse.generation_source schema tests
# ---------------------------------------------------------------------------

class TestRAGResponseGenerationSource:
    """Verify the generation_source field on RAGResponse (REQ-1209)."""

    def _minimal_response(self, **kwargs) -> RAGResponse:
        """Build the smallest valid RAGResponse, optionally overriding fields."""
        defaults = dict(
            query="q",
            processed_query="q",
            query_confidence=0.5,
            action="search",
        )
        defaults.update(kwargs)
        return RAGResponse(**defaults)

    def test_generation_source_default_none(self):
        """RAGResponse with minimal fields has generation_source=None by default."""
        resp = self._minimal_response()
        assert resp.generation_source is None

    def test_generation_source_retrieval(self):
        """generation_source can be set to 'retrieval' and is accessible."""
        resp = self._minimal_response(generation_source="retrieval")
        assert resp.generation_source == "retrieval"

    def test_generation_source_memory(self):
        """generation_source can be set to 'memory' and is accessible."""
        resp = self._minimal_response(generation_source="memory")
        assert resp.generation_source == "memory"

    def test_generation_source_retrieval_plus_memory(self):
        """generation_source can be set to 'retrieval+memory' and is accessible."""
        resp = self._minimal_response(generation_source="retrieval+memory")
        assert resp.generation_source == "retrieval+memory"


# ---------------------------------------------------------------------------
# Class 2: Generation source routing decision tests
# ---------------------------------------------------------------------------

class TestGenerationSourceRouting:
    """Verify the routing logic that sets generation_source inside chain.run().

    Each test stubs out process_query, search, reranker, and generator to
    control the exact conditions that drive the routing decision, then asserts
    on the generation_source field of the returned RAGResponse.
    """

    # ------------------------------------------------------------------
    # Helper — patches common to every routing test
    # ------------------------------------------------------------------

    def _run_with_mocks(
        self,
        monkeypatch,
        query_result: QueryResult,
        reranked_results: list,
        *,
        generator_answer: str = "Generated answer.",
        memory_context: str = None,
        memory_recent_turns: list = None,
    ) -> RAGResponse:
        """Wire all heavy collaborators to stubs and invoke chain.run()."""
        chain = _build_chain()

        # Stub process_query
        monkeypatch.setattr(_PROCESS_QUERY_TARGET, lambda *a, **kw: query_result)

        # Stub embedding (returns a dummy vector so search can be called)
        dummy_embeddings = MagicMock()
        dummy_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
        chain.embeddings = dummy_embeddings

        # Stub search (returns raw rows, will be converted by _ranked_from_search_results)
        raw_rows = [
            MagicMock(text=r.text, score=r.score, metadata=r.metadata)
            for r in reranked_results
        ]
        monkeypatch.setattr(_SEARCH_TARGET, lambda **kw: raw_rows)
        monkeypatch.setattr(_ENSURE_COLLECTION_TARGET, lambda *a, **kw: None)

        # Stub reranker (identity — keeps the same scores)
        dummy_reranker = MagicMock()
        dummy_reranker.rerank.return_value = reranked_results
        chain.reranker = dummy_reranker

        # Stub generator
        dummy_generator = MagicMock()
        dummy_generator.is_available.return_value = True
        dummy_generator.model = "llama3"
        dummy_generator.generate.return_value = generator_answer
        chain._generator = dummy_generator

        return chain.run(
            "test query",
            memory_context=memory_context,
            memory_recent_turns=memory_recent_turns,
            fast_path=True,
        )

    # ------------------------------------------------------------------
    # Suppress-memory path (REQ-1205)
    # ------------------------------------------------------------------

    def test_suppress_memory_source_retrieval(self, monkeypatch):
        """When suppress_memory=True and retrieval succeeds, source must be 'retrieval'.

        REQ-1205: context reset strips all memory from generation;
        generation_source reflects that only retrieved documents were used.
        """
        qr = _strong_query_result(suppress_memory=True)
        resp = self._run_with_mocks(
            monkeypatch,
            qr,
            _make_strong_reranked(),
            memory_context="prior summary",
            memory_recent_turns=[{"role": "user", "content": "hi"}],
        )
        assert resp.generation_source == "retrieval"

    # ------------------------------------------------------------------
    # Strong retrieval paths (REQ-1204)
    # ------------------------------------------------------------------

    def test_strong_retrieval_with_backward_ref_source(self, monkeypatch):
        """Strong retrieval + backward_ref=True → generation_source='retrieval+memory'.

        REQ-1204: Hybrid path — doc retrieval succeeded AND the query refers
        back to prior turns; both document context and memory are used.
        """
        qr = _strong_query_result(has_backward_reference=True)
        resp = self._run_with_mocks(
            monkeypatch,
            qr,
            _make_strong_reranked(),
            memory_context="Summary of prior turns.",
            memory_recent_turns=[{"role": "user", "content": "Tell me more"}],
        )
        assert resp.generation_source == "retrieval+memory"

    def test_strong_retrieval_no_backward_ref_with_memory(self, monkeypatch):
        """Strong retrieval + no backward ref + memory present → 'retrieval+memory'.

        Standard retrieval path: when memory context is available it is included
        in generation, so the source tracks both inputs.
        """
        qr = _strong_query_result(has_backward_reference=False)
        resp = self._run_with_mocks(
            monkeypatch,
            qr,
            _make_strong_reranked(),
            memory_context="Previous conversation summary.",
        )
        assert resp.generation_source == "retrieval+memory"

    def test_strong_retrieval_no_backward_ref_no_memory(self, monkeypatch):
        """Strong retrieval + no backward ref + no memory → source='retrieval'.

        Standard retrieval path with no memory context: generation draws
        only from retrieved documents.
        """
        qr = _strong_query_result(has_backward_reference=False)
        resp = self._run_with_mocks(
            monkeypatch,
            qr,
            _make_strong_reranked(),
            memory_context=None,
            memory_recent_turns=None,
        )
        assert resp.generation_source == "retrieval"

    # ------------------------------------------------------------------
    # Weak retrieval paths (REQ-1203)
    # ------------------------------------------------------------------

    def _weak_reranked(self) -> list:
        """Return RankedResult objects with scores below the strong/moderate thresholds."""
        # RAG_RETRIEVAL_QUALITY_WEAK_THRESHOLD is 0.4 by default; 0.45 is weak.
        return [RankedResult(text="Loosely related doc.", score=0.45, metadata={"source": "x.pdf"})]

    def test_weak_retrieval_backward_ref_memory_present(self, monkeypatch):
        """Weak retrieval + backward_ref + non-empty memory → source='memory'.

        REQ-1203: Memory-generation path — weak doc retrieval falls back to
        generating exclusively from conversation history.
        """
        qr = _strong_query_result(has_backward_reference=True)
        resp = self._run_with_mocks(
            monkeypatch,
            qr,
            self._weak_reranked(),
            memory_context="Prior topic summary.",
            memory_recent_turns=[{"role": "user", "content": "And that?"}],
        )
        assert resp.generation_source == "memory"

    def test_weak_retrieval_backward_ref_empty_memory(self, monkeypatch):
        """Weak retrieval + backward_ref + empty memory → BLOCK (generation_source=None).

        REQ-1203 guard: a backward-reference on a fresh conversation has no
        memory to fall back to; the pipeline emits a deterministic block answer
        and leaves generation_source as None.
        """
        qr = _strong_query_result(has_backward_reference=True)
        resp = self._run_with_mocks(
            monkeypatch,
            qr,
            self._weak_reranked(),
            memory_context=None,
            memory_recent_turns=None,
        )
        assert resp.generation_source is None

    def test_weak_retrieval_no_backward_ref(self, monkeypatch):
        """Weak retrieval + no backward ref → source='retrieval' (FLAG/suppress path).

        Per spec line 91: weak retrieval without a backward reference suppresses
        recent_turns but still uses retrieval docs; generation_source is 'retrieval'.
        """
        qr = _strong_query_result(has_backward_reference=False)
        resp = self._run_with_mocks(
            monkeypatch,
            qr,
            self._weak_reranked(),
            memory_context=None,
            memory_recent_turns=None,
        )
        assert resp.generation_source == "retrieval"


# ---------------------------------------------------------------------------
# Class 3: BLOCK / FLAG post-guardrail action schema contract tests
# ---------------------------------------------------------------------------

class TestBlockFlagMemoryContract:
    """Verify that RAGResponse can carry BLOCK and FLAG post-guardrail actions.

    These are schema/contract tests only: they confirm the field exists and
    accepts the documented values, without requiring the full confidence-routing
    pipeline to run.
    """

    def _response_with_action(self, action: str) -> RAGResponse:
        return RAGResponse(
            query="q",
            processed_query="q",
            query_confidence=0.5,
            action="search",
            post_guardrail_action=action,
        )

    def test_block_response_has_post_guardrail_action(self):
        """RAGResponse.post_guardrail_action can express 'block'."""
        resp = self._response_with_action("block")
        assert resp.post_guardrail_action == "block"

    def test_flag_response_has_post_guardrail_action(self):
        """RAGResponse.post_guardrail_action can express 'flag'."""
        resp = self._response_with_action("flag")
        assert resp.post_guardrail_action == "flag"

    def test_post_guardrail_action_default_none(self):
        """RAGResponse.post_guardrail_action defaults to None when not provided."""
        resp = RAGResponse(
            query="q",
            processed_query="q",
            query_confidence=0.5,
            action="search",
        )
        assert resp.post_guardrail_action is None

    def test_block_and_generation_source_coexist(self):
        """A response can simultaneously carry generation_source and post_guardrail_action."""
        resp = RAGResponse(
            query="q",
            processed_query="q",
            query_confidence=0.4,
            action="search",
            generated_answer=(
                "Insufficient documentation found to provide a reliable answer. "
                "Please try a more specific query."
            ),
            post_guardrail_action="block",
            generation_source="retrieval",
        )
        assert resp.post_guardrail_action == "block"
        assert resp.generation_source == "retrieval"
        assert resp.generated_answer is not None

    def test_flag_with_memory_source_coexist(self):
        """FLAG action is compatible with generation_source='memory' on the memory path."""
        resp = RAGResponse(
            query="Tell me more",
            processed_query="Tell me more",
            query_confidence=0.35,
            action="search",
            generated_answer="Based on our conversation...",
            post_guardrail_action="flag",
            generation_source="memory",
        )
        assert resp.post_guardrail_action == "flag"
        assert resp.generation_source == "memory"
