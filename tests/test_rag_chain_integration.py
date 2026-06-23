import pytest

from src.retrieval.query.schemas import QueryAction, QueryResult
from src.retrieval.pipeline.rag_chain import RAGChain


class _DummySpan:
    def set_attribute(self, key, value):
        return None

    def end(self, status="ok", error=None):
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class _DummyTracer:
    def span(self, name, attributes=None, parent=None):
        return _DummySpan()

    def start_span(self, name, attrs=None, parent=None):
        return _DummySpan()


class _DummyRetryProvider:
    def execute(self, operation_name, fn, policy, idempotency_key):
        return fn()


def _build_chain_without_model_init() -> RAGChain:
    from collections import OrderedDict
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
    chain._guardrails_input_executor = None
    chain._guardrails_output_executor = None
    chain._guardrails_merge_gate = None
    chain._visual_retrieval_enabled = False
    chain._embedding_cache = OrderedDict()
    chain._embedding_cache_max = 128
    return chain


def test_rag_chain_returns_ask_user(monkeypatch):
    monkeypatch.setattr(
        "src.retrieval.pipeline.rag_chain.process_query",
        lambda *args, **kwargs: QueryResult(
            processed_query="x",
            confidence=0.1,
            action=QueryAction.ASK_USER,
            clarification_message="clarify",
            iterations=1,
        ),
    )
    chain = _build_chain_without_model_init()
    response = chain.run("x")
    assert response.action == "ask_user"
    assert response.clarification_message == "clarify"


# ---------------------------------------------------------------------------
# Hybrid search alpha blending tests
# ---------------------------------------------------------------------------


class _DummyCtxMgr:
    """Context manager that returns a dummy Weaviate client."""

    def __enter__(self):
        return object()

    def __exit__(self, *_args):
        return False


@pytest.mark.parametrize("alpha,label", [
    (0.0, "BM25 only"),
    (1.0, "vector only"),
    (0.5, "balanced blend"),
])
def test_do_search_passes_alpha_to_search(monkeypatch, alpha, label):
    """alpha value must be forwarded to the search layer unchanged."""
    captured = {}

    def _fake_search(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr("src.retrieval.pipeline.rag_chain.search", _fake_search)
    monkeypatch.setattr("src.retrieval.pipeline.rag_chain.get_client", _DummyCtxMgr)
    monkeypatch.setattr("src.retrieval.pipeline.rag_chain.ensure_collection", lambda *a, **k: None)

    chain = _build_chain_without_model_init()
    chain._do_search("test query", [0.1, 0.2, 0.3], alpha=alpha, search_limit=5, filters=None)

    assert abs(captured.get("alpha") - alpha) < 1e-9, f"{label}: expected alpha={alpha}"


# ---------------------------------------------------------------------------
# Citation indexing — the returned results expose the SAME 1-based number the
# generator cites ([N] in build_messages), in post-rerank order.
# ---------------------------------------------------------------------------

def test_stamp_citation_indices_marks_1based_position_in_order():
    from src.retrieval.common.schemas import RankedResult

    results = [
        RankedResult(text="a", score=0.9, metadata={"source": "x.pdf"}),
        RankedResult(text="b", score=0.8, metadata={"source": "y.pdf"}),
        RankedResult(text="c", score=0.7, metadata={"source": "z.pdf"}),
    ]
    RAGChain._stamp_citation_indices(results)
    assert [r.metadata["citation_index"] for r in results] == [1, 2, 3]


def test_stamp_citation_indices_skips_non_dict_metadata():
    """A result whose metadata isn't a dict must not raise — it's just skipped."""
    class _Weird:
        metadata = None  # not a dict

    from src.retrieval.common.schemas import RankedResult
    ok = RankedResult(text="a", score=0.5, metadata={"source": "x.pdf"})
    RAGChain._stamp_citation_indices([_Weird(), ok])
    assert ok.metadata["citation_index"] == 2  # position preserved, weird one skipped


def test_stamp_citation_indices_restamps_replacement_retry_list():
    """The confidence-routing retry path reassigns ``reranked = retry_reranked`` —
    a freshly reranked list that was never stamped. The final pre-return re-stamp
    must (re)number the NEW list 1..N so the returned results still match the [N]
    the model cited in the retry answer, with no stale index leaking through.
    """
    from src.retrieval.common.schemas import RankedResult

    # Primary list stamped earlier in the pipeline (Stage 5.5).
    primary = [
        RankedResult(text="p1", score=0.9, metadata={"source": "p.pdf"}),
        RankedResult(text="p2", score=0.8, metadata={"source": "q.pdf"}),
    ]
    RAGChain._stamp_citation_indices(primary)

    # Retry produced a different, UNSTAMPED list that replaces `reranked`.
    retry = [
        RankedResult(text="r1", score=0.95, metadata={"source": "a.pdf"}),
        RankedResult(text="r2", score=0.85, metadata={"source": "b.pdf"}),
        RankedResult(text="r3", score=0.75, metadata={"source": "c.pdf"}),
    ]
    reranked = retry  # reassignment, as in the retry branch (rag_chain.py ~2815)

    # The final pre-return re-stamp (the fix) runs unconditionally on `reranked`.
    RAGChain._stamp_citation_indices(reranked)

    assert [r.metadata["citation_index"] for r in reranked] == [1, 2, 3]
    # The returned (retry) list is numbered 1..N by position — exactly what the
    # generator emitted as [N] when it numbered ``retry_reranked``.
    assert all(r.metadata["citation_index"] == i + 1 for i, r in enumerate(reranked))
