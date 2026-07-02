# @summary
# Unit tests for the turn-loop retrieval primitive: RAGChain.retrieve_primitive
# (hyde-vs-query embed selection through the shared LRU, rerank anchored to
# query_text not the HyDE text, stable weaviate-uuid chunk_id in the output
# dicts, top_k truncation, EvidenceChunk-shaped output, and — loop rule — no
# ignored_doc_ids suppression filter ever injected) plus request validation of
# the retrieve_ranked Temporal activity (ValueError = non-retryable).
# Deps: pytest, src.retrieval.pipeline.rag_chain.RAGChain, server.activities,
#       src.vector_db.common.schemas, src.retrieval.common.schemas
# @end-summary
"""Tests for ``RAGChain.retrieve_primitive`` and the ``retrieve_ranked`` activity.

Follows the bare-chain pattern of ``tests/retrieval/test_role_query_filter.py``:
``object.__new__(RAGChain)`` with stubbed embed/search/rerank collaborators —
no Weaviate, no GPU models, fully deterministic. The activity tests call the
``@activity.defn`` function directly (it is a plain sync function) against a
patched chain singleton.
"""
from __future__ import annotations

from collections import OrderedDict

import pytest

import server.activities as activities
from src.retrieval.common.schemas import RankedResult
from src.retrieval.pipeline.rag_chain import RAGChain
from src.vector_db.common.schemas import SearchResult


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #


class _DummySpan:
    def set_attribute(self, *_a, **_kw):
        return None

    def end(self, *_a, **_kw):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class _DummyTracer:
    def span(self, *_a, **_kw):
        return _DummySpan()


class _StubEmbeddings:
    """Records every text embedded; returns a distinct vector per text."""

    def __init__(self):
        self.calls: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.calls.append(text)
        return [float(len(text)), 1.0]


class _StubReranker:
    """Records the rerank anchor query; preserves input order, caps to top_k."""

    def __init__(self):
        self.queries: list[str] = []

    def rerank(self, *, query: str, documents: list, top_k: int) -> list[RankedResult]:
        self.queries.append(query)
        return [
            RankedResult(text=d.text, score=1.0 - i * 0.01, metadata=d.metadata)
            for i, d in enumerate(documents[:top_k])
        ]


class _PassthroughRetry:
    """Retry-provider stub: executes the wrapped fn once, no policy logic."""

    def execute(self, *, operation_name, fn, policy, idempotency_key):
        return fn()


# Body text long enough that the thin/nav candidate filter never drops it.
_BODY = (
    "The verification environment is created by instantiating the base test "
    "class and connecting the interface agents to the DUT signals. Each agent "
    "drives stimulus through its sequencer while the scoreboard compares the "
    "observed transactions against the reference model predictions in detail."
)


def _hit(i: int, *, chunk_id: bool = True) -> SearchResult:
    """Build one fake hybrid-search hit with full turn-loop metadata."""
    md = {
        "source": f"doc{i}.pdf",
        "source_key": f"corpus/doc{i}.pdf",
        "document_id": f"docid-{i}",
        "heading": f"Section {i}",
        "chunk_index": i,
        "refactored_char_start": 100 + i,
        "refactored_char_end": 500 + i,
    }
    if chunk_id:
        md["chunk_id"] = f"00000000-0000-0000-0000-00000000000{i}"
    return SearchResult(text=f"{_BODY} Variant {i}.", score=0.9 - i * 0.05, metadata=md)


def _make_chain(hits: list[SearchResult]) -> tuple[RAGChain, dict]:
    """Bare RAGChain with stubbed collaborators; returns (chain, capture box)."""
    box: dict = {"search_calls": []}
    chain = object.__new__(RAGChain)
    chain.tracer = _DummyTracer()
    chain.embeddings = _StubEmbeddings()
    chain.reranker = _StubReranker()
    chain.retry_provider = _PassthroughRetry()
    chain.retry_policy = object()
    chain._embedding_cache = OrderedDict()
    chain._embedding_cache_max = 128

    def _fake_do_search(bm25_query, query_embedding, alpha, search_limit, filters):
        box["search_calls"].append(
            {
                "bm25_query": bm25_query,
                "embedding": query_embedding,
                "alpha": alpha,
                "limit": search_limit,
                "filters": filters,
            }
        )
        return list(hits)

    chain._do_search = _fake_do_search
    return chain, box


# --------------------------------------------------------------------------- #
# retrieve_primitive — embed selection
# --------------------------------------------------------------------------- #


def test_embeds_query_text_when_no_hyde():
    chain, _ = _make_chain([_hit(1)])
    chain.retrieve_primitive("how to build a verification environment", None, top_k=3)
    assert chain.embeddings.calls == ["how to build a verification environment"]


def test_embeds_hyde_text_when_given():
    chain, box = _make_chain([_hit(1)])
    chain.retrieve_primitive(
        "how to build a verification environment",
        "You create a UVM environment by extending uvm_env and ...",
        top_k=3,
    )
    # Dense side follows the hypothetical answer ...
    assert chain.embeddings.calls == [
        "You create a UVM environment by extending uvm_env and ..."
    ]
    # ... while the BM25/lexical side stays anchored to the user's query.
    assert box["search_calls"][0]["bm25_query"] == (
        "how to build a verification environment"
    )


def test_embedding_lru_keyed_on_embedded_text():
    # The HyDE vector must be cached under the HyDE text, never under the
    # query text — a follow-up plain-query call must embed the query fresh.
    chain, _ = _make_chain([_hit(1)])
    chain.retrieve_primitive("q anchor", "hypothetical answer", top_k=2)
    chain.retrieve_primitive("q anchor", None, top_k=2)
    assert chain.embeddings.calls == ["hypothetical answer", "q anchor"]
    # Exact repeats hit the LRU (no new embed calls).
    chain.retrieve_primitive("q anchor", "hypothetical answer", top_k=2)
    assert chain.embeddings.calls == ["hypothetical answer", "q anchor"]


# --------------------------------------------------------------------------- #
# retrieve_primitive — rerank anchor
# --------------------------------------------------------------------------- #


def test_rerank_anchored_to_query_text_not_hyde():
    chain, _ = _make_chain([_hit(1), _hit(2)])
    chain.retrieve_primitive("the real user query", "a hypothetical answer", top_k=2)
    assert chain.reranker.queries == ["the real user query"]


# --------------------------------------------------------------------------- #
# retrieve_primitive — chunk identity
# --------------------------------------------------------------------------- #


def test_chunk_id_is_weaviate_uuid_and_stable():
    chain, _ = _make_chain([_hit(1), _hit(2)])
    first = chain.retrieve_primitive("q", None, top_k=2)
    second = chain.retrieve_primitive("q", None, top_k=2)
    assert [c["chunk_id"] for c in first] == [
        "00000000-0000-0000-0000-000000000001",
        "00000000-0000-0000-0000-000000000002",
    ]
    # Stable across calls — this is the pool dedup key.
    assert [c["chunk_id"] for c in first] == [c["chunk_id"] for c in second]


def test_chunk_id_falls_back_deterministically_without_uuid():
    # A hit whose metadata lacks the plumbed uuid still gets a non-empty,
    # call-stable id (the same metadata composite the agentic dedup uses).
    chain, _ = _make_chain([_hit(3, chunk_id=False)])
    first = chain.retrieve_primitive("q", None, top_k=1)
    second = chain.retrieve_primitive("q", None, top_k=1)
    assert first[0]["chunk_id"]
    assert first[0]["chunk_id"] == second[0]["chunk_id"]


# --------------------------------------------------------------------------- #
# retrieve_primitive — shape and truncation
# --------------------------------------------------------------------------- #


def test_top_k_truncation():
    chain, _ = _make_chain([_hit(i) for i in range(1, 7)])
    chunks = chain.retrieve_primitive("q", None, top_k=2)
    assert len(chunks) == 2


def test_output_shape_matches_evidence_chunk_contract():
    # Exactly EvidenceChunk minus provenance/round_added (schemas.py contract).
    chain, _ = _make_chain([_hit(1)])
    (chunk,) = chain.retrieve_primitive("q", None, top_k=1)
    assert set(chunk) == {
        "chunk_id",
        "document_id",
        "source_key",
        "source",
        "heading",
        "text",
        "score",
        "refactored_char_start",
        "refactored_char_end",
    }
    assert chunk["document_id"] == "docid-1"
    assert chunk["source_key"] == "corpus/doc1.pdf"
    assert chunk["source"] == "doc1.pdf"
    assert chunk["heading"] == "Section 1"
    assert chunk["refactored_char_start"] == 101
    assert chunk["refactored_char_end"] == 501
    assert isinstance(chunk["score"], float)


# --------------------------------------------------------------------------- #
# retrieve_primitive — filters (loop rule: no suppression)
# --------------------------------------------------------------------------- #


def test_no_suppression_filter_by_default():
    chain, box = _make_chain([_hit(1)])
    chain.retrieve_primitive("q", None, top_k=1)
    assert box["search_calls"][0]["filters"] is None


def test_scoping_filters_present_but_never_not_in():
    chain, box = _make_chain([_hit(1)])
    chain.retrieve_primitive(
        "q", None, top_k=1, source_filter="doc1.pdf", heading_filter="Section 1"
    )
    filters = box["search_calls"][0]["filters"]
    assert filters is not None
    assert {(f.property, f.operator) for f in filters} == {
        ("source", "eq"),
        ("heading", "eq"),
    }
    # Loop-level rule (design §5): no ignored_doc_ids hard suppression, ever.
    assert all(f.operator != "not_in" for f in filters)
    assert all(f.property != "document_id" for f in filters)


# --------------------------------------------------------------------------- #
# retrieve_primitive — validation
# --------------------------------------------------------------------------- #


def test_empty_query_text_raises_value_error():
    chain, _ = _make_chain([_hit(1)])
    with pytest.raises(ValueError):
        chain.retrieve_primitive("   ", None, top_k=1)


def test_non_positive_top_k_raises_value_error():
    chain, _ = _make_chain([_hit(1)])
    with pytest.raises(ValueError):
        chain.retrieve_primitive("q", None, top_k=0)


# --------------------------------------------------------------------------- #
# retrieve_ranked activity — request validation + passthrough
# --------------------------------------------------------------------------- #


class _StubChain:
    def __init__(self):
        self.calls: list[dict] = []

    def retrieve_primitive(self, query_text, hyde_text=None, *, top_k,
                           source_filter=None, heading_filter=None):
        self.calls.append(
            {
                "query_text": query_text,
                "hyde_text": hyde_text,
                "top_k": top_k,
                "source_filter": source_filter,
                "heading_filter": heading_filter,
            }
        )
        return [{"chunk_id": "u1", "text": "t", "score": 0.5}]


@pytest.fixture
def stub_chain(monkeypatch):
    chain = _StubChain()
    monkeypatch.setattr(activities, "_rag_chain", chain)
    return chain


def test_activity_happy_path_shape(stub_chain):
    result = activities.retrieve_ranked(
        {
            "query_text": "q",
            "hyde_text": "h",
            "top_k": 4,
            "source_filter": "s.pdf",
            "heading_filter": "H",
        }
    )
    assert set(result) == {"chunks", "timings"}
    assert result["chunks"] == [{"chunk_id": "u1", "text": "t", "score": 0.5}]
    assert "total_ms" in result["timings"]
    assert stub_chain.calls == [
        {
            "query_text": "q",
            "hyde_text": "h",
            "top_k": 4,
            "source_filter": "s.pdf",
            "heading_filter": "H",
        }
    ]


def test_activity_top_k_defaults_from_settings(stub_chain):
    from config.settings import RAG_TURN_LOOP_RETRIEVE_TOP_K

    activities.retrieve_ranked({"query_text": "q"})
    assert stub_chain.calls[0]["top_k"] == RAG_TURN_LOOP_RETRIEVE_TOP_K
    assert stub_chain.calls[0]["hyde_text"] is None


@pytest.mark.parametrize(
    "request_body",
    [
        {},
        {"query_text": ""},
        {"query_text": "   "},
        {"query_text": 42},
        {"query_text": "q", "top_k": 0},
        {"query_text": "q", "top_k": -3},
        {"query_text": "q", "top_k": True},
        {"query_text": "q", "top_k": "5"},
        {"query_text": "q", "hyde_text": 7},
        {"query_text": "q", "source_filter": ["a"]},
        {"query_text": "q", "heading_filter": 1.5},
    ],
)
def test_activity_invalid_request_raises_value_error(stub_chain, request_body):
    with pytest.raises(ValueError):
        activities.retrieve_ranked(request_body)


def test_activity_empty_optional_strings_normalize_to_none(stub_chain):
    activities.retrieve_ranked(
        {"query_text": "q", "hyde_text": "  ", "source_filter": "", "heading_filter": None}
    )
    call = stub_chain.calls[0]
    assert call["hyde_text"] is None
    assert call["source_filter"] is None
    assert call["heading_filter"] is None
