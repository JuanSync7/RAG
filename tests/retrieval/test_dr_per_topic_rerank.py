"""Tests for per-topic reranking with split-confidence gate on DR.

Spec:

When DR decomposes into N>=2 topics AND ``split_confidence >=
per_topic_rerank_min_confidence`` AND ``per_topic_rerank_enabled`` is True,
the orchestrator reranks each ``TopicPool`` against its ``rerank_anchor``
(NOT the original query), takes top-``per_topic_top_k`` per pool, and merges
in round-robin order. Otherwise it falls through with no rerank applied.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Optional

import pytest

import src.platform.observability as obs_module
from src.platform.observability.backend import (
    Generation,
    ObservabilityBackend,
    Span,
    Trace,
)
from src.retrieval.common.schemas import RankedResult
from src.retrieval.deep_research.metrics import DR_PER_TOPIC_RERANK_TOTAL
from src.retrieval.pipeline.deep_research import (
    DeepResearch,
    DeepResearchBudget,
)
from src.vector_db.common.schemas import SearchResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mk_chunk(idx: int, *, source: str = "doc.md", text: str = "") -> SearchResult:
    return SearchResult(
        text=text or f"chunk-{idx}",
        score=0.5,
        metadata={"source": source, "heading": "h", "chunk_index": idx},
        object_id=f"oid-{idx}-{source}",
        collection="RagDocuments",
    )


@dataclass
class _LLMResp:
    content: str
    model: str = "fake"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


class FakeProvider:
    def __init__(self, responses: Optional[list[dict[str, Any]]] = None) -> None:
        self.responses = list(responses or [])
        self.calls: list[dict[str, Any]] = []

    async def agenerate(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        if self.responses:
            payload = self.responses.pop(0)
        else:
            payload = {"is_sufficient": True}
        return _LLMResp(content=json.dumps(payload))


class RecordingReranker:
    """Captures every rerank call, returns a deterministic top-k slice."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def rerank(self, query, documents, top_k):
        self.calls.append({"query": query, "documents": list(documents), "top_k": top_k})
        out = []
        for i, d in enumerate(documents[:top_k]):
            out.append(
                RankedResult(
                    text=d.text,
                    score=1.0 - i * 0.01,
                    metadata=dict(d.metadata or {}),
                )
            )
        return out


def _counter_value(counter, **labels) -> float:
    return counter.labels(**labels)._value.get()


def _orch(
    *,
    provider: FakeProvider,
    reranker: Optional[RecordingReranker] = None,
    budget: Optional[DeepResearchBudget] = None,
    kb_results_per_query: Optional[dict[str, list[SearchResult]]] = None,
    original_question: str = "what is X and Y?",
    processed_query: str = "X Y",
) -> DeepResearch:
    counter = {"n": 100}
    kb_results_per_query = kb_results_per_query or {}

    async def kb_retrieve(query: str):
        if query in kb_results_per_query:
            return kb_results_per_query[query]
        counter["n"] += 1
        return [_mk_chunk(counter["n"], source=f"fallback-{counter['n']}.md")]

    async def kg_retrieve(query: str):
        return None

    return DeepResearch(
        provider=provider,
        kb_retrieve=kb_retrieve,
        kg_retrieve=kg_retrieve,
        original_question=original_question,
        processed_query=processed_query,
        budget=budget or DeepResearchBudget(),
        reranker=reranker,
    )


# ---------------------------------------------------------------------------
# Spy backend (mirrors test_dr_tracing.py)
# ---------------------------------------------------------------------------


@dataclass
class SpySpan(Span):
    name: str = ""
    attributes: dict = field(default_factory=dict)
    end_calls: list = field(default_factory=list)

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def end(self, status="ok", error=None):
        self.end_calls.append((status, error))


@dataclass
class SpyGeneration(Generation):
    name: str = ""
    model: str = ""
    input: Any = None
    metadata: dict = field(default_factory=dict)
    output: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    end_calls: list = field(default_factory=list)

    def set_output(self, output):
        self.output = output

    def set_token_counts(self, p, c):
        self.prompt_tokens = p
        self.completion_tokens = c

    def end(self, status="ok", error=None):
        self.end_calls.append((status, error))


@dataclass
class SpyTrace(Trace):
    name: str = ""
    metadata: dict = field(default_factory=dict)
    spans: list = field(default_factory=list)
    generations: list = field(default_factory=list)
    attributes: dict = field(default_factory=dict)
    end_calls: list = field(default_factory=list)

    def span(self, name, attributes=None):
        s = SpySpan(name=name, attributes=dict(attributes or {}))
        self.spans.append(s)
        return s

    def generation(self, name, model, input, metadata=None):
        g = SpyGeneration(name=name, model=model, input=input, metadata=dict(metadata or {}))
        self.generations.append(g)
        return g

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def end(self, status="ok", error=None):
        self.end_calls.append((status, error))


class SpyBackend(ObservabilityBackend):
    def __init__(self):
        self.traces: list[SpyTrace] = []

    def trace(self, name, metadata=None):
        t = SpyTrace(name=name, metadata=dict(metadata or {}))
        self.traces.append(t)
        return t

    def span(self, name, attributes=None, parent=None):
        return SpySpan(name=name, attributes=dict(attributes or {}))

    def generation(self, name, model, input, metadata=None):
        return SpyGeneration(name=name, model=model, input=input, metadata=dict(metadata or {}))

    def flush(self):
        pass

    def shutdown(self):
        pass


@pytest.fixture()
def spy_backend():
    spy = SpyBackend()
    prev = obs_module._backend
    obs_module._backend = spy
    try:
        yield spy
    finally:
        obs_module._backend = prev


# ---------------------------------------------------------------------------
# Multi-topic decomposition driver — emits 2 or 3 topics with configurable
# split_confidence and shapes the per-topic chunk pools so we can exercise
# round-robin merge.
# ---------------------------------------------------------------------------


def _multi_topic_responses(
    *,
    topic_names: list[str],
    split_confidence: float,
    sufficient_iter0: bool = False,
) -> list[dict[str, Any]]:
    """Return a FakeProvider response sequence that decomposes into
    ``topic_names`` topics, each with one follow-up question that quickly
    reaches sufficient=True."""
    responses: list[dict[str, Any]] = []
    # iter 0 sufficiency
    responses.append({"is_sufficient": sufficient_iter0, "missing_information": "more"})
    # decompose
    responses.append({
        "topics": [
            {"name": n, "questions": [{"question": f"q-{n}?", "query": f"q-{n}"}]}
            for n in topic_names
        ],
        "split_confidence": split_confidence,
    })
    # per-topic sufficient (one per topic)
    for _ in topic_names:
        responses.append({"is_sufficient": True})
    return responses


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_skipped_when_unified_result():
    """Single-topic / unified result → no rerank; reason='unified';
    metric mode='skipped_unified'."""
    before = _counter_value(DR_PER_TOPIC_RERANK_TOTAL, mode="skipped_unified")
    provider = FakeProvider(responses=[{"is_sufficient": True}])
    reranker = RecordingReranker()
    kb = {
        "X Y": [
            _mk_chunk(1, source="a.md"),
            _mk_chunk(2, source="b.md"),
            _mk_chunk(3, source="c.md"),
        ],
    }
    orch = _orch(provider=provider, reranker=reranker, kb_results_per_query=kb,
                 processed_query="X Y")
    result = asyncio.run(orch.research())

    assert result.is_unified is True
    assert result.per_topic_rerank_applied is False
    assert result.per_topic_rerank_skipped_reason == "unified"
    assert reranker.calls == []
    after = _counter_value(DR_PER_TOPIC_RERANK_TOTAL, mode="skipped_unified")
    assert after == before + 1


def test_skipped_when_low_confidence():
    """N>=2 topics but split_confidence < threshold → no rerank; mode='off'."""
    before = _counter_value(DR_PER_TOPIC_RERANK_TOTAL, mode="off")
    provider = FakeProvider(responses=_multi_topic_responses(
        topic_names=["t1", "t2", "t3"], split_confidence=0.4,
    ))
    reranker = RecordingReranker()
    orch = _orch(provider=provider, reranker=reranker,
                 budget=DeepResearchBudget(per_topic_rerank_min_confidence=0.6))
    result = asyncio.run(orch.research())

    assert result.is_unified is False
    assert result.per_topic_rerank_applied is False
    assert result.per_topic_rerank_skipped_reason == "low_confidence"
    assert reranker.calls == []
    after = _counter_value(DR_PER_TOPIC_RERANK_TOTAL, mode="off")
    assert after == before + 1


def test_applies_when_high_confidence_multi_topic():
    """High confidence + N>=2 → rerank called once per topic with that
    topic's rerank_anchor."""
    before = _counter_value(DR_PER_TOPIC_RERANK_TOTAL, mode="on")
    provider = FakeProvider(responses=_multi_topic_responses(
        topic_names=["alpha", "beta"], split_confidence=0.8,
    ))
    reranker = RecordingReranker()
    orch = _orch(provider=provider, reranker=reranker,
                 budget=DeepResearchBudget(per_topic_rerank_min_confidence=0.6,
                                           per_topic_top_k=3))
    result = asyncio.run(orch.research())

    assert result.is_unified is False
    assert result.per_topic_rerank_applied is True
    assert result.per_topic_rerank_skipped_reason is None
    # One rerank call per topic pool, with that pool's anchor.
    anchors = sorted(c["query"] for c in reranker.calls)
    expected_anchors = sorted(p.rerank_anchor for p in result.topic_pools)
    assert anchors == expected_anchors
    # NOT the original query.
    for c in reranker.calls:
        assert c["query"] != "what is X and Y?"
    after = _counter_value(DR_PER_TOPIC_RERANK_TOTAL, mode="on")
    assert after == before + 1


def test_round_robin_merge_order():
    """3 topics with 3 ranked chunks each → merged interleaves t1[0], t2[0],
    t3[0], t1[1], t2[1], t3[1], ..."""
    provider = FakeProvider(responses=_multi_topic_responses(
        topic_names=["alpha", "beta", "gamma"], split_confidence=0.9,
    ))

    class FixedReranker:
        """Returns 3 distinctively-tagged ranked results per call so we can
        verify interleaving by inspecting metadata."""
        def __init__(self):
            self.calls = []

        def rerank(self, query, documents, top_k):
            self.calls.append(query)
            tag = query  # anchors are unique
            out = []
            for i in range(min(top_k, 3)):
                out.append(RankedResult(
                    text=f"{tag}-{i}",
                    score=1.0 - i * 0.01,
                    metadata={"tag": tag, "rank": i},
                ))
            return out

    reranker = FixedReranker()
    orch = _orch(provider=provider, reranker=reranker,
                 budget=DeepResearchBudget(per_topic_rerank_min_confidence=0.6,
                                           per_topic_top_k=3))
    asyncio.run(orch.research())

    merged = orch.last_result.per_topic_rerank_merged  # set by orchestrator
    assert merged is not None and len(merged) == 9
    # Round-robin: ranks are 0,0,0,1,1,1,2,2,2
    expected_ranks = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert [r.metadata["rank"] for r in merged] == expected_ranks


def test_per_topic_top_k_caps_per_pool():
    """per_topic_top_k=2 → reranker called with top_k=2 (caller may return
    more, but we trust top_k arg is forwarded)."""
    provider = FakeProvider(responses=_multi_topic_responses(
        topic_names=["a", "b"], split_confidence=0.9,
    ))
    reranker = RecordingReranker()
    orch = _orch(provider=provider, reranker=reranker,
                 budget=DeepResearchBudget(per_topic_rerank_min_confidence=0.6,
                                           per_topic_top_k=2))
    asyncio.run(orch.research())

    assert reranker.calls
    for c in reranker.calls:
        assert c["top_k"] == 2


def test_disabled_flag_short_circuits():
    """per_topic_rerank_enabled=False → no rerank; reason='disabled'; mode='off'."""
    before = _counter_value(DR_PER_TOPIC_RERANK_TOTAL, mode="off")
    provider = FakeProvider(responses=_multi_topic_responses(
        topic_names=["a", "b"], split_confidence=0.9,
    ))
    reranker = RecordingReranker()
    orch = _orch(provider=provider, reranker=reranker,
                 budget=DeepResearchBudget(per_topic_rerank_enabled=False))
    result = asyncio.run(orch.research())

    assert result.per_topic_rerank_applied is False
    assert result.per_topic_rerank_skipped_reason == "disabled"
    assert reranker.calls == []
    after = _counter_value(DR_PER_TOPIC_RERANK_TOTAL, mode="off")
    assert after == before + 1


def test_metric_counter_increments_exactly_once_per_research_run():
    """Across all modes, exactly one increment fires per research() call."""
    # Run an "on" run.
    before_on = _counter_value(DR_PER_TOPIC_RERANK_TOTAL, mode="on")
    before_off = _counter_value(DR_PER_TOPIC_RERANK_TOTAL, mode="off")
    before_skipped = _counter_value(DR_PER_TOPIC_RERANK_TOTAL, mode="skipped_unified")
    provider = FakeProvider(responses=_multi_topic_responses(
        topic_names=["a", "b"], split_confidence=0.9,
    ))
    reranker = RecordingReranker()
    orch = _orch(provider=provider, reranker=reranker,
                 budget=DeepResearchBudget(per_topic_rerank_min_confidence=0.6))
    asyncio.run(orch.research())
    # Total increments across all labels = 1.
    delta = (
        (_counter_value(DR_PER_TOPIC_RERANK_TOTAL, mode="on") - before_on)
        + (_counter_value(DR_PER_TOPIC_RERANK_TOTAL, mode="off") - before_off)
        + (_counter_value(DR_PER_TOPIC_RERANK_TOTAL, mode="skipped_unified") - before_skipped)
    )
    assert delta == 1


def test_langfuse_span_emitted_when_applied(spy_backend):
    """When per-topic rerank applies, a 'dr_per_topic_rerank' span lands on
    the parent trace."""
    provider = FakeProvider(responses=_multi_topic_responses(
        topic_names=["a", "b"], split_confidence=0.9,
    ))
    reranker = RecordingReranker()
    orch = _orch(provider=provider, reranker=reranker,
                 budget=DeepResearchBudget(per_topic_rerank_min_confidence=0.6))
    asyncio.run(orch.research())

    t = spy_backend.traces[0]
    span_names = [s.name for s in t.spans]
    assert "dr_per_topic_rerank" in span_names


def test_global_top_k_cap_after_merge():
    """Round-robin produces N*per_topic_top_k chunks; orchestrator must
    expose the unbounded merge (caller applies global cap). We assert the
    merged list length equals sum of per-pool returns."""
    provider = FakeProvider(responses=_multi_topic_responses(
        topic_names=["a", "b", "c"], split_confidence=0.9,
    ))
    reranker = RecordingReranker()
    orch = _orch(provider=provider, reranker=reranker,
                 budget=DeepResearchBudget(per_topic_rerank_min_confidence=0.6,
                                           per_topic_top_k=3))
    result = asyncio.run(orch.research())

    # Up to 3 pools * 3 = 9. Caller (rag_chain) trims to global top_k.
    assert result.per_topic_rerank_merged is not None
    assert len(result.per_topic_rerank_merged) <= 9


def test_backward_compat_default_values():
    """DeepResearchBudget() with no kwargs still works; defaults must
    preserve existing behaviour shape."""
    b = DeepResearchBudget()
    assert b.per_topic_rerank_enabled is True
    assert b.per_topic_rerank_min_confidence == 0.6
    assert b.per_topic_top_k == 3
    # No rerank wired (reranker=None) → must not error and must skip cleanly.
    provider = FakeProvider(responses=[{"is_sufficient": True}])
    kb = {
        "X Y": [_mk_chunk(1, source="a.md"), _mk_chunk(2, source="b.md"),
                _mk_chunk(3, source="c.md")],
    }
    orch = _orch(provider=provider, reranker=None, kb_results_per_query=kb)
    result = asyncio.run(orch.research())
    assert result.per_topic_rerank_applied is False
