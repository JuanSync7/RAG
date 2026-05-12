"""Tests for Langfuse-compatible tracing in the deep-research orchestrator.

We inject a Spy ObservabilityBackend via the module-level singleton
(``src.platform.observability._backend``) and assert that the orchestrator
emits a parent trace ``deep_research`` with one Generation per ``_call_json``
invocation, plus per-iteration spans.
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
from src.retrieval.pipeline.deep_research import DeepResearch, DeepResearchBudget
from src.vector_db.common.schemas import SearchResult


# ---------------------------------------------------------------------------
# Spy backend
# ---------------------------------------------------------------------------


@dataclass
class SpySpan(Span):
    name: str = ""
    attributes: dict = field(default_factory=dict)
    end_calls: list[tuple[str, Optional[Exception]]] = field(default_factory=list)

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value

    def end(self, status: str = "ok", error: Optional[Exception] = None) -> None:
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
    end_calls: list[tuple[str, Optional[Exception]]] = field(default_factory=list)

    def set_output(self, output: str) -> None:
        self.output = output

    def set_token_counts(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens

    def end(self, status: str = "ok", error: Optional[Exception] = None) -> None:
        self.end_calls.append((status, error))


@dataclass
class SpyTrace(Trace):
    name: str = ""
    metadata: dict = field(default_factory=dict)
    spans: list[SpySpan] = field(default_factory=list)
    generations: list[SpyGeneration] = field(default_factory=list)
    attributes: dict = field(default_factory=dict)
    end_calls: list[tuple[str, Optional[Exception]]] = field(default_factory=list)

    def span(self, name: str, attributes: Optional[dict] = None) -> Span:
        s = SpySpan(name=name, attributes=dict(attributes or {}))
        self.spans.append(s)
        return s

    def generation(
        self,
        name: str,
        model: str,
        input: str,
        metadata: Optional[dict] = None,
    ) -> Generation:
        g = SpyGeneration(name=name, model=model, input=input, metadata=dict(metadata or {}))
        self.generations.append(g)
        return g

    # The orchestrator may call set_attribute / end on the trace via duck-typing
    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value

    def end(self, status: str = "ok", error: Optional[Exception] = None) -> None:
        self.end_calls.append((status, error))


class SpyBackend(ObservabilityBackend):
    def __init__(self) -> None:
        self.traces: list[SpyTrace] = []
        self.spans: list[SpySpan] = []
        self.generations: list[SpyGeneration] = []

    def trace(self, name: str, metadata: Optional[dict] = None) -> Trace:
        t = SpyTrace(name=name, metadata=dict(metadata or {}))
        self.traces.append(t)
        return t

    def span(self, name, attributes=None, parent=None) -> Span:
        s = SpySpan(name=name, attributes=dict(attributes or {}))
        self.spans.append(s)
        return s

    def generation(self, name, model, input, metadata=None) -> Generation:
        g = SpyGeneration(name=name, model=model, input=input, metadata=dict(metadata or {}))
        self.generations.append(g)
        return g

    def flush(self) -> None:
        pass

    def shutdown(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Helpers (mirror test_dr_metrics.py)
# ---------------------------------------------------------------------------


def _mk_chunk(idx: int, text: str = "") -> SearchResult:
    return SearchResult(
        text=text or f"chunk-{idx}",
        score=0.5,
        metadata={"source": "doc.md", "heading": "h", "chunk_index": idx},
        object_id=f"oid-{idx}",
        collection="RagDocuments",
    )


@dataclass
class _LLMResp:
    content: str
    model: str = "fake"
    prompt_tokens: int = 7
    completion_tokens: int = 11
    total_tokens: int = 18
    cost_usd: float = 0.0


class FakeProvider:
    def __init__(self, responses: Optional[list[dict]] = None) -> None:
        self.responses = list(responses or [])
        self.calls: list[dict] = []

    async def agenerate(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        if self.responses:
            payload = self.responses.pop(0)
        else:
            payload = {"is_sufficient": True}
        return _LLMResp(content=json.dumps(payload))


def _orch(*, provider, budget=None) -> DeepResearch:
    counter = {"n": 0}

    async def kb_retrieve(query: str):
        counter["n"] += 1
        return [_mk_chunk(counter["n"], text=f"text for {query}")]

    async def kg_retrieve(query: str):
        return None

    return DeepResearch(
        provider=provider,
        kb_retrieve=kb_retrieve,
        kg_retrieve=kg_retrieve,
        original_question="what is X and Y?",
        processed_query="X Y context",
        budget=budget or DeepResearchBudget(),
        model_alias="fake-model",
    )


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


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
# Tests
# ---------------------------------------------------------------------------


def test_dr_run_creates_one_parent_trace_with_attributes(spy_backend):
    provider = FakeProvider(responses=[{"is_sufficient": True}])
    orch = _orch(provider=provider)
    asyncio.run(orch.research())

    assert len(spy_backend.traces) == 1
    t = spy_backend.traces[0]
    assert t.name == "deep_research"
    md = t.metadata
    assert md["original_question"] == "what is X and Y?"
    assert md["processed_query"] == "X Y context"
    assert md["max_topics"] == 3
    assert md["max_depth"] == 3


def test_dr_emits_one_generation_per_call_json_with_correct_name(spy_backend):
    provider = FakeProvider(responses=[{"is_sufficient": True}])
    orch = _orch(provider=provider, budget=DeepResearchBudget(early_stop_enabled=False))
    asyncio.run(orch.research())

    t = spy_backend.traces[0]
    assert len(t.generations) == 1
    g = t.generations[0]
    assert g.name == "dr_sufficiency"
    assert g.model == "fake-model"
    assert g.input  # non-empty rendered prompt
    assert g.output is not None
    assert "is_sufficient" in g.output
    assert g.prompt_tokens == 7
    assert g.completion_tokens == 11
    assert g.end_calls == [("ok", None)]


def test_dr_decompose_generation_named_dr_decompose(spy_backend):
    provider = FakeProvider(
        responses=[
            {"is_sufficient": False, "missing_information": "more"},
            {"topics": [
                {"name": "alpha", "questions": [{"question": "qA?", "query": "qA"}]},
                {"name": "beta",  "questions": [{"question": "qB?", "query": "qB"}]},
            ]},
            {"is_sufficient": True},
            {"is_sufficient": True},
        ]
    )
    orch = _orch(provider=provider, budget=DeepResearchBudget(max_depth=2, max_topics=3, per_topic_questions=2))
    asyncio.run(orch.research())

    t = spy_backend.traces[0]
    names = [g.name for g in t.generations]
    assert "dr_sufficiency" in names
    assert "dr_decompose" in names
    # All generations ended with ok
    for g in t.generations:
        assert g.end_calls and g.end_calls[0][0] == "ok"
    # topic_count attribute on parent trace
    assert t.attributes.get("topic_count") == 2


def test_dr_iteration_spans_emitted(spy_backend):
    """Recursion levels emit dr_iteration_depth_{depth} spans on the trace."""
    provider = FakeProvider(
        responses=[
            {"is_sufficient": False, "missing_information": "more"},
            {"topics": [
                {"name": "alpha", "questions": [{"question": "qA?", "query": "qA"}]},
                {"name": "beta",  "questions": [{"question": "qB?", "query": "qB"}]},
            ]},
            {"is_sufficient": True},
            {"is_sufficient": True},
        ]
    )
    orch = _orch(provider=provider, budget=DeepResearchBudget(max_depth=2, max_topics=3, per_topic_questions=2))
    asyncio.run(orch.research())

    t = spy_backend.traces[0]
    iter_spans = [s for s in t.spans if s.name.startswith("dr_iteration_depth_")]
    assert len(iter_spans) >= 1


def test_dr_trace_ends_with_error_on_exception(spy_backend):
    provider = FakeProvider()
    orch = _orch(provider=provider)

    def boom():
        raise RuntimeError("synthetic")

    orch._budget_check = boom  # type: ignore[assignment]
    with pytest.raises(RuntimeError):
        asyncio.run(orch.research())

    assert len(spy_backend.traces) == 1
    t = spy_backend.traces[0]
    assert t.end_calls
    assert t.end_calls[-1][0] == "error"


def test_dr_trace_ends_ok_on_success(spy_backend):
    provider = FakeProvider(responses=[{"is_sufficient": True}])
    orch = _orch(provider=provider)
    asyncio.run(orch.research())

    t = spy_backend.traces[0]
    assert t.end_calls and t.end_calls[-1][0] == "ok"


def test_dr_generation_records_error_on_provider_failure(spy_backend):
    class ExplodingProvider(FakeProvider):
        async def agenerate(self, messages, **kwargs):
            raise RuntimeError("llm down")

    orch = _orch(provider=ExplodingProvider())
    asyncio.run(orch.research())

    t = spy_backend.traces[0]
    assert len(t.generations) == 1
    g = t.generations[0]
    assert g.end_calls and g.end_calls[-1][0] == "error"


def test_dr_split_confidence_attribute_set_on_trace(spy_backend):
    """After the first decompose, the parent trace gets a split_confidence
    attribute reflecting the LLM's reported confidence in the split."""
    provider = FakeProvider(
        responses=[
            {"is_sufficient": False, "missing_information": "more"},
            {"topics": [
                {"name": "alpha", "questions": [{"question": "qA?", "query": "qA"}]},
                {"name": "beta",  "questions": [{"question": "qB?", "query": "qB"}]},
            ], "split_confidence": 0.85},
            {"is_sufficient": True},
            {"is_sufficient": True},
        ]
    )
    orch = _orch(provider=provider, budget=DeepResearchBudget(max_depth=2, max_topics=3, per_topic_questions=2))
    asyncio.run(orch.research())

    t = spy_backend.traces[0]
    assert t.attributes.get("split_confidence") == pytest.approx(0.85)


def test_dr_early_stop_reason_attribute_set_on_trace(spy_backend):
    """When the tightened early-stop gate fires, the parent trace records
    early_stop_reason."""
    # Build an orch where the root retrieval returns 3 chunks across 3 sources
    # so the gate is satisfied immediately.
    provider = FakeProvider(responses=[{"is_sufficient": True}])

    async def kb_retrieve(query: str):
        return [
            _mk_chunk(1, text="t1"),
            SearchResult(text="t2", score=0.5,
                          metadata={"source": "b.md", "heading": "h", "chunk_index": 2},
                          object_id="oid-2", collection="RagDocuments"),
            SearchResult(text="t3", score=0.5,
                          metadata={"source": "c.md", "heading": "h", "chunk_index": 3},
                          object_id="oid-3", collection="RagDocuments"),
        ]

    async def kg_retrieve(query: str):
        return None

    orch = DeepResearch(
        provider=provider,
        kb_retrieve=kb_retrieve,
        kg_retrieve=kg_retrieve,
        original_question="what is X?",
        processed_query="X",
        model_alias="fake-model",
    )
    asyncio.run(orch.research())
    t = spy_backend.traces[0]
    assert t.attributes.get("early_stop_reason") == "iter0_sufficient"


def test_dr_noop_backend_does_not_break(monkeypatch):
    """With the default noop backend, the orchestrator must run without errors."""
    # Ensure noop singleton in place
    obs_module._backend = None
    provider = FakeProvider(responses=[{"is_sufficient": True}])
    orch = _orch(provider=provider)
    result = asyncio.run(orch.research())
    assert result.is_unified is True
