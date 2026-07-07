"""OTel instrumentation contract for DeepResearch orchestrator.

This exercises the OTel backend path: the orchestrator opens a
``retrieval.deep_research`` trace and emits per-recursion-depth
``retrieval.deep_research.iteration`` spans. The Langfuse-shaped behaviour
remains covered by the Spy backend in test_dr_tracing.py.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Optional

from src.retrieval.pipeline.deep_research import DeepResearch, DeepResearchBudget
from src.vector_db.common.schemas import SearchResult


def _spans_named(exporter, name):
    return [s for s in exporter.get_finished_spans() if s.name == name]


def _mk_chunk(idx: int) -> SearchResult:
    return SearchResult(
        text=f"chunk-{idx}",
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

    async def agenerate(self, messages, **kwargs):
        payload = self.responses.pop(0) if self.responses else {"is_sufficient": True}
        return _LLMResp(content=json.dumps(payload))


def _orch(*, provider, budget=None) -> DeepResearch:
    counter = {"n": 0}

    async def kb_retrieve(query: str):
        counter["n"] += 1
        return [_mk_chunk(counter["n"])]

    async def kg_retrieve(query: str):
        return None

    return DeepResearch(
        provider=provider,
        kb_retrieve=kb_retrieve,
        kg_retrieve=kg_retrieve,
        original_question="what is X?",
        processed_query="X context",
        budget=budget or DeepResearchBudget(),
        model_alias="fake-model",
    )


def test_dr_emits_retrieval_deep_research_root(otel_capture):
    provider = FakeProvider(responses=[{"is_sufficient": True}])
    orch = _orch(provider=provider)
    asyncio.run(orch.research())

    spans = _spans_named(otel_capture, "retrieval.deep_research")
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert attrs["original_question"] == "what is X?"
    assert attrs["processed_query"] == "X context"


def test_dr_emits_iteration_spans(otel_capture):
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
    orch = _orch(
        provider=provider,
        budget=DeepResearchBudget(max_depth=2, max_topics=3, per_topic_questions=2),
    )
    asyncio.run(orch.research())

    iter_spans = _spans_named(otel_capture, "retrieval.deep_research.iteration")
    assert len(iter_spans) >= 1
    # Each iteration span carries depth/topic/question_count attrs.
    for s in iter_spans:
        assert "depth" in s.attributes
        assert "topic" in s.attributes
        assert "question_count" in s.attributes


def test_dr_iteration_nests_under_deep_research_root(otel_capture):
    """Context propagation: iteration spans must have the dr root as parent."""
    provider = FakeProvider(
        responses=[
            {"is_sufficient": False, "missing_information": "more"},
            {"topics": [
                {"name": "alpha", "questions": [{"question": "qA?", "query": "qA"}]},
            ]},
            {"is_sufficient": True},
        ]
    )
    orch = _orch(
        provider=provider,
        budget=DeepResearchBudget(max_depth=2, max_topics=1, per_topic_questions=1),
    )
    asyncio.run(orch.research())

    spans = otel_capture.get_finished_spans()
    by_id = {s.context.span_id: s for s in spans}
    dr_root = next(s for s in spans if s.name == "retrieval.deep_research")
    iter_spans = [s for s in spans if s.name == "retrieval.deep_research.iteration"]
    assert iter_spans
    for s in iter_spans:
        # Walk parent chain — at minimum some ancestor is the dr root.
        cur = s
        seen_root = False
        while cur is not None and cur.parent is not None:
            parent = by_id.get(cur.parent.span_id)
            if parent is dr_root:
                seen_root = True
                break
            cur = parent
        assert seen_root, f"iteration span {s.name} not nested under retrieval.deep_research"
