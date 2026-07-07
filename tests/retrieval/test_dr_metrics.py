"""Tests for Prometheus instrumentation of the deep-research orchestrator.

Strategy:

- Drive the existing ``DeepResearch`` class with the in-test ``FakeProvider`` /
  fake KB / fake KG already used by ``test_deep_research.py`` so we exercise
  the real orchestrator code paths.
- Read the metric values directly from the prometheus_client objects (no HTTP
  scrape required for unit tests). End-to-end scraping via ``/metrics`` is
  covered by the harness verification step described in the task.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Optional

import pytest

from src.retrieval.deep_research.metrics import (
    DR_ITERATION_DURATION,
    DR_ITERATIONS,
    DR_LLM_CALLS_TOTAL,
    DR_RUN_DURATION,
    DR_RUNS_TOTAL,
    DR_TOPIC_COUNT,
)
from src.retrieval.pipeline.deep_research import (
    DeepResearch,
    DeepResearchBudget,
)
from src.vector_db.common.schemas import SearchResult


# ---------------------------------------------------------------------------
# Helpers (mirrors test_deep_research.py)
# ---------------------------------------------------------------------------


def _mk_chunk(idx: int, text: str = "", source: str = "doc.md") -> SearchResult:
    return SearchResult(
        text=text or f"chunk-{idx}",
        score=0.5,
        metadata={"source": source, "heading": "h", "chunk_index": idx},
        object_id=f"oid-{idx}",
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
            payload = {"is_sufficient": True, "missing_information": "", "coverage_notes": ""}
        return _LLMResp(content=json.dumps(payload))


def _orch(
    *,
    provider: FakeProvider,
    budget: Optional[DeepResearchBudget] = None,
) -> DeepResearch:
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
    )


# ---------------------------------------------------------------------------
# Metric snapshot helpers
# ---------------------------------------------------------------------------


def _counter_value(counter, **labels) -> float:
    """Read a labeled Counter's current value."""
    return counter.labels(**labels)._value.get()


def _hist_count(hist, **labels) -> int:
    """Total observation count for a Histogram (with or without labels)."""
    if labels:
        h = hist.labels(**labels)
    else:
        h = hist
    # _sum and _buckets exist; the easy path is to walk samples.
    total = 0
    for metric in hist.collect():
        for sample in metric.samples:
            if sample.name.endswith("_count") and (not labels or all(sample.labels.get(k) == v for k, v in labels.items())):
                total += int(sample.value)
    return total


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_metrics_module_registers_expected_metrics():
    """Sanity: importing the module exposes all six metrics with correct names."""
    # Counters store name without `_total` suffix in `_name`; histograms keep the full name.
    assert DR_RUNS_TOTAL._name == "rag_dr_runs"
    assert DR_LLM_CALLS_TOTAL._name == "rag_dr_llm_calls"
    assert DR_ITERATIONS._name == "rag_dr_iterations"
    assert DR_TOPIC_COUNT._name == "rag_dr_topic_count"
    assert DR_ITERATION_DURATION._name == "rag_dr_iteration_duration_seconds"
    assert DR_RUN_DURATION._name == "rag_dr_run_duration_seconds"


def test_dr_run_increments_ok_outcome_when_sufficient_first_round():
    """Sufficient on round 0 → outcome=early_stop, one sufficiency call, one iteration."""
    before_early_stop = _counter_value(DR_RUNS_TOTAL, outcome="early_stop")
    before_suff = _counter_value(DR_LLM_CALLS_TOTAL, stage="sufficiency")
    before_iter = _hist_count(DR_ITERATIONS)
    before_run = _hist_count(DR_RUN_DURATION)

    provider = FakeProvider(responses=[{"is_sufficient": True, "missing_information": ""}])
    orch = _orch(provider=provider, budget=DeepResearchBudget(early_stop_enabled=False))
    result = asyncio.get_event_loop().run_until_complete(orch.research()) if False else asyncio.run(orch.research())

    assert result.is_unified is True
    assert result.llm_call_count == 1

    assert _counter_value(DR_RUNS_TOTAL, outcome="early_stop") == before_early_stop + 1
    assert _counter_value(DR_LLM_CALLS_TOTAL, stage="sufficiency") == before_suff + 1
    assert _hist_count(DR_ITERATIONS) == before_iter + 1
    assert _hist_count(DR_RUN_DURATION) == before_run + 1


def test_dr_multi_topic_run_increments_ok_and_decompose_metrics():
    """Multi-topic split → sufficiency + multi_queries calls counted, topic_count observed."""
    before_ok = _counter_value(DR_RUNS_TOTAL, outcome="ok")
    before_suff = _counter_value(DR_LLM_CALLS_TOTAL, stage="sufficiency")
    before_decomp = _counter_value(DR_LLM_CALLS_TOTAL, stage="multi_queries")
    before_topics = _hist_count(DR_TOPIC_COUNT)

    provider = FakeProvider(
        responses=[
            # round-0 sufficiency: not sufficient
            {"is_sufficient": False, "missing_information": "need more"},
            # decomposition into 2 topics
            {
                "topics": [
                    {"name": "alpha", "questions": [{"question": "qA?", "query": "qA"}]},
                    {"name": "beta", "questions": [{"question": "qB?", "query": "qB"}]},
                ]
            },
            # subsequent sufficiency calls: all sufficient (terminate quickly)
            {"is_sufficient": True},
            {"is_sufficient": True},
        ]
    )
    budget = DeepResearchBudget(max_depth=2, max_topics=3, per_topic_questions=2)
    orch = _orch(provider=provider, budget=budget)
    result = asyncio.run(orch.research())

    assert result.is_unified is False
    assert len(result.topic_pools) == 2

    assert _counter_value(DR_RUNS_TOTAL, outcome="ok") == before_ok + 1
    assert _counter_value(DR_LLM_CALLS_TOTAL, stage="sufficiency") >= before_suff + 1
    assert _counter_value(DR_LLM_CALLS_TOTAL, stage="multi_queries") == before_decomp + 1
    assert _hist_count(DR_TOPIC_COUNT) == before_topics + 1


def test_dr_budget_exhausted_outcome():
    """When wall-clock / nodes cap fires, outcome must be budget_exhausted."""
    before_be = _counter_value(DR_RUNS_TOTAL, outcome="budget_exhausted")

    provider = FakeProvider(
        responses=[
            {"is_sufficient": False, "missing_information": "x"},
            {
                "topics": [
                    {"name": "alpha", "questions": [{"question": "qA?", "query": "qA"}]},
                    {"name": "beta", "questions": [{"question": "qB?", "query": "qB"}]},
                ]
            },
        ]
    )
    # max_nodes=1 trips immediately after the root retrieval.
    budget = DeepResearchBudget(max_nodes=1, max_depth=3, max_topics=2, per_topic_questions=2)
    orch = _orch(provider=provider, budget=budget)
    result = asyncio.run(orch.research())

    assert result.budget_exhausted is True
    assert _counter_value(DR_RUNS_TOTAL, outcome="budget_exhausted") == before_be + 1


def test_dr_error_outcome_on_orchestrator_exception():
    """If the orchestrator raises, outcome=error must be incremented and exception re-raised."""
    before_err = _counter_value(DR_RUNS_TOTAL, outcome="error")

    class ExplodingProvider(FakeProvider):
        async def agenerate(self, messages, **kwargs):
            raise RuntimeError("boom")

    async def kb_retrieve(query: str):
        # Force the orchestrator to fail by raising during retrieval.
        raise RuntimeError("kb boom")

    async def kg_retrieve(query: str):
        return None

    orch = DeepResearch(
        provider=ExplodingProvider(),
        kb_retrieve=lambda q: (_ for _ in ()).throw(RuntimeError("kb boom")),  # type: ignore[arg-type]
        kg_retrieve=kg_retrieve,
        original_question="q",
        processed_query="q",
    )
    # The orchestrator currently swallows kb errors, so to force a true raise
    # we monkeypatch the build path. Instead drive an exception via _build_result.
    # Simpler: patch the budget check to raise.
    original = orch._budget_check

    def boom():
        raise RuntimeError("synthetic")

    orch._budget_check = boom  # type: ignore[assignment]

    with pytest.raises(RuntimeError):
        asyncio.run(orch.research())

    assert _counter_value(DR_RUNS_TOTAL, outcome="error") == before_err + 1
