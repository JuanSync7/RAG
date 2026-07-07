"""Tests for the tightened early-stop gate and adaptive-depth heuristics
on the deep-research orchestrator.

Two coupled efficiency improvements are exercised:

1. **Early-stop gate**: ``is_sufficient=True`` alone is no longer enough to
   stop. The pool must also satisfy a chunk floor and source-diversity
   floor, AND confidence must be monotonically non-decreasing.
2. **Adaptive depth**: after the first decompose, the LLM's
   ``split_confidence`` and topic count cap ``max_depth`` to avoid wasted
   recursion on coherent or soft splits.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Optional

import pytest

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


def _orch_with_kb(
    *,
    provider: FakeProvider,
    kb_results_per_query: dict[str, list[SearchResult]],
    budget: Optional[DeepResearchBudget] = None,
    original_question: str = "what is X?",
    processed_query: str = "X",
) -> DeepResearch:
    """Build an orchestrator whose KB returns the configured chunks per
    query, falling back to a single fresh-source chunk for any other query."""
    counter = {"n": 100}

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
    )


# ---------------------------------------------------------------------------
# Early-stop tests
# ---------------------------------------------------------------------------


class TestEarlyStop:
    @pytest.mark.asyncio
    async def test_early_stop_fires_when_sufficient_floor_diversity_met(self) -> None:
        """3+ chunks across 2+ sources + sufficient → 1 LLM call, early-stop."""
        provider = FakeProvider(responses=[
            {"is_sufficient": True, "missing_information": "", "coverage_notes": "ok"},
        ])
        kb = {
            "X": [
                _mk_chunk(1, source="a.md"),
                _mk_chunk(2, source="b.md"),
                _mk_chunk(3, source="c.md"),
            ],
        }
        orch = _orch_with_kb(provider=provider, kb_results_per_query=kb, processed_query="X")
        result = await orch.research()

        assert result.llm_call_count == 1
        assert result.early_stop_reason == "iter0_sufficient"
        assert result.is_unified is True

    @pytest.mark.asyncio
    async def test_early_stop_blocked_when_chunk_floor_unmet(self) -> None:
        """sufficient=True but only 1 chunk → DR continues to decompose."""
        provider = FakeProvider(responses=[
            {"is_sufficient": True, "missing_information": "", "coverage_notes": ""},
            # If we fall through, DR will decompose; return an empty decomp
            # so the loop terminates without infinite work.
            {"topics": [], "split_confidence": 0.0},
        ])
        kb = {"X": [_mk_chunk(1, source="a.md")]}
        orch = _orch_with_kb(provider=provider, kb_results_per_query=kb, processed_query="X")
        result = await orch.research()

        assert result.llm_call_count >= 2
        assert result.early_stop_reason is None

    @pytest.mark.asyncio
    async def test_early_stop_blocked_when_source_diversity_unmet(self) -> None:
        """5 chunks all from one source → DR continues."""
        provider = FakeProvider(responses=[
            {"is_sufficient": True, "missing_information": "", "coverage_notes": ""},
            {"topics": [], "split_confidence": 0.0},
        ])
        kb = {
            "X": [_mk_chunk(i, source="x.md") for i in range(5)],
        }
        orch = _orch_with_kb(provider=provider, kb_results_per_query=kb, processed_query="X")
        result = await orch.research()

        assert result.llm_call_count >= 2
        assert result.early_stop_reason is None

    @pytest.mark.asyncio
    async def test_early_stop_blocked_when_confidence_oscillates(self) -> None:
        """Confident-true at iter 0 fails the chunk floor (so no early-stop),
        we decompose, then a deeper sufficiency check returns confidence
        lower than the prior max — trend check rejects it. The loop should
        therefore not early-stop on the second sufficiency."""
        provider = FakeProvider(responses=[
            # iter 0 — sufficient + high confidence, but only 1 chunk → blocked.
            {"is_sufficient": True, "confidence": 0.9, "missing_information": ""},
            # decompose to one topic with one follow-up.
            {"topics": [
                {"name": "t", "questions": [
                    {"question": "follow up?", "query": "X-followup"},
                ]},
            ], "split_confidence": 0.8},
            # iter 1 on the same pool — sufficient but lower confidence → trend-blocked.
            {"is_sufficient": True, "confidence": 0.4, "missing_information": ""},
        ])
        kb = {"X": [_mk_chunk(1, source="a.md")]}
        orch = _orch_with_kb(provider=provider, kb_results_per_query=kb, processed_query="X")
        result = await orch.research()

        # Trend check should have fired — early_stop_reason still None.
        assert result.early_stop_reason is None

    @pytest.mark.asyncio
    async def test_early_stop_disabled_via_budget_flag(self) -> None:
        """early_stop_enabled=False reverts to legacy single-bool stop."""
        provider = FakeProvider(responses=[
            {"is_sufficient": True, "missing_information": "", "coverage_notes": ""},
        ])
        kb = {"X": [_mk_chunk(1, source="a.md")]}
        budget = DeepResearchBudget(early_stop_enabled=False)
        orch = _orch_with_kb(
            provider=provider, kb_results_per_query=kb,
            budget=budget, processed_query="X",
        )
        result = await orch.research()

        # Legacy: single chunk, single source, but early-stop still fires.
        assert result.llm_call_count == 1
        assert result.early_stop_reason == "iter0_sufficient"


# ---------------------------------------------------------------------------
# Adaptive-depth tests
# ---------------------------------------------------------------------------


class TestAdaptiveDepth:
    @pytest.mark.asyncio
    async def test_n1_topic_caps_max_depth(self) -> None:
        """Decompose returns exactly 1 topic → max_depth capped to 2."""
        provider = FakeProvider(responses=[
            # iter 0: sufficient but blocked by gate (1 chunk, 1 source).
            {"is_sufficient": True, "missing_information": ""},
            # First decompose: one topic.
            {"topics": [
                {"name": "only", "questions": [
                    {"question": "q?", "query": "qq"},
                ]},
            ], "split_confidence": 0.9},
            # Subsequent sufficiency on the only topic — terminate quickly.
            {"is_sufficient": False, "missing_information": "x"},
            # If allowed to recurse further, would emit more — return [] to stop.
            {"topics": [], "split_confidence": 0.0},
        ])
        kb = {"X": [_mk_chunk(1, source="a.md")]}
        # Start with max_depth=3 — assert it is capped to 2 mid-run.
        budget = DeepResearchBudget(max_depth=3)
        orch = _orch_with_kb(
            provider=provider, kb_results_per_query=kb,
            budget=budget, processed_query="X",
        )
        await orch.research()
        assert orch._budget.max_depth == 2

    @pytest.mark.asyncio
    async def test_high_confidence_split_preserves_depth(self) -> None:
        """N=2 topics with split_confidence=0.9 → max_depth unchanged."""
        provider = FakeProvider(responses=[
            {"is_sufficient": True, "missing_information": ""},
            {"topics": [
                {"name": "a", "questions": [{"question": "qa?", "query": "qa"}]},
                {"name": "b", "questions": [{"question": "qb?", "query": "qb"}]},
            ], "split_confidence": 0.9},
            # Quickly terminate sub-pools.
            {"is_sufficient": True, "missing_information": ""},
            {"is_sufficient": True, "missing_information": ""},
        ])
        kb = {"X": [_mk_chunk(1, source="a.md")]}
        budget = DeepResearchBudget(max_depth=3)
        orch = _orch_with_kb(
            provider=provider, kb_results_per_query=kb,
            budget=budget, processed_query="X",
        )
        await orch.research()
        assert orch._budget.max_depth == 3

    @pytest.mark.asyncio
    async def test_low_confidence_split_caps_depth(self) -> None:
        """N=2, split_confidence=0.4 → max_depth capped to 2."""
        provider = FakeProvider(responses=[
            {"is_sufficient": True, "missing_information": ""},
            {"topics": [
                {"name": "a", "questions": [{"question": "qa?", "query": "qa"}]},
                {"name": "b", "questions": [{"question": "qb?", "query": "qb"}]},
            ], "split_confidence": 0.4},
            {"is_sufficient": True, "missing_information": ""},
            {"is_sufficient": True, "missing_information": ""},
        ])
        kb = {"X": [_mk_chunk(1, source="a.md")]}
        budget = DeepResearchBudget(max_depth=3)
        orch = _orch_with_kb(
            provider=provider, kb_results_per_query=kb,
            budget=budget, processed_query="X",
        )
        await orch.research()
        assert orch._budget.max_depth == 2

    @pytest.mark.asyncio
    async def test_split_confidence_missing_defaults_to_low(self) -> None:
        """LLM omits split_confidence for a 2-topic split → low-confidence path
        (capped depth)."""
        provider = FakeProvider(responses=[
            {"is_sufficient": True, "missing_information": ""},
            # No split_confidence key.
            {"topics": [
                {"name": "a", "questions": [{"question": "qa?", "query": "qa"}]},
                {"name": "b", "questions": [{"question": "qb?", "query": "qb"}]},
            ]},
            {"is_sufficient": True, "missing_information": ""},
            {"is_sufficient": True, "missing_information": ""},
        ])
        kb = {"X": [_mk_chunk(1, source="a.md")]}
        budget = DeepResearchBudget(max_depth=3)
        orch = _orch_with_kb(
            provider=provider, kb_results_per_query=kb,
            budget=budget, processed_query="X",
        )
        await orch.research()
        assert orch._budget.max_depth == 2


# ---------------------------------------------------------------------------
# Result-field test
# ---------------------------------------------------------------------------


class TestResultFields:
    @pytest.mark.asyncio
    async def test_result_surfaces_early_stop_reason(self) -> None:
        """early_stop_reason is set when the gate fires; None otherwise."""
        # Case A: early-stop fires.
        provider_a = FakeProvider(responses=[
            {"is_sufficient": True, "missing_information": ""},
        ])
        kb_a = {"X": [
            _mk_chunk(1, source="a.md"),
            _mk_chunk(2, source="b.md"),
            _mk_chunk(3, source="c.md"),
        ]}
        orch_a = _orch_with_kb(provider=provider_a, kb_results_per_query=kb_a, processed_query="X")
        ra = await orch_a.research()
        assert ra.early_stop_reason == "iter0_sufficient"

        # Case B: early-stop blocked, run to multi-topic completion → reason None.
        provider_b = FakeProvider(responses=[
            {"is_sufficient": False, "missing_information": "more"},
            {"topics": [
                {"name": "a", "questions": [{"question": "qa?", "query": "qa"}]},
                {"name": "b", "questions": [{"question": "qb?", "query": "qb"}]},
            ], "split_confidence": 0.9},
            {"is_sufficient": True},
            {"is_sufficient": True},
        ])
        kb_b = {"X": [_mk_chunk(1, source="a.md")]}
        orch_b = _orch_with_kb(provider=provider_b, kb_results_per_query=kb_b, processed_query="X")
        rb = await orch_b.research()
        assert rb.early_stop_reason is None
