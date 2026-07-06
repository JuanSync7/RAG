"""Tests for the deep-research orchestrator.

Focus is on the loop's structural behavior, not on prompt quality:

- JSON parsing tolerates fenced output and salvage cases.
- TopicPool dedupe + evidence truncation.
- "Sufficient on first round" → one unified pool, no decomposition.
- Single-topic decomposition stays unified.
- Multi-topic split fans out and returns N pools (is_unified=False).
- Budget caps terminate the loop.
- Original question is pinned across recursion depths (sufficiency calls
  always see the same anchor question).
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
    KGSnippet,
    TopicPool,
    _parse_json_object,
)
from src.vector_db.common.schemas import SearchResult


# ---------------------------------------------------------------------------
# Helpers
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
    """Programmable LLM provider — returns canned JSON in order.

    `responses` is a list of dicts (each gets json.dumps'ed). The provider
    pops from the front on each `agenerate` call. If the list is exhausted,
    it returns a benign "is_sufficient=true" response so the loop terminates.
    """

    def __init__(self, responses: Optional[list[dict[str, Any]]] = None) -> None:
        self.responses = list(responses or [])
        self.calls: list[dict[str, Any]] = []  # (purpose-detected) record of inputs

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
    kb_results_per_query: Optional[dict[str, list[SearchResult]]] = None,
    kg_results_per_query: Optional[dict[str, KGSnippet]] = None,
    original_question: str = "what is X and Y?",
    processed_query: str = "X Y context",
    budget: Optional[DeepResearchBudget] = None,
    model_alias: str = "default",
) -> DeepResearch:
    kb_map = dict(kb_results_per_query or {})
    kg_map = dict(kg_results_per_query or {})
    kb_calls: list[str] = []
    kg_calls: list[str] = []

    async def kb_retrieve(query: str):
        kb_calls.append(query)
        # Default: one fresh chunk per unique query.
        return kb_map.get(query, [_mk_chunk(idx=len(kb_calls), text=f"text for {query}")])

    async def kg_retrieve(query: str):
        kg_calls.append(query)
        return kg_map.get(query)

    o = DeepResearch(
        provider=provider,  # type: ignore[arg-type]
        kb_retrieve=kb_retrieve,
        kg_retrieve=kg_retrieve,
        original_question=original_question,
        processed_query=processed_query,
        budget=budget or DeepResearchBudget(
            max_nodes=10, max_llm_calls=20, wall_clock_ms=5000,
            max_depth=3, max_topics=3, per_topic_questions=3,
        ),
        model_alias=model_alias,
    )
    o._kb_calls = kb_calls  # type: ignore[attr-defined] — for test introspection
    o._kg_calls = kg_calls  # type: ignore[attr-defined]
    return o


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


class TestParseJsonObject:
    def test_plain_object(self) -> None:
        assert _parse_json_object('{"a": 1}') == {"a": 1}

    def test_code_fenced_json(self) -> None:
        assert _parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}

    def test_code_fenced_no_lang(self) -> None:
        assert _parse_json_object('```\n{"a": 1}\n```') == {"a": 1}

    def test_salvage_with_surrounding_prose(self) -> None:
        assert _parse_json_object('here is the answer {"a": 1} hope it helps') == {"a": 1}

    def test_returns_none_for_array(self) -> None:
        # Top-level must be an object, not an array.
        assert _parse_json_object("[1, 2, 3]") is None

    def test_returns_none_for_garbage(self) -> None:
        assert _parse_json_object("not json at all") is None


# ---------------------------------------------------------------------------
# TopicPool
# ---------------------------------------------------------------------------


class TestTopicPool:
    def test_dedupe_by_object_id(self) -> None:
        pool = TopicPool(name="t", rerank_anchor="a")
        c1 = _mk_chunk(1, text="first")
        c1_dup = _mk_chunk(1, text="first-different-text")  # same oid → dedup
        c2 = _mk_chunk(2, text="second")
        pool.merge_chunks([c1, c1_dup, c2])
        assert len(pool.chunks_by_id) == 2
        # First-write wins (preserves earliest reference).
        first = next(iter(pool.chunks_by_id.values()))
        assert first.text == "first"

    def test_evidence_text_caps_chars(self) -> None:
        pool = TopicPool(name="t", rerank_anchor="a")
        # 5 chunks of ~50 chars each.
        pool.merge_chunks([_mk_chunk(i, text="x" * 50) for i in range(5)])
        out = pool.evidence_text(max_chars=80)
        # Hard cap allows some slack, but we shouldn't see all 5 chunks.
        assert "more chunks omitted" in out

    def test_evidence_text_when_empty(self) -> None:
        pool = TopicPool(name="t", rerank_anchor="a")
        out = pool.evidence_text(max_chars=1000)
        assert "no evidence" in out


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class TestOrchestrator:
    @pytest.mark.asyncio
    async def test_llm_calls_use_configured_model_alias(self) -> None:
        """Deep-research controller calls run on the configured alias, so it can be
        pointed at the instruct model instead of the reasoning/gen default."""
        provider = FakeProvider(responses=[
            {"is_sufficient": True, "missing_information": "", "coverage_notes": "ok"},
        ])
        o = _orch(provider=provider, model_alias="judge", budget=DeepResearchBudget(
            max_nodes=10, max_llm_calls=20, wall_clock_ms=5000,
            max_depth=3, max_topics=3, per_topic_questions=3, early_stop_enabled=False,
        ))
        await o.research()
        assert provider.calls, "expected at least one LLM call"
        assert all(c["kwargs"].get("model_alias") == "judge" for c in provider.calls)

    @pytest.mark.asyncio
    async def test_domain_is_injected_into_controller_prompts(self) -> None:
        """The corpus domain is rendered into the sufficiency/decomposition prompts
        so DR resolves domain-ambiguous acronyms in-corpus (same fix as HyDE)."""
        provider = FakeProvider(responses=[
            {"is_sufficient": True, "missing_information": "", "coverage_notes": "ok"},
        ])
        o = _orch(provider=provider, budget=DeepResearchBudget(
            max_nodes=10, max_llm_calls=20, wall_clock_ms=5000,
            max_depth=3, max_topics=3, per_topic_questions=3, early_stop_enabled=False,
        ))
        o._domain = "Silicon/SoC design: DFT (Design-for-Test), MBIST, AMBA."
        await o.research()
        assert provider.calls, "expected at least one controller LLM call"
        prompt_text = provider.calls[0]["messages"][0]["content"]
        assert "Design-for-Test" in prompt_text  # domain reached the prompt

    @pytest.mark.asyncio
    async def test_sufficient_on_first_round_returns_unified(self) -> None:
        provider = FakeProvider(responses=[
            {"is_sufficient": True, "missing_information": "", "coverage_notes": "covered"},
        ])
        # Legacy single-bool stop semantics — disable the tightened gate.
        o = _orch(provider=provider, budget=DeepResearchBudget(
            max_nodes=10, max_llm_calls=20, wall_clock_ms=5000,
            max_depth=3, max_topics=3, per_topic_questions=3,
            early_stop_enabled=False,
        ))
        result = await o.research()

        assert result.is_unified is True
        assert len(result.topic_pools) == 1
        assert result.topic_pools[0].name == "<root>"
        # One root retrieval + one sufficiency check.
        assert result.node_count == 1
        assert result.llm_call_count == 1

    @pytest.mark.asyncio
    async def test_single_topic_decomposition_stays_unified(self) -> None:
        provider = FakeProvider(responses=[
            # Round 0: insufficient, missing UVM aspects.
            {"is_sufficient": False, "missing_information": "missing UVM aspects", "coverage_notes": ""},
            # Decomposition: single topic with 2 questions.
            {"topics": [
                {"name": "UVM coverage", "questions": [
                    {"question": "What is functional coverage?", "query": "UVM functional coverage"},
                    {"question": "How are gates enforced?", "query": "UVM coverage gates"},
                ]},
            ]},
            # Sufficiency for both deeper retrievals — both sufficient.
            {"is_sufficient": True, "missing_information": "", "coverage_notes": ""},
            {"is_sufficient": True, "missing_information": "", "coverage_notes": ""},
        ])
        o = _orch(provider=provider)
        result = await o.research()

        assert result.is_unified is True
        assert len(result.topic_pools) == 1
        # Root + 2 sub-questions = 3 retrievals.
        assert result.node_count >= 3

    @pytest.mark.asyncio
    async def test_multi_topic_split_returns_n_pools(self) -> None:
        provider = FakeProvider(responses=[
            # Round 0: insufficient.
            {"is_sufficient": False, "missing_information": "two distinct topics", "coverage_notes": ""},
            # Decomposition: two distinct topics.
            {"topics": [
                {"name": "verification", "questions": [
                    {"question": "verification methodology?", "query": "verification methodology UVM"},
                ]},
                {"name": "lint flow", "questions": [
                    {"question": "lint flow?", "query": "lint flow Spyglass"},
                ]},
            ]},
            # Sufficiency on each leaf retrieval — both sufficient.
            {"is_sufficient": True, "missing_information": "", "coverage_notes": ""},
            {"is_sufficient": True, "missing_information": "", "coverage_notes": ""},
        ])
        o = _orch(
            provider=provider,
            original_question="how does verification work and how does lint flow work",
            processed_query="verification lint flow",
        )
        result = await o.research()

        assert result.is_unified is False
        assert len(result.topic_pools) == 2
        names = {p.name for p in result.topic_pools}
        assert "verification" in names and "lint flow" in names

    @pytest.mark.asyncio
    async def test_budget_max_nodes_terminates(self) -> None:
        # Provider that always says "insufficient" + always emits 3 sub-questions.
        # If the loop respected nothing, it would explode. Budget should cap it.
        responses = []
        for _ in range(40):
            responses.append({"is_sufficient": False, "missing_information": "more", "coverage_notes": ""})
            responses.append({"topics": [
                {"name": "always-more", "questions": [
                    {"question": "q1", "query": "qq1-{i}"},
                    {"question": "q2", "query": "qq2-{i}"},
                    {"question": "q3", "query": "qq3-{i}"},
                ]},
            ]})
        provider = FakeProvider(responses=responses)
        budget = DeepResearchBudget(
            max_nodes=4, max_llm_calls=20, wall_clock_ms=5000,
            max_depth=5, max_topics=3, per_topic_questions=3,
        )
        o = _orch(provider=provider, budget=budget)
        result = await o.research()

        assert result.budget_exhausted is True
        assert result.budget_exhausted_reason == "max_nodes"
        # Node count is bounded but may be off-by-one due to in-flight tasks.
        assert result.node_count <= 4 + 3  # generous slack for parallel siblings

    @pytest.mark.asyncio
    async def test_original_question_is_pinned_across_depths(self) -> None:
        """Sufficiency_check should always receive the SAME original question
        even after decomposition recursions."""
        provider = FakeProvider(responses=[
            # Round 0: insufficient.
            {"is_sufficient": False, "missing_information": "more", "coverage_notes": ""},
            # Decomposition: one topic with two follow-ups.
            {"topics": [{"name": "topic", "questions": [
                {"question": "q1?", "query": "qq1"},
                {"question": "q2?", "query": "qq2"},
            ]}]},
            # All deeper sufficiencies return sufficient.
            {"is_sufficient": True, "missing_information": "", "coverage_notes": ""},
            {"is_sufficient": True, "missing_information": "", "coverage_notes": ""},
        ])
        original = "compare X to Y on Z"
        o = _orch(provider=provider, original_question=original, processed_query="X Y Z")
        result = await o.research()
        assert result.is_unified is True

        # Inspect every sufficiency-style call: prompt should contain the
        # original question verbatim, never the local sub-question.
        sufficiency_calls = [
            c for c in provider.calls
            if "Sufficiency Check" in c["messages"][0]["content"]
        ]
        assert len(sufficiency_calls) >= 1
        for c in sufficiency_calls:
            content = c["messages"][0]["content"]
            assert original in content, "original question must be pinned in every sufficiency check"

    @pytest.mark.asyncio
    async def test_kg_graph_context_accumulated(self) -> None:
        provider = FakeProvider(responses=[
            {"is_sufficient": True, "missing_information": "", "coverage_notes": ""},
        ])
        kg_snippets = {"X Y context": KGSnippet(terms=["x"], graph_context="GRAPH-X-Y")}
        o = _orch(provider=provider, kg_results_per_query=kg_snippets)
        result = await o.research()

        assert "GRAPH-X-Y" in result.graph_context

    @pytest.mark.asyncio
    async def test_dedup_query_string_avoids_double_retrieval(self) -> None:
        """If the LLM emits a sub-question whose `query` matches the root
        query string, the orchestrator should not re-issue retrieval."""
        provider = FakeProvider(responses=[
            {"is_sufficient": False, "missing_information": "more", "coverage_notes": ""},
            {"topics": [{"name": "topic", "questions": [
                # Same query as processed_query — should be deduped.
                {"question": "same question", "query": "X Y context"},
            ]}]},
            {"is_sufficient": True, "missing_information": "", "coverage_notes": ""},
        ])
        o = _orch(provider=provider)
        result = await o.research()

        # KB called once for the root only, not twice.
        assert len(o._kb_calls) == 1  # type: ignore[attr-defined]
        assert result.node_count == 1


    @pytest.mark.asyncio
    async def test_result_exposes_iteration_decomposed_topic_count_unified(self) -> None:
        """Deep-research result must surface iteration_count, decomposed,
        topic_count so workflow observability (Temporal search attributes,
        Prom labels) can read them off response.metadata.deep_research."""
        provider = FakeProvider(responses=[
            {"is_sufficient": True, "missing_information": "", "coverage_notes": "ok"},
        ])
        o = _orch(provider=provider, budget=DeepResearchBudget(
            max_nodes=10, max_llm_calls=20, wall_clock_ms=5000,
            max_depth=3, max_topics=3, per_topic_questions=3,
            early_stop_enabled=False,
        ))
        result = await o.research()

        # Sufficient on round 0 = no decomposition.
        assert result.decomposed is False
        assert result.topic_count == 1
        # iteration_count counts every sufficiency-check iteration. Round 0
        # root retrieval counts as 1 iteration.
        assert result.iteration_count >= 1

    @pytest.mark.asyncio
    async def test_result_fields_on_multi_topic_decomposition(self) -> None:
        provider = FakeProvider(responses=[
            {"is_sufficient": False, "missing_information": "two topics", "coverage_notes": ""},
            {"topics": [
                {"name": "alpha", "questions": [
                    {"question": "a?", "query": "alpha-q"},
                ]},
                {"name": "beta", "questions": [
                    {"question": "b?", "query": "beta-q"},
                ]},
            ]},
            {"is_sufficient": True, "missing_information": "", "coverage_notes": ""},
            {"is_sufficient": True, "missing_information": "", "coverage_notes": ""},
        ])
        o = _orch(
            provider=provider,
            original_question="alpha and beta",
            processed_query="alpha beta",
        )
        result = await o.research()

        assert result.decomposed is True
        assert result.topic_count == 2
        assert result.iteration_count >= 1
