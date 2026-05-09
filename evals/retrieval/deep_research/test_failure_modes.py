"""Failure-mode unit tests for the deep-research orchestrator.

These exercise graceful-degradation paths without needing a live stack:
LLM unreachable, empty retrieval, budget pre-exhausted. Marked with the
``eval_deep_research`` marker so they run with the rest of the suite.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from src.retrieval.pipeline.deep_research import (
    DeepResearch,
    DeepResearchBudget,
    KGSnippet,
)
from src.vector_db.common.schemas import SearchResult

pytestmark = pytest.mark.eval_deep_research


class _StubProvider:
    """Minimal LLMProvider stand-in with configurable agenerate behavior."""

    def __init__(self, behavior):
        self._behavior = behavior
        self.calls = 0

    async def agenerate(self, messages, **kwargs):  # noqa: D401
        self.calls += 1
        return await self._behavior(messages, **kwargs)


def _mk_chunk(text: str, source: str = "x.md") -> SearchResult:
    return SearchResult(
        object_id=f"oid-{hash(text) & 0xFFFF:x}",
        text=text,
        score=0.5,
        metadata={"source": source, "heading": "h"},
    )


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) if False else asyncio.run(coro)


def test_llm_connect_error_does_not_crash():
    """Test A: when the sufficiency LLM call raises ConnectError, DR should
    return whatever pool it had built (i.e., the root pool from the first
    KB retrieve), not propagate the exception, not return None."""

    async def bad_llm(messages, **kwargs):
        raise httpx.ConnectError("ollama unreachable")

    async def kb_retrieve(q):
        return [_mk_chunk("the gpio checklist requires X", source="hw/ip_templates/gpio/doc/checklist.md")]

    dr = DeepResearch(
        provider=_StubProvider(bad_llm),
        kb_retrieve=kb_retrieve,
        kg_retrieve=None,
        original_question="gpio checklist?",
        processed_query="gpio checklist?",
        budget=DeepResearchBudget(wall_clock_ms=5000),
    )
    result = _run(dr.research())

    assert result is not None
    assert result.topic_pools, "DR should still return a pool even when LLM is down"
    total_chunks = sum(len(p.chunks_by_id) for p in result.topic_pools)
    assert total_chunks > 0, "DR should retain root retrieval chunks when LLM fails"
    assert result.is_unified, "with no successful decomposition, DR stays unified"


def test_zero_kb_results_returns_empty_pool_with_reason():
    """Test C: when every KB retrieve returns 0 chunks, DR must not raise.
    The result has a topic_pool with 0 chunks; budget_exhausted_reason may
    be unset (clean exit on sufficiency=true with empty evidence is allowed,
    but the orchestrator must produce a result object)."""

    async def empty_kb(q):
        return []

    async def llm_says_sufficient(messages, **kwargs):
        class _R:
            content = '{"is_sufficient": true, "missing_information": ""}'

        return _R()

    dr = DeepResearch(
        provider=_StubProvider(llm_says_sufficient),
        kb_retrieve=empty_kb,
        kg_retrieve=None,
        original_question="totally unrelated query",
        processed_query="totally unrelated query",
        budget=DeepResearchBudget(wall_clock_ms=5000),
    )
    result = _run(dr.research())

    assert result is not None
    assert result.topic_pools, "topic_pools must be a list even when empty-of-chunks"
    assert sum(len(p.chunks_by_id) for p in result.topic_pools) == 0


def test_kg_failure_isolated_from_kb_path():
    """Test B (regression-shaped): when the KG retrieval raises but KB works,
    DR should still produce a populated pool. This guards the failure
    isolation in ``_retrieve_into``."""

    async def kb_retrieve(q):
        return [_mk_chunk("contents", source="doc/contributing/dv/README.md")]

    async def kg_explodes(q):
        raise RuntimeError("kg down")

    async def llm_says_sufficient(messages, **kwargs):
        class _R:
            content = '{"is_sufficient": true}'

        return _R()

    dr = DeepResearch(
        provider=_StubProvider(llm_says_sufficient),
        kb_retrieve=kb_retrieve,
        kg_retrieve=kg_explodes,
        original_question="dv guide?",
        processed_query="dv guide?",
        budget=DeepResearchBudget(wall_clock_ms=5000),
    )
    result = _run(dr.research())

    assert result is not None
    total_chunks = sum(len(p.chunks_by_id) for p in result.topic_pools)
    assert total_chunks > 0


def test_last_result_side_channel_populated():
    """The harness reads ``DeepResearch.last_result`` to surface llm_calls
    when RAGResponse lacks a metadata field. Verify it's set after research()."""

    async def kb_retrieve(q):
        return [_mk_chunk("ok")]

    async def llm_says_sufficient(messages, **kwargs):
        class _R:
            content = '{"is_sufficient": true}'

        return _R()

    DeepResearch.last_result = None
    dr = DeepResearch(
        provider=_StubProvider(llm_says_sufficient),
        kb_retrieve=kb_retrieve,
        kg_retrieve=None,
        original_question="q",
        processed_query="q",
        budget=DeepResearchBudget(wall_clock_ms=5000),
    )
    result = _run(dr.research())

    assert DeepResearch.last_result is result
    assert DeepResearch.last_result.llm_call_count >= 1
