"""S4: retrieval-mode query processor dispatch.

Hard sub-mode must NOT call the LLM. Auto sub-mode must run a single
retrieval-tuned LLM pass and populate ``history_decision`` /
``history_turns_used`` on the QueryResult. Malformed LLM output must
fall back to ``use_as_is`` with the literal query.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.retrieval.query.nodes import query_processor as qp
from src.retrieval.query.schemas import QueryAction


class _RecordingProvider:
    def __init__(self, content: str = ""):
        self.calls: list[dict] = []
        self._content = content

    def generate(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return SimpleNamespace(content=self._content)


def test_hard_mode_returns_literal_query_with_zero_llm_calls(monkeypatch):
    provider = _RecordingProvider(content="should never run")
    monkeypatch.setattr(qp, "get_llm_provider", lambda: provider)

    result = qp.process_query(
        raw_query="raw with prepended context",
        user_query="LITERAL USER TEXT",
        memory_context="prior turn 1\nprior turn 2",
        mode="retrieval",
        retrieval_sub_mode="hard",
    )

    assert result.processed_query == "LITERAL USER TEXT"
    assert result.standalone_query == "LITERAL USER TEXT"
    assert result.action == QueryAction.SEARCH
    assert result.history_decision == "hard_query"
    assert result.history_turns_used == 0
    assert result.suppress_memory is True
    # Critical: zero LLM calls in hard mode.
    assert provider.calls == []


def test_auto_mode_parses_llm_json_and_populates_history_fields(monkeypatch):
    provider = _RecordingProvider(
        content='{"decision": "partial_history", "processed_query": "transformer attention papers", "history_turns_used": 2}'
    )
    monkeypatch.setattr(qp, "get_llm_provider", lambda: provider)
    monkeypatch.setattr(qp, "_check_llm_available", lambda: True)
    # Force prompt cache to a known stub so we don't read the file.
    monkeypatch.setattr(qp, "_get_retrieval_rewriter_prompt", lambda: "<RETRIEVAL_PROMPT>")

    result = qp.process_query(
        raw_query="show me more like that",
        user_query="show me more like that",
        memory_context="USER: explain attention\nASSISTANT: ...",
        mode="retrieval",
        retrieval_sub_mode="auto",
    )

    assert result.processed_query == "transformer attention papers"
    assert result.history_decision == "partial_history"
    assert result.history_turns_used == 2
    assert result.action == QueryAction.SEARCH
    assert result.has_backward_reference is True
    assert len(provider.calls) == 1


def test_auto_mode_use_as_is_zeros_turns_used_even_if_llm_lies(monkeypatch):
    """If the LLM picks use_as_is but reports turns_used > 0, we override to 0
    — turns can't be 'used' under the no-history strategy."""
    provider = _RecordingProvider(
        content='{"decision": "use_as_is", "processed_query": "photosynthesis docs", "history_turns_used": 5}'
    )
    monkeypatch.setattr(qp, "get_llm_provider", lambda: provider)
    monkeypatch.setattr(qp, "_check_llm_available", lambda: True)
    monkeypatch.setattr(qp, "_get_retrieval_rewriter_prompt", lambda: "<RETRIEVAL_PROMPT>")

    result = qp.process_query(
        raw_query="photosynthesis", user_query="photosynthesis",
        mode="retrieval", retrieval_sub_mode="auto",
    )
    assert result.history_decision == "use_as_is"
    assert result.history_turns_used == 0
    assert result.has_backward_reference is False


def test_auto_mode_malformed_json_falls_back_to_literal_query(monkeypatch):
    provider = _RecordingProvider(content="this is not JSON at all")
    monkeypatch.setattr(qp, "get_llm_provider", lambda: provider)
    monkeypatch.setattr(qp, "_check_llm_available", lambda: True)
    monkeypatch.setattr(qp, "_get_retrieval_rewriter_prompt", lambda: "<RETRIEVAL_PROMPT>")

    result = qp.process_query(
        raw_query="literal text", user_query="literal text",
        mode="retrieval", retrieval_sub_mode="auto",
    )
    assert result.processed_query == "literal text"
    assert result.history_decision == "use_as_is"
    assert result.history_turns_used == 0


def test_auto_mode_unknown_decision_value_falls_back_to_use_as_is(monkeypatch):
    provider = _RecordingProvider(
        content='{"decision": "deep_history", "processed_query": "x", "history_turns_used": 9}'
    )
    monkeypatch.setattr(qp, "get_llm_provider", lambda: provider)
    monkeypatch.setattr(qp, "_check_llm_available", lambda: True)
    monkeypatch.setattr(qp, "_get_retrieval_rewriter_prompt", lambda: "<RETRIEVAL_PROMPT>")

    result = qp.process_query(
        raw_query="x", user_query="x",
        mode="retrieval", retrieval_sub_mode="auto",
    )
    assert result.history_decision == "use_as_is"


def test_auto_mode_llm_unavailable_returns_literal(monkeypatch):
    provider = _RecordingProvider(content="never")
    monkeypatch.setattr(qp, "get_llm_provider", lambda: provider)
    monkeypatch.setattr(qp, "_check_llm_available", lambda: False)
    monkeypatch.setattr(qp, "_get_retrieval_rewriter_prompt", lambda: "<RETRIEVAL_PROMPT>")

    result = qp.process_query(
        raw_query="anything", user_query="anything",
        mode="retrieval", retrieval_sub_mode="auto",
    )
    assert result.processed_query == "anything"
    assert result.history_decision == "use_as_is"
    # Critical: provider was not called when LLM is unavailable.
    assert provider.calls == []


def test_query_mode_unchanged_by_new_args(monkeypatch):
    """The default mode='query' path must continue to drive the LangGraph
    creation/verification loop and ignore retrieval_sub_mode."""
    provider = _RecordingProvider(content="never")
    monkeypatch.setattr(qp, "get_llm_provider", lambda: provider)

    # We can't easily run the full LangGraph in a unit test, so just confirm
    # the early-return retrieval branch is NOT taken for mode='query'. We do
    # this by checking that a stub _get_compiled_graph is invoked.
    invoked = {"called": False}

    class _StubGraph:
        def invoke(self, state):
            invoked["called"] = True
            return {
                **state,
                "current_query": state["original_query"],
                "confidence": 1.0,
                "iteration": 0,
                "action": "search",
                "clarification_message": "",
                "standalone_query": state["original_query"],
            }

    monkeypatch.setattr(qp, "_get_compiled_graph", lambda: _StubGraph())
    monkeypatch.setattr(qp, "_check_llm_available", lambda: True)

    result = qp.process_query(
        raw_query="test query",
        user_query="test query",
        mode="query",
        retrieval_sub_mode="auto",  # Should be ignored.
    )
    assert invoked["called"] is True
    assert result.history_decision is None
    assert result.history_turns_used == 0
