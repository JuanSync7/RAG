# @summary
# Tests for turn-loop TurnContext assembly: tolerant coercion of the memory
# layer's structured dict (missing keys, None, malformed entry types all
# degrade to empty defaults), the most-recent chunk_refs cap, pending
# clarification passthrough, and the render_for_prompt digest surfacing every
# populated section.
# @end-summary
"""Tests for ``turn_loop/context.py``."""

from __future__ import annotations

from src.retrieval.pipeline.turn_loop import build_turn_context


def test_none_memory_builds_empty_context():
    context = build_turn_context("conv-1", None)

    assert context.conversation_id == "conv-1"
    assert context.rolling_summary == ""
    assert context.recent_turns == []
    assert context.chunk_refs == []
    assert context.docs_studied == []
    assert context.pending_clarification is None


def test_malformed_entries_are_dropped_not_raised():
    context = build_turn_context(
        "conv-1",
        {
            "rolling_summary": None,
            "recent_turns": [{"query": "q", "answer": "a"}, "not-a-dict", 7],
            "chunk_refs": "not-a-list",
            "docs_studied": [{"document_id": "d1"}, None],
            "pending_clarification": "not-a-dict",
        },
    )

    assert context.rolling_summary == ""
    assert context.recent_turns == [{"query": "q", "answer": "a"}]
    assert context.chunk_refs == []
    assert context.docs_studied == [{"document_id": "d1"}]
    assert context.pending_clarification is None


def test_chunk_refs_keep_the_most_recent_up_to_the_cap(monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "RAG_TURN_CONTEXT_MAX_CHUNK_REFS", 2, raising=False)
    refs = [{"chunk_id": f"c{i}"} for i in range(5)]

    context = build_turn_context("conv-1", {"chunk_refs": refs})

    assert [r["chunk_id"] for r in context.chunk_refs] == ["c3", "c4"]


def test_pending_clarification_and_digest_rendering():
    context = build_turn_context(
        "conv-1",
        {
            "rolling_summary": "we discussed setup",
            "recent_turns": [{"query": "how?", "answer": "like this"}],
            "chunk_refs": [
                {
                    "chunk_id": "c1",
                    "document_id": "doc-1",
                    "source_key": "docs/d1.md",
                    "heading": "Setup",
                    "preview": "the setup steps",
                }
            ],
            "docs_studied": [{"document_id": "doc-2", "conclusion": "covers flow"}],
            "pending_clarification": {"question": "Which one?", "hints": ["A", "B"]},
        },
    )

    digest = context.render_for_prompt(0)
    assert "we discussed setup" in digest
    assert "how?" in digest
    assert "doc-1" in digest and "docs/d1.md" in digest
    assert "covers flow" in digest
    assert "Which one?" in digest and "Hint 2: B" in digest
