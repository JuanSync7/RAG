# @summary
# Shared fakes and factories for the turn-loop tests: FakeProvider (scripted
# agenerate responses + scripted generate_stream deltas, call log), make_deps
# (TurnLoopDeps over async stubs with an event collector), make_budget /
# make_chunk factories, and JSON helpers for scripting controller / judge /
# self-score / deep-study / clarify responses. No infrastructure — the whole
# loop runs on these fakes.
# @end-summary
"""Fixtures and fakes for turn-loop tests (pure DI, zero infrastructure)."""

from __future__ import annotations

import json
from typing import Any, Optional

import pytest

from src.retrieval.pipeline.turn_loop.schemas import (
    EvidenceChunk,
    TurnBudget,
    TurnContext,
    TurnLoopDeps,
)


class FakeLLMResponse:
    """Minimal LLMResponse stand-in (content + token counts read via getattr)."""

    def __init__(self, content: str, prompt_tokens: int = 7, completion_tokens: int = 3):
        self.content = content
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class FakeProvider:
    """Scripted LLM provider double.

    ``responses`` is a FIFO of str contents (or Exception instances to raise)
    consumed by ``agenerate``; ``streams`` is a FIFO of delta scripts (each a
    list of ``(kind, text)`` tuples, or an Exception raised mid-stream)
    consumed by ``generate_stream``. Every call is appended to ``calls`` as
    ``(method, messages, kwargs)`` for assertion.
    """

    def __init__(
        self,
        responses: Optional[list] = None,
        streams: Optional[list] = None,
    ):
        self.responses = list(responses or [])
        self.streams = list(streams or [])
        self.calls: list[tuple[str, list, dict]] = []

    async def agenerate(self, messages, **kwargs) -> FakeLLMResponse:
        self.calls.append(("agenerate", messages, kwargs))
        if not self.responses:
            raise AssertionError("FakeProvider ran out of scripted responses")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return FakeLLMResponse(item)

    def generate_stream(self, messages, **kwargs):
        self.calls.append(("generate_stream", messages, kwargs))
        script = self.streams.pop(0) if self.streams else [("content", "draft")]

        def _gen():
            for item in script:
                if isinstance(item, Exception):
                    raise item
                yield item

        return _gen()


def make_chunk(
    chunk_id: str,
    *,
    document_id: str = "doc-1",
    source_key: str = "docs/doc-1.md",
    source: str = "Doc One",
    heading: str = "Section A",
    text: str = "evidence text",
    score: float = 0.8,
    refactored_char_start: int = -1,
    refactored_char_end: int = -1,
) -> EvidenceChunk:
    """Build a retrieve-provenance EvidenceChunk with test defaults."""
    return EvidenceChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        source_key=source_key,
        source=source,
        heading=heading,
        text=text,
        score=score,
        refactored_char_start=refactored_char_start,
        refactored_char_end=refactored_char_end,
        provenance=EvidenceChunk.PROVENANCE_RETRIEVE,
        round_added=0,
    )


def make_budget(**overrides) -> TurnBudget:
    """Hand-built TurnBudget with small, test-friendly defaults."""
    values = dict(
        max_actions=5,
        max_llm_calls=20,
        wall_clock_ms=60_000,
        max_answer_attempts=2,
        deep_study_max_docs=2,
        deep_study_window_chars=100,
        deep_study_window_overlap_chars=20,
        deep_study_max_windows=3,
        clarify_max_hints=2,
        retrieve_top_k=5,
        answer_confidence_threshold=0.6,
        gate_weight_judge=0.5,
        gate_weight_self=0.3,
        gate_weight_citation=0.2,
    )
    values.update(overrides)
    return TurnBudget(**values)


class FakeDoc:
    """Stored-document stand-in for the fetch_document seam."""

    def __init__(self, content: str, title: Optional[str] = "Fake Doc"):
        self.content = content
        self.title = title


def make_deps(
    provider: FakeProvider,
    *,
    retrieve_batches: Optional[list[list[EvidenceChunk]]] = None,
    documents: Optional[dict[str, FakeDoc]] = None,
    emitted: Optional[list] = None,
    emit_error: Optional[Exception] = None,
) -> tuple[TurnLoopDeps, list]:
    """Build TurnLoopDeps over async fakes; returns ``(deps, emitted_events)``.

    ``retrieve_batches`` is a FIFO of chunk lists returned per retrieve call
    (empty list when exhausted). ``documents`` maps document_id/source_key to
    a FakeDoc. ``emit_error`` makes the sink raise (emitter-swallow tests).
    """
    batches = list(retrieve_batches or [])
    docs = dict(documents or {})
    events = emitted if emitted is not None else []

    async def retrieve_ranked(query_text, hyde_text, top_k):
        return batches.pop(0) if batches else []

    async def fetch_document(document_id, source_key):
        return docs.get(document_id) or docs.get(source_key)

    async def emit(event):
        if emit_error is not None:
            raise emit_error
        events.append(event)

    return (
        TurnLoopDeps(
            retrieve_ranked=retrieve_ranked,
            fetch_document=fetch_document,
            llm_provider=provider,
            emit=emit,
        ),
        events,
    )


def decision_json(action: str, reason: str = "scripted", confidence: float = 0.9, **args) -> str:
    """Serialize a controller decision response."""
    return json.dumps(
        {"action": action, "reason": reason, "confidence": confidence, "args": args}
    )


def judge_json(
    keep_indexes: list[int],
    total: int,
    *,
    sufficient: bool = True,
    confidence: float = 0.9,
    missing: str = "",
) -> str:
    """Serialize an agentic-judge response keeping ``keep_indexes`` in order."""
    return json.dumps(
        {
            "chunks": [
                {
                    "i": i,
                    "relevance": 1.0 if i in keep_indexes else 0.0,
                    "faithfulness": 1.0 if i in keep_indexes else 0.0,
                    "keep": i in keep_indexes,
                }
                for i in range(total)
            ],
            "ranking": keep_indexes,
            "pool": {
                "sufficient": sufficient,
                "confidence": confidence,
                "missing_information": missing,
            },
        }
    )


def selfscore_json(score: float, claims: Optional[list[str]] = None) -> str:
    """Serialize a turn_answer_selfscore response."""
    return json.dumps({"self_score": score, "unsupported_claims": claims or []})


def deep_read_json(notes: str, answer_found: bool = False, hint: str = "") -> str:
    """Serialize a turn_deep_study_read window response."""
    return json.dumps(
        {"notes": notes, "answer_found": answer_found, "next_window_hint": hint}
    )


def clarify_json(question: str, hints: list[str], scoping: Optional[list[str]] = None) -> str:
    """Serialize a turn_clarify_generate response."""
    return json.dumps(
        {"question": question, "hints": hints, "scoping_questions": scoping or []}
    )


@pytest.fixture
def empty_context() -> TurnContext:
    """A fresh-conversation TurnContext."""
    return TurnContext(conversation_id="conv-1")


def event_types(events: list) -> list[str]:
    """Project a TurnEvent list to its type names (ordering assertions)."""
    return [event.type for event in events]


def events_of(events: list, event_type: str) -> list[Any]:
    """Payloads of every event of ``event_type`` in emission order."""
    return [event.payload for event in events if event.type == event_type]
