# @summary
# Unit tests for the suggested follow-up questions generator (src/retrieval/
# pipeline/follow_up.py): count cap, dedup, original-question echo suppression,
# grounding headings reaching the prompt, and FAIL-OPEN on every failure mode
# (empty answer, provider error, unparseable JSON) — an advisory feature must
# never raise. No live models — the provider is a stub.
# Deps: pytest, src.retrieval.pipeline.follow_up
# @end-summary
"""Unit tests for follow-up-question generation."""
from __future__ import annotations

import asyncio
import json
from typing import Any

from src.retrieval.pipeline import follow_up
from src.retrieval.pipeline.follow_up import (
    _coerce_questions,
    _headings_block,
    generate_follow_ups,
)


class _Resp:
    def __init__(self, content: str) -> None:
        self.content = content


class _StubProvider:
    """Records the prompt it saw and returns a canned payload (dict → JSON)."""

    def __init__(self, payload: Any) -> None:
        self._payload = payload
        self.prompts: list[str] = []

    async def agenerate(self, messages, **kwargs):
        self.prompts.append(messages[0]["content"])
        body = self._payload if isinstance(self._payload, str) else json.dumps(self._payload)
        return _Resp(body)


def _gen(provider, **kw):
    defaults = dict(question="What is AXI?", answer="AXI has 5 channels.",
                    headings=["A1 About AXI", "A2 Signals"], count=3)
    defaults.update(kw)
    return asyncio.run(generate_follow_ups(provider, **defaults))


def test_returns_questions_and_caps_to_count():
    p = _StubProvider({"questions": ["Q1?", "Q2?", "Q3?", "Q4?"]})
    out = _gen(p, count=3)
    assert out == ["Q1?", "Q2?", "Q3?"]  # capped at count


def test_dedupes_and_drops_original_question_echo():
    p = _StubProvider({"questions": ["What is AXI?", "Q2?", "q2?", "Q3?"]})
    out = _gen(p, question="What is AXI?", count=3)
    assert "What is AXI?" not in out          # original echo removed
    assert out == ["Q2?", "Q3?"]              # case-insensitive dedupe of Q2


def test_empty_answer_short_circuits_no_call():
    p = _StubProvider({"questions": ["Q1?"]})
    out = _gen(p, answer="   ")
    assert out == []
    assert p.prompts == []  # no LLM call when there's no answer


def test_fail_open_on_provider_error():
    class _Boom:
        async def agenerate(self, messages, **kwargs):
            raise RuntimeError("model down")
    assert _gen(_Boom()) == []


def test_fail_open_on_unparseable_json():
    assert _gen(_StubProvider("not json at all")) == []
    assert _gen(_StubProvider("{}")) == []            # no 'questions' key
    assert _gen(_StubProvider({"questions": []})) == []


def test_headings_reach_the_prompt_for_grounding():
    p = _StubProvider({"questions": ["Q1?"]})
    _gen(p, headings=["B1.5 Coherence overview", "B2 Channels"])
    assert p.prompts, "expected a prompt to be captured"
    assert "B1.5 Coherence overview" in p.prompts[0]  # grounding headings injected


def test_headings_block_dedupes_and_caps():
    block = _headings_block(["A", "A", "B", "", "  "])
    assert block == "- A\n- B"
    assert _headings_block([]) == "(no section headings available)"


def test_coerce_ignores_non_dict_and_non_list():
    assert _coerce_questions("nope", 3, "orig") == []
    assert _coerce_questions({"questions": "single string"}, 3, "orig") == ["single string"]
