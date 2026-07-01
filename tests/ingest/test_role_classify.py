# @summary
# Unit tests for the shared LLM chunk-role classifier (src/ingest/common/
# role_classify.py): correct index->role mapping, fail-open to "content" on
# empty/garbage/missing-index/exception, length/order preservation, batching of
# a >batch_size input into multiple calls, and prefix-capping of chunk text sent
# to the model. No live models — a FAKE provider returns canned JSON.
# @end-summary
"""Contract tests for classify_roles (the model-driven chunk-role tagger).

The classifier NEVER drops a chunk on its own; it only labels. The single most
important invariant under test is FAIL-OPEN: any classifier error, parse
failure, missing index, or ambiguity yields role="content" so real content is
never silently dropped downstream.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import pytest

from src.ingest.common.role_classify import classify_roles


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class _LLMResp:
    content: str
    model: str = "fake"
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class _Chunk:
    """Minimal object exposing ``.text`` (the classifier accepts either an
    object with ``.text`` or a dict with ``"text"``)."""

    text: str


class FakeProvider:
    """Records every call and returns a canned response.

    ``responses`` may be:
    - a single str (same response for every batch call), or
    - a list[str] consumed one per call (so multi-batch inputs can be driven
      with a distinct payload per batch), or
    - a callable(prompt) -> str.

    Every prompt is recorded in ``self.prompts`` and every kwargs dict in
    ``self.calls`` so tests can assert batching and prefix-capping.
    """

    def __init__(self, responses: Any) -> None:
        self._responses = responses
        self._idx = 0
        self.prompts: list[str] = []
        self.calls: list[dict[str, Any]] = []

    async def agenerate(self, messages, **kwargs):
        prompt = str(messages[-1].get("content", ""))
        self.prompts.append(prompt)
        self.calls.append(kwargs)
        if callable(self._responses):
            content = self._responses(prompt)
        elif isinstance(self._responses, list):
            content = self._responses[min(self._idx, len(self._responses) - 1)]
            self._idx += 1
        else:
            content = self._responses
        return _LLMResp(content=str(content))


def _roles_payload(roles: list[str]) -> str:
    """Build a well-formed classifier JSON response for the given roles."""
    return json.dumps(
        {"roles": [{"i": i, "role": r} for i, r in enumerate(roles)]}
    )


def _run(coro):
    return asyncio.run(coro)


_KW = dict(
    model_alias="judge",
    batch_size=40,
    prefix_chars=350,
    timeout_s=120,
    max_output_tokens=1500,
)


# ---------------------------------------------------------------------------
# (a) correct role mapping by index
# ---------------------------------------------------------------------------


def test_correct_role_mapping_by_index():
    chunks = [_Chunk("table of contents ...... 5"), _Chunk("The bus has 5 channels."),
              _Chunk("Copyright 2026 ACME")]
    provider = FakeProvider(_roles_payload(["navigation", "content", "boilerplate"]))
    roles = _run(classify_roles(provider, chunks, **_KW))
    assert roles == ["navigation", "content", "boilerplate"]


def test_accepts_dict_chunks():
    chunks = [{"text": "see chapter 4"}, {"text": "Register field WSTRB is 4 bits."}]
    provider = FakeProvider(_roles_payload(["navigation", "content"]))
    roles = _run(classify_roles(provider, chunks, **_KW))
    assert roles == ["navigation", "content"]


def test_out_of_order_indices_are_mapped_not_positional():
    # Model returns entries in a scrambled order; mapping must be BY index "i".
    chunks = [_Chunk("a"), _Chunk("b"), _Chunk("c")]
    payload = json.dumps({"roles": [
        {"i": 2, "role": "boilerplate"},
        {"i": 0, "role": "navigation"},
        {"i": 1, "role": "content"},
    ]})
    provider = FakeProvider(payload)
    roles = _run(classify_roles(provider, chunks, **_KW))
    assert roles == ["navigation", "content", "boilerplate"]


# ---------------------------------------------------------------------------
# (b) fail-open to "content"
# ---------------------------------------------------------------------------


def test_fail_open_on_empty_response():
    chunks = [_Chunk("x"), _Chunk("y")]
    provider = FakeProvider("")
    roles = _run(classify_roles(provider, chunks, **_KW))
    assert roles == ["content", "content"]


def test_fail_open_on_garbage_response():
    chunks = [_Chunk("x"), _Chunk("y")]
    provider = FakeProvider("not json at all {{{")
    roles = _run(classify_roles(provider, chunks, **_KW))
    assert roles == ["content", "content"]


def test_fail_open_on_missing_index():
    # Model only labels index 0; index 1 is omitted -> defaults to content.
    chunks = [_Chunk("x"), _Chunk("y")]
    payload = json.dumps({"roles": [{"i": 0, "role": "navigation"}]})
    provider = FakeProvider(payload)
    roles = _run(classify_roles(provider, chunks, **_KW))
    assert roles == ["navigation", "content"]


def test_fail_open_on_unknown_role_value():
    # An out-of-vocabulary role string is ambiguity -> content.
    chunks = [_Chunk("x"), _Chunk("y")]
    payload = json.dumps({"roles": [
        {"i": 0, "role": "garbage-role"},
        {"i": 1, "role": "content"},
    ]})
    provider = FakeProvider(payload)
    roles = _run(classify_roles(provider, chunks, **_KW))
    assert roles == ["content", "content"]


def test_fail_open_on_provider_exception():
    class BoomProvider:
        async def agenerate(self, messages, **kwargs):
            raise RuntimeError("model down")

    chunks = [_Chunk("x"), _Chunk("y"), _Chunk("z")]
    roles = _run(classify_roles(BoomProvider(), chunks, **_KW))
    assert roles == ["content", "content", "content"]


def test_fail_open_only_affects_failing_batch():
    # batch_size=2 -> 2 batches. First batch good, second batch returns garbage.
    chunks = [_Chunk("a"), _Chunk("b"), _Chunk("c"), _Chunk("d")]
    provider = FakeProvider([
        _roles_payload(["navigation", "boilerplate"]),
        "garbage",
    ])
    roles = _run(classify_roles(provider, chunks, **{**_KW, "batch_size": 2}))
    assert roles == ["navigation", "boilerplate", "content", "content"]


def test_custom_default_role_is_honoured_on_failure():
    chunks = [_Chunk("x"), _Chunk("y")]
    provider = FakeProvider("")
    roles = _run(classify_roles(provider, chunks, **{**_KW, "default_role": "content"}))
    assert roles == ["content", "content"]


# ---------------------------------------------------------------------------
# (c) output length == input length (and order preserved)
# ---------------------------------------------------------------------------


def test_output_length_matches_input():
    chunks = [_Chunk(f"chunk {i}") for i in range(7)]
    provider = FakeProvider(_roles_payload(["content"] * 7))
    roles = _run(classify_roles(provider, chunks, **_KW))
    assert len(roles) == 7


def test_empty_input_returns_empty():
    provider = FakeProvider(_roles_payload([]))
    roles = _run(classify_roles(provider, [], **_KW))
    assert roles == []


# ---------------------------------------------------------------------------
# (d) batching splits a >batch_size input into multiple calls
# ---------------------------------------------------------------------------


def test_batching_splits_into_multiple_calls():
    # 5 chunks, batch_size=2 -> ceil(5/2) = 3 calls.
    chunks = [_Chunk(f"c{i}") for i in range(5)]
    provider = FakeProvider([
        _roles_payload(["content", "content"]),
        _roles_payload(["navigation", "content"]),
        _roles_payload(["boilerplate"]),
    ])
    roles = _run(classify_roles(provider, chunks, **{**_KW, "batch_size": 2}))
    assert len(provider.calls) == 3
    assert roles == ["content", "content", "navigation", "content", "boilerplate"]


def test_single_call_when_within_batch_size():
    chunks = [_Chunk(f"c{i}") for i in range(3)]
    provider = FakeProvider(_roles_payload(["content"] * 3))
    _run(classify_roles(provider, chunks, **{**_KW, "batch_size": 40}))
    assert len(provider.calls) == 1


# ---------------------------------------------------------------------------
# (e) only the first prefix_chars are sent
# ---------------------------------------------------------------------------


def test_prefix_chars_caps_text_sent_to_model():
    long_tail = "ZZTAIL_MARKER"
    head = "H" * 20
    chunks = [_Chunk(head + ("x" * 1000) + long_tail)]
    provider = FakeProvider(_roles_payload(["content"]))
    _run(classify_roles(provider, chunks, **{**_KW, "prefix_chars": 10}))
    prompt = provider.prompts[0]
    # The capped head is present; the far tail beyond prefix_chars is not.
    assert head[:10] in prompt
    assert long_tail not in prompt


def test_prefix_chars_keeps_short_text_intact():
    chunks = [_Chunk("short")]
    provider = FakeProvider(_roles_payload(["content"]))
    _run(classify_roles(provider, chunks, **{**_KW, "prefix_chars": 350}))
    assert "short" in provider.prompts[0]


# ---------------------------------------------------------------------------
# json_mode / response_format plumbing
# ---------------------------------------------------------------------------


def test_json_mode_requests_response_format():
    chunks = [_Chunk("x")]
    provider = FakeProvider(_roles_payload(["content"]))
    _run(classify_roles(provider, chunks, **{**_KW, "json_mode": True}))
    assert provider.calls[0].get("response_format") == {"type": "json_object"}


def test_json_mode_off_omits_response_format():
    chunks = [_Chunk("x")]
    provider = FakeProvider(_roles_payload(["content"]))
    _run(classify_roles(provider, chunks, **{**_KW, "json_mode": False}))
    assert "response_format" not in provider.calls[0]


def test_model_alias_and_timeout_forwarded():
    chunks = [_Chunk("x")]
    provider = FakeProvider(_roles_payload(["content"]))
    _run(classify_roles(provider, chunks, **{**_KW, "model_alias": "judge", "timeout_s": 99}))
    assert provider.calls[0].get("model_alias") == "judge"
    assert provider.calls[0].get("timeout") == 99
