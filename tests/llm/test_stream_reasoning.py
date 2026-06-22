"""Streaming reasoning-split tests for ``LLMProvider.generate_stream``.

Reasoning models (deepseek-r1 / qwen via a vLLM reasoning parser) emit their
chain-of-thought on ``delta.reasoning_content`` and the final answer on
``delta.content``. ``generate_stream`` must:

* by default yield plain content strings (the backward-compatible contract the
  LangChain adapter and other string callers depend on), dropping reasoning;
* when ``include_reasoning=True``, yield ``(kind, text)`` tuples that keep the
  two streams distinct so a UI can show the model "thinking".
"""
from __future__ import annotations

import types

from src.platform.llm.provider import LLMProvider
from src.platform.llm.schemas import LLMConfig


def _delta(*, content=None, reasoning=None):
    d = types.SimpleNamespace(content=content)
    if reasoning is not None:
        d.reasoning_content = reasoning
    return d


def _chunk(*, content=None, reasoning=None):
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(delta=_delta(content=content, reasoning=reasoning))]
    )


def _make_streaming_provider(chunks):
    config = LLMConfig(
        model="ollama/qwen",
        api_base=None,
        api_key=None,
        max_tokens=128,
        temperature=0.0,
        num_retries=0,
        fallback_models=[],
        vision_model=None,
        query_model=None,
    )

    class _FakeRouter:
        def completion(self, **kwargs):
            assert kwargs.get("stream") is True
            return iter(chunks)

    return LLMProvider(config=config, router=_FakeRouter())


def test_generate_stream_default_yields_content_strings_only():
    """Default contract: plain content strings; reasoning deltas are dropped."""
    provider = _make_streaming_provider([
        _chunk(reasoning="let me think"),
        _chunk(content="hel"),
        _chunk(content="lo"),
    ])
    out = list(provider.generate_stream([{"role": "user", "content": "q"}]))
    assert out == ["hel", "lo"]  # strings, no tuples, no reasoning


def test_generate_stream_include_reasoning_yields_kind_text_tuples():
    """With include_reasoning, reasoning and content are kept distinct and ordered."""
    provider = _make_streaming_provider([
        _chunk(reasoning="let me "),
        _chunk(reasoning="think"),
        _chunk(content="final "),
        _chunk(content="answer"),
    ])
    out = list(provider.generate_stream(
        [{"role": "user", "content": "q"}], include_reasoning=True
    ))
    assert out == [
        ("reasoning", "let me "),
        ("reasoning", "think"),
        ("content", "final "),
        ("content", "answer"),
    ]


def test_generate_stream_include_reasoning_handles_combined_delta():
    """A single delta carrying both reasoning and content emits both, reasoning first."""
    provider = _make_streaming_provider([
        _chunk(reasoning="hmm", content="ok"),
    ])
    out = list(provider.generate_stream(
        [{"role": "user", "content": "q"}], include_reasoning=True
    ))
    assert out == [("reasoning", "hmm"), ("content", "ok")]


def test_generate_stream_include_reasoning_skips_empty_deltas():
    """Empty/None content and absent reasoning_content produce no spurious yields."""
    provider = _make_streaming_provider([
        _chunk(),                       # nothing
        _chunk(content=""),             # empty content
        _chunk(content="real"),
    ])
    out = list(provider.generate_stream(
        [{"role": "user", "content": "q"}], include_reasoning=True
    ))
    assert out == [("content", "real")]
