"""OTel instrumentation tests for src.common.llm.provider.ChatLLMAdapter."""
from __future__ import annotations

import sys
from typing import Iterator

# tests/conftest.py installs a minimal langchain_core stub. Evict it before
# importing the real package needed by these tests. Also evict the src LLM
# modules that capture langchain symbols at import time: if a sibling test
# imported them while the (stubbed or earlier-real) langchain was live, a bare
# re-import here would be a stale sys.modules cache hit bound to a different
# langchain identity, desyncing isinstance/pydantic checks. Dropping them forces
# a fresh re-import against the real langchain we load below.
for _mod in list(sys.modules):
    if (
        _mod == "langchain_core"
        or _mod.startswith("langchain_core.")
        or _mod == "langgraph"
        or _mod.startswith("langgraph.")
        or _mod == "langchain_text_splitters"
        or _mod == "src.common.llm"
        or _mod.startswith("src.common.llm.")
        or _mod == "src.platform.llm"
        or _mod.startswith("src.platform.llm.")
    ):
        del sys.modules[_mod]

import pytest

from opentelemetry.trace import StatusCode

from langchain_core.messages import HumanMessage

from src.common.llm.provider import ChatLLMAdapter
from src.platform.llm.schemas import LLMResponse


@pytest.fixture
def stub_platform_provider(monkeypatch):
    """Replace src.platform.llm.get_llm_provider with a stub returning a programmable response."""
    state = {"response": None, "raises": None, "stream": None}

    class _StubProvider:
        def generate(self, msg_dicts, **kwargs):
            if state["raises"] is not None:
                raise state["raises"]
            return state["response"]

        def generate_stream(self, msg_dicts, **kwargs) -> Iterator[str]:
            if state["raises"] is not None:
                raise state["raises"]
            yield from (state["stream"] or [])

    from src.platform import llm as llm_pkg
    monkeypatch.setattr(llm_pkg.provider, "_provider", _StubProvider())
    return state


def test_chat_adapter_generate_emits_adapter_sync_span(otel_capture, stub_platform_provider):
    """ChatLLMAdapter._generate must emit a 'llm.adapter.sync' span; inner LLMProvider span also fires."""
    stub_platform_provider["response"] = LLMResponse(
        content="reply",
        model="ollama/llama3",
        prompt_tokens=2,
        completion_tokens=3,
        total_tokens=5,
        cost_usd=0.0,
    )
    adapter = ChatLLMAdapter(model_alias="default")
    result = adapter.invoke([HumanMessage(content="hello")])
    assert result.content == "reply"

    spans = otel_capture.get_finished_spans()
    names = [s.name for s in spans]
    assert "llm.adapter.sync" in names
    sync = next(s for s in spans if s.name == "llm.adapter.sync")
    attrs = dict(sync.attributes)
    assert attrs["gen_ai.system"] == "langchain"
    assert attrs["gen_ai.request.model"] == "default"
    assert attrs["gen_ai.completion"] == "reply"
    assert attrs["gen_ai.usage.input_tokens"] == 2
    assert attrs["gen_ai.usage.output_tokens"] == 3


def test_chat_adapter_generate_records_error(otel_capture, stub_platform_provider):
    """When the underlying provider raises, the adapter.sync span must record ERROR."""
    stub_platform_provider["raises"] = RuntimeError("ad-boom")
    adapter = ChatLLMAdapter(model_alias="default")
    with pytest.raises(RuntimeError, match="ad-boom"):
        adapter.invoke([HumanMessage(content="hi")])
    spans = otel_capture.get_finished_spans()
    sync_spans = [s for s in spans if s.name == "llm.adapter.sync"]
    assert len(sync_spans) == 1
    assert sync_spans[0].status.status_code == StatusCode.ERROR


def test_chat_adapter_stream_emits_adapter_stream_span(otel_capture, stub_platform_provider):
    """ChatLLMAdapter._stream must emit a 'llm.adapter.stream' span that ends on stream-end."""
    stub_platform_provider["stream"] = ["he", "llo"]
    adapter = ChatLLMAdapter(model_alias="default")
    chunks = list(adapter.stream([HumanMessage(content="hi")]))
    assert "".join(c.content for c in chunks) == "hello"

    spans = otel_capture.get_finished_spans()
    stream_spans = [s for s in spans if s.name == "llm.adapter.stream"]
    assert len(stream_spans) == 1
    attrs = dict(stream_spans[0].attributes)
    assert attrs["gen_ai.system"] == "langchain"
    assert attrs["gen_ai.request.model"] == "default"
    assert attrs["gen_ai.completion"] == "hello"


def test_chat_adapter_stream_records_error(otel_capture, stub_platform_provider):
    """When generate_stream raises, the adapter.stream span must record ERROR."""
    stub_platform_provider["raises"] = RuntimeError("stream-fail")
    adapter = ChatLLMAdapter(model_alias="default")
    with pytest.raises(RuntimeError, match="stream-fail"):
        list(adapter.stream([HumanMessage(content="hi")]))
    spans = otel_capture.get_finished_spans()
    stream_spans = [s for s in spans if s.name == "llm.adapter.stream"]
    assert len(stream_spans) == 1
    assert stream_spans[0].status.status_code == StatusCode.ERROR
