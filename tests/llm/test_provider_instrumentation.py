"""OTel instrumentation tests for src.platform.llm.provider.LLMProvider.

These tests run the real OTelBackend against an in-memory span exporter
and only mock the litellm Router's network call. Span shape, attributes,
parent linkage, and error status are verified end-to-end.
"""
from __future__ import annotations

import types

import pytest

from opentelemetry.trace import StatusCode

from src.platform.llm.provider import LLMProvider
from src.platform.llm.schemas import LLMConfig
from src.platform.observability import get_tracer


def _make_provider(monkeypatch, *, response=None, raises=None, async_response=None):
    """Construct an LLMProvider whose Router.completion is stubbed."""
    config = LLMConfig(
        model="ollama/llama3",
        api_base=None,
        api_key=None,
        max_tokens=128,
        temperature=0.0,
        num_retries=0,
        fallback_models=[],
        vision_model=None,
        query_model=None,
    )

    # Fake router that bypasses litellm entirely.
    class _FakeRouter:
        def completion(self, **kwargs):
            if raises is not None:
                raise raises
            return response

        async def acompletion(self, **kwargs):
            if raises is not None:
                raise raises
            return async_response if async_response is not None else response

    return LLMProvider(config=config, router=_FakeRouter())


def _fake_completion_response(content="hello there", model="ollama/llama3", pt=5, ct=7):
    """Minimal litellm-like ModelResponse shape used by LLMProvider.generate()."""
    usage = types.SimpleNamespace(prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct)
    message = types.SimpleNamespace(content=content)
    choice = types.SimpleNamespace(message=message)
    return types.SimpleNamespace(choices=[choice], model=model, usage=usage)


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------

def test_generate_emits_llm_generate_span_with_gen_ai_attrs(otel_capture, monkeypatch):
    """LLMProvider.generate must emit a span named 'llm.generate' with gen_ai.* attrs."""
    provider = _make_provider(monkeypatch, response=_fake_completion_response())
    out = provider.generate([{"role": "user", "content": "hi"}])
    assert out.content == "hello there"

    spans = otel_capture.get_finished_spans()
    gen_spans = [s for s in spans if s.name == "llm.generate"]
    assert len(gen_spans) == 1
    attrs = dict(gen_spans[0].attributes)
    assert attrs["gen_ai.system"] == "litellm"
    assert attrs["gen_ai.request.model"] == "ollama/llama3"
    assert "gen_ai.prompt" in attrs
    assert attrs["gen_ai.completion"] == "hello there"
    assert attrs["gen_ai.usage.input_tokens"] == 5
    assert attrs["gen_ai.usage.output_tokens"] == 7


def test_generate_records_error_status_on_raise(otel_capture, monkeypatch):
    """When Router.completion raises, the generation span must be ERROR and re-raise."""
    provider = _make_provider(monkeypatch, raises=RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        provider.generate([{"role": "user", "content": "hi"}])

    spans = otel_capture.get_finished_spans()
    gen_spans = [s for s in spans if s.name == "llm.generate"]
    assert len(gen_spans) == 1
    assert gen_spans[0].status.status_code == StatusCode.ERROR
    assert any(ev.name == "exception" for ev in gen_spans[0].events)


def test_generate_nests_under_active_trace(otel_capture, monkeypatch):
    """A llm.generate emitted under tracer.trace() must parent under the trace root.

    Note: OTelBackend nests via Trace/parent linkage, not via OTel ambient
    current-context (the backend does not call use_span). Production callers
    that want nesting must wire it through tracer.trace() or by passing
    parent=. See observability-otel summary notes.
    """
    provider = _make_provider(monkeypatch, response=_fake_completion_response())
    tracer = get_tracer()
    trace = tracer.trace("pipeline.run")
    # Bind generation under the trace root by using trace.generation directly;
    # LLMProvider does not know about the outer trace, so we verify the span
    # is emitted at all — and that, when the inner span is started inside the
    # OTel current-span context that an outer use_span call activates,
    # parent linkage is established by OTel itself.
    from opentelemetry import trace as otel_trace
    with otel_trace.use_span(trace._span, end_on_exit=False):  # type: ignore[attr-defined]
        provider.generate([{"role": "user", "content": "hi"}])
    trace.__exit__(None, None, None)

    spans = otel_capture.get_finished_spans()
    by_name = {s.name: s for s in spans}
    outer_sp = by_name["pipeline.run"]
    gen_sp = by_name["llm.generate"]
    assert gen_sp.context.trace_id == outer_sp.context.trace_id
    assert gen_sp.parent is not None
    assert gen_sp.parent.span_id == outer_sp.context.span_id


# ---------------------------------------------------------------------------
# agenerate()
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_agenerate_emits_llm_agenerate_span(otel_capture, monkeypatch):
    """LLMProvider.agenerate must emit 'llm.agenerate' with gen_ai.* attrs."""
    provider = _make_provider(monkeypatch, async_response=_fake_completion_response(content="async-out", pt=3, ct=4))
    out = await provider.agenerate([{"role": "user", "content": "yo"}])
    assert out.content == "async-out"

    spans = otel_capture.get_finished_spans()
    gen_spans = [s for s in spans if s.name == "llm.agenerate"]
    assert len(gen_spans) == 1
    attrs = dict(gen_spans[0].attributes)
    assert attrs["gen_ai.system"] == "litellm"
    assert attrs["gen_ai.request.model"] == "ollama/llama3"
    assert attrs["gen_ai.completion"] == "async-out"
    assert attrs["gen_ai.usage.input_tokens"] == 3
    assert attrs["gen_ai.usage.output_tokens"] == 4


@pytest.mark.asyncio
async def test_agenerate_records_error_status_on_raise(otel_capture, monkeypatch):
    """When acompletion raises, the agenerate span must record ERROR."""
    provider = _make_provider(monkeypatch, raises=RuntimeError("akaboom"))
    with pytest.raises(RuntimeError, match="akaboom"):
        await provider.agenerate([{"role": "user", "content": "hi"}])
    spans = otel_capture.get_finished_spans()
    gen_spans = [s for s in spans if s.name == "llm.agenerate"]
    assert len(gen_spans) == 1
    assert gen_spans[0].status.status_code == StatusCode.ERROR


# ---------------------------------------------------------------------------
# call_oneshot()
# ---------------------------------------------------------------------------

def test_call_oneshot_emits_inner_generate_span(otel_capture, monkeypatch):
    """call_oneshot delegates to LLMProvider.generate so a llm.generate span is emitted.

    No outer 'llm.oneshot' span is added — the inner llm.generate already
    carries gen_ai.* attrs and the OTel backend does not implicitly parent
    spans by ambient context, so a redundant outer span would not nest.
    """
    from src.platform import llm as llm_pkg

    provider = _make_provider(monkeypatch, response=_fake_completion_response(content="done"))
    monkeypatch.setattr(llm_pkg.provider, "_provider", provider)

    out = llm_pkg.call_oneshot("question?")
    assert out == "done"

    spans = otel_capture.get_finished_spans()
    by_name = {s.name: s for s in spans}
    assert "llm.generate" in by_name
