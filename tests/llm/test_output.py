"""Unit tests for ``src.common.llm.output.structured_output``.

Covers the three-tier extraction strategy:

1. native ``with_structured_output`` success,
2. fallback to plain ``invoke`` + JSON parse + validate,
3. auto-fix via a cheap secondary model when validation fails,

plus the ``auto_fix=False`` re-raise branch and the ``model_used`` /
``auto_fixed`` metadata.  The LLM and the secondary fix model are the only
boundaries: a hand-rolled fake LLM object stands in for the LangChain
``BaseChatModel``, and ``get_llm`` is monkeypatched for the auto-fix path.
The real ``parse_json_object`` and Pydantic validation run unmocked.
"""
from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

import src.common.llm.output as output_mod
from src.common.llm.output import structured_output


class _Schema(BaseModel):
    name: str
    count: int


class _Msg:
    """Minimal stand-in for a LangChain message with a ``.content`` attr."""

    def __init__(self, content):
        self.content = content


class _FakeLLM:
    """Programmable fake BaseChatModel.

    ``structured_return`` controls the native path; ``structured_raises``
    forces the native path to throw; ``invoke_return`` is the plain-text
    fallback payload.
    """

    model_alias = "fake-model"

    def __init__(self, *, structured_return=..., structured_raises=None, invoke_return=None):
        self._structured_return = structured_return
        self._structured_raises = structured_raises
        self._invoke_return = invoke_return
        self.invoke_calls = 0

    def with_structured_output(self, schema):
        if self._structured_raises is not None:
            raise self._structured_raises
        outer = self

        class _Structured:
            def invoke(self, _messages):
                return outer._structured_return

        return _Structured()

    def invoke(self, _messages):
        self.invoke_calls += 1
        return self._invoke_return


# ── 1. Native structured output ─────────────────────────────────────────


def test_native_structured_output_success():
    """When with_structured_output returns a non-None object, use it directly."""
    obj = _Schema(name="ada", count=3)
    llm = _FakeLLM(structured_return=obj)
    res = structured_output(llm, _Schema, "prompt")
    assert res.parsed is obj
    assert res.auto_fixed is False
    assert res.raw == ""
    assert res.model_used == "fake-model"
    assert llm.invoke_calls == 0  # plain invoke never reached


def test_native_none_falls_through_to_plain_invoke():
    """A None native result is treated as a miss; plain invoke runs next."""
    llm = _FakeLLM(
        structured_return=None,
        invoke_return=_Msg('{"name": "bob", "count": 7}'),
    )
    res = structured_output(llm, _Schema, "prompt")
    assert res.parsed.name == "bob"
    assert res.parsed.count == 7
    assert res.auto_fixed is False
    assert llm.invoke_calls == 1


# ── 2. Plain invoke → parse → validate ──────────────────────────────────


def test_plain_invoke_after_native_raises():
    """A raising native path falls back to plain invoke + JSON parse."""
    llm = _FakeLLM(
        structured_raises=RuntimeError("no tool support"),
        invoke_return=_Msg('{"name": "cleo", "count": 1}'),
    )
    res = structured_output(llm, _Schema, "prompt")
    assert res.parsed.name == "cleo"
    assert res.raw == '{"name": "cleo", "count": 1}'
    assert res.auto_fixed is False


def test_plain_invoke_handles_response_without_content_attr():
    """A raw string response (no .content) is coerced via str()."""
    llm = _FakeLLM(
        structured_raises=RuntimeError("x"),
        invoke_return='{"name": "dot", "count": 9}',
    )
    res = structured_output(llm, _Schema, "prompt")
    assert res.parsed.name == "dot"
    assert res.parsed.count == 9


# ── 3. auto_fix=False re-raise ──────────────────────────────────────────


def test_validation_failure_without_autofix_reraises():
    """auto_fix=False re-raises the validation error instead of repairing."""
    llm = _FakeLLM(
        structured_raises=RuntimeError("x"),
        invoke_return=_Msg('{"name": "missing count"}'),
    )
    with pytest.raises(ValidationError):
        structured_output(llm, _Schema, "prompt", auto_fix=False)


# ── 4. Auto-fix path ────────────────────────────────────────────────────


def test_autofix_repairs_invalid_json(monkeypatch):
    """When validation fails and auto_fix is on, the fix model repairs it."""
    llm = _FakeLLM(
        structured_raises=RuntimeError("x"),
        invoke_return=_Msg("not json at all"),
    )

    class _FixLLM:
        def invoke(self, _messages):
            return _Msg('{"name": "fixed", "count": 42}')

    captured = {}

    def fake_get_llm(alias):
        captured["alias"] = alias
        return _FixLLM()

    monkeypatch.setattr(output_mod, "get_llm", fake_get_llm)

    res = structured_output(llm, _Schema, "prompt", fix_model_alias="cheap")
    assert res.parsed.name == "fixed"
    assert res.parsed.count == 42
    assert res.auto_fixed is True
    # raw is the ORIGINAL bad text, not the fixed text
    assert res.raw == "not json at all"
    assert res.model_used == "fake-model"
    assert captured["alias"] == "cheap"


def test_autofix_uses_default_alias_when_unspecified(monkeypatch):
    """The fix model alias defaults to 'default'."""
    llm = _FakeLLM(structured_raises=RuntimeError("x"), invoke_return=_Msg("garbage"))

    class _FixLLM:
        def invoke(self, _messages):
            return _Msg('{"name": "z", "count": 0}')

    captured = {}

    def fake_get_llm(alias):
        captured["alias"] = alias
        return _FixLLM()

    monkeypatch.setattr(output_mod, "get_llm", fake_get_llm)
    structured_output(llm, _Schema, "prompt")
    assert captured["alias"] == "default"
