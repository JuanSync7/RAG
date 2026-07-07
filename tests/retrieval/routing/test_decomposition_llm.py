# @summary
# Unit tests for the Tier-2 glossary-grounded LLM comparison-decomposer
# (``llm_decompose``) and its failure-signalling exception ``DecompositionError``.
# Pins the C2 contract from DOCUMENT_ROUTING_DESIGN.md §6.5: on success return
# 2-5 validated standalone sub-queries; on ANY failure (provider error/timeout,
# unparseable, wrong shape, out-of-range count, empty/whitespace/over-long item)
# RAISE ``DecompositionError`` so the C3 orchestrator can fall back. The LLM
# provider is always MOCKED — no live calls.
# Deps: pytest, src.retrieval.routing.decomposition, src.platform.llm.schemas
# @end-summary
"""Unit tests for ``src.retrieval.routing.decomposition.llm_decompose``.

The Tier-2 decomposer is the *primary* comparison-decomposition path. It calls
the LLM provider's ``json_completion`` grounded in ``PROTOCOL_GLOSSARY`` and
expects a JSON object ``{"subqueries": [...]}`` with 2-5 short, non-empty
standalone sub-queries (one per compared entity). Any deviation must RAISE
``DecompositionError`` — C2 never falls back itself; it only signals failure.

Mocking strategy: ``llm_decompose`` resolves the provider via
``get_llm_provider`` *as imported into the decomposition module*, so these tests
monkeypatch ``src.retrieval.routing.decomposition.get_llm_provider`` with a fake
factory returning a fake provider whose ``json_completion`` returns a real
``LLMResponse`` (matching the provider's actual return shape) whose ``.content``
is the canned JSON — or raises, to simulate a provider error/timeout.

Bare-list JSON decision (documented + tested): the project's canonical parser
``src.common.parse_json_object`` only ever yields a ``dict`` (a top-level JSON
array decodes to ``{}``). ``llm_decompose`` therefore does NOT tolerate a bare
JSON list — such a response is treated as a failure and RAISES.
"""
from __future__ import annotations

import pytest

from config import settings
from src.platform.llm.schemas import LLMResponse
from src.retrieval.routing import decomposition
from src.retrieval.routing.decomposition import DecompositionError, llm_decompose


# ─── Fakes ──────────────────────────────────────────────────────────────────


class _FakeProvider:
    """Fake LLM provider capturing the call and returning/raising on demand."""

    def __init__(self, *, content: str | None = None, exc: Exception | None = None):
        self._content = content
        self._exc = exc
        self.calls: list[dict] = []

    def json_completion(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        if self._exc is not None:
            raise self._exc
        return LLMResponse(content=self._content or "", model="fake-model")


def _install_provider(monkeypatch, provider: _FakeProvider) -> _FakeProvider:
    """Monkeypatch ``get_llm_provider`` (as imported in decomposition) → provider."""
    monkeypatch.setattr(decomposition, "get_llm_provider", lambda: provider)
    return provider


def _install_content(monkeypatch, content: str) -> _FakeProvider:
    return _install_provider(monkeypatch, _FakeProvider(content=content))


# ─── POSITIVE: valid JSON object → cleaned sub-queries ──────────────────────


def test_valid_subqueries_returns_cleaned_strings(monkeypatch):
    _install_content(
        monkeypatch,
        '{"subqueries": ["AXI4 burst transaction ordering", "CHI coherency model"]}',
    )
    result = llm_decompose("compare AXI4 vs CHI for coherency")
    assert result == [
        "AXI4 burst transaction ordering",
        "CHI coherency model",
    ]


def test_valid_subqueries_strips_whitespace(monkeypatch):
    _install_content(
        monkeypatch,
        '{"subqueries": ["  AXI4 ordering  ", "\\tCHI coherency\\n"]}',
    )
    result = llm_decompose("AXI4 vs CHI")
    assert result == ["AXI4 ordering", "CHI coherency"]


def test_valid_three_items_in_range(monkeypatch):
    _install_content(
        monkeypatch,
        '{"subqueries": ["AXI4 details", "AXI5 details", "CHI details"]}',
    )
    result = llm_decompose("AXI4 / AXI5 / CHI")
    assert result == ["AXI4 details", "AXI5 details", "CHI details"]


def test_tolerates_prose_wrapped_json(monkeypatch):
    # parse_json_object tolerates surrounding prose / fences; llm_decompose
    # should accept it too since it reuses the canonical parser.
    _install_content(
        monkeypatch,
        'Here is the JSON:\n{"subqueries": ["AXI4 spec", "CHI spec"]}\nDone.',
    )
    assert llm_decompose("AXI4 vs CHI") == ["AXI4 spec", "CHI spec"]


# ─── PROMPT GROUNDING / CALL SHAPE ──────────────────────────────────────────


def test_prompt_is_glossary_grounded_and_capped(monkeypatch):
    provider = _install_content(
        monkeypatch, '{"subqueries": ["AXI4 x", "CHI y"]}'
    )
    llm_decompose("AXI4 vs CHI")

    assert len(provider.calls) == 1
    call = provider.calls[0]
    messages = call["messages"]
    # System + user messages present.
    roles = [m["role"] for m in messages]
    assert "system" in roles and "user" in roles
    # The user query is forwarded verbatim somewhere in the messages.
    joined = "\n".join(m["content"] for m in messages)
    assert "AXI4 vs CHI" in joined
    # Grounding: at least some canonical glossary entities appear in the prompt.
    assert "CHI" in joined and "AXI4" in joined
    # Capped output + low temperature + JSON-only intent.
    kwargs = call["kwargs"]
    assert kwargs["max_tokens"] <= 256
    assert kwargs["temperature"] <= 0.1
    # Default timeout flows from settings when not overridden.
    assert kwargs["timeout"] == settings.RAG_DECOMPOSITION_LLM_TIMEOUT_SECONDS


def test_explicit_timeout_overrides_default(monkeypatch):
    provider = _install_content(
        monkeypatch, '{"subqueries": ["AXI4 x", "CHI y"]}'
    )
    llm_decompose("AXI4 vs CHI", timeout=3)
    assert provider.calls[0]["kwargs"]["timeout"] == 3


# ─── NEGATIVE: shape / parse failures → DecompositionError ──────────────────


def test_unparseable_content_raises(monkeypatch):
    _install_content(monkeypatch, "this is not json at all")
    with pytest.raises(DecompositionError):
        llm_decompose("AXI4 vs CHI")


def test_empty_content_raises(monkeypatch):
    _install_content(monkeypatch, "")
    with pytest.raises(DecompositionError):
        llm_decompose("AXI4 vs CHI")


def test_wrong_shape_no_subqueries_key_raises(monkeypatch):
    _install_content(monkeypatch, '{"foo": 1}')
    with pytest.raises(DecompositionError):
        llm_decompose("AXI4 vs CHI")


def test_subqueries_not_a_list_raises(monkeypatch):
    _install_content(monkeypatch, '{"subqueries": "AXI4"}')
    with pytest.raises(DecompositionError):
        llm_decompose("AXI4 vs CHI")


def test_bare_list_json_is_rejected(monkeypatch):
    # Documented decision: bare list is NOT tolerated (parse_json_object yields
    # {} for a top-level array), so it raises.
    _install_content(monkeypatch, '["AXI4", "CHI"]')
    with pytest.raises(DecompositionError):
        llm_decompose("AXI4 vs CHI")


# ─── NEGATIVE: count out of range → DecompositionError ──────────────────────


def test_too_few_items_raises(monkeypatch):
    _install_content(monkeypatch, '{"subqueries": ["AXI4 only"]}')
    with pytest.raises(DecompositionError):
        llm_decompose("AXI4 vs CHI")


def test_empty_list_raises(monkeypatch):
    _install_content(monkeypatch, '{"subqueries": []}')
    with pytest.raises(DecompositionError):
        llm_decompose("AXI4 vs CHI")


def test_too_many_items_raises(monkeypatch):
    over_max = settings.RAG_DECOMPOSITION_MAX_SUBQUERIES + 1
    items = ", ".join(f'"entity {i}"' for i in range(over_max))
    _install_content(monkeypatch, '{"subqueries": [' + items + "]}")
    with pytest.raises(DecompositionError):
        llm_decompose("a / b / c / d / e / f")


# ─── NEGATIVE: bad individual items → DecompositionError ────────────────────


def test_empty_string_item_raises(monkeypatch):
    _install_content(monkeypatch, '{"subqueries": ["AXI4 ok", ""]}')
    with pytest.raises(DecompositionError):
        llm_decompose("AXI4 vs CHI")


def test_whitespace_only_item_raises(monkeypatch):
    _install_content(monkeypatch, '{"subqueries": ["AXI4 ok", "   "]}')
    with pytest.raises(DecompositionError):
        llm_decompose("AXI4 vs CHI")


def test_non_string_item_raises(monkeypatch):
    _install_content(monkeypatch, '{"subqueries": ["AXI4 ok", 123]}')
    with pytest.raises(DecompositionError):
        llm_decompose("AXI4 vs CHI")


def test_absurdly_long_item_raises(monkeypatch):
    long_item = "word " * 40  # ~40 words, far over the short-string cap
    _install_content(
        monkeypatch, '{"subqueries": ["AXI4 ok", "' + long_item.strip() + '"]}'
    )
    with pytest.raises(DecompositionError):
        llm_decompose("AXI4 vs CHI")


# ─── NEGATIVE: provider error / timeout → DecompositionError (cause chained) ─


def test_provider_exception_raises_with_cause(monkeypatch):
    boom = RuntimeError("provider exploded")
    _install_provider(monkeypatch, _FakeProvider(exc=boom))
    with pytest.raises(DecompositionError) as excinfo:
        llm_decompose("AXI4 vs CHI")
    assert excinfo.value.__cause__ is boom


def test_provider_timeout_raises_with_cause(monkeypatch):
    timeout_err = TimeoutError("llm timed out")
    _install_provider(monkeypatch, _FakeProvider(exc=timeout_err))
    with pytest.raises(DecompositionError) as excinfo:
        llm_decompose("AXI4 vs CHI", timeout=1)
    assert excinfo.value.__cause__ is timeout_err
