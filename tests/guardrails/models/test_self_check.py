"""Unit tests for :class:`SelfCheckGuardian`.

The guardian wraps :func:`src.platform.llm.call_oneshot` (imported lazily
inside ``classify``), so all tests patch it at its definition site
``src.platform.llm.call_oneshot``. A small recorder fake captures the prompt
and system instruction so prompt-selection branches can be asserted exactly.
"""

from __future__ import annotations

import pytest

from src.guardrails.models.base import (
    GuardianRisk,
    GuardianUnavailable,
    GuardianVerdict,
)
from src.guardrails.models.self_check import (
    SelfCheckGuardian,
    _SYSTEM_INSTR,
)


class _Recorder:
    """Stand-in for ``call_oneshot`` that records args and returns/raises."""

    def __init__(self, *, returns=None, raises=None):
        self.returns = returns
        self.raises = raises
        self.calls: list[tuple[str, str]] = []

    def __call__(self, prompt, *, system=None):
        self.calls.append((prompt, system))
        if self.raises is not None:
            raise self.raises
        return self.returns

    @property
    def prompt(self) -> str:
        assert self.calls, "call_oneshot was never invoked"
        return self.calls[-1][0]

    @property
    def system(self):
        assert self.calls, "call_oneshot was never invoked"
        return self.calls[-1][1]


def _patch(monkeypatch, rec: _Recorder) -> None:
    monkeypatch.setattr("src.platform.llm.call_oneshot", rec)


# ---------------------------------------------------------------------------
# 1. Identity / contract
# ---------------------------------------------------------------------------


def test_name_and_supported_risks_exact():
    g = SelfCheckGuardian(threshold=0.5)
    assert g.name == "self_check"
    assert SelfCheckGuardian.name == "self_check"
    assert g.supported_risks == frozenset(
        {GuardianRisk.HARM, GuardianRisk.GROUNDEDNESS}
    )


# ---------------------------------------------------------------------------
# 2. Unsupported risk
# ---------------------------------------------------------------------------


def test_unsupported_risk_raises_value_error(monkeypatch):
    # call_oneshot must NOT be reached; if it is, fail loudly.
    rec = _Recorder(raises=AssertionError("call_oneshot should not run"))
    _patch(monkeypatch, rec)
    g = SelfCheckGuardian(threshold=0.5)
    with pytest.raises(ValueError) as ei:
        g.classify("hi", risk=GuardianRisk.JAILBREAK, direction="input")
    # Message names the offending risk.
    assert "JAILBREAK" in repr(GuardianRisk.JAILBREAK)
    assert repr(GuardianRisk.JAILBREAK) in str(ei.value)
    assert rec.calls == []


# ---------------------------------------------------------------------------
# 3. input-direction HARM, "yes" -> safe
# ---------------------------------------------------------------------------


def test_input_harm_yes_is_safe(monkeypatch):
    rec = _Recorder(returns="yes")
    _patch(monkeypatch, rec)
    g = SelfCheckGuardian(threshold=0.5)
    v = g.classify("please be nice", risk=GuardianRisk.HARM, direction="input")

    assert isinstance(v, GuardianVerdict)
    assert v.safe is True
    assert v.score == 0.1
    assert v.risk == GuardianRisk.HARM
    assert v.raw == {"answer": "yes"}

    # Input prompt was used: distinctive input-prompt text + interpolated text.
    assert "Your task is to check if the user message below" in rec.prompt
    assert "User message:" in rec.prompt
    assert "please be nice" in rec.prompt
    # Not the output/grounded prompts.
    assert "bot response" not in rec.prompt
    assert "fully grounded" not in rec.prompt
    assert rec.system == _SYSTEM_INSTR


# ---------------------------------------------------------------------------
# 4. output-direction HARM, "no" -> unsafe + OUTPUT prompt
# ---------------------------------------------------------------------------


def test_output_harm_no_is_unsafe(monkeypatch):
    rec = _Recorder(returns="no")
    _patch(monkeypatch, rec)
    g = SelfCheckGuardian(threshold=0.5)
    v = g.classify("here is how to do harm", risk=GuardianRisk.HARM, direction="output")

    assert v.safe is False
    assert v.score == 0.9
    assert v.risk == GuardianRisk.HARM
    assert v.raw == {"answer": "no"}

    # Output prompt was used, distinct from the input prompt.
    assert "Your task is to check if the bot response below" in rec.prompt
    assert "Bot response:" in rec.prompt
    assert "here is how to do harm" in rec.prompt
    assert "user message below" not in rec.prompt


# ---------------------------------------------------------------------------
# 5. case + whitespace insensitivity / startswith semantics
# ---------------------------------------------------------------------------


def test_yes_with_whitespace_caps_and_suffix_is_safe(monkeypatch):
    # Leading/trailing whitespace, mixed case, AND a trailing clause.
    # Strips+lowers to "yes, it is safe" which startswith "yes" but is not
    # equal to "yes" -- this gives the yes-branch ``startswith`` teeth.
    rec = _Recorder(returns="  YES, it is safe\n")
    _patch(monkeypatch, rec)
    g = SelfCheckGuardian(threshold=0.5)
    v = g.classify("q", risk=GuardianRisk.HARM, direction="input")
    assert v.safe is True
    assert v.score == 0.1
    # raw answer is stripped+lowered
    assert v.raw == {"answer": "yes, it is safe"}


def test_no_with_trailing_punctuation_is_unsafe(monkeypatch):
    rec = _Recorder(returns="No.")
    _patch(monkeypatch, rec)
    g = SelfCheckGuardian(threshold=0.5)
    v = g.classify("q", risk=GuardianRisk.HARM, direction="output")
    assert v.safe is False
    assert v.score == 0.9
    assert v.raw == {"answer": "no."}


# ---------------------------------------------------------------------------
# 6. GROUNDEDNESS prompt building
# ---------------------------------------------------------------------------


def test_groundedness_joins_context_with_separator(monkeypatch):
    rec = _Recorder(returns="yes")
    _patch(monkeypatch, rec)
    g = SelfCheckGuardian(threshold=0.5)
    v = g.classify(
        "the sky is blue",
        risk=GuardianRisk.GROUNDEDNESS,
        context=["fact one", "fact two"],
        direction="output",
    )
    assert v.safe is True

    # Grounded prompt, regardless of direction.
    assert "fully grounded in the provided context" in rec.prompt
    assert "the sky is blue" in rec.prompt
    # Context joined by the exact separator.
    assert "fact one\n---\nfact two" in rec.prompt
    # Not the harm prompts.
    assert "user message below" not in rec.prompt
    assert "bot response below" not in rec.prompt


def test_groundedness_empty_context_uses_placeholder(monkeypatch):
    rec = _Recorder(returns="no")
    _patch(monkeypatch, rec)
    g = SelfCheckGuardian(threshold=0.5)
    v = g.classify(
        "claim",
        risk=GuardianRisk.GROUNDEDNESS,
        context=None,
        direction="output",
    )
    assert v.safe is False
    assert "(no context)" in rec.prompt
    assert "fully grounded in the provided context" in rec.prompt


# ---------------------------------------------------------------------------
# 7. call_oneshot raises -> GuardianUnavailable
# ---------------------------------------------------------------------------


def test_call_oneshot_raises_maps_to_unavailable(monkeypatch):
    rec = _Recorder(raises=RuntimeError("network down"))
    _patch(monkeypatch, rec)
    g = SelfCheckGuardian(threshold=0.5)
    with pytest.raises(GuardianUnavailable) as ei:
        g.classify("q", risk=GuardianRisk.HARM, direction="input")
    assert "network down" in str(ei.value)


# ---------------------------------------------------------------------------
# 8. empty / falsy result -> GuardianUnavailable
# ---------------------------------------------------------------------------


def test_empty_string_result_is_unavailable(monkeypatch):
    rec = _Recorder(returns="")
    _patch(monkeypatch, rec)
    g = SelfCheckGuardian(threshold=0.5)
    with pytest.raises(GuardianUnavailable) as ei:
        g.classify("q", risk=GuardianRisk.HARM, direction="input")
    assert "empty response" in str(ei.value)


def test_none_result_is_unavailable(monkeypatch):
    rec = _Recorder(returns=None)
    _patch(monkeypatch, rec)
    g = SelfCheckGuardian(threshold=0.5)
    with pytest.raises(GuardianUnavailable) as ei:
        g.classify("q", risk=GuardianRisk.HARM, direction="input")
    assert "empty response" in str(ei.value)


# ---------------------------------------------------------------------------
# 9. legacy JSON payload (toxicity_score) + clamp + threshold boundary
# ---------------------------------------------------------------------------


def test_legacy_json_high_score_above_threshold_is_unsafe(monkeypatch):
    # 0.8 >= 0.5 -> NOT (score < threshold) -> unsafe
    rec = _Recorder(returns='{"toxicity_score": 0.8}')
    _patch(monkeypatch, rec)
    g = SelfCheckGuardian(threshold=0.5)
    v = g.classify("q", risk=GuardianRisk.HARM, direction="input")
    assert v.safe is False
    assert v.score == 0.8
    assert v.raw == {"toxicity_score": 0.8}


def test_legacy_json_low_score_below_threshold_is_safe(monkeypatch):
    # 0.2 < 0.5 -> safe
    rec = _Recorder(returns='{"toxicity_score": 0.2}')
    _patch(monkeypatch, rec)
    g = SelfCheckGuardian(threshold=0.5)
    v = g.classify("q", risk=GuardianRisk.HARM, direction="input")
    assert v.safe is True
    assert v.score == 0.2


def test_legacy_json_score_equal_threshold_is_unsafe(monkeypatch):
    # Boundary: 0.5 is NOT < 0.5 -> unsafe. Gives `<` teeth vs `<=`.
    rec = _Recorder(returns='{"toxicity_score": 0.5}')
    _patch(monkeypatch, rec)
    g = SelfCheckGuardian(threshold=0.5)
    v = g.classify("q", risk=GuardianRisk.HARM, direction="input")
    assert v.safe is False
    assert v.score == 0.5


def test_legacy_json_score_clamped_above_one(monkeypatch):
    rec = _Recorder(returns='{"toxicity_score": 1.5}')
    _patch(monkeypatch, rec)
    g = SelfCheckGuardian(threshold=0.5)
    v = g.classify("q", risk=GuardianRisk.HARM, direction="input")
    assert v.score == 1.0
    assert v.safe is False


def test_legacy_json_score_clamped_below_zero(monkeypatch):
    rec = _Recorder(returns='{"toxicity_score": -0.3}')
    _patch(monkeypatch, rec)
    g = SelfCheckGuardian(threshold=0.5)
    v = g.classify("q", risk=GuardianRisk.HARM, direction="input")
    assert v.score == 0.0
    assert v.safe is True


# ---------------------------------------------------------------------------
# 10. unparseable non-yes/no non-JSON -> GuardianUnavailable
# ---------------------------------------------------------------------------


def test_unparseable_response_is_unavailable(monkeypatch):
    rec = _Recorder(returns="maybe")
    _patch(monkeypatch, rec)
    g = SelfCheckGuardian(threshold=0.5)
    with pytest.raises(GuardianUnavailable) as ei:
        g.classify("q", risk=GuardianRisk.HARM, direction="input")
    assert "unparseable" in str(ei.value)
    assert "maybe" in str(ei.value)
