"""Unit tests for the guardian models package.

Covers the GuardianModel contract, GraniteGuardian (vLLM HTTP path with a
mocked endpoint), the SelfCheckGuardian, and the ToxicityFilter rail with
guardian injection / fallback behaviour.

Transformers-mode Granite is NOT exercised here — it requires a 5B-param
download. Integration tests in ``tests/integration/guardrails/`` cover that
path against a running vLLM container.
"""
from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest

from src.guardrails.common import RailVerdict
from src.guardrails.models import (
    GRANITE_RISK_MAP,
    GraniteGuardian,
    GuardianModel,
    GuardianRisk,
    GuardianUnavailable,
    GuardianVerdict,
    SelfCheckGuardian,
)
from src.guardrails.shared import ToxicityFilter


# ── GuardianModel contract ────────────────────────────────────────────────


def test_guardian_verdict_is_immutable():
    v = GuardianVerdict(safe=True, risk=GuardianRisk.HARM, score=0.1)
    with pytest.raises(Exception):
        v.safe = False  # frozen dataclass


def test_supports_returns_false_for_unknown_risk():
    g = SelfCheckGuardian()
    assert g.supports(GuardianRisk.HARM)
    assert not g.supports(GuardianRisk.FUNCTION_CALL)


# ── SelfCheckGuardian ─────────────────────────────────────────────────────


@patch("src.platform.llm.call_oneshot")
def test_self_check_yes_means_safe(mock_call):
    mock_call.return_value = "yes"
    g = SelfCheckGuardian()
    v = g.classify("hello", risk=GuardianRisk.HARM)
    assert v.safe is True
    assert v.score == pytest.approx(0.1)


@patch("src.platform.llm.call_oneshot")
def test_self_check_no_means_unsafe(mock_call):
    mock_call.return_value = "no"
    g = SelfCheckGuardian()
    v = g.classify("burn it down", risk=GuardianRisk.HARM)
    assert v.safe is False
    assert v.score == pytest.approx(0.9)


@patch("src.platform.llm.call_oneshot")
def test_self_check_empty_response_is_unavailable(mock_call):
    mock_call.return_value = None
    g = SelfCheckGuardian()
    with pytest.raises(GuardianUnavailable):
        g.classify("x", risk=GuardianRisk.HARM)


def test_self_check_rejects_unsupported_risk():
    g = SelfCheckGuardian()
    with pytest.raises(ValueError):
        g.classify("x", risk=GuardianRisk.FUNCTION_CALL)


# ── GraniteGuardian (vLLM HTTP) ───────────────────────────────────────────


def _granite_response(yes_lp: float, no_lp: float) -> dict:
    """Build a vLLM/OpenAI-shaped chat completion response."""
    return {
        "choices": [
            {
                "message": {"content": "Yes" if yes_lp > no_lp else "No"},
                "logprobs": {
                    "content": [
                        {
                            "token": "Yes" if yes_lp > no_lp else "No",
                            "logprob": max(yes_lp, no_lp),
                            "top_logprobs": [
                                {"token": "Yes", "logprob": yes_lp},
                                {"token": "No", "logprob": no_lp},
                            ],
                        }
                    ]
                },
            }
        ]
    }


def _make_granite_with_mock(response_body: dict, status: int = 200):
    g = GraniteGuardian(mode="vllm", endpoint="http://localhost:8000/v1")
    mock_resp = MagicMock()
    mock_resp.status_code = status
    mock_resp.json.return_value = response_body
    mock_resp.text = "" if status == 200 else "boom"
    g._http_client = MagicMock()
    g._http_client.post.return_value = mock_resp
    return g


def test_granite_unsafe_high_yes_logprob():
    # log(0.9) ≈ -0.105, log(0.1) ≈ -2.30
    g = _make_granite_with_mock(_granite_response(yes_lp=math.log(0.9), no_lp=math.log(0.1)))
    v = g.classify("kill them all", risk=GuardianRisk.HARM)
    assert v.safe is False
    assert v.score > 0.5
    assert v.raw["risk_name"] == "harm"


def test_granite_safe_high_no_logprob():
    g = _make_granite_with_mock(_granite_response(yes_lp=math.log(0.05), no_lp=math.log(0.95)))
    v = g.classify("how do I bake bread", risk=GuardianRisk.HARM)
    assert v.safe is True
    assert v.score < 0.5


def test_granite_http_error_raises_unavailable():
    g = _make_granite_with_mock({}, status=503)
    with pytest.raises(GuardianUnavailable):
        g.classify("x", risk=GuardianRisk.HARM)


def test_granite_unsupported_risk_raises():
    g = GraniteGuardian(mode="vllm", endpoint="http://localhost:8000/v1")
    # PII isn't in GRANITE_RISK_MAP for our default mapping
    bad_risk = next(
        (r for r in GuardianRisk if r not in GRANITE_RISK_MAP), None
    )
    if bad_risk is None:
        pytest.skip("no unsupported risk in current map")
    with pytest.raises(ValueError):
        g.classify("x", risk=bad_risk)


def test_granite_vllm_requires_endpoint():
    with pytest.raises(ValueError):
        GraniteGuardian(mode="vllm", endpoint="")


def test_granite_groundedness_includes_context():
    g = GraniteGuardian(mode="vllm", endpoint="http://x/v1")
    msgs = g._build_messages(
        "Paris is the capital of France",
        context=["France's capital is Paris."],
        direction="output",
    )
    assert any(m["role"] == "context" for m in msgs)
    assert any(m["role"] == "assistant" for m in msgs)


# ── ToxicityFilter with guardian ──────────────────────────────────────────


class _FakeGuardian(GuardianModel):
    """In-memory guardian for rail-level tests."""

    name = "fake"
    supported_risks = frozenset({GuardianRisk.HARM})

    def __init__(self, *, safe: bool = True, raise_unavailable: bool = False):
        self._safe = safe
        self._raise = raise_unavailable

    def classify(self, text, *, risk, context=None, direction="input"):
        if self._raise:
            raise GuardianUnavailable("mock")
        return GuardianVerdict(
            safe=self._safe,
            risk=risk,
            score=0.1 if self._safe else 0.9,
        )


def test_toxicity_uses_guardian_when_present():
    f = ToxicityFilter(guardian=_FakeGuardian(safe=False))
    r = f.check("anything")
    assert r.verdict == RailVerdict.REJECT
    assert r.score == pytest.approx(0.9)


def test_toxicity_passes_when_guardian_says_safe():
    f = ToxicityFilter(guardian=_FakeGuardian(safe=True))
    r = f.check("hello")
    assert r.verdict == RailVerdict.PASS


def test_toxicity_falls_back_to_regex_on_guardian_unavailable():
    f = ToxicityFilter(guardian=_FakeGuardian(raise_unavailable=True))
    # Trips the regex pattern
    r = f.check("kill you")
    assert r.verdict == RailVerdict.REJECT


def test_toxicity_regex_floor_when_no_guardian():
    f = ToxicityFilter(guardian=None)
    assert f.check("hello there").verdict == RailVerdict.PASS
    assert f.check("kill them all").verdict == RailVerdict.REJECT


def test_toxicity_filter_output_replaces_unsafe():
    f = ToxicityFilter(guardian=_FakeGuardian(safe=False))
    assert f.filter_output("anything") == "[CONTENT_FILTERED]"


def test_toxicity_filter_output_passes_safe():
    f = ToxicityFilter(guardian=_FakeGuardian(safe=True))
    assert f.filter_output("hello") == "hello"


# ── NemoBackend wiring smoke test ─────────────────────────────────────────


def test_nemo_backend_propagates_guardian_to_all_rails():
    """End-to-end wiring check: a guardian built by ``build_guardian`` must
    reach the toxicity, injection, and faithfulness rails. Catches
    regressions where someone updates the backend but forgets to thread
    the guardian into a new rail.
    """
    sentinel = MagicMock(name="guardian-sentinel")
    sentinel.supports.return_value = False  # don't trigger any classify path

    # Patch the runtime so GuardrailsRuntime.initialize() is a no-op and
    # ``runtime.initialized`` / ``runtime.rails`` look truthy enough to
    # reach the rail constructors without loading NeMo.
    fake_runtime = MagicMock()
    fake_runtime.initialized = True
    fake_runtime.rails = MagicMock()

    with patch(
        "src.guardrails.nemo_guardrails.runtime.GuardrailsRuntime"
    ) as mock_rt_cls, patch(
        "src.guardrails.models.build_guardian",
        return_value=sentinel,
    ):
        mock_rt_cls.get.return_value = fake_runtime

        from src.guardrails.nemo_guardrails.backend import NemoBackend

        backend = NemoBackend()

    # Pull each rail off the executors and assert the guardian made it through.
    in_exec = backend._input_executor
    out_exec = backend._output_executor

    assert in_exec._toxicity is not None
    assert in_exec._toxicity._guardian is sentinel
    assert in_exec._injection is not None
    assert in_exec._injection._guardian is sentinel
    assert out_exec._faithfulness is not None
    assert out_exec._faithfulness._guardian is sentinel
