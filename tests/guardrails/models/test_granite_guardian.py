"""Tests for the IBM Granite Guardian classifier (vLLM/HTTP backend + pure logic).

The transformers backend requires a GPU and is out of scope here. Everything
exercised below routes through the HTTP path (driven by an injected fake httpx
client) or through pure helpers (`_Probs`, `_build_messages`).
"""

from __future__ import annotations

import math

import pytest

from src.guardrails.models.base import (
    GuardianRisk,
    GuardianUnavailable,
    GuardianVerdict,
)
from src.guardrails.models.granite_guardian import (
    GRANITE_RISK_MAP,
    GraniteGuardian,
    _Probs,
)


# ── fakes ───────────────────────────────────────────────────────────────────


class FakeResponse:
    """Mimics the bits of an httpx.Response that _classify_http reads."""

    def __init__(self, status_code: int, json_body: object = None, text: str = ""):
        self.status_code = status_code
        self._json_body = json_body
        self.text = text

    def json(self):
        return self._json_body


class FakeClient:
    """Records POST payloads and returns a queued FakeResponse."""

    def __init__(self, response: FakeResponse):
        self._response = response
        self.posts: list[tuple[str, dict]] = []
        self.closed = 0

    def post(self, path, json):  # noqa: A002 - mirror httpx.Client.post signature
        self.posts.append((path, json))
        return self._response

    def close(self):
        self.closed += 1

    @property
    def last_payload(self) -> dict:
        return self.posts[-1][1]


def _logprobs_body(top_logprobs, content="Yes"):
    """Build an OpenAI-compatible chat.completions body with first-token logprobs."""
    return {
        "choices": [
            {
                "message": {"content": content},
                "logprobs": {"content": [{"top_logprobs": top_logprobs}]},
            }
        ]
    }


def _make_http_guardian(**kwargs) -> GraniteGuardian:
    kwargs.setdefault("endpoint", "http://granite:8000/v1")
    return GraniteGuardian(mode="vllm", **kwargs)


# ── constructor / config ────────────────────────────────────────────────────


def test_unknown_mode_raises_value_error():
    with pytest.raises(ValueError, match="unknown mode"):
        GraniteGuardian(mode="bogus", endpoint="http://x")  # type: ignore[arg-type]


def test_vllm_mode_without_endpoint_raises():
    with pytest.raises(ValueError, match="requires `endpoint`"):
        GraniteGuardian(mode="vllm")


def test_endpoint_trailing_slash_stripped():
    g = GraniteGuardian(mode="vllm", endpoint="http://x/")
    assert g._endpoint == "http://x"


def test_supported_risks_and_name():
    g = _make_http_guardian()
    assert g.supported_risks == frozenset(GRANITE_RISK_MAP.keys())
    assert GraniteGuardian.name == "granite_guardian"
    assert g.name == "granite_guardian"


# ── _Probs.unsafe_score ─────────────────────────────────────────────────────


def test_unsafe_score_renormalises():
    assert _Probs(yes=0.8, no=0.2).unsafe_score == pytest.approx(0.8)
    assert _Probs(yes=3.0, no=1.0).unsafe_score == pytest.approx(0.75)


def test_unsafe_score_zero_total_returns_zero_no_div_error():
    assert _Probs(yes=0.0, no=0.0).unsafe_score == 0.0


# ── _build_messages ─────────────────────────────────────────────────────────


def test_build_messages_input_single_user_turn():
    msgs = GraniteGuardian._build_messages("hello", context=None, direction="input")
    assert msgs == [{"role": "user", "content": "hello"}]


def test_build_messages_output_user_then_assistant():
    msgs = GraniteGuardian._build_messages("answer", context=None, direction="output")
    assert msgs == [
        {"role": "user", "content": ""},
        {"role": "assistant", "content": "answer"},
    ]
    # The assistant turn (last) is what Granite scores and must carry the text.
    assert msgs[-1] == {"role": "assistant", "content": "answer"}


def test_build_messages_context_leads_and_joins():
    msgs = GraniteGuardian._build_messages(
        "q", context=["c1", "c2"], direction="input"
    )
    assert msgs[0] == {"role": "context", "content": "c1\n---\nc2"}
    assert msgs[1] == {"role": "user", "content": "q"}
    assert len(msgs) == 2


# ── classify ────────────────────────────────────────────────────────────────


def test_classify_unsupported_risk_raises_value_error():
    g = _make_http_guardian()
    # GuardianRisk.PII is intentionally absent from GRANITE_RISK_MAP.
    assert GuardianRisk.PII not in GRANITE_RISK_MAP
    with pytest.raises(ValueError, match="does not support risk"):
        g.classify("text", risk=GuardianRisk.PII)


def test_classify_unsafe_when_score_at_or_above_threshold():
    # yes=3, no=1 -> unsafe_score 0.75 >= threshold 0.5 -> unsafe.
    g = _make_http_guardian(threshold=0.5)
    body = _logprobs_body(
        [
            {"token": "Yes", "logprob": math.log(3.0)},
            {"token": "No", "logprob": math.log(1.0)},
        ]
    )
    g._http_client = FakeClient(FakeResponse(200, body))
    v = g.classify("danger", risk=GuardianRisk.HARM)
    assert isinstance(v, GuardianVerdict)
    assert v.safe is False
    assert v.risk is GuardianRisk.HARM
    assert v.score == pytest.approx(0.75)
    assert v.raw["yes_prob"] == pytest.approx(3.0)
    assert v.raw["no_prob"] == pytest.approx(1.0)
    assert v.raw["risk_name"] == "harm"
    assert v.raw["mode"] == "vllm"


def test_classify_safe_when_score_below_threshold():
    # yes=1, no=3 -> unsafe_score 0.25 < threshold 0.5 -> safe.
    g = _make_http_guardian(threshold=0.5)
    body = _logprobs_body(
        [
            {"token": "Yes", "logprob": math.log(1.0)},
            {"token": "No", "logprob": math.log(3.0)},
        ]
    )
    g._http_client = FakeClient(FakeResponse(200, body))
    v = g.classify("benign", risk=GuardianRisk.HATE)
    assert v.safe is True
    assert v.score == pytest.approx(0.25)
    assert v.risk is GuardianRisk.HATE
    assert v.raw["risk_name"] == "social_bias"


def test_classify_guardian_unavailable_propagates_unchanged():
    g = _make_http_guardian()
    sentinel = GuardianUnavailable("backend down")

    def boom(messages, guardian_config):
        raise sentinel

    g._classify_http = boom  # type: ignore[assignment]
    with pytest.raises(GuardianUnavailable) as exc:
        g.classify("x", risk=GuardianRisk.HARM)
    assert exc.value is sentinel


def test_classify_generic_exception_wrapped_as_unavailable():
    g = _make_http_guardian()

    def boom(messages, guardian_config):
        raise RuntimeError("kaboom-but-not-guardian")

    g._classify_http = boom  # type: ignore[assignment]
    with pytest.raises(GuardianUnavailable, match="kaboom-but-not-guardian"):
        g.classify("x", risk=GuardianRisk.HARM)


# ── _classify_http ──────────────────────────────────────────────────────────


def test_classify_http_picks_max_logprob_per_token():
    g = _make_http_guardian()
    # Duplicate "yes" entries; the HIGHER one (-0.1) comes SECOND so a
    # "take first" mutation would pick the lower (-2.0) and differ.
    body = _logprobs_body(
        [
            {"token": "Yes", "logprob": -2.0},
            {"token": "No", "logprob": -1.0},
            {"token": " yes", "logprob": -0.1},
        ]
    )
    g._http_client = FakeClient(FakeResponse(200, body))
    probs = g._classify_http([], {"risk_name": "harm"})
    assert probs.yes == pytest.approx(math.exp(-0.1))
    assert probs.no == pytest.approx(math.exp(-1.0))


def test_classify_http_status_4xx_raises_unavailable():
    g = _make_http_guardian()
    g._http_client = FakeClient(FakeResponse(400, None, text="bad request body"))
    with pytest.raises(GuardianUnavailable, match="400"):
        g._classify_http([], {"risk_name": "harm"})


def test_classify_http_malformed_body_raises_unexpected_shape():
    g = _make_http_guardian()
    g._http_client = FakeClient(FakeResponse(200, {"nope": True}))
    with pytest.raises(GuardianUnavailable, match="unexpected response shape"):
        g._classify_http([], {"risk_name": "harm"})


def test_classify_http_literal_fallback_yes():
    g = _make_http_guardian()
    body = _logprobs_body([], content="Yes, this is unsafe")
    g._http_client = FakeClient(FakeResponse(200, body))
    probs = g._classify_http([], {"risk_name": "harm"})
    assert probs.yes == pytest.approx(0.9)
    assert probs.no == pytest.approx(0.1)


def test_classify_http_literal_fallback_no():
    g = _make_http_guardian()
    body = _logprobs_body([], content="No, looks fine")
    g._http_client = FakeClient(FakeResponse(200, body))
    probs = g._classify_http([], {"risk_name": "harm"})
    assert probs.yes == pytest.approx(0.1)
    assert probs.no == pytest.approx(0.9)


def test_classify_http_literal_fallback_neither_raises():
    g = _make_http_guardian()
    body = _logprobs_body([], content="maybe?")
    g._http_client = FakeClient(FakeResponse(200, body))
    with pytest.raises(GuardianUnavailable, match="no Yes/No"):
        g._classify_http([], {"risk_name": "harm"})


def test_classify_http_payload_fields_and_risk_name():
    g = _make_http_guardian()
    body = _logprobs_body(
        [
            {"token": "Yes", "logprob": -0.5},
            {"token": "No", "logprob": -0.5},
        ]
    )
    client = FakeClient(FakeResponse(200, body))
    g._http_client = client
    g._classify_http(
        [{"role": "user", "content": "hi"}], {"risk_name": "jailbreak"}
    )
    path, payload = client.posts[-1]
    assert path == "/chat/completions"
    assert payload["max_tokens"] == 1
    assert payload["temperature"] == 0.0
    assert payload["logprobs"] is True
    assert (
        payload["chat_template_kwargs"]["guardian_config"]["risk_name"]
        == "jailbreak"
    )


def test_classify_drives_full_http_path_with_risk_name_in_payload():
    # End-to-end through classify -> _classify_http with a real fake response,
    # asserting the mapped risk_name reaches the posted payload.
    g = _make_http_guardian(threshold=0.5)
    body = _logprobs_body(
        [
            {"token": "Yes", "logprob": math.log(4.0)},
            {"token": "No", "logprob": math.log(1.0)},
        ]
    )
    client = FakeClient(FakeResponse(200, body))
    g._http_client = client
    v = g.classify("attack prompt", risk=GuardianRisk.JAILBREAK, direction="input")
    assert v.safe is False
    assert v.score == pytest.approx(0.8)
    assert (
        client.last_payload["chat_template_kwargs"]["guardian_config"]["risk_name"]
        == "jailbreak"
    )


# ── close ───────────────────────────────────────────────────────────────────


def test_close_closes_client_and_is_idempotent():
    g = _make_http_guardian()
    client = FakeClient(FakeResponse(200, {}))
    g._http_client = client
    g.close()
    assert client.closed == 1
    assert g._http_client is None
    # Second call is a no-op (no error, client not re-closed).
    g.close()
    assert client.closed == 1
    assert g._http_client is None
