# @summary
# IBM Granite Guardian classifier integration. Two execution modes:
#   - "transformers": local HuggingFace pipeline (Yes/No-probability via softmax)
#   - "vllm": OpenAI-compatible HTTP endpoint (Yes/No probability via logprobs)
# Maps GuardianRisk enum to Granite's native risk_name values.
# Exports: GraniteGuardian, GraniteMode, GRANITE_RISK_MAP
# Deps: src.guardrails.models.base, httpx, threading, logging, math
# @end-summary
"""IBM Granite Guardian — dedicated safety/groundedness classifier.

Granite Guardian is a fine-tuned Granite LLM that emits a single "Yes"/"No"
token answering whether a piece of text violates a named risk. We extract
a calibrated probability from the model's first-token distribution rather
than relying on the literal answer token.

Two backends are supported behind one class:

* ``transformers`` — load the model locally via the HuggingFace
  ``transformers`` library. Best for development, on-prem GPU, or air-gapped
  deployments. A module-level lock serializes inference per replica so
  multiple rails calling concurrently do not race the model.
* ``vllm`` — call a self-hosted OpenAI-compatible endpoint (vLLM, TGI, or
  IBM watsonx). Best for production: the server batches requests across
  rails and tenants. Probability is read from the OpenAI ``logprobs``
  field on the first generated token.

The class exposes a uniform :meth:`classify` matching the
:class:`GuardianModel` contract; rails do not need to know which backend
is active.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from typing import Any, Literal, Optional

from src.guardrails.models.base import (
    Direction,
    GuardianModel,
    GuardianRisk,
    GuardianUnavailable,
    GuardianVerdict,
)

logger = logging.getLogger("rag.guardrails.models.granite_guardian")


GraniteMode = Literal["transformers", "vllm"]


# Map our backend-agnostic risks to Granite Guardian's native risk_name slots.
# Values come from the Granite Guardian 3.x model card; missing entries indicate
# risks Granite does not natively support (rails will route to another judge).
GRANITE_RISK_MAP: dict[GuardianRisk, str] = {
    GuardianRisk.HARM: "harm",
    GuardianRisk.JAILBREAK: "jailbreak",
    GuardianRisk.VIOLENCE: "violence",
    GuardianRisk.SEXUAL: "sexual_content",
    GuardianRisk.HATE: "social_bias",
    GuardianRisk.PROFANITY: "profanity",
    GuardianRisk.GROUNDEDNESS: "groundedness",
    GuardianRisk.FUNCTION_CALL: "function_call",
}


_DEFAULT_MODEL_ID = "ibm-granite/granite-guardian-3.2-5b"


@dataclass
class _Probs:
    """First-token probabilities for the "Yes"/"No" answer."""

    yes: float
    no: float

    @property
    def unsafe_score(self) -> float:
        """Probability that Granite Guardian judges the input UNSAFE.

        Granite Guardian convention: ``Yes`` = unsafe, ``No`` = safe.
        We renormalize over just the two answer tokens so noisy logprob
        distributions still produce a calibrated [0, 1] score.
        """
        total = self.yes + self.no
        if total <= 0:
            return 0.0
        return self.yes / total


class GraniteGuardian(GuardianModel):
    """Granite Guardian classifier with transformers and vLLM backends.

    Args:
        mode: ``"transformers"`` for local HF inference, ``"vllm"`` for an
            OpenAI-compatible HTTP endpoint.
        model_id: HF model id (transformers) or served model name (vllm).
        endpoint: Base URL for vllm mode (ignored in transformers mode).
            Example: ``"http://granite-guardian:8000/v1"``.
        api_key: Optional bearer token for the HTTP endpoint.
        timeout_s: Per-request timeout for HTTP calls. Transformers mode is
            not bound by this; protect call sites with rail-level timeouts.
        device: Override device for transformers mode (e.g. ``"cuda:0"``).
        threshold: Score threshold above which a verdict is unsafe. Kept
            tunable so calibration drift between Granite versions doesn't
            require rail-side patches.

    Raises:
        ImportError: If the chosen backend's dependencies aren't installed
            (``transformers``+``torch`` for local; ``httpx`` for vLLM).
        GuardianUnavailable: From :meth:`classify` on transient backend
            failures (network, OOM, etc.).
    """

    name = "granite_guardian"
    supported_risks = frozenset(GRANITE_RISK_MAP.keys())

    def __init__(
        self,
        *,
        mode: GraniteMode = "vllm",
        model_id: str = _DEFAULT_MODEL_ID,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_s: float = 5.0,
        device: Optional[str] = None,
        threshold: float = 0.5,
    ) -> None:
        if mode not in ("transformers", "vllm"):
            raise ValueError(f"unknown mode: {mode!r}")
        self._mode = mode
        self._model_id = model_id
        self._endpoint = (endpoint or "").rstrip("/")
        self._api_key = api_key
        self._timeout_s = timeout_s
        self._threshold = threshold
        self._device = device

        # Backend-specific lazy state (initialised on first classify call to
        # avoid loading the model in processes that never use it)
        self._tf_lock = threading.Lock()
        self._tf_model = None
        self._tf_tokenizer = None
        self._http_client = None  # httpx.Client

        if mode == "vllm" and not self._endpoint:
            raise ValueError("granite_guardian: vllm mode requires `endpoint`")

    # ── public ────────────────────────────────────────────────────────────

    def classify(
        self,
        text: str,
        *,
        risk: GuardianRisk,
        context: Optional[list[str]] = None,
        direction: Direction = "input",
    ) -> GuardianVerdict:
        risk_name = GRANITE_RISK_MAP.get(risk)
        if risk_name is None:
            raise ValueError(
                f"GraniteGuardian does not support risk {risk!r}; "
                f"supported: {sorted(r.value for r in self.supported_risks)}"
            )

        messages = self._build_messages(text, context=context, direction=direction)
        guardian_config = {"risk_name": risk_name}

        try:
            if self._mode == "transformers":
                probs = self._classify_local(messages, guardian_config)
            else:
                probs = self._classify_http(messages, guardian_config)
        except GuardianUnavailable:
            raise
        except Exception as e:
            # Wrap any other backend failure as Unavailable so rails can fall
            # back without surfacing transient infra errors to callers.
            logger.warning("Granite Guardian (%s) failed: %s", self._mode, e)
            raise GuardianUnavailable(str(e)) from e

        score = probs.unsafe_score
        return GuardianVerdict(
            safe=score < self._threshold,
            risk=risk,
            score=score,
            raw={
                "yes_prob": probs.yes,
                "no_prob": probs.no,
                "risk_name": risk_name,
                "mode": self._mode,
            },
        )

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _build_messages(
        text: str,
        *,
        context: Optional[list[str]],
        direction: Direction,
    ) -> list[dict[str, str]]:
        """Construct the chat-template messages Granite Guardian expects.

        Granite Guardian inspects the *last* turn against the configured
        risk. For input direction we emit a single user message; for output
        direction we emit ``user``→``assistant`` so the assistant turn is
        what gets judged. Groundedness needs context as a system message.
        """
        messages: list[dict[str, str]] = []
        if context:
            messages.append(
                {"role": "context", "content": "\n---\n".join(context)}
            )
        if direction == "output":
            # The "user" turn is a placeholder; Granite scores the assistant turn.
            messages.append({"role": "user", "content": ""})
            messages.append({"role": "assistant", "content": text})
        else:
            messages.append({"role": "user", "content": text})
        return messages

    # ── transformers backend ──────────────────────────────────────────────

    def _ensure_local_loaded(self) -> None:
        """Load tokenizer + model on first use; serialised by ``_tf_lock``."""
        if self._tf_model is not None:
            return
        try:
            import torch  # noqa: F401 — required by transformers backend
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "granite_guardian transformers mode requires `transformers` "
                "and `torch` — install with: uv pip install transformers torch"
            ) from e

        logger.info("Loading Granite Guardian locally: %s", self._model_id)
        self._tf_tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        kwargs: dict[str, Any] = {"torch_dtype": "auto"}
        if self._device:
            kwargs["device_map"] = self._device
        else:
            kwargs["device_map"] = "auto"
        self._tf_model = AutoModelForCausalLM.from_pretrained(
            self._model_id, **kwargs
        )
        self._tf_model.eval()

    def _classify_local(
        self,
        messages: list[dict[str, str]],
        guardian_config: dict[str, str],
    ) -> _Probs:
        """Run a forward pass and read Yes/No probabilities from logits.

        The lock prevents concurrent rails from interleaving forward passes
        on a single model replica (which would either OOM or serialise on
        the GPU mutex anyway, but with worse error messages).
        """
        with self._tf_lock:
            self._ensure_local_loaded()
            import torch

            tok = self._tf_tokenizer
            model = self._tf_model

            input_ids = tok.apply_chat_template(
                messages,
                guardian_config=guardian_config,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                logits = model(input_ids=input_ids).logits[0, -1, :]

            yes_id = self._token_id("Yes")
            no_id = self._token_id("No")
            probs = torch.softmax(
                torch.tensor([logits[yes_id], logits[no_id]]), dim=0
            )
            return _Probs(yes=float(probs[0]), no=float(probs[1]))

    def _token_id(self, literal: str) -> int:
        """Resolve the first sub-token id for a literal answer word.

        Granite tokenizers may split a leading-space or capitalisation
        variant differently; we try the most likely encoding first.
        """
        tok = self._tf_tokenizer
        for variant in (literal, " " + literal, literal.lower(), literal.upper()):
            ids = tok.encode(variant, add_special_tokens=False)
            if ids:
                return ids[0]
        raise GuardianUnavailable(f"could not resolve token id for {literal!r}")

    # ── vLLM / OpenAI-compat backend ──────────────────────────────────────

    def _ensure_http(self):
        if self._http_client is not None:
            return self._http_client
        try:
            import httpx
        except ImportError as e:
            raise ImportError(
                "granite_guardian vllm mode requires `httpx` — "
                "install with: uv pip install httpx"
            ) from e
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        self._http_client = httpx.Client(
            base_url=self._endpoint,
            headers=headers,
            timeout=self._timeout_s,
        )
        return self._http_client

    def _classify_http(
        self,
        messages: list[dict[str, str]],
        guardian_config: dict[str, str],
    ) -> _Probs:
        """POST to ``/chat/completions`` with logprobs, read Yes/No probs.

        We pass ``guardian_config`` via ``extra_body`` — vLLM forwards
        unknown OpenAI fields to the chat template which Granite Guardian's
        tokenizer consumes.
        """
        client = self._ensure_http()
        payload: dict[str, Any] = {
            "model": self._model_id,
            "messages": messages,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": 5,
            # vLLM-specific passthrough so the chat template sees the risk
            "chat_template_kwargs": {"guardian_config": guardian_config},
        }

        response = client.post("/chat/completions", json=payload)
        if response.status_code >= 400:
            raise GuardianUnavailable(
                f"granite endpoint {response.status_code}: {response.text[:200]}"
            )
        body = response.json()

        try:
            choice = body["choices"][0]
            top = choice["logprobs"]["content"][0]["top_logprobs"]
        except (KeyError, IndexError, TypeError) as e:
            raise GuardianUnavailable(f"unexpected response shape: {e}") from e

        yes_lp = no_lp = float("-inf")
        for entry in top:
            tok = entry.get("token", "").strip().lower()
            lp = entry.get("logprob", float("-inf"))
            if tok == "yes" and lp > yes_lp:
                yes_lp = lp
            elif tok == "no" and lp > no_lp:
                no_lp = lp

        # If only the literal answer was returned without alternatives, infer
        # the missing branch as ~0 probability rather than raising.
        if yes_lp == float("-inf") and no_lp == float("-inf"):
            literal = choice["message"].get("content", "").strip().lower()
            if literal.startswith("yes"):
                return _Probs(yes=0.9, no=0.1)
            if literal.startswith("no"):
                return _Probs(yes=0.1, no=0.9)
            raise GuardianUnavailable(f"no Yes/No in response: {literal!r}")

        yes_p = math.exp(yes_lp) if yes_lp > float("-inf") else 0.0
        no_p = math.exp(no_lp) if no_lp > float("-inf") else 0.0
        return _Probs(yes=yes_p, no=no_p)

    def close(self) -> None:
        """Release HTTP / model resources. Safe to call multiple times."""
        if self._http_client is not None:
            try:
                self._http_client.close()
            except Exception:
                pass
            self._http_client = None
        # Transformers model is left to GC — explicit unload is rarely useful
        # since the next classify call would just reload it.
