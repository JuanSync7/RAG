# @summary
# LLM generator for RAG answer synthesis, backed by LiteLLM Router.
# Main exports: OllamaGenerator, get_system_prompt, reload_system_prompt,
#   GenerationResult, GenerationError, GenerationErrorKind,
#   StreamEvent, TokenEvent, ErrorEvent.
# Deps: typing, dataclasses, enum, config.settings, src.platform.llm
# @end-summary
"""LLM generator for RAG answer synthesis, backed by LiteLLM Router."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional, Tuple, Union

from config.settings import (
    GENERATION_MAX_TOKENS,
    GENERATION_TEMPERATURE,
    PROMPTS_DIR,
)
from src.platform.llm import get_llm_provider
from src.platform.observability import get_tracer


def _load_system_prompt() -> str:
    """Load the RAG system prompt from an external file (REQ-601).

    Falls back to a minimal inline prompt if the file is missing,
    ensuring the pipeline never crashes due to a missing prompt file.
    """
    path = PROMPTS_DIR / "rag_system.md"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    logging.getLogger("rag.generator").warning(
        "System prompt file not found at %s — using minimal fallback", path
    )
    return (
        "You are a helpful assistant. Answer questions using ONLY the provided context. "
        "Cite sources using [1], [2], etc. If context is insufficient, say so."
    )


_SYSTEM_PROMPT: Optional[str] = None


def get_system_prompt() -> str:
    """Return the cached system prompt, loading it from disk on first call.

    Public helper used by callers that need the canonical RAG system prompt
    without instantiating an `OllamaGenerator` (e.g. token-budget calculation
    and output sanitization in `rag_chain.py`).
    """
    global _SYSTEM_PROMPT
    if _SYSTEM_PROMPT is None:
        _SYSTEM_PROMPT = _load_system_prompt()
    return _SYSTEM_PROMPT


def reload_system_prompt() -> str:
    """Clear the in-memory prompt cache and re-read it from disk.

    Used as a hot-reload hook so prompt edits are picked up without restarting
    the process. Returns the freshly-loaded prompt.
    """
    global _SYSTEM_PROMPT
    _SYSTEM_PROMPT = None
    return get_system_prompt()


# Known confidence levels — the only valid values for the confidence field.
_CONFIDENCE_LEVELS = {"high", "medium", "low"}

# Strict schema format — used when the provider supports json_schema (e.g., OpenAI GPT-4o).
# Enforces confidence as an enum at the token level — the LLM literally cannot output other values.
_RAG_RESPONSE_FORMAT_STRICT = {
    "type": "json_schema",
    "json_schema": {
        "name": "rag_answer",
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            },
            "required": ["answer", "confidence"],
            "additionalProperties": False,
        },
    },
}

# Basic JSON format — used when the provider only supports json_object (e.g., Ollama).
# The prompt instructs the schema; validation catches bad values.
_RAG_RESPONSE_FORMAT_BASIC = {"type": "json_object"}


# ── Prompt fence markers ──────────────────────────────────────────────────
# These delimiters wrap retrieved context so the LLM treats the inner block
# as opaque content. They prevent collisions when documents/graph content
# happen to contain "Question:" or "Answer:" markers (issue #12).
# The fences are intentionally unlikely to appear in real prose, so the LLM
# can use them as structural boundaries without us parsing them in code.
_GRAPH_FENCE_BEGIN = "<<<GRAPH_CONTEXT_BEGIN>>>"
_GRAPH_FENCE_END = "<<<GRAPH_CONTEXT_END>>>"
_DOC_FENCE_BEGIN = "<<<DOCUMENT_CONTEXT_BEGIN>>>"
_DOC_FENCE_END = "<<<DOCUMENT_CONTEXT_END>>>"


# ── Typed result + error contract ─────────────────────────────────────────


class GenerationErrorKind(str, Enum):
    """Discriminator for `GenerationError`. String-valued for easy serialization."""

    PROVIDER_UNAVAILABLE = "provider_unavailable"
    AUTH_FAILED = "auth_failed"
    CONTEXT_LENGTH_EXCEEDED = "context_length_exceeded"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    MALFORMED_RESPONSE = "malformed_response"
    REFUSED = "refused"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class GenerationError:
    """Typed failure surface from `OllamaGenerator.generate`.

    `user_message` is safe to surface in UIs; `internal_detail` is for logs.
    """

    kind: GenerationErrorKind
    user_message: str
    internal_detail: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind.value,
            "user_message": self.user_message,
            "internal_detail": self.internal_detail,
        }


@dataclass(frozen=True)
class GenerationResult:
    """Return value of `OllamaGenerator.generate`.

    Always populated — generate() never raises and never returns None.
    On failure, `answer` is "" and `error` carries the typed reason.
    """

    answer: str
    confidence: str
    raw_response: Any
    error: Optional[GenerationError] = None


# ── Stream events ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TokenEvent:
    """A single streamed token chunk."""

    text: str


@dataclass(frozen=True)
class ErrorEvent:
    """Terminal error event for streaming."""

    error: GenerationError


StreamEvent = Union[TokenEvent, ErrorEvent]


# ── Default user-facing messages per error kind ───────────────────────────

_DEFAULT_USER_MESSAGES: dict[GenerationErrorKind, str] = {
    GenerationErrorKind.PROVIDER_UNAVAILABLE: (
        "The language model service is currently unreachable. Please try again shortly."
    ),
    GenerationErrorKind.AUTH_FAILED: (
        "The language model rejected our credentials. Please contact an administrator."
    ),
    GenerationErrorKind.CONTEXT_LENGTH_EXCEEDED: (
        "The model couldn't process the request because the input was too long. "
        "Please ask a more focused question or reduce the retrieved context."
    ),
    GenerationErrorKind.RATE_LIMITED: (
        "The language model is rate-limiting requests. Please wait a moment and try again."
    ),
    GenerationErrorKind.TIMEOUT: (
        "The language model took too long to respond. Please try again."
    ),
    GenerationErrorKind.MALFORMED_RESPONSE: (
        "The language model returned an unexpected response. Please try again."
    ),
    GenerationErrorKind.REFUSED: (
        "The language model declined to answer this query."
    ),
    GenerationErrorKind.UNKNOWN: (
        "Answer generation failed due to an unexpected error. Please try again."
    ),
}


def _classify_provider_exception(exc: BaseException) -> GenerationErrorKind:
    """Map a provider/litellm exception to a `GenerationErrorKind`.

    Falls back to UNKNOWN when the exception type is unrecognized.
    """
    # TimeoutError from stdlib — and litellm's Timeout subclass of openai.APIError
    if isinstance(exc, TimeoutError):
        return GenerationErrorKind.TIMEOUT
    try:
        import litellm.exceptions as lex
    except Exception:
        return GenerationErrorKind.UNKNOWN

    if isinstance(exc, lex.ContextWindowExceededError):
        return GenerationErrorKind.CONTEXT_LENGTH_EXCEEDED
    if isinstance(exc, lex.AuthenticationError):
        return GenerationErrorKind.AUTH_FAILED
    if isinstance(exc, lex.PermissionDeniedError):
        return GenerationErrorKind.AUTH_FAILED
    if isinstance(exc, lex.RateLimitError):
        return GenerationErrorKind.RATE_LIMITED
    if isinstance(exc, lex.BudgetExceededError):
        return GenerationErrorKind.RATE_LIMITED
    if isinstance(exc, lex.ContentPolicyViolationError):
        return GenerationErrorKind.REFUSED
    if isinstance(exc, lex.RejectedRequestError):
        return GenerationErrorKind.REFUSED
    if isinstance(exc, (lex.ServiceUnavailableError, lex.BadGatewayError, lex.InternalServerError, lex.APIConnectionError)):
        return GenerationErrorKind.PROVIDER_UNAVAILABLE
    if isinstance(exc, lex.NotFoundError):
        return GenerationErrorKind.PROVIDER_UNAVAILABLE
    if isinstance(exc, (lex.JSONSchemaValidationError, lex.APIResponseValidationError)):
        return GenerationErrorKind.MALFORMED_RESPONSE
    # Generic timeout flavours from litellm/openai
    name = type(exc).__name__.lower()
    if "timeout" in name:
        return GenerationErrorKind.TIMEOUT
    return GenerationErrorKind.UNKNOWN


def _make_error(exc: BaseException) -> GenerationError:
    kind = _classify_provider_exception(exc)
    return GenerationError(
        kind=kind,
        user_message=_DEFAULT_USER_MESSAGES[kind],
        internal_detail=f"{type(exc).__name__}: {exc}",
    )


def _empty_context_error() -> GenerationError:
    return GenerationError(
        kind=GenerationErrorKind.UNKNOWN,
        user_message="No context was available to answer the question.",
        internal_detail="generate() called with empty context_chunks",
    )


def _render_graph_context_section(graph_context: str) -> str:
    """Render graph context for prompt injection, fenced as an opaque block.

    REQ-KG-794: Positioned before document chunks.
    REQ-KG-796: When empty, returns "" — no placeholder, no heading.

    The graph_context string already includes section markers from
    GraphContextFormatter. We wrap it in `<<<GRAPH_CONTEXT_BEGIN>>>` /
    `<<<GRAPH_CONTEXT_END>>>` so the LLM treats inner content as opaque
    even if it contains tokens that resemble prompt scaffolding such as
    "Question:" or "Answer:" (issue #12).
    """
    if not graph_context:
        return ""
    return f"{_GRAPH_FENCE_BEGIN}\n{graph_context}\n{_GRAPH_FENCE_END}"


def _build_user_prompt(context: str, question: str) -> str:
    """Build user prompt via concatenation — safe against curly braces in documents.

    The retrieved document context is fenced in an opaque block; the question
    sits outside the fence so prompt scaffolding markers ("Question:" /
    "Answer:") cannot collide with content inside the fence (issue #12).
    Using string concatenation instead of .format() also prevents
    KeyError/IndexError when documents contain Python format specifiers
    like {variable}, JSON examples, or template syntax (REQ-602).
    """
    return (
        _DOC_FENCE_BEGIN
        + "\n"
        + context
        + "\n"
        + _DOC_FENCE_END
        + "\n\nQuestion: "
        + question
        + "\n\nAnswer:"
    )


logger = logging.getLogger("rag.generator")


class OllamaGenerator:
    """Generate answers using LiteLLM Router (provider-agnostic).

    Retains the OllamaGenerator name for backward compatibility — callers
    continue to use the same class, but all HTTP calls now go through
    LLMProvider instead of raw urllib to Ollama's /api/chat.

    Instances are safe to share across concurrent requests: `generate()`
    returns a `GenerationResult` instead of stashing per-call state on the
    instance (issue #6).
    """

    def __init__(
        self,
        max_tokens: int = GENERATION_MAX_TOKENS,
        temperature: float = GENERATION_TEMPERATURE,
    ):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._provider = get_llm_provider()
        self.model = self._provider.config.model
        try:
            import litellm
            if litellm.supports_response_schema(self.model):
                self._response_format = _RAG_RESPONSE_FORMAT_STRICT
                logger.info("Model %s supports json_schema — using strict response format", self.model)
            else:
                self._response_format = _RAG_RESPONSE_FORMAT_BASIC
                logger.info("Model %s uses json_object — using basic response format", self.model)
        except Exception:
            self._response_format = _RAG_RESPONSE_FORMAT_BASIC

    @property
    def system_prompt(self) -> str:
        """Public accessor for the lazily-loaded system prompt."""
        return get_system_prompt()

    @classmethod
    def build_messages(
        cls,
        query: str,
        context_chunks: List[str],
        scores: Optional[List[float]] = None,
        memory_context: Optional[str] = None,
        recent_turns: Optional[List[dict]] = None,
        graph_context: str = "",
    ) -> list[dict]:
        """Assemble chat messages for the RAG generation prompt.

        Public surface used by both `OllamaGenerator.generate()` /
        `generate_stream()` and external streaming callers that need the
        same prompt shape but call the LLM provider directly.
        """
        if scores:
            doc_context = "\n\n".join(
                f"[{i+1}] (relevance: {score:.0%}) {chunk}"
                for i, (chunk, score) in enumerate(zip(context_chunks, scores))
            )
        else:
            doc_context = "\n\n".join(
                f"[{i+1}] {chunk}" for i, chunk in enumerate(context_chunks)
            )
        graph_section = _render_graph_context_section(graph_context)
        if graph_section:
            context = graph_section + "\n\n" + doc_context
        else:
            context = doc_context
        user_message = _build_user_prompt(context, query)
        messages: list[dict] = [{"role": "system", "content": get_system_prompt()}]
        if memory_context:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Use the conversation context below only as supporting context for follow-up intent.\n"
                        + memory_context
                    ),
                }
            )
        for turn in recent_turns or []:
            role = str(turn.get("role", "user"))
            if role not in {"user", "assistant", "system"}:
                continue
            content = str(turn.get("content", "")).strip()
            if not content:
                continue
            messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_message})
        return messages

    def generate(
        self,
        query: str,
        context_chunks: List[str],
        scores: Optional[List[float]] = None,
        memory_context: Optional[str] = None,
        recent_turns: Optional[List[dict]] = None,
        graph_context: str = "",
    ) -> GenerationResult:
        """Generate an answer using retrieved context chunks.

        Returns a `GenerationResult` in all cases — never raises, never None.
        On failure the result has `error` set and `answer == ""`.
        """
        if not context_chunks:
            return GenerationResult(
                answer="",
                confidence="medium",
                raw_response=None,
                error=_empty_context_error(),
            )

        messages = self.build_messages(
            query,
            context_chunks,
            scores,
            memory_context=memory_context,
            recent_turns=recent_turns,
            graph_context=graph_context,
        )

        with get_tracer().span(
            "generator.generate",
            {
                "model": self.model,
                "context_chunk_count": len(context_chunks),
            },
        ) as span:
            try:
                response = self._provider.generate(
                    messages,
                    model_alias="default",
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format=self._response_format,
                )
            except Exception as exc:
                err = _make_error(exc)
                logger.warning(
                    "LLM generation failed: kind=%s detail=%s",
                    err.kind.value,
                    err.internal_detail,
                )
                span.set_attribute("generation_error_kind", err.kind.value)
                return GenerationResult(
                    answer="",
                    confidence="medium",
                    raw_response=None,
                    error=err,
                )

            raw_content = response.content or None
            if not raw_content:
                err = GenerationError(
                    kind=GenerationErrorKind.MALFORMED_RESPONSE,
                    user_message=_DEFAULT_USER_MESSAGES[GenerationErrorKind.MALFORMED_RESPONSE],
                    internal_detail="provider returned empty content",
                )
                span.set_attribute("generation_error_kind", err.kind.value)
                return GenerationResult(
                    answer="",
                    confidence="medium",
                    raw_response=response,
                    error=err,
                )

            answer, confidence = self._parse_structured_response(raw_content)
            span.set_attribute("llm_confidence", confidence)
            return GenerationResult(
                answer=answer,
                confidence=confidence,
                raw_response=response,
                error=None,
            )

    @staticmethod
    def _parse_structured_response(response_text: str) -> Tuple[str, str]:
        """Parse the LLM response as structured JSON with answer + confidence.

        Strategy (issue #5):
          1) try direct json.loads,
          2) strip ```json fences and retry,
          3) extract the first balanced {...} block (brace-depth scan that
             ignores braces inside string literals),
          4) fall back to free-text confidence extraction.

        Returns:
            Tuple of (answer_text, confidence_level).
            Confidence defaults to "medium" if not parseable.
        """
        import json

        candidates = []
        text = response_text.strip()
        candidates.append(text)
        stripped = _strip_code_fences(text)
        if stripped != text:
            candidates.append(stripped)
        balanced = _extract_first_json_object(stripped)
        if balanced and balanced not in candidates:
            candidates.append(balanced)

        for candidate in candidates:
            try:
                data = json.loads(candidate)
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(data, dict):
                continue
            answer = str(data.get("answer", "")).strip()
            confidence = str(data.get("confidence", "medium")).strip().lower()
            if confidence not in _CONFIDENCE_LEVELS:
                confidence = "medium"
            return (answer or response_text, confidence)

        return OllamaGenerator._extract_confidence_from_text(response_text)

    @staticmethod
    def _extract_confidence_from_text(response_text: str) -> Tuple[str, str]:
        """Fallback extraction when structured output is not available.

        Scans for "confidence: high|medium|low" anywhere in a line (issue #5),
        tolerating leading prose, markdown bold (`**Confidence:** ...`),
        labelled variants ("Confidence level: high"), and trailing punctuation.
        Strips matching lines from the answer text.
        """
        confidence = "medium"
        lines = response_text.splitlines()
        clean_lines: list[str] = []
        for line in lines:
            normalized = line.strip().lower().replace("*", "")
            # match if "confidence" appears and there's a colon to split on
            if "confidence" in normalized and ":" in normalized:
                # take the tail after the LAST ':' so "Confidence level: high" works
                tail = normalized.rsplit(":", 1)[1].strip()
                # strip trailing punctuation/whitespace
                tail = tail.rstrip(" .,;!?")
                # drop a leading "level " if any leftovers slipped through
                if tail in _CONFIDENCE_LEVELS:
                    confidence = tail
                    continue  # drop this line from the answer
            clean_lines.append(line)
        return "\n".join(clean_lines).strip(), confidence

    def generate_stream(
        self,
        query: str,
        context_chunks: List[str],
        scores: Optional[List[float]] = None,
        memory_context: Optional[str] = None,
        recent_turns: Optional[List[dict]] = None,
        graph_context: str = "",
    ):
        """Stream tokens from LLM as discriminated `StreamEvent` values.

        Yields `TokenEvent(text=...)` for each chunk and, on failure, a final
        `ErrorEvent(error=...)` so callers can surface a typed error to the UI.
        """
        if not context_chunks:
            yield ErrorEvent(error=_empty_context_error())
            return

        messages = self.build_messages(
            query,
            context_chunks,
            scores,
            memory_context=memory_context,
            recent_turns=recent_turns,
            graph_context=graph_context,
        )

        try:
            for chunk in self._provider.generate_stream(
                messages,
                model_alias="default",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            ):
                yield TokenEvent(text=chunk)
        except Exception as exc:
            err = _make_error(exc)
            logger.warning(
                "LLM streaming failed: kind=%s detail=%s",
                err.kind.value,
                err.internal_detail,
            )
            yield ErrorEvent(error=err)

    def is_available(self) -> bool:
        """Check if the LLM provider is reachable."""
        with get_tracer().span("generator.is_available", {"model": self.model}):
            try:
                return self._provider.is_available(model_alias="default")
            except Exception:
                return False


# ── helpers used by _parse_structured_response ────────────────────────────


def _strip_code_fences(text: str) -> str:
    """Remove leading/trailing ```json``` or ``` fences if present.

    Pure string operations — no regex.
    """
    s = text.strip()
    if not s.startswith("```"):
        return s
    # drop the opening fence line (``` or ```json)
    newline = s.find("\n")
    if newline == -1:
        return s
    body = s[newline + 1 :]
    # drop the trailing fence
    if body.endswith("```"):
        body = body[:-3]
    return body.strip()


def _extract_first_json_object(text: str) -> Optional[str]:
    """Return the first balanced {...} block in `text`, or None.

    Brace-depth scan that ignores braces inside JSON string literals
    (handles escaped quotes). Pure string ops — no regex.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
            continue
        if c == '"':
            in_string = True
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


__all__ = [
    "OllamaGenerator",
    "GenerationResult",
    "GenerationError",
    "GenerationErrorKind",
    "StreamEvent",
    "TokenEvent",
    "ErrorEvent",
    "get_system_prompt",
    "reload_system_prompt",
]
