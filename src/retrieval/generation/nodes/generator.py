# @summary
# LLM generator for RAG answer synthesis, backed by LiteLLM Router.
# Main exports: OllamaGenerator, get_system_prompt.
# Deps: typing, config.settings, src.platform.llm
# @end-summary
"""LLM generator for RAG answer synthesis, backed by LiteLLM Router."""

import logging
from typing import List, Optional, Tuple

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

def _render_graph_context_section(graph_context: str) -> str:
    """Render graph context for prompt injection.

    REQ-KG-794: Positioned before document chunks.
    REQ-KG-796: When empty, returns "" — no placeholder, no heading.

    The graph_context string already includes section markers from
    GraphContextFormatter (e.g. "## Graph Context\\n### Entities\\n..."),
    so this helper simply passes it through when non-empty.
    """
    if not graph_context:
        return ""
    return graph_context


def _build_user_prompt(context: str, question: str) -> str:
    """Build user prompt via concatenation — safe against curly braces in documents.

    Using string concatenation instead of .format() prevents KeyError/IndexError
    when retrieved documents contain Python format specifiers like {variable},
    JSON examples, or template syntax (REQ-602).
    """
    return "Context:\n" + context + "\n\nQuestion: " + question + "\n\nAnswer:"

logger = logging.getLogger("rag.generator")


class OllamaGenerator:
    """Generate answers using LiteLLM Router (provider-agnostic).

    Retains the OllamaGenerator name for backward compatibility — callers
    continue to use the same class, but all HTTP calls now go through
    LLMProvider instead of raw urllib to Ollama's /api/chat.
    """

    def __init__(
        self,
        max_tokens: int = GENERATION_MAX_TOKENS,
        temperature: float = GENERATION_TEMPERATURE,
    ):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._provider = get_llm_provider()
        # Expose model name for logging (matches old interface)
        self.model = self._provider.config.model
        # Last LLM response — populated after generate() for token tracking
        self._last_response = None
        # Last LLM self-reported confidence — read by rag_chain.py for composite scoring
        self._last_llm_confidence: str = "medium"
        # Auto-detect structured output support for this model
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
        # REQ-KG-794: graph context section positioned before document chunks.
        # REQ-KG-796: omitted entirely when empty — no placeholder or heading.
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
    ) -> Optional[str]:
        """Generate an answer using retrieved context chunks.

        Args:
            query: The user's question.
            context_chunks: List of relevant text chunks from retrieval.
            scores: Optional reranker scores (0.0-1.0) for each chunk.
            graph_context: Optional pre-formatted KG context string.
                When non-empty it is placed before document chunks in the
                prompt (REQ-KG-794).  When empty, it is omitted entirely
                with no placeholder or heading (REQ-KG-796).

        Returns:
            Generated answer string, or None if generation fails.
        """
        if not context_chunks:
            return None

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
                self._last_response = response
                raw_content = response.content or None
                if raw_content:
                    answer, confidence = self._parse_structured_response(raw_content)
                    self._last_llm_confidence = confidence
                    span.set_attribute("llm_confidence", confidence)
                    return answer
                return None
            except Exception as e:
                logger.warning("LLM generation failed: %s", e)
                return None

    @staticmethod
    def _parse_structured_response(response_text: str) -> Tuple[str, str]:
        """Parse the LLM response as structured JSON with answer + confidence.

        The LLM is called with response_format=json_schema, which constrains
        the output to {"answer": str, "confidence": "high"|"medium"|"low"}.
        Falls back to text extraction if JSON parsing fails (e.g., provider
        doesn't support structured output).

        Args:
            response_text: Raw LLM response (expected JSON).

        Returns:
            Tuple of (answer_text, confidence_level).
            Confidence defaults to "medium" if not parseable.
        """
        import json
        try:
            data = json.loads(response_text)
            answer = data.get("answer", "").strip()
            confidence = data.get("confidence", "medium").strip().lower()
            return answer or response_text, confidence
        except (json.JSONDecodeError, AttributeError):
            # Fallback: provider didn't return JSON — extract from text
            return OllamaGenerator._extract_confidence_from_text(response_text)

    @staticmethod
    def _extract_confidence_from_text(response_text: str) -> Tuple[str, str]:
        """Fallback extraction when structured output is not available.

        Scans for "CONFIDENCE: high|medium|low" anywhere in the text and
        strips it from the answer.
        """
        confidence = "medium"
        lines = response_text.splitlines()
        clean_lines = []
        for line in lines:
            stripped = line.strip().lower().replace("*", "")
            if stripped.startswith("confidence:"):
                level = stripped.split(":", 1)[1].strip()
                if level in _CONFIDENCE_LEVELS:
                    confidence = level
                continue
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
        """Stream tokens from LLM. Yields content strings as they arrive.

        Same prompt as generate(), but uses streaming mode so callers
        can display tokens incrementally.  Accepts graph_context for
        consistency with generate() (REQ-KG-794, REQ-KG-796).
        """
        if not context_chunks:
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
                yield chunk
        except Exception as e:
            logger.warning("LLM streaming failed: %s", e)

    def is_available(self) -> bool:
        """Check if the LLM provider is reachable."""
        with get_tracer().span("generator.is_available", {"model": self.model}):
            try:
                return self._provider.is_available(model_alias="default")
            except Exception:
                return False
