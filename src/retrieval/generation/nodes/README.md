<!-- @summary
Pipeline stage nodes for RAG answer generation: document formatting, LLM-backed answer synthesis,
and output sanitization. These nodes sit between reranking and confidence evaluation in the
generation pipeline.
@end-summary -->

# retrieval/generation/nodes

This package contains the individual stage nodes that transform ranked retrieval results into a
clean, cite-grounded answer. The nodes are pure-function or class-based and are composed by the
generation pipeline orchestrator.

## Contents

| Path | Purpose |
| --- | --- |
| `document_formatter.py` | `format_context` — converts `RankedResult` objects into a structured context string with metadata headers; detects multi-version conflicts and prepends a warning block (REQ-501, REQ-502, REQ-503) |
| `generator.py` | `OllamaGenerator` — LiteLLM Router-backed class that builds structured prompts (with optional graph context and conversation history), calls the LLM for JSON-formatted answers, and returns a typed `GenerationResult` (REQ-601, REQ-602, REQ-KG-794, REQ-KG-796) |
| `output_sanitizer.py` | `sanitize_answer` — strips document boundary markers, unreplaced template variables, and system prompt fragments from generated answers using structural detection rather than regex (REQ-704) |
| `__init__.py` | Package facade re-exporting `OllamaGenerator`, `GenerationResult`, `GenerationError`, `GenerationErrorKind`, `StreamEvent` / `TokenEvent` / `ErrorEvent`, `reload_system_prompt`, `format_context`, `FormattedContext`, `VersionConflict`, and `sanitize_answer` |

## Generator return contract

`OllamaGenerator.generate()` always returns a `GenerationResult` — never `None`,
never raises. The result is a frozen dataclass:

```python
@dataclass(frozen=True)
class GenerationResult:
    answer: str            # "" on failure
    confidence: str        # "high" | "medium" | "low" — defaults to "medium"
    raw_response: Any      # the underlying LLMResponse (or None on failure)
    error: Optional[GenerationError] = None
```

On failure, `error` carries a typed `GenerationErrorKind` plus a
`user_message` (UI-safe) and `internal_detail` (logs/observability).
Replaces the prior `_last_response` / `_last_llm_confidence` instance
attributes that were unsafe to share across concurrent requests.

`generate_stream()` yields a discriminated union of `TokenEvent(text=...)`
during normal token flow and a single terminal `ErrorEvent(error=...)` on
failure, so streaming consumers can surface a typed error to the UI.

Retrieved context is wrapped in opaque fence delimiters
(`<<<DOCUMENT_CONTEXT_BEGIN/END>>>` and, when graph context is non-empty,
`<<<GRAPH_CONTEXT_BEGIN/END>>>`) so document content cannot collide with
prompt scaffolding markers like `Question:` or `Answer:`.

`reload_system_prompt()` is a public hot-reload hook that clears the
in-memory cache and re-reads `prompts/rag_system.md`.
