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
| `generator.py` | `OllamaGenerator` — LiteLLM Router-backed class that builds structured prompts (with optional graph context and conversation history), calls the LLM for JSON-formatted answers, and captures self-reported confidence for downstream scoring (REQ-601, REQ-602, REQ-KG-794, REQ-KG-796) |
| `output_sanitizer.py` | `sanitize_answer` — strips document boundary markers, unreplaced template variables, and system prompt fragments from generated answers using structural detection rather than regex (REQ-704) |
| `__init__.py` | Package facade re-exporting `OllamaGenerator`, `format_context`, `FormattedContext`, `VersionConflict`, and `sanitize_answer` |
