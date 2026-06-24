<!-- @summary
Token budget tracker backed by litellm.token_counter(): accurate per-model
tokenization, model capability discovery (litellm + Ollama fallback), and
context window usage calculation for console, CLI, and API responses.
@end-summary -->

# Token Budget

Provides accurate token counting and context window usage percentage so users
know when to compact conversations or adjust query parameters. Backed by
`litellm.token_counter()` for per-model tokenization accuracy, with character
heuristic fallback for models not in litellm's registry.

## Files

| File | Purpose | Key Exports |
|------|---------|-------------|
| `__init__.py` | Public API facade | `count_tokens`, `calculate_budget`, `get_capabilities`, etc. |
| `schemas.py` | Frozen dataclasses for budget state | `ModelCapabilities`, `TokenBreakdown`, `TokenBudgetSnapshot` |
| `provider.py` | Model capability discovery + budget calculation | `get_capabilities`, `refresh_capabilities`, `calculate_budget` |
| `utils.py` | Token counting via litellm (heuristic fallback) | `count_tokens` |

## Token Counting

Primary: `litellm.token_counter(model, text=...)` — uses the correct tokenizer
for each provider (tiktoken for OpenAI, SentencePiece for Ollama/Llama, etc.).

Fallback: character heuristic (`len(text) // chars_per_token`) for models
litellm doesn't recognize.

```python
from src.platform.token_budget import count_tokens

# Accurate per-model count
tokens = count_tokens(text="What is RAG?", model="ollama/qwen2.5:3b")

# Message-level count (includes role/template overhead)
tokens = count_tokens(messages=[{"role": "user", "content": "Hello"}], model="ollama/qwen2.5:3b")
```

## Model Capability Discovery

Resolution order:
1. `litellm.get_model_info()` — works for cloud & known open-weight models
2. Ollama `/api/show` — works for locally-pulled Ollama models
3. Default fallback (`TOKEN_BUDGET_DEFAULT_CONTEXT_LENGTH`)

## Budget Calculation

```python
from src.platform.token_budget import calculate_budget

snapshot = calculate_budget(
    system_prompt="You are a helpful assistant...",
    memory_context="Prior conversation summary...",
    chunks=["chunk1", "chunk2"],
    query="What is RAG?",
    model="ollama/qwen2.5:3b",
)
print(f"{snapshot.usage_percent}% of context window used")
print(f"{snapshot.input_tokens}/{snapshot.context_length} tokens")
```

## Integration

The token budget flows through the pipeline:

```
RAGChain.run()
  └─ calculate_budget() after retrieval + generation
     ├─ Estimates: system_prompt, memory, chunks, query tokens
     └─ Actuals: prompt_tokens, completion_tokens from LLM response
       ↓
RAGResponse.token_budget (TokenBudgetSnapshot)
       ↓
QueryResponse.token_budget (serialized for API)
       ↓
CLI display_results() → color-coded context window %
```

## Configuration

| Env Var | Default | Purpose |
|---------|---------|---------|
| `RAG_TOKEN_BUDGET_DEFAULT_CONTEXT_LENGTH` | `2048` | Fallback when model info unavailable |
| `RAG_TOKEN_BUDGET_CHARS_PER_TOKEN` | `4` | Heuristic ratio (fallback only) |
| `RAG_TOKEN_BUDGET_WARN_PERCENT` | `70` | Yellow warning threshold |
| `RAG_TOKEN_BUDGET_CRITICAL_PERCENT` | `90` | Red critical threshold |
