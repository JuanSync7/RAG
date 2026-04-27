### `src/core/embeddings.py` — Embedding Providers

#### Purpose

This module provides two interchangeable embedding backends and a factory function that selects between them at runtime based on configuration. Both backends implement LangChain's `Embeddings` interface so the rest of the system can treat them identically.

The two backends are:

- **`LocalBGEEmbeddings`** — runs BAAI/bge-m3 in-process using `sentence-transformers`. Suitable for single-node or development deployments where a GPU is available on the same host.
- **`LiteLLMEmbeddings`** — calls an OpenAI-compatible `/v1/embeddings` endpoint (typically a vLLM container) over HTTP via the `litellm` library. Suitable for distributed deployments where embedding is offloaded to a separate inference server.

The module also exposes an internal method `encode_sentences` on both classes, used by the semantic chunking stage to compute cosine similarities without going through the LangChain interface.

---

#### How it works

**Step 1 — Factory selection (`get_embedding_provider`)**

The entry point for all callers is `get_embedding_provider()`. It reads the `INFERENCE_BACKEND` setting and returns the appropriate instance:

```python
def get_embedding_provider() -> Embeddings:
    if INFERENCE_BACKEND == "vllm":
        return LiteLLMEmbeddings(timeout=VLLM_TIMEOUT_SECONDS)
    return LocalBGEEmbeddings()
```

Callers receive a concrete `Embeddings` object and never need to branch on backend type themselves.

**Step 2 — Local path: `LocalBGEEmbeddings`**

Construction loads the model from disk via a lazy import of `sentence-transformers`:

```python
def _load_sentence_transformer(model_path: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_path)
```

The lazy import is intentional — it ensures the worker image (used in vLLM deployments) can start without `sentence-transformers` installed.

`embed_documents` batches the input texts and calls `model.encode` with `normalize_embeddings=True` and a batch size of 32. The normalization step produces unit-norm vectors, which is required for the semantic chunking contract (see `encode_sentences` below).

`embed_query` encodes a single string. It uses the same normalization flag but omits the batch size and progress bar for single-item efficiency.

`encode_sentences` is called by the semantic chunking stage. It uses a larger batch size (64) and disables the progress bar:

```python
def encode_sentences(self, sentences: list[str]) -> np.ndarray:
    return self.model.encode(
        sentences,
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=False,
    )
```

**Step 3 — Remote path: `LiteLLMEmbeddings`**

`litellm` is imported lazily inside `embed_documents`, so the API container never pays its import cost unless the vLLM backend is active:

```python
def embed_documents(self, texts: list[str]) -> list[list[float]]:
    import litellm
    resp = litellm.embedding(model=self.model, input=texts, timeout=self.timeout)
    return [d["embedding"] for d in resp.data]
```

`embed_query` delegates to `embed_documents` with a single-element list and returns the first result.

`encode_sentences` calls `embed_documents` and then L2-normalizes the resulting numpy array in-place, because vLLM's endpoint does not normalize by default:

```python
def encode_sentences(self, sentences: list[str]) -> np.ndarray:
    vecs = np.array(self.embed_documents(sentences))
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vecs / norms
```

**Step 4 — Observability**

Both classes obtain a tracer from `src.platform.observability.get_tracer()` at construction time. Every `embed_documents` and `embed_query` call opens a span, records either the batch size or the text length as an attribute, and closes the span with `status="ok"` on success. Spans are not closed on exception — any exception propagates to the caller with the span left open, which the observability layer treats as an implicit error signal.

---

#### Key design decisions

| Decision | Choice | Alternatives considered | Rationale |
|----------|--------|------------------------|-----------|
| Lazy imports for `sentence-transformers` and `litellm` | Import inside the function/method body, not at module top | Top-level imports | Each dependency is heavy and only needed in one deployment mode. A vLLM deployment should not need `sentence-transformers` installed, and the API container should not need `litellm`. Lazy importing keeps the container images lean and prevents import-time failures in the wrong environment. |
| Single factory function rather than a registry or plugin system | `get_embedding_provider()` reads one env var and returns one of two hardcoded classes | A plugin registry, abstract factory class, or entry-points mechanism | The system currently has exactly two backends. A registry adds indirection without benefit at this scale. The factory is easy to extend when a third backend is needed — add one `elif` branch. |
| Both classes expose `encode_sentences` returning `np.ndarray` | Defined on both `LocalBGEEmbeddings` and `LiteLLMEmbeddings` | Only on `LocalBGEEmbeddings`; semantic chunking uses local only | Semantic chunking must work in both deployment modes. Exposing `encode_sentences` on both classes — with explicit L2 normalization on the vLLM path — keeps the chunking stage backend-agnostic. |
| L2 normalization in `LiteLLMEmbeddings.encode_sentences` | Normalize after the API call using numpy | Rely on vLLM normalization config; document that callers must normalize | vLLM's `/v1/embeddings` endpoint does not normalize by default and the setting is server-side, not per-request. Normalizing client-side in `encode_sentences` satisfies the cosine-sim contract without requiring server configuration. The zero-norm guard (`np.where(norms == 0, 1.0, norms)`) prevents division-by-zero on degenerate inputs. |
| LangChain `Embeddings` as the base class | Both classes inherit from `langchain_core.embeddings.Embeddings` | A custom abstract base class; no base class at all | LangChain's `Embeddings` interface is the standard contract for embeddings in LangChain retrieval chains and vector store integrations. Conforming to it means the classes are directly usable anywhere LangChain expects an `Embeddings` object, with no adapter layer. |

---

#### Configuration

| Parameter | Env var | Type | Default | Effect |
|-----------|---------|------|---------|--------|
| `INFERENCE_BACKEND` | `RAG_INFERENCE_BACKEND` | `str` | `"local"` | Selects the active backend. `"local"` → `LocalBGEEmbeddings`; `"vllm"` → `LiteLLMEmbeddings`. Any other value raises `ValueError` in `config/settings.py` at startup, before any embedding call is made. |
| `EMBEDDING_MODEL_PATH` | derived from `RAG_MODEL_ROOT` | `str` | path under `RAG_MODEL_ROOT/baai/bge-m3` | Filesystem path passed to `SentenceTransformer()` when the local backend is active. Has no effect when `INFERENCE_BACKEND="vllm"`. |
| `VLLM_TIMEOUT_SECONDS` | `RAG_VLLM_TIMEOUT_SECONDS` | `int` | `30` | HTTP timeout (in seconds) applied to every `litellm.embedding()` call when the vLLM backend is active. Has no effect on the local backend. |

The `model` name passed to `LiteLLMEmbeddings` defaults to `"embedding"` and is resolved by the LiteLLM router configuration (`RAG_LLM_ROUTER_CONFIG`) to the actual backend URL. If the router config is absent or misconfigured, `litellm.embedding()` will raise at call time.

---

#### Error behavior

**`LocalBGEEmbeddings`**

- **Model not found at path**: `SentenceTransformer(model_path)` raises `OSError` or an internal `sentence-transformers` exception at construction time (inside `__init__`). The error propagates out of `get_embedding_provider()` to the caller. The process will typically fail to start if this path is wrong.
- **`sentence-transformers` not installed**: The lazy import inside `_load_sentence_transformer` raises `ModuleNotFoundError` at first construction. This is expected when the worker is configured for vLLM — the factory will never call this path in that case.
- **Encoding errors**: `model.encode()` propagates exceptions from the underlying PyTorch/numpy stack (for example `RuntimeError` on CUDA OOM). No retry logic is applied; callers are responsible for handling these.

**`LiteLLMEmbeddings`**

- **`litellm` not installed**: The deferred `import litellm` inside `embed_documents` raises `ModuleNotFoundError` at the first call, not at construction time. This is expected when running with the local backend — the factory will never instantiate this class in that case.
- **Request timeout**: `litellm.embedding()` raises `litellm.Timeout` (or a transport-level exception) if the vLLM server does not respond within `VLLM_TIMEOUT_SECONDS`. Callers should handle this and apply appropriate retry or circuit-breaking logic at their layer.
- **HTTP / model routing errors**: LiteLLM surfaces these as `litellm.BadRequestError`, `litellm.AuthenticationError`, or similar subclasses of `litellm.exceptions.APIError`. A misconfigured `RAG_LLM_ROUTER_CONFIG` (wrong model name, unreachable URL) will surface as one of these errors on the first embedding call.
- **Zero-norm vectors in `encode_sentences`**: The explicit guard `np.where(norms == 0, 1.0, norms)` silently treats a zero vector as already normalized (divides by 1.0), returning a zero vector unchanged. This is a safe fallback rather than an error; the semantic chunking stage should treat any resulting zero-similarity scores as expected.

In both classes, observability spans are opened before the backend call and closed with `status="ok"` only on success. An exception escaping the method leaves the span unclosed, which the observability backend records as an error. No exceptions are caught or swallowed within this module.
