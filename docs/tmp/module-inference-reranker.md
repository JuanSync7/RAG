### `src/retrieval/query/nodes/reranker.py` — Reranker Providers

#### Purpose

This module provides the reranking layer of the retrieval pipeline. After an initial vector search returns a candidate set of documents, the reranker rescores each candidate against the query using a cross-encoder model, which reads the query and document together (rather than embedding them independently). The top-K highest-scoring documents are returned as `RankedResult` objects for downstream answer generation.

The module exposes two provider implementations and a factory function that selects between them based on runtime configuration:

- **`LocalBGEReranker`** — loads `BAAI/bge-reranker-v2-m3` in-process via Hugging Face `transformers` and scores batches on the local GPU or CPU.
- **`LiteLLMReranker`** — delegates to a separately-deployed vLLM container via the `litellm.rerank()` API, keeping the main process free of model weights.

---

#### How it works

**Factory dispatch — `get_reranker_provider()`**

The public entry point for the rest of the pipeline is `get_reranker_provider()`. It reads the `INFERENCE_BACKEND` config value (set by `RAG_INFERENCE_BACKEND`, validated at settings load time) and returns the matching provider:

```python
def get_reranker_provider():
    if INFERENCE_BACKEND == "vllm":
        return LiteLLMReranker(timeout=VLLM_TIMEOUT_SECONDS)
    return LocalBGEReranker()
```

Any value other than `"vllm"` routes to `LocalBGEReranker`. An unknown backend value raises `ValueError` inside `config/settings.py` before this function is ever reached.

---

**`LocalBGEReranker` — load sequence**

Construction performs the full model load synchronously and logs elapsed time for observability:

1. **Device selection** — checks `torch.cuda.is_available()` and sets `self.device` to `"cuda"` or `"cpu"`.
2. **Dtype resolution** — looks up `RERANKER_PRECISION` (e.g. `"fp16"`) in `_TORCH_DTYPE_NAME_BY_PRECISION`, which maps precision labels to `torch` attribute *name strings* (e.g. `"float16"`). The actual `torch.float16` object is resolved at runtime via `getattr(torch, _dtype_name)`. This deferred resolution is intentional (see Key Design Decisions).
3. **Model load — two paths:**
   - If dtype is `fp16` or `bf16`: loads directly in the reduced precision with `torch_dtype=torch_dtype` and `attn_implementation="sdpa"` to activate PyTorch's Scaled Dot-Product Attention kernel. This avoids materializing a wasteful full-precision copy first.
   - If dtype is `fp32` (or unrecognized, such as `int8`/`int4`): loads in default fp32. Unrecognized precision labels emit a `logger.warning` because `int8`/`int4` require `bitsandbytes` quantization wiring that is not yet present.
4. **Error handling** — any exception during tokenizer or model load is caught, logged, and re-raised as `ModelLoadError` (a typed exception that carries `model_path` for structured error handling upstream).
5. **`model.eval()`** — disables dropout and batch normalization update in inference mode.

---

**`LocalBGEReranker.rerank()` — batching loop and manual sigmoid**

```python
def rerank(self, query, documents, top_k=RERANK_TOP_K) -> list[RankedResult]:
```

The method pairs the query with every document text, then scores them in fixed-size batches to bound peak VRAM usage:

```python
pairs = [[query, doc.text] for doc in documents]
for i in range(0, len(pairs), RERANKER_BATCH_SIZE):
    batch = pairs[i : i + RERANKER_BATCH_SIZE]
    inputs = self.tokenizer(batch, padding=True, truncation=True,
                            max_length=RERANKER_MAX_LENGTH,
                            return_tensors="pt").to(self.device)
    logits = self.model(**inputs).logits.squeeze(-1).float()
    scores_raw = logits.cpu().tolist()
    if isinstance(scores_raw, float):
        scores_raw = [scores_raw]
    batch_scores = [1.0 / (1.0 + math.exp(-x)) for x in scores_raw]
    scores.extend(batch_scores)
```

After all batches, each `(document, score)` pair is assembled into a `RankedResult`, the list is sorted descending by score, and the top-K slice is returned.

The entire body runs under `torch.inference_mode()` as a context manager (not a decorator — see Key Design Decisions), combined with an observability span that records `input_count`, `top_k`, `score_min`, `score_max`, and `output_count`.

---

**`LiteLLMReranker.rerank()` — deferred import and results guard**

```python
def rerank(self, query, documents, top_k=RERANK_TOP_K) -> list[RankedResult]:
    import litellm  # deferred so API container never imports it
```

`litellm` is imported inside the method body, not at module load time. This prevents the API server — which does not use vLLM — from incurring the `litellm` import cost or failing on environments where `litellm` is not installed.

The call delegates to:

```python
resp = litellm.rerank(
    model=self.model,
    query=query,
    documents=[d.text for d in documents],
    top_n=top_k,
    timeout=self.timeout,
)
raw_results = resp.results or []
```

`resp.results or []` guards against a `None` response body from the vLLM endpoint (possible on transient routing failures or empty result sets). The raw results carry an `"index"` field pointing back into the original `documents` list, which the method uses to reconstruct full `RankedResult` objects with metadata intact.

---

#### Key design decisions

| Decision | Choice | Alternatives considered | Rationale |
|----------|--------|------------------------|-----------|
| `_TORCH_DTYPE_NAME_BY_PRECISION` stores strings, not `torch.dtype` objects | `{"fp16": "float16", ...}` with `getattr(torch, name)` at use time | Store `torch.float16` etc. directly at module level | Module-level `torch.float32` access crashed CI pytest collection on incomplete torch wheel builds. Strings are safe at import time; `getattr` is deferred until the constructor actually runs on a complete environment. |
| `torch.inference_mode()` used as a context manager, not a `@torch.inference_mode()` decorator | `with torch.inference_mode():` inside `rerank()` | `@torch.inference_mode()` decorator on the method | The decorator form evaluates `torch.inference_mode()` at class-body definition time (i.e., module import time). CI intermittently had incomplete torch wheels at that point, breaking pytest collection. The context manager defers evaluation until the method is called. |
| Manual sigmoid via `1.0 / (1.0 + math.exp(-x))` instead of `torch.sigmoid` | Python `math.exp` on CPU-extracted floats | `logits.sigmoid()` or `torch.sigmoid(logits)` | `transformers >= 4.50` returns logits as a `_Logits` proxy object that lacks `__neg__` and `.sigmoid()`. `torch.sigmoid` was removed in `torch 2.x`. Both built-in approaches would break on modern toolchains. Converting to Python floats via `.cpu().tolist()` escapes the proxy and is numerically equivalent. |
| `resp.results or []` guard in `LiteLLMReranker` | Falsy check on the results field | Assert non-None / raise on empty | `litellm.rerank()` returns a response object whose `.results` field can be `None` on routing failures or empty document inputs. A hard assert would turn a recoverable transient failure into a crash. The `or []` sentinel degrades gracefully to an empty result list, consistent with the empty-input fast path in both providers. |
| `litellm` imported inside `rerank()`, not at module top | Deferred `import litellm` | Top-level import | The API server container does not include `litellm` in its dependency set. A top-level import would make the entire `reranker` module unimportable in that environment, even when `INFERENCE_BACKEND` is `"local"`. Deferring the import scopes the dependency to the code path that actually needs it. |
| `LocalBGEReranker` uses `transformers` directly instead of `FlagEmbedding` | `AutoModelForSequenceClassification` + `AutoTokenizer` | `FlagEmbedding.FlagReranker` | `FlagEmbedding` has compatibility issues with `transformers >= 5.x`. Using `transformers` directly removes the third-party wrapper dependency and gives full control over dtype, attention implementation, and inference mode. |
| `attn_implementation="sdpa"` on all load paths | SDPA for both fp32 and reduced-precision | Default attention / flash-attn | PyTorch's built-in SDPA fuses the attention kernel and reduces VRAM for long sequences without requiring a separate `flash-attn` wheel, which has strict CUDA version constraints. |

---

#### Configuration

| Parameter | Env var | Type | Default | Effect |
|-----------|---------|------|---------|--------|
| `INFERENCE_BACKEND` | `RAG_INFERENCE_BACKEND` | `str` | `"local"` | Selects provider: `"local"` → `LocalBGEReranker`; `"vllm"` → `LiteLLMReranker`. Any other value raises `ValueError` at settings load. |
| `RERANKER_MODEL_PATH` | Derived from `RAG_MODEL_ROOT` | `str` | *(configured under model root)* | Filesystem path to the `BAAI/bge-reranker-v2-m3` model directory. Used only by `LocalBGEReranker`. |
| `RERANK_TOP_K` | `RAG_RERANK_TOP_K` | `int` | `5` | Maximum number of `RankedResult` objects returned by either provider's `rerank()` call. |
| `RERANKER_MAX_LENGTH` | `RAG_RERANKER_MAX_LENGTH` | `int` | `512` | Token truncation limit passed to the tokenizer in `LocalBGEReranker`. Prevents OOM on unexpectedly long documents. |
| `RERANKER_BATCH_SIZE` | `RAG_RERANKER_BATCH_SIZE` | `int` | `32` | Number of query-document pairs processed per forward pass in `LocalBGEReranker`. Lower values reduce peak VRAM at the cost of more kernel launches. |
| `RERANKER_PRECISION` | `RAG_RERANKER_PRECISION` | `str` | `"fp32"` | Weight precision for `LocalBGEReranker`. Supported: `"fp32"`, `"fp16"`, `"bf16"`. `"int8"` / `"int4"` are recognized keys but fall through to fp32 with a warning until `bitsandbytes` is wired. |
| `VLLM_TIMEOUT_SECONDS` | `RAG_VLLM_TIMEOUT_SECONDS` | `int` | `30` | HTTP timeout (seconds) for `LiteLLMReranker` requests to the vLLM `/v1/rerank` endpoint. |

---

#### Error behavior

**`ModelLoadError`** — raised by `LocalBGEReranker.__init__()` if the tokenizer or model fails to load from `RERANKER_MODEL_PATH` (e.g., missing files, corrupted weights, incompatible architecture). The exception carries a `model_path` attribute for structured upstream handling. The original exception is chained via `raise ... from exc` so the full traceback is preserved. The error is also logged at `ERROR` level before being raised.

**`RuntimeError` from CUDA OOM or inference failure** — `LocalBGEReranker.rerank()` documents that it may raise `RuntimeError` if the model inference step fails. Common causes include GPU out-of-memory (reduce `RERANKER_BATCH_SIZE` or switch to `"fp16"`/`"bf16"`), tokenizer version mismatches producing unexpected tensor shapes, or a corrupted model producing an unexpected output shape from `.logits.squeeze(-1)`. These are not caught inside the method and propagate to the caller.

**Empty-input fast path** — both `LocalBGEReranker.rerank()` and `LiteLLMReranker.rerank()` return `[]` immediately when `documents` is empty, before any tokenization or remote call. This is a deliberate sentinel, not an error, and avoids unnecessary work when the upstream vector search returns no candidates.

**`ValueError` from unknown `INFERENCE_BACKEND`** — validation occurs inside `config/settings.py` at application startup, not inside this module. If `RAG_INFERENCE_BACKEND` is set to an unrecognized value (anything other than `"local"` or `"vllm"`), a `ValueError` is raised before `get_reranker_provider()` is ever called, so the misconfiguration is surfaced at startup rather than at the first rerank request.

**`resp.results` being `None` in `LiteLLMReranker`** — the `resp.results or []` guard silently converts a `None` response from the vLLM endpoint into an empty result list. This is treated as a graceful degradation (no ranked results surfaced) rather than a hard failure, consistent with the behavior when the document set is empty.
