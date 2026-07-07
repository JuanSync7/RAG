# Observability Playbook (LLM-in-the-Loop Datasheet)

> Reference card for adding tracing/observability to new LLM-in-the-loop code in
> this repo. Optimized for an LLM coding assistant — scan top-down, copy the
> matching pattern, follow the gotcha list before shipping.

---

## 1. Mental model

- **One trace = one user-facing request or pipeline run.** Trace root lives at the entry point (API request, CLI command, Temporal activity, eval driver).
- **Spans** are timed sub-operations under the trace. Convention: `component.operation` (`retrieval.rerank.local`, `ingest.parse.docling`).
- **Generations** are spans specialised for LLM calls — they carry `model`, `input`, `output`, `prompt_tokens`, `completion_tokens`.
- **Ambient context** (OTel `contextvars`) means a `with tracer.span(...)` block is automatically the parent of any span opened inside it — across `await`, **not** across thread/process boundaries.
- **Backends are swappable** via the `OBSERVABILITY_PROVIDER` env var (`noop` | `otel`). All consumer code talks only to the public API.

## 2. Public API (the only surface you touch)

```python
from src.platform.observability import get_tracer

tracer = get_tracer()  # NEVER cache at module level — see §6.1

# Trace root (entry point)
with tracer.trace("pipeline.run", metadata={"user_id": uid}) as t:
    ...

# Plain span (any sub-operation)
with tracer.span("retrieval.rerank.local", {"top_k": 20, "model": name}):
    ...

# LLM call
with tracer.generation(
    name="llm.generate",
    model=model_id,
    input=prompt,
    metadata={"gen_ai.system": "litellm", "temperature": 0.2},
) as gen:
    out = client.complete(prompt)
    gen.set_output(out.text)
    gen.set_token_counts(prompt_tokens=out.usage.input, completion_tokens=out.usage.output)

# API boundary: continue an inbound W3C traceparent
trace = tracer.start_trace_from_carrier("api.query.post", carrier=headers, metadata=md)

# Cross-thread propagation
from src.platform.observability import submit_with_context
fut = submit_with_context(pool, worker_fn, *args)  # NOT pool.submit
```

## 3. Span-naming and attribute conventions

### Span names (snake_case, dot-separated)

| Layer | Pattern | Example |
| --- | --- | --- |
| API root | `api.<resource>.<verb>` | `api.query.post` |
| Pipeline root | `<pipeline>.run` | `ingest.directory`, `retrieval.rag` |
| LangGraph node | `node.<stage_name>` | `node.chunking` |
| Sub-operation | `<component>.<op>[.<variant>]` | `retrieval.search.weaviate`, `ingest.parse.docling` |
| LLM call | `llm.generate` / `llm.adapter.sync` / domain-specific (`dr_sufficiency`) | `llm.generate` |

### GenAI semantic-convention attribute keys

Use these exact keys on LLM-call spans/generations so backends (Langfuse, Phoenix, Braintrust, Honeycomb) auto-parse them:

| Key | Meaning |
| --- | --- |
| `gen_ai.system` | `"litellm"`, `"langchain"`, `"sentence-transformers"`, `"tei"`, `"ollama"`, `"openai"`… |
| `gen_ai.request.model` | Model identifier, e.g. `"qwen2.5:32b"` |
| `gen_ai.request.temperature` | Decoding temperature |
| `gen_ai.request.max_tokens` | Max output tokens |
| `gen_ai.prompt` | Prompt text (set via `Generation.input` arg) |
| `gen_ai.completion` | Output (set via `gen.set_output(...)`) |
| `gen_ai.usage.input_tokens` | Set via `set_token_counts(prompt_tokens=...)` |
| `gen_ai.usage.output_tokens` | Set via `set_token_counts(completion_tokens=...)` |

For HTTP boundaries also use OTel HTTP keys: `http.method`, `http.route`, `http.status_code`.

## 4. Pattern cookbook

### 4.1 New LLM call

```python
def generate_answer(prompt: str, model: str) -> str:
    with get_tracer().generation(
        name="llm.generate",
        model=model,
        input=prompt,
        metadata={"gen_ai.system": "litellm"},
    ) as gen:
        out = litellm.completion(model=model, messages=[{"role":"user","content":prompt}])
        text = out.choices[0].message.content
        gen.set_output(text)
        gen.set_token_counts(
            prompt_tokens=out.usage.prompt_tokens,
            completion_tokens=out.usage.completion_tokens,
        )
        return text
```

### 4.2 New LangGraph node

Use the `@node_span` decorator — it emits both a JSON log and an OTel span:

```python
from src.ingest.common.observability import node_span

@node_span("my_new_stage")
def my_new_stage(state: dict) -> dict:
    # any tracer.span() opened here nests under node.my_new_stage automatically
    ...
```

### 4.3 New pipeline root / orchestrator entry point

```python
def run_my_pipeline(payload: dict) -> Result:
    with get_tracer().trace("mypipeline.run", metadata={
        "user_id": payload.get("user_id"),
        "request_id": payload.get("request_id"),
    }):
        return _do_work(payload)
```

### 4.4 New retrieval / tool operation (non-LLM)

```python
def hybrid_search(query: str, k: int) -> list[Doc]:
    with get_tracer().span("retrieval.search.hybrid", {
        "query_len": len(query),
        "top_k": k,
    }) as span:
        docs = do_search(query, k)
        span.set_attribute("result_count", len(docs))
        return docs
```

### 4.5 New API endpoint

Nothing to do — `ObservabilityMiddleware` already wraps every request in
`api.<resource>.<verb>` and propagates inbound `traceparent`. Your handler can
attach domain attributes via `request.state.observability_trace`.

### 4.6 ThreadPoolExecutor fan-out

```python
from src.platform.observability import submit_with_context
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as pool:
    futures = [submit_with_context(pool, worker_fn, item) for item in items]
    results = [f.result() for f in futures]
```

### 4.7 SSE / streaming response

Nothing extra to do — the ASGI middleware defers root-span close until
`more_body=False`. Any span opened inside your async generator nests correctly.

## 5. Decision flowchart

```
Adding a new ...                     Do this
─────────────────────────────────────────────────────────────────────────
LLM call                           → tracer.generation(...) + set_output + set_token_counts
Embedding call                     → tracer.generation(name="embeddings.<provider>", ...)
Tool / retrieval / DB call         → tracer.span("<component>.<op>", attrs)
LangGraph node                     → @node_span("<name>") decorator
Pipeline entry point               → tracer.trace("<pipeline>.run", metadata)
API endpoint                       → nothing (middleware handles it)
SSE/streaming endpoint             → nothing (raw ASGI middleware handles it)
ThreadPoolExecutor fan-out         → submit_with_context, NOT pool.submit
asyncio.to_thread / anyio          → nothing (3.9+ propagates contextvars)
Subprocess / Temporal activity     → re-root: tracer.start_trace_from_carrier or fresh trace
Cross-service HTTP call            → inject W3C traceparent header (TODO: helper not yet added)
```

## 6. Gotchas — read before shipping

### 6.1 Never capture `get_tracer()` at module level

```python
# WRONG — locks singleton at import time, test fixtures break
tracer = get_tracer()
def f(): tracer.span(...)

# RIGHT — fetch lazily inside the function
def f(): get_tracer().span(...)

# ACCEPTABLE — capture in __init__ (instance is constructed after fixtures swap singleton)
class C:
    def __init__(self): self._tracer = get_tracer()
```

Why: test fixtures (`tests/observability/conftest.py`) swap `_backend` *after*
modules import. Module-level capture freezes a stale noop reference.

### 6.2 `ThreadPoolExecutor.submit` drops OTel context

`contextvars` (and therefore OTel ambient span context) does **not** propagate
into worker threads. Use `submit_with_context` (see §4.6). `asyncio.to_thread`
and `anyio.to_thread.run_sync` already copy context — they are safe.

### 6.3 `BaseHTTPMiddleware` closes streaming traces early

If you're writing custom middleware that needs to live for the full response
body (e.g. for streaming endpoints), use raw ASGI middleware and gate close on
`http.response.body` with `more_body=False`. Don't use
`starlette.middleware.base.BaseHTTPMiddleware`.

### 6.4 `Trace` and `Generation` ABCs do NOT expose `set_attribute`

Only `Span` does. To attach an attribute discovered post-construction to a
generation or trace, emit a sibling/child `tracer.span(...)` carrying the
attribute. Example:

```python
with trace.span("api.response", {"http.status_code": 200}):
    pass
```

### 6.5 `OTLPSpanExporter` does not take an `endpoint=` kwarg

Configure via env var only: `OTEL_EXPORTER_OTLP_ENDPOINT`. Passing
`endpoint=...` to the constructor is silently ignored on some versions.

### 6.6 Live tests need `source .env` first

The OTel exporter reads `OTEL_EXPORTER_OTLP_ENDPOINT` etc. from the process
environment. CI / local pytest runs targeting Langfuse must export env first.
Mark such tests `@pytest.mark.integration` so they don't run by default.

### 6.7 `OTelTrace` has no `.end()` method

Always use the context-manager form (`with tracer.trace(...) as t:`). Manual
`t.end()` doesn't exist and will silently no-op.

### 6.8 Fail-open is the contract

Every observability call must catch its own exceptions and never propagate
them to business code. When constructing a span where backend init might
throw, wrap with `contextlib.nullcontext()` as the fallback:

```python
try:
    cm = get_tracer().span("x", {...})
except Exception:
    cm = contextlib.nullcontext()
with cm:
    ...
```

### 6.9 Don't import `opentelemetry` outside `src/platform/observability/otel/`

Encapsulation. All other code uses the public API. This keeps backend swap
painless.

### 6.10 Avoid high-cardinality attributes

Never put user IDs, query text, or document text into span/operation names.
Names are dimensions — they explode cardinality. Put them in attributes,
where backends can downsample or filter.

## 7. Testing observability

### 7.1 Unit-test span emission with `InMemorySpanExporter`

Pattern from `tests/observability/test_otel_backend.py`:

```python
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

def _build_otel_backend():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    backend = OTelBackend(provider=provider)  # bypass global SDK setup
    return backend, exporter

def test_my_op_emits_span():
    backend, exporter = _build_otel_backend()
    _obs_module._backend = backend
    try:
        my_op()
        spans = exporter.get_finished_spans()
        assert any(s.name == "my.op" for s in spans)
    finally:
        _obs_module._backend = None
```

### 7.2 Watch for `tests/observability/conftest.py`

That conftest has an autouse fixture that forces `OBSERVABILITY_PROVIDER=noop`
**only within `tests/observability/`**. Tests in other directories
(`tests/server/`, `tests/ingest/`, `tests/guardrails/`) can directly install an
OTel backend without fighting it.

### 7.3 Live e2e tests against Langfuse

Mark with `@pytest.mark.integration`. Source env first. Use
`_poll_for_trace` / `_poll_for_observations` helpers (TODO: extract to
`tests/observability/_live_helpers.py`).

## 8. Backend / deployment notes

### 8.1 Provider swap

| Env | Result |
| --- | --- |
| `OBSERVABILITY_PROVIDER=noop` | Default. Tracing is a no-op. |
| `OBSERVABILITY_PROVIDER=otel` + `OTEL_EXPORTER_OTLP_ENDPOINT=https://...` | Spans export via OTLP/HTTP. |

Any OTLP-compatible collector works: Langfuse v3, Phoenix, Braintrust,
Honeycomb, Datadog, Grafana Tempo, self-hosted OTel Collector.

### 8.2 Resource attributes

Set on the OTel `Resource` at backend init: `service.name`, `service.version`,
`deployment.environment`. Already wired in `OTelBackend.__init__`.

## 9. Checklist before merging observability changes

- [ ] No module-level `tracer = get_tracer()` calls in new code (§6.1).
- [ ] Every LLM call uses `tracer.generation(...)` with `set_output` + `set_token_counts`.
- [ ] Span names follow `component.operation` convention; no high-cardinality values in names (§6.10).
- [ ] LLM spans carry `gen_ai.system` and `gen_ai.request.model` attrs (§3).
- [ ] All new fan-out via `ThreadPoolExecutor` uses `submit_with_context` (§4.6).
- [ ] No `opentelemetry` imports outside `src/platform/observability/otel/` (§6.9).
- [ ] Fail-open: all observability calls in hot paths wrapped against backend init errors (§6.8).
- [ ] Unit tests use `InMemorySpanExporter` to assert span shape (§7.1).
- [ ] Live e2e test added if the change introduces a new trace root or major span subtree.
- [ ] Docs updated if architecture / public API changed.

## 10. Pointers

- Engineering guide: `docs/observability.md`
- ABC contracts: `src/platform/observability/backend.py`
- OTel impl: `src/platform/observability/otel/backend.py`
- Threadpool helper: `src/platform/observability/concurrency.py`
- API middleware: `server/observability_middleware.py`
- Test patterns: `tests/observability/test_otel_backend.py`,
  `tests/observability/test_otel_live_e2e.py`
- LangGraph node decorator: `src/ingest/common/observability.py`
