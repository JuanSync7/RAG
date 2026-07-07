<!-- @summary
RagWeave observability engineering guide: OTel-native tracing backend, OTLP/HTTP export to any compatible collector (Langfuse, Phoenix, Honeycomb, ...), public API surface, span-tree reference, backend swap procedure, extension patterns, and known gotchas.
@end-summary -->

# Observability — Engineering Guide

RagWeave emits OpenTelemetry traces for every meaningful execution surface:
API requests, retrieval, LLM / embedding calls, ingest, and guardrails.
There is **one** backend (`OTelBackend`) which speaks OTLP/HTTP to any
OTel-compatible collector. Swap collectors with **one** env var.

This guide is the entry-point for instrumentation work. For deeper design
context see [`docs/observability/`](observability/) (design, spec,
implementation notes). For per-component span inventories see
`src/platform/observability/README.md`.

Related memory:
[`~/.claude/projects/-home-kok-shew-juan-RagWeave/memory/project_observability_otel.md`](../../../.claude/projects/-home-kok-shew-juan-RagWeave/memory/project_observability_otel.md).

---

## 1. Architecture

```
┌───────────┐    public API     ┌──────────────┐   OTLP/HTTP   ┌────────────┐
│ call site │ ───────────────▶  │ OTelBackend  │ ─────────────▶│  collector │
│  (Python) │  get_tracer()     │  (OTel SDK)  │  /v1/traces   │ (Langfuse, │
│           │  .trace/.span/    │              │               │  Phoenix,  │
│           │  .generation/...  │              │               │  ...)      │
└───────────┘                   └──────────────┘               └────────────┘
```

* The only mandatory dependency is `opentelemetry-{api,sdk,exporter-otlp-proto-http}`.
* No vendor SDKs (Langfuse / Phoenix / Braintrust) are imported.
* Collector swap = `OTEL_EXPORTER_OTLP_ENDPOINT` + headers. Application code is
  unchanged.
* Provider selection lives in `src/platform/observability/__init__.py::_init_backend`.

### Provider selector

| `RAG_OBSERVABILITY_PROVIDER` | Backend       | Notes |
| ---------------------------- | ------------- | ----- |
| (unset / empty)              | `OTelBackend` | Default. Drops silently if no collector listening. |
| `otel`                       | `OTelBackend` | Canonical. |
| `langfuse`                   | `OTelBackend` | Deprecated alias; emits `DeprecationWarning`. |
| `noop`                       | `NoopBackend` | Explicit opt-out. |
| anything else                | —             | `ValueError` at first `get_tracer()`. |

---

## 2. Public API

Only import from `src.platform.observability`. Everything below is the stable
surface; never reach into `src.platform.observability.otel.*`.

```python
from src.platform.observability import get_tracer

tracer = get_tracer()

# 2.1 Root trace (one per logical request)
with tracer.trace("retrieval.rag", metadata={"query": q}) as t:
    # 2.2 Child span
    with t.span("retrieval.search.hybrid", attributes={"top_k": 10}) as s:
        ...
    # 2.3 LLM generation (gen_ai.* semantic conventions)
    gen = t.generation(
        "llm.generate",
        model="claude-3-7-sonnet",
        input=prompt,
        metadata={"system": "anthropic"},
    )
    gen.set_output(answer)
    gen.set_token_counts(prompt_tokens, completion_tokens)
    gen.end(status="ok")
```

### 2.4 Distributed propagation — W3C `traceparent`

For inbound HTTP requests, parent the trace under the caller's
`traceparent` header so the trace_id is shared across services:

```python
trace = tracer.start_trace_from_carrier(
    "api.query.post",
    carrier=dict(request.headers),
    metadata={"http.method": "POST", "http.route": "/query"},
)
with trace as t:
    ...
```

If the carrier has no valid `traceparent`, `start_trace_from_carrier` falls
back to a fresh root (see `OTelBackend.start_trace_from_carrier` at
`src/platform/observability/otel/backend.py:456`).

### 2.5 Ambient context

`OTelTrace.__enter__` attaches the root span as the OTel ambient context
(`opentelemetry.context.attach`). Any code running inside the `with` block
that uses OTel directly will be parented correctly — **including new spans
emitted by libraries we don't own**. `__exit__` detaches.

### 2.6 `gen_ai.*` attribute conventions

`Trace.generation(name, model, input, metadata)` sets the standard
OpenTelemetry gen_ai semantic-convention attributes on the span:

| Attribute              | Source                                         |
| ---------------------- | ---------------------------------------------- |
| `gen_ai.system`        | `metadata["gen_ai.system"]` or `metadata["system"]`, else `"unknown"` |
| `gen_ai.request.model` | the `model` argument                           |
| `gen_ai.prompt`        | the `input` argument (truncated)               |
| `gen_ai.completion`    | `generation.set_output(...)`                   |
| `gen_ai.usage.input_tokens` / `output_tokens` | `generation.set_token_counts(prompt, completion)` |

Langfuse, Phoenix, and Braintrust all map these to their own native
generation/usage UI.

---

## 3. Span-tree reference

What lands in the collector for one real `POST /query`:

```
api.query.post                            # Starlette middleware, parented to inbound traceparent
└── retrieval.rag                         # RAGChain.run root
    ├── retrieval.rag.process_query
    │   ├── retrieval.query.process       # query_processor
    │   ├── retrieval.query.call_llm      # if rewrite/expand uses LLM
    │   │   └── llm.generate              # LLM call (GENERATION)
    │   └── retrieval.query.llm_healthcheck
    ├── retrieval.rag.pii_gate            # input guardrails
    │   └── guardrails.input
    │       └── guardrails.rail.<name>    # one per rail
    ├── retrieval.embed_query
    │   └── embeddings.{local,tei}.query  # GENERATION
    ├── retrieval.search.hybrid
    │   └── retrieval.search.weaviate
    ├── retrieval.kg_expand                # optional
    ├── retrieval.collect_candidates
    ├── retrieval.rerank
    │   ├── retrieval.rerank.local        # ce-rerank
    │   └── retrieval.rerank.tei          # if remote
    ├── retrieval.tree.descent             # tree-RAG, optional
    │   └── retrieval.tree.lift
    ├── retrieval.visual                   # ColQwen, optional
    │   ├── retrieval.visual.model_load
    │   ├── retrieval.visual.text_encode
    │   ├── retrieval.visual.search
    │   └── retrieval.visual.presigned_urls
    ├── retrieval.deep_research            # optional
    │   └── retrieval.deep_research.iteration (xN)
    ├── retrieval.fallback                 # if all retrievers empty
    ├── retrieval.generation
    │   ├── retrieval.generation.format_context
    │   ├── retrieval.generation.is_available
    │   ├── retrieval.generation.answer
    │   │   └── llm.generate / llm.agenerate / llm.adapter.{sync,stream}
    │   ├── retrieval.generation.stream
    │   └── retrieval.generation.sanitize
    ├── retrieval.confidence.score
    ├── retrieval.confidence.routing
    ├── retrieval.re_retrieval             # confidence-driven retry, optional
    └── retrieval.rag.output_rails
        └── guardrails.output
            └── guardrails.rail.<name>
```

### Ingest tree (per `POST /ingest`)

```
api.ingest.post
└── ingest.file  /  ingest.directory
    └── ingest.doc_processing
        ├── ingest.parse.docling
        ├── ingest.chunk.docling
        ├── ingest.vision.caption_batch
        │   ├── ingest.vision.caption
        │   └── ingest.vision.notes
        └── ingest.embedding
            ├── ingest.embedding.colqwen
            └── ingest.embedding.store
                └── embeddings.{local,tei}.batch
```

### API roots (all share the same naming convention)

`api.{resource}.{verb}` — `api.query.post`, `api.ingest.post`,
`api.health.get`, etc. All parented under inbound traceparent when present.
See the Starlette middleware in `server/observability_middleware.py`.

---

## 4. Backend swapping

One env var controls the destination collector. Headers are signal-specific.

| Collector            | `OTEL_EXPORTER_OTLP_ENDPOINT`                    | Auth header                                                                  |
| -------------------- | ------------------------------------------------ | ---------------------------------------------------------------------------- |
| Langfuse self-hosted | `http://localhost:3000/api/public/otel`          | Auto-derived from `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` (Basic).      |
| Langfuse Cloud       | `https://cloud.langfuse.com/api/public/otel`     | Same as above.                                                               |
| Phoenix self-hosted  | `http://localhost:6006`                          | None for local dev.                                                          |
| Braintrust           | `https://api.braintrust.dev/otel`                | `OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer <key>"`                    |
| Honeycomb            | `https://api.honeycomb.io`                       | `OTEL_EXPORTER_OTLP_HEADERS="x-honeycomb-team=<key>"`                        |
| Datadog (Agent)      | `http://<agent>:4318`                            | Configure on the Datadog Agent side.                                         |

The OTLP/HTTP exporter automatically appends `/v1/traces` to the endpoint
when **read from the env var**. See gotcha §6.1.

For Langfuse, if `OTEL_EXPORTER_OTLP_HEADERS` is unset but
`LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are present,
`OTelBackend.__init__` calls
`src.platform.observability.otel.auth.ensure_otlp_headers_from_env` which
builds the `Authorization=Basic ...` header.

---

## 5. Extending instrumentation

### 5.1 Instrumenting a new function

```python
from src.platform.observability import get_tracer

def my_function(x):
    tracer = get_tracer()
    with tracer.span("subsystem.my_function", attributes={"x_len": len(x)}) as s:
        result = do_work(x)
        s.set_attribute("result_size", len(result))
        return result
```

Or use the `@observe` decorator for thin wrappers:

```python
from src.platform.observability import observe

@observe("subsystem.my_function", capture_input=True, capture_output=True)
def my_function(self, x):
    return do_work(x)
```

### 5.2 Adding a new LLM call site

Use `tracer.generation(...)` if there is no active trace (rare). Inside an
active trace, use `trace.generation(...)` so the span is parented:

```python
gen = tracer.generation(
    "llm.generate",
    model=model_name,
    input=prompt,
    metadata={"system": provider_name},
)
try:
    completion = call_provider(prompt)
    gen.set_output(completion.text)
    gen.set_token_counts(completion.input_tokens, completion.output_tokens)
    gen.end(status="ok")
except Exception as exc:
    gen.end(status="error", error=exc)
    raise
```

### 5.3 Span naming convention

`<component>.<area>.<verb>` — e.g. `retrieval.rerank.local`,
`ingest.embedding.store`, `guardrails.rail.pii`. Stay under 64 characters,
lowercase, dots-as-separators.

### 5.4 What NOT to do

* **Do not** call `tracer.start_span(...)` directly (it does not exist on
  our backend abstraction). Use `tracer.span(...)`.
* **Do not** thread `parent=` arguments around manually. Use the context
  manager pattern (`with trace: with trace.span: ...`); ambient OTel
  context handles parenting.
* **Do not** write helpers around `set_attribute`. The `Span.set_attribute`
  contract is already minimal. New helpers fragment the surface.
* **Do not** call `set_attribute` on a `Trace` or `Generation`. Those ABCs
  do not expose it. Use `tracer.span(...)` for post-hoc attributes (see
  §6.4).
* **Do not** pass `endpoint=` to `OTLPSpanExporter` directly (§6.1).
* **Do not** import the `langfuse` Python SDK. It is no longer a dep.

---

## 6. Caveats & gotchas

### 6.1 `OTLPSpanExporter(endpoint=...)` kwarg silently breaks Langfuse

The exporter has two paths:

* env var (`OTEL_EXPORTER_OTLP_ENDPOINT`) → `_append_trace_path()` appends
  `/v1/traces` automatically.
* explicit `endpoint=` kwarg → used **verbatim**.

If you pass `endpoint="http://localhost:3000/api/public/otel"` as a kwarg,
the exporter POSTs to that URL (no `/v1/traces` suffix) and Langfuse
returns 404. **Never pass `endpoint=` as a kwarg.** See
`src/platform/observability/otel/backend.py:411`.

### 6.2 Live tests need `source .env`

The integration test modules
(`tests/observability/test_otel_live*.py`) read `LANGFUSE_PUBLIC_KEY` /
`LANGFUSE_SECRET_KEY` / `OTEL_EXPORTER_OTLP_ENDPOINT` from the
environment. Run as:

```bash
set -a; source .env; set +a
uv run --extra dev pytest tests/observability/test_otel_live.py \
                          tests/observability/test_otel_live_e2e.py \
                          -m integration -v
```

The module fixture skips with an actionable message when the stack is
unreachable or keys are absent.

### 6.3 `tests/observability/conftest.py` forces `noop`

Every test in `tests/observability/` runs with
`RAG_OBSERVABILITY_PROVIDER=noop` (autouse fixture) to prevent stray OTLP
exports during the unit suite. The live tests override this in their own
`_reset_otel_global_state` fixture; new unit tests that need real
OTel-shaped output should use the `otel_capture` fixture (`InMemorySpanExporter`)
rather than instantiating `OTelBackend` directly.

### 6.4 `OTelTrace` has no `.end()` method — drive as context manager

`OTelTrace` ends its root span in `__exit__`. There is no public `.end()`.
Always use:

```python
with backend.trace("name") as t:
    ...
# root span is ended here
```

### 6.5 `Generation` ABC has no `set_attribute`

`Generation` exposes only `set_output`, `set_token_counts`, `end`. If you
need an arbitrary attribute on a generation span, wrap the generation in a
`tracer.span(...)` and set the attribute there, or pass it via the
`metadata=` dict at construction time (it will be emitted as a span
attribute by `_apply_gen_ai_attributes`).

### 6.6 `ThreadPoolExecutor.submit` does NOT propagate ambient OTel context

OTel's ambient context is thread-local. `submit`-ed work runs on a worker
thread that sees no parent span, so spans emitted there become orphans
(new trace_id). **Pattern**: emit child spans in the result-collecting
main thread instead — collect raw results from workers, then build the
span tree around `result` once back on the parent thread.

There is an OTel-recommended workaround (`contextvars.copy_context()` +
`executor.submit(ctx.run, fn, ...)`); we have not adopted it because the
"build spans in the collector" pattern reads more clearly.

### 6.7 FastAPI streaming responses (SSE) — current limitation

`BaseHTTPMiddleware` closes the response before the SSE generator
finishes. The `api.<resource>.<verb>` root trace therefore ends *before*
streamed `llm.generate` spans complete. Those spans still share the
trace_id via ambient context, but they appear as orphan-rooted under the
trace in the collector UI.

Mitigations:
* Acceptable for dev. The downstream spans are still queryable by
  trace_id.
* For prod-grade fix: emit the root span inside an ASGI middleware (not
  `BaseHTTPMiddleware`) that defers `__exit__` until `send` emits
  `http.response.body` with `more_body=False`.

### 6.8 `LANGFUSE_INIT_ORG_ID` required for headless bootstrap

Headless first-boot of the Langfuse stack requires
`LANGFUSE_INIT_ORG_ID` in `docker-compose.observability.yml`
(see commit 41588f5). Without it, the Langfuse web UI never finishes
bootstrap and the API returns 401s on `/api/public/traces`.

---

## 7. Operational checks

### 7.1 Bring the stack up

```bash
docker compose --profile observability up -d
# Wait ~10s for health, then:
curl -sf http://localhost:3000/api/public/health
```

### 7.2 Live-test the full path

```bash
set -a; source .env; set +a
uv run --extra dev pytest tests/observability/test_otel_live_e2e.py \
                          tests/observability/test_otel_live.py \
                          -m integration -v
```

Expected: 5 passed. The first run after `up -d` may take an extra
second or two while the Langfuse worker warms.

### 7.3 Sanity check from a Python REPL

```bash
set -a; source .env; set +a
uv run --extra dev python -c "
from src.platform.observability import get_tracer
b = get_tracer()
with b.trace('manual.smoke', metadata={'note': 'hello'}) as t:
    t.span('child').end(status='ok')
b.flush()
print('OK — check Langfuse')
"
```

---

## 8. Reference

| File                                                          | What lives there                                  |
| ------------------------------------------------------------- | ------------------------------------------------- |
| `src/platform/observability/__init__.py`                      | Public API (`get_tracer`, `observe`).             |
| `src/platform/observability/backend.py`                       | ABCs (`ObservabilityBackend`, `Span`, `Trace`, `Generation`) + `start_trace_from_carrier` fallback. |
| `src/platform/observability/otel/backend.py`                  | `OTelBackend` + `OTelSpan` + `OTelTrace` + `OTelGeneration`. All `opentelemetry` imports gated here. |
| `src/platform/observability/otel/auth.py`                     | `ensure_otlp_headers_from_env` (Langfuse Basic auth derivation). |
| `src/platform/observability/noop/backend.py`                  | `NoopBackend` — safe no-ops.                      |
| `server/observability_middleware.py`                          | Starlette middleware — `api.<resource>.<verb>` roots + traceparent extraction. |
| `tests/observability/test_otel_live.py`                       | Live smoke tests (trace/span/generation/error).   |
| `tests/observability/test_otel_live_e2e.py`                   | Live e2e shape tests (RAGChain replay, traceparent round-trip). |
| `docs/observability/`                                         | Deeper design / spec docs.                        |
