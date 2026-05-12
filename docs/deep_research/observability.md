# Deep Research — Observability

The Deep Research orchestrator (`src/retrieval/pipeline/deep_research.py`) is
instrumented with the swappable observability subsystem
(`src/platform/observability`). When `OBSERVABILITY_PROVIDER=langfuse` is set,
all spans, generations, and traces emitted below are forwarded to Langfuse.
When the env var is unset (or set to `noop`), the calls are no-ops with zero
runtime cost.

No new env vars are introduced — the orchestrator reuses the existing
`get_tracer()` singleton.

## What gets emitted

### Parent trace: `deep_research`

Created at the top of `DeepResearch.research()`. One per orchestrator run.

Metadata on creation:

- `original_question` — pinned user question
- `processed_query` — query after pre-processing
- `max_topics` — budget cap on depth-1 topic split
- `max_depth` — budget cap on recursion depth

Attributes set during the run:

- `topic_count` — number of topics returned by the first decomposition
  (set after the depth-0→1 split, capped by `max_topics`)

End status:

- `ok` on successful completion
- `error` if `_research_impl` raises (the exception is re-raised)

### Generation spans (one per LLM call)

Emitted by `_call_json` for every sufficiency / decomposition LLM call.

| Field | Value |
| --- | --- |
| `name` | `dr_sufficiency` or `dr_decompose` |
| `model` | `self._model_alias` |
| `input` | rendered user-prompt content |
| `metadata.iteration` | iteration counter at call time |
| `metadata.node_count` | retrieval node counter at call time |
| `metadata.purpose` | `sufficiency` \| `decompose` |
| `output` | provider response content (raw JSON string) |
| `prompt_tokens` / `completion_tokens` | from `LLMResp` |
| end status | `ok` on success, `error` on provider exception |

### Iteration spans

Emitted by `_recurse_topic` once per recursion level entered.

| Field | Value |
| --- | --- |
| `name` | `dr_iteration_depth_{depth}` |
| `attributes.depth` | recursion depth (positive int) |
| `attributes.topic` | owning topic-pool name |
| `attributes.question_count` | number of incoming sub-questions |

## Enabling Langfuse

Set the standard observability env vars (already documented in
`docs/observability/`):

```sh
export OBSERVABILITY_PROVIDER=langfuse
export LANGFUSE_PUBLIC_KEY=...
export LANGFUSE_SECRET_KEY=...
export LANGFUSE_HOST=https://cloud.langfuse.com
```

Then run any flow that exercises the Deep Research orchestrator (CLI, server,
or tests). Each DR run will appear as a single trace with nested generations
and iteration spans.

## Failure-mode contract

All instrumentation is fail-open — an exception inside the tracer never
propagates to the orchestrator. If `get_tracer()` raises during initialization
or if any `trace.generation()` / `span.end()` call fails, the orchestrator
proceeds as if no backend were configured. Existing Prometheus metrics
(`DR_RUNS_TOTAL` etc.) are unaffected by tracing state.

## Tests

- `tests/retrieval/test_dr_tracing.py` — 8 tests covering trace creation,
  generation naming, attributes, error propagation, and noop fallback.
- `tests/retrieval/test_dr_metrics.py` — 5 tests; unchanged by the tracing
  layer (verifies the Prometheus instrumentation still works alongside the
  spans).

## Temporal search attributes

Deep Research runs are indexed in Temporal via custom search attributes so
operators can query DR-specific properties from the Temporal UI / `tctl`.

| Attribute | Type | Meaning |
| --- | --- | --- |
| `DREnabled` | Bool | True when the request had `deep_research=True`. |
| `DRIterations` | Int | Sufficiency-check iterations the orchestrator drove. Round 0 root retrieval counts as 1. |
| `DRLLMCalls` | Int | LLM calls (sufficiency + decomposition) consumed by the run. |
| `DREarlyStopped` | Bool | True if the orchestrator never ran a multi-topic decomposition (sufficient on round 0, sanitizer reject, or early budget exhaust). |
| `DRTopicCount` | Int | Number of topic pools in the result (>=1 always; >1 indicates fan-out). |

Definitions live in [`server/search_attributes.py`](../../server/search_attributes.py).
The workflow upserts them after the activity returns (see
`server/workflows.py::RAGQueryWorkflow.run`). The values come from
`response.metadata["deep_research"]`, populated by `RAGChain.run`.

### One-time registration

Custom search attributes must be declared to the cluster before workflows can
upsert them. Run the admin script once per cluster / namespace:

```bash
# Defaults to localhost:7233 and namespace "default".
python -m ops.temporal.register_search_attributes

# Or with overrides:
RAG_TEMPORAL_TARGET_HOST=temporal.prod:7233 \
  RAG_TEMPORAL_NAMESPACE=ragweave \
  python -m ops.temporal.register_search_attributes
```

The script is idempotent — re-running it after a successful registration
treats the "already exists" error as success.

### Querying DR runs

Once registered, query in the Temporal UI's "Workflows" filter, or via `tctl`:

```bash
# Find runs that looped a lot.
tctl workflow list --query 'WorkflowType="RAGQueryWorkflow" AND DRIterations > 5'

# Find DR runs that early-stopped (sanitizer reject or sufficient on round 0).
tctl workflow list --query 'DREnabled = true AND DREarlyStopped = true'

# Multi-topic fan-outs only.
tctl workflow list --query 'DRTopicCount > 1'
```
