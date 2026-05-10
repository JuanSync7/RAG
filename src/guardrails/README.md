<!-- @summary
Swappable guardrails subsystem for input/output safety: config-driven backend dispatcher,
GuardrailBackend ABC, NeMo backend, and backend-agnostic ML rails (intent, injection, PII,
toxicity, topic safety, faithfulness). Rails run in parallel with per-rail timeouts and a
consensus merge gate. Backend is selected via GUARDRAIL_BACKEND config key.
@end-summary -->

# src/guardrails

## Overview

Swappable guardrails subsystem for RAG query safety. The backend is selected at config level
via `GUARDRAIL_BACKEND` — swapping backends requires zero changes to retrieval code.

**Input rails** (run on user query before retrieval): intent, injection, PII, toxicity, topic safety.
**Output rails** (run on generated answer before delivery): faithfulness, PII, toxicity.

## Files

| File | Purpose | Key Exports |
| --- | --- | --- |
| `__init__.py` | Public API + config-driven backend dispatcher | `run_input_rails`, `run_output_rails`, `redact_pii`, `register_rag_chain`, `RailMergeGate` |
| `backend.py` | `GuardrailBackend` ABC — formal backend contract | `GuardrailBackend` |

## Subdirectories

| Directory | Purpose |
| --- | --- |
| `common/` | Shared typed contracts (`RailVerdict`, `InputRailResult`, `OutputRailResult`, `GuardrailsMetadata`, `RailMergeGate`) |
| `shared/` | Backend-agnostic ML rail modules (PII, injection, toxicity, intent, topic safety, faithfulness) |
| `nemo_guardrails/` | NeMo Guardrails backend: `NemoBackend`, `GuardrailsRuntime`, `InputRailExecutor`, `OutputRailExecutor` |

## Configuration

Select backend via `GUARDRAIL_BACKEND` environment variable (default: `"nemo"`):

| Value | Behavior |
| --- | --- |
| `"nemo"` | NeMo Guardrails backend (default) |
| `""` or `"none"` | Disabled — all rail calls are no-ops |

Optional ML dependencies:
- `presidio`, `spacy` — required for PII detection (`shared/pii.py`)
- `gliner` — required for GLiNER supplementary PII layer (`shared/gliner_pii.py`)

## Architecture

```
rag_chain.py
    │
    ├─ redact_pii(query)                    # pre-LLM PII gate (synchronous)
    ├─ run_input_rails(query, tenant_id)    # parallel: intent, injection, PII, toxicity, topic
    │       └─ GuardrailBackend [dispatched by GUARDRAIL_BACKEND]
    │               └─ NemoBackend → nemo_guardrails/executor.py → shared/ rails
    │
    ├─ RailMergeGate.merge(query_result, rail_result)  # routing: reject / canned / search
    │
    └─ run_output_rails(answer, context_chunks)        # parallel: faithfulness, PII, toxicity
            └─ GuardrailBackend [dispatched by GUARDRAIL_BACKEND]
```

## Adding a New Backend

1. Create `src/guardrails/<backend_name>/backend.py` with a class that subclasses `GuardrailBackend`
2. Implement `run_input_rails`, `run_output_rails`, `redact_pii`
3. Add an `elif GUARDRAIL_BACKEND == "<backend_name>":` branch to `_get_backend()` in `__init__.py`
4. Set `GUARDRAIL_BACKEND=<backend_name>` — no other changes needed

See `docs/guardrails/` for full design and engineering docs.
