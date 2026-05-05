<!-- @summary
Guardian classifier model implementations. One guardian instance is shared
across rails (toxicity, injection, topic_safety, faithfulness) so a single
backend choice (Granite Guardian, self-check, etc.) propagates everywhere.
@end-summary -->

# models

Backend-agnostic **judge model** wrappers used by guardrail rails. A
"guardian" answers a single question about text — *is it safe along risk
dimension X?* — and returns a calibrated probability.

This package solves a different problem from `shared/`: rails *orchestrate*
the safety decision; guardians *make* it. One guardian instance is reused
across rails so swapping providers is a one-line config change.

## Files

| File | Purpose |
| --- | --- |
| `base.py` | `GuardianModel` ABC, `GuardianRisk` enum, `GuardianVerdict`, `GuardianUnavailable` |
| `granite_guardian.py` | IBM Granite Guardian — `transformers` (local) and `vllm` (HTTP) modes |
| `self_check.py` | Legacy yes/no LLM prompt via `call_oneshot`, wrapped behind `GuardianModel` |
| `__init__.py` | `build_guardian()` factory driven by `RAG_GUARDIAN_*` settings |

## Configuration

| Setting | Default | Notes |
| --- | --- | --- |
| `RAG_GUARDIAN_ENABLED` | `false` | Master switch. When false, rails skip the guardian and use deterministic floors. |
| `RAG_GUARDIAN_PROVIDER` | `granite` | `granite` or `self_check`. |
| `RAG_GUARDIAN_MODE` | `vllm` | `vllm` (HTTP) or `transformers` (local HF). |
| `RAG_GUARDIAN_MODEL_ID` | `ibm-granite/granite-guardian-3.2-5b` | HF id or served model name. |
| `RAG_GUARDIAN_ENDPOINT` | — | Required for `vllm` mode. Example: `http://granite-guardian:8000/v1`. |
| `RAG_GUARDIAN_API_KEY` | — | Optional bearer token. |
| `RAG_GUARDIAN_TIMEOUT_S` | `5.0` | Per-request HTTP timeout. |
| `RAG_GUARDIAN_THRESHOLD` | `0.5` | Unsafe-probability cutoff. |

## Adding a new guardian

1. Create `src/guardrails/models/<provider>.py` with a class subclassing
   `GuardianModel`. Declare `name` and `supported_risks`.
2. Implement `classify(...)` returning a `GuardianVerdict`. Raise
   `GuardianUnavailable` for transient failures so rails can fall back.
3. Add a branch in `build_guardian()` keyed on `RAG_GUARDIAN_PROVIDER`.
4. Map the provider's risk vocabulary in your module (see
   `GRANITE_RISK_MAP` for the pattern).

## Risk dimensions

The `GuardianRisk` enum is provider-agnostic. Not every guardian supports
every risk — rails check `guardian.supports(risk)` before calling.

| Risk | Used by | Granite | Self-check |
| --- | --- | --- | --- |
| `HARM` | toxicity | yes | yes |
| `JAILBREAK` | injection | yes | — |
| `VIOLENCE` / `SEXUAL` / `HATE` / `PROFANITY` | toxicity (subcategories) | yes | — |
| `PII` | pii | — | — |
| `GROUNDEDNESS` | faithfulness | yes | yes |
| `FUNCTION_CALL` | (output, future) | yes | — |

## Concurrency / latency notes

Granite Guardian is a 5B model — local inference is ~200–800ms on a single
GPU. The `transformers` backend serializes inference behind a module-level
lock so concurrent rails don't race the model. For production prefer
`vllm` mode: the server batches across rails and tenants. Rail-level
timeouts (`RAG_NEMO_RAIL_TIMEOUT_SECONDS`) still apply.

## Testing

`tests/guardrails/test_models.py` covers the contract and the vLLM HTTP
path with mocked responses. The `transformers` path is reserved for
integration tests that bring up a real vLLM/HF instance.
