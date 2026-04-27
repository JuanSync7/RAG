<!-- @summary
NeMo Guardrails backend implementation. Wires the NeMo runtime to the shared ML rails, runs input and output rails in parallel via executor classes, and manages the NeMo LLMRails singleton lifecycle.
@end-summary -->

# nemo_guardrails

This package provides the NeMo Guardrails backend — the concrete implementation of the `GuardrailBackend` interface that connects NeMo's LLM-based runtime to the backend-agnostic rails defined in `src/guardrails/shared/`.

`NemoBackend` reads all `RAG_NEMO_*` configuration at construction, initialises the `GuardrailsRuntime` singleton once, and injects it into each rail. The executor classes run rails concurrently via `ThreadPoolExecutor` with per-rail timeouts and configurable fail-open / fail-closed behavior.

## Files

| File | Purpose |
| --- | --- |
| `__init__.py` | Package init; re-exports `NemoBackend`, `InputRailExecutor`, `OutputRailExecutor` |
| `backend.py` | `NemoBackend` — `GuardrailBackend` implementation; reads config and wires all rails |
| `executor.py` | `InputRailExecutor` and `OutputRailExecutor` — parallel rail execution with consensus gate and Prometheus metrics |
| `runtime.py` | `GuardrailsRuntime` — process-wide singleton managing the NeMo `LLMRails` lifecycle (lazy-import, fail-open, async generation, action registration) |
