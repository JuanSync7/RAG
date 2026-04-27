<!-- @summary
Test suite for the guardrails subsystem, covering the backend ABC contract,
Colang 2.0 action wrappers, flow registration, syntax validation, rail-class
wrappers, end-to-end pipeline behavior, and the dispatcher facade.
@end-summary -->

# tests/guardrails

Tests for `src/guardrails` and the companion `config/guardrails` Colang 2.0
configuration. The suite covers every layer: the abstract backend contract,
individual Colang action implementations, `.co` file syntax and flow parsing,
higher-level rail wrappers, full end-to-end pipeline scenarios, and the
dispatcher that routes calls to the configured backend.

`conftest.py` removes broken `langchain_core` ghost modules injected by the
langsmith pytest plugin so that NeMo Guardrails can import cleanly.

## Contents

| Path | Purpose |
| --- | --- |
| `conftest.py` | Module-level cleanup that removes broken `langchain_core` ghost modules so NeMo Guardrails can import cleanly |
| `test_backend_abc.py` | Verifies `GuardrailBackend` ABC enforcement and correct return types for all abstract methods |
| `test_colang_actions.py` | Unit tests for every Colang action in `config/guardrails/actions.py` (length checks, citation checks, PII, jailbreak escalation, abuse rate-limiting, etc.) |
| `test_colang_e2e.py` | End-to-end and regression tests for the full `generate_async()` pipeline; includes NeMo-disabled fallback behavior |
| `test_colang_flows.py` | Integration tests verifying all `.co` files parse and register their flows correctly |
| `test_colang_rail_wrappers.py` | Tests that rail-class-wrapping actions delegate correctly to the underlying action functions |
| `test_colang_syntax.py` | Validates Colang 2.0 syntax of all `.co` configuration files against the installed nemoguardrails parser |
| `test_dispatcher.py` | Tests the `src.guardrails` public dispatcher facade: NoOp backend passthrough, unknown-backend error, and `RailMergeGate` re-export |
