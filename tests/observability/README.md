<!-- @summary
Test suite for the platform observability subsystem, covering the backend ABC,
NoopBackend, LangfuseBackend, the public facade, schema dataclasses, provider
shims, and end-to-end swappable-backend integration.
@end-summary -->

# tests/observability

Tests for `src/platform/observability`. The suite covers every layer of the
observability stack: abstract backend contract enforcement, the no-op and
Langfuse concrete backends, the public `get_tracer()` facade, schema
dataclasses, the deprecated providers shim, and full end-to-end integration
flows that verify backend swapping at runtime.

`conftest.py` resets the module-level backend singleton and forces
`OBSERVABILITY_PROVIDER=noop` around every test to prevent cross-suite
contamination from tests that configure a real backend.

## Contents

| Path | Purpose |
| --- | --- |
| `conftest.py` | Auto-use fixture that resets the backend singleton and `OBSERVABILITY_PROVIDER` before and after each test |
| `test_backend.py` | ABC enforcement, context manager protocol for `Span`/`Generation`/`Trace`, and base-class behavior |
| `test_init.py` | Public `src.platform.observability` facade: `get_tracer()` dispatch, singleton caching, and re-exports |
| `test_integration.py` | End-to-end flows for the swappable observability subsystem (provider switching, full trace/span lifecycle) |
| `test_langfuse_backend.py` | `LangfuseBackend` and its wrapper classes: initialization, tracing calls, and error handling |
| `test_noop_backend.py` | `NoopBackend`, `NoopSpan`, `NoopTrace`, and `NoopGeneration` — verifies all no-op types are safe and return sensible defaults |
| `test_providers.py` | Deprecated `providers` backward-compat shim — verifies re-exports and deprecation warnings |
| `test_schemas.py` | Schema record dataclasses — field defaults, validation, and serialization contracts |
