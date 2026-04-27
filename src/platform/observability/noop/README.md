<!-- @summary
No-op observability backend adapter. Implements the ObservabilityBackend interface with zero-cost stubs that accept any input and return typed no-op objects without performing any I/O. Used as the default backend when no provider is configured and as the fallback when a real backend fails to initialize.
@end-summary -->

# noop

Concrete no-op implementation of the observability backend interface. All methods return typed stub objects (`NoopSpan`, `NoopTrace`, `NoopGeneration`) immediately with zero I/O. Active when `OBSERVABILITY_PROVIDER` is unset or set to `"noop"`, and also serves as the safe fallback when the configured backend (e.g., Langfuse) fails to initialize.

## Contents

| Path | Purpose |
| --- | --- |
| `__init__.py` | Package entry point; exports `NoopBackend`, `NoopSpan`, `NoopTrace`, `NoopGeneration` |
| `backend.py` | No-op backend implementation (`NoopBackend`, `NoopTrace`, `NoopSpan`, `NoopGeneration`) |
