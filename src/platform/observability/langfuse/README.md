<!-- @summary
Langfuse v3 observability backend adapter. Implements the ObservabilityBackend interface using the Langfuse SDK, with all SDK imports confined to backend.py. All SDK calls are fail-open — exceptions are caught and logged, never propagated to callers.
@end-summary -->

# langfuse

Concrete implementation of the observability backend interface backed by Langfuse v3. The package exports `LangfuseBackend`, which connects to Langfuse via the SDK's `get_client()` singleton and reads credentials from environment variables. All SDK calls are fail-open: on error, methods return noop objects rather than raising, ensuring observability failures never affect pipeline execution.

Consumer code must not import from this package directly — use the public API at `src.platform.observability`.

## Contents

| Path | Purpose |
| --- | --- |
| `__init__.py` | Package entry point; exports `LangfuseBackend` |
| `backend.py` | Langfuse v3 backend implementation (`LangfuseBackend`, `LangfuseTrace`, `LangfuseSpan`, `LangfuseGeneration`) |
