<!-- @summary
Smoke tests for the RagWeave Docker stack: nginx tier-routing behavior and
stack cold-start budget. Gated behind the `smoke` pytest marker; not part of
default CI because they require a running Docker daemon.
@end-summary -->

# tests/smoke

End-to-end smoke tests requiring a working Docker daemon. Not run on every
PR — too slow, too environment-dependent. Run nightly or before merging
ops/infra changes.

## Files

| File | What it asserts |
| --- | --- |
| `test_nginx_tier_routing.py` | nginx fallback semantics: GPU 5xx → CPU fallback; ingest-pinned 5xx fails fast (does NOT retry the same CPU pool). Regression guard for PR 38 review issue 1. |
| `test_stack_cold_start.py` | rag-nginx reaches healthy state within 90s. Regression guard for PR 38 review issue 2 (CPU pool's `service_healthy` dep gating the whole stack). |
| `conftest.py` | Stub HTTP server fixture so nginx tests don't need the real TEI image. |

## Running

```bash
# All smoke tests:
uv run pytest -m smoke

# Just the nginx routing (fast, ~5s):
uv run pytest -m smoke tests/smoke/test_nginx_tier_routing.py

# Cold-start (slow, requires populated .tei_cache):
uv run pytest -m smoke tests/smoke/test_stack_cold_start.py
```

The cold-start test skips when `.tei_cache/` is empty so first-run BGE-M3
download time doesn't pollute the budget. Populate by running
`docker compose up -d rag-embed` once before the test.
