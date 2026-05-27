---
slice_id: P0-collection-selection
validable_outcome: |
  `uv run pytest tests/vector_db/test_collection_selection.py -q` reports all tests green, including:
    1. `test_isolation_writes_do_not_leak` (slow + integration): ingests one chunk into collection A and zero into B via programmatic collection_name override; asserts A has chunks, B has zero.
    2. `test_rag_chain_accepts_collection_name`: constructs `RAGChain(collection_name="custom_foo")` and verifies the resolved collection is `custom_foo`, not the env default.
    3. `test_ingest_cli_accepts_collection_flag`: parser accepts `--collection custom_foo` and the parsed arg surfaces as `args.collection == "custom_foo"`.
    4. `test_collection_exists_helper`: `collection_exists(client, name)` returns True for an existing collection and False for a non-existent one.
  PLUS: `uv run pytest tests/vector_db/ tests/retrieval/ -q -m "not slow and not integration"` stays fully green (no regressions in default-path tests).
touches:
  - src/vector_db/__init__.py
  - src/vector_db/backend.py
  - src/vector_db/weaviate/backend.py
  - src/vector_db/weaviate/store.py
  - src/retrieval/pipeline/rag_chain.py
  - src/ingest/cli.py
  - src/ingest/impl.py
  - config/settings.py
  - tests/vector_db/test_collection_selection.py
depends_on: []
---

# P0 — Collection-selection plumbing

## End-state (validable, machine-checkable)

A caller can programmatically route reads and writes to an arbitrary Weaviate collection without env-var manipulation, with full isolation between collections proven by an E2E test. Default behavior (no override) is unchanged.

## Current state (surveyed before slice opens)

Already plumbed (do NOT redo):
- `config/settings.py:48` — `VECTOR_COLLECTION_DEFAULT` is env-overridable via `RAG_VECTOR_COLLECTION_DEFAULT`, defaulting to `WEAVIATE_COLLECTION_NAME = "RAGDocuments"`.
- `src/vector_db/weaviate/store.py` — most read/write/ensure functions already accept `collection: str = WEAVIATE_COLLECTION_NAME`.
- `src/vector_db/weaviate/backend.py` — `WeaviateBackend` already exposes `ensure_collection`, `delete_collection`, `list_collections`.
- `src/vector_db/__init__.py` — re-exports admin functions.

Real gaps (this slice closes):
1. **`RAGChain.__init__`** does not accept `collection_name`. It implicitly uses `VECTOR_COLLECTION_DEFAULT` via internal calls. → Add `collection_name: Optional[str] = None` parameter; default falls back to `VECTOR_COLLECTION_DEFAULT`. Thread it through every internal retrieval/admin call inside the class.
2. **Ingest CLI** has no `--collection` flag. → Add `--collection`/`-c` arg to `src/ingest/cli.py` parser; thread through to `src/ingest/impl.py` so the chosen collection is honored.
3. **`collection_exists(client, name) -> bool`** helper does not exist on the admin surface. → Add to `WeaviateBackend.collection_exists` + re-export from `src/vector_db/__init__.py`. Implementation: thin wrapper around `client.collections.exists(name)`.
4. **E2E isolation test** does not exist. → Author `tests/vector_db/test_collection_selection.py` with the four tests enumerated in `validable_outcome`.

## Constraints

- **Do NOT introduce a new env var name** (`RAG_COLLECTION_NAME` etc.). Keep using the existing `RAG_VECTOR_COLLECTION_DEFAULT`. The plan doc has been corrected.
- **Default behavior unchanged.** `RAGChain()` with no args, `python -m src.ingest.cli <doc>` with no `--collection` flag, must both route to `VECTOR_COLLECTION_DEFAULT` exactly as today. Regression sweep gates merge.
- **Test gating.** Live-Weaviate tests must carry **both** markers per project memory `feedback_dual_marker_gating`: `pytestmark = [pytest.mark.slow, pytest.mark.integration]`. Single-marker gating slips through `-m "not slow"`.
- **Live-Weaviate test isolation.** The E2E test must use unique collection names (UUID-suffixed) and clean up both collections on teardown — pass or fail — so re-runs and concurrent runs are safe.
- **Boundary discipline.** Modifications confined to `touches`. No drive-by refactors.

## Anti-gaming guards

- The E2E isolation test must actually exercise the ingest write path (chunks land in Weaviate) and a real read path back (count via Weaviate client). Not a mocked test that pretends collection_name was honored.
- The `test_rag_chain_accepts_collection_name` test must observe the resolved collection through an existing public-ish attribute or a read-only accessor — not by mock-patching the resolution logic the slice just implemented.
- Red-reason proof: before any implementation, run the new test file. Failure output must include either `AttributeError` / `TypeError` on the new `collection_name` kwarg or `pytest: error: unrecognized arguments: --collection` — NOT `ImportError` from a missing test fixture, NOT `ModuleNotFoundError`. Capture the failure output in the closeout.

## Out of scope (defer)

- `RAG_EVAL_COLLECTION_NAME` env override (lives in a later eval-pack slice).
- Auto-GC of `ragweave_test_*` collections (separate slice, ops concern).
- Collection schema migration across renames (existing schema_migrations module is unchanged here).
