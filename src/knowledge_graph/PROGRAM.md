# knowledge_graph — Error Handling Coverage Program

## Objective

Reach 100% error-handling coverage on the 8 identified high-risk hot-path
functions. Baseline: 1/8 (12%) — only `EmbeddingResolver.find_candidates`
is currently guarded. Every unguarded path is a crash risk on the query
hot path or during backend I/O.

Use case: ASIC design fix-policy knowledge graph. A dropped Neo4j connection
or a malformed LLM response during a query must never propagate as an
unhandled exception to the caller.

## Scoring mode

Numerical

## Metric

Score = number of guarded hot-path functions / 8. Higher is better.
A function is "guarded" when it contains at least one `try/except` block
that covers the high-risk operation(s) in that function.
Improvement = strict increase in score. Target = 8/8 (100%).

## Scoring mechanism

```
python3 src/knowledge_graph/score_error_handling.py
```

Run from the RagWeave project root. Prints `SCORE: N/8 (X%)` on success.
Exits with code 1 and prints `GUARD FAILED:` if the package is not importable
(correctness guard — see below).

Baseline verified: 1/8 (12%).

### Correctness guard

The scorer verifies that `src.knowledge_graph`, `src.knowledge_graph.backend`,
and `src.knowledge_graph.common` are importable before reporting any score.
An iteration that breaks imports is recorded as `crash` and reverted,
even if the AST check would have passed.

## Mutable files

Only modify these — everything else is locked:

- `query/expander.py`
- `query/entity_matcher.py`
- `backends/networkx_backend.py`
- `community/summarizer.py`
- `resolution/embedding_resolver.py`
- `community/detector.py`
- `tests/knowledge_graph/` — new or existing test files may be added or
  modified to cover the new error-handling branches

## Immutable files (DO NOT MODIFY)

- `score_error_handling.py` — the scorer; never modify during the loop
- `PROGRAM.md` — this file
- `backend.py` — ABC contract (GraphStorageBackend)
- `common/` — all files (data contracts, types, schemas, utils)
- `extraction/` — all files (out of scope for this loop)
- `backends/neo4j_backend.py` — Neo4j backend (only NetworkX is in scope)
- `resolution/resolver.py`, `resolution/alias_resolver.py`, `resolution/schemas.py`
- `query/sanitizer.py`
- `community/detector.py` (except for `_to_igraph` — see Exploration directions)

## Exploration directions

Work through these groups in order. One group = one logical unit of work,
but may span multiple iterations if the group has several functions:

1. **Query path (A)** — guard `GraphQueryExpander.expand()`,
   `EntityMatcher._match_spacy()`, and `EntityMatcher._llm_match()`.
   These are called on every user query; an unguarded exception here
   kills the entire RAG response. Wrap backend calls and spaCy calls in
   try/except, log the error, and return a safe degraded result (empty
   expansion / empty match list) rather than re-raising.

2. **Backend I/O (B)** — guard `NetworkXBackend.save()` and
   `NetworkXBackend.load()`. Wrap `orjson` serialize/deserialize and
   `Path.write_bytes` / `Path.read_bytes` in try/except with
   OSError/ValueError/orjson.JSONDecodeError coverage. Log and re-raise
   (callers handle the failure, but the backend must log context first).

3. **Inference / model operations (C)** — guard
   `CommunitySummarizer._call_llm()`. Wrap the LLM call in try/except,
   log the failure, and raise a clear application-level exception so the
   per-community retry in `summarize_all()` can catch it cleanly.

4. **Community graph construction (D)** — guard
   `CommunityDetector._to_igraph()`. Wrap igraph operations in try/except
   covering MemoryError and igraph-specific exceptions. Log and re-raise
   so the caller knows community detection failed rather than silently
   getting an empty graph.

## Constraints

- Do NOT swallow exceptions silently. Every `except` clause must either:
  (a) log the error and return a defined safe fallback, or
  (b) log and re-raise (or raise a more specific exception).
- Do NOT change function signatures or return types.
- Error messages must include the relevant context (entity name, source
  file path, model alias, etc.) — not just `str(exc)`.
- Do not add retry logic — that belongs in a separate concern.
- The public API surface (`__init__.py` exports) must remain unchanged.

## Stop conditions

- Score reaches 8/8 (100%) — success, stop
- 2 consecutive iterations with no improvement — stop, log conclusion
  in `research/changelog.md`
- Correctness guard fails 3 times in a row — scorer infrastructure is
  broken, stop and report to user
