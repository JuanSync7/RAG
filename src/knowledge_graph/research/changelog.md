# knowledge_graph — Error Handling Research Changelog

## Run: 2026-04-12 — worktree-ResearchAgent-KG

**Scoring mode:** Numerical
**Starting score:** 1/8 (12%)
**Final score:** 8/8 (100%)
**Iterations:** 7 (7 kept, 0 discarded, 0 crashes)

---

### What improved

- **`GraphQueryExpander.expand()`** (iter 002) — Full body wrapped in `try/except Exception`. Logs at ERROR with query string, returns `[]`. Previously, any backend error (Neo4j connection drop, NetworkX index failure) would propagate as an unhandled exception through every RAG query.

- **`EntityMatcher._match_spacy()`** (iter 003) — Full body wrapped in `try/except Exception`. Logs at ERROR with query string, returns `[]`. Protects against spaCy model errors on malformed ASIC signal name tokens (special characters, unicode).

- **`EntityMatcher._llm_match()`** (iter 004) — Inner `try/except` wraps `provider.json_completion()` and `json.loads()`. Catches `json.JSONDecodeError` specifically, then broad `Exception`. Logs at WARNING (outer `match_with_llm_fallback` provides the final safety net), returns `[]`. Malformed LLM JSON responses no longer crash the query path.

- **`NetworkXBackend.save()`** (iter 005) — `try/except` wraps `orjson.dumps()` + `path.write_bytes()`. Catches `OSError`, `ValueError`, broad `Exception`. Logs at ERROR with path context, re-raises. Disk-full or permission errors now produce a clear log instead of a raw traceback.

- **`NetworkXBackend.load()`** (iter 005) — `try/except` wraps `path.read_bytes()` + `orjson.loads()` + `nx.node_link_graph()`. Catches `OSError`, `orjson.JSONDecodeError`/`ValueError`, broad `Exception`. Logs at ERROR with path context, re-raises. Corrupt on-disk graphs fail with context instead of an opaque JSON error.

- **`CommunitySummarizer._call_llm()`** (iter 006) — `try/except` wraps `self._provider.generate()`. Logs at ERROR with `max_tokens` and `temperature` config values, re-raises as `RuntimeError` with context. Allows the per-community retry loop in `summarize_all()` to catch and continue rather than aborting the full summarization pass.

- **`CommunityDetector._to_igraph()`** (iter 007) — `try/except` wraps the entire method body. `MemoryError` caught at CRITICAL with entity/edge counts (large ASIC graphs can exhaust memory); all other exceptions caught at ERROR with type/message/counts. Both re-raise so the community detection caller knows the conversion failed.

---

### What didn't work

Nothing was discarded. Every iteration produced a strict score improvement. The strategies were well-scoped (one function per iteration, except B which did both backend functions in one pass since they were identical in structure and counted as two scorer targets).

---

### Hypothesis mispredictions

None. All 7 hypotheses predicted the correct score delta before the change was made. The pattern held cleanly: each `try/except` addition guarded exactly one scorer target and the score advanced by the predicted amount.

The only non-obvious outcome was iteration 005 (save + load in one commit): the hypothesis correctly predicted +2/8 by treating two scorer targets as one logical unit. This was valid because both functions are in the same file, same class, and follow the same error pattern.

---

### Remaining gaps

All 8 hot paths are now guarded. No scorer targets remain.

The following non-scorer gaps from the original audit are NOT addressed by this run (out of scope by PROGRAM.md design):

1. **`query/sanitizer.py`** — no logger at all; query normalization transformations are invisible in logs.
2. **`extraction/regex_extractor.py`** — no logger; extraction counts and regex match details are invisible.
3. **Missing timing instrumentation** — `expand()`, `community/detector.py` (Leiden), `resolution/embedding_resolver.py` (embedding computation) have no `time.monotonic()` timing logs.
4. **O(N) alias scan per match in `entity_matcher._match_spacy()`** — should be a pre-built lowercase dict; untouched by this run.
5. **CRITICAL Graph-RAG gaps** — path-based retrieval, edge-type filtering, graph-context injection into LLM prompt — all out of scope; need a separate implementation session.
