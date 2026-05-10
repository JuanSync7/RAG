# knowledge_graph — Observability Research Changelog

## Run: 2026-04-12 — worktree-ResearchAgent-KG / branch autoresearch/observability-2026-04-12

**Scoring mode:** Numerical
**Starting score:** 0/7 (0%)
**Final score:** 7/7 (100%)
**Iterations:** 6 (6 kept, 0 discarded, 0 crashes)

---

### What improved

- **`QuerySanitizer` in `query/sanitizer.py`** (iter 002) — Added `import logging` and
  `logger = logging.getLogger(__name__)` at module level. Added `logger.debug()` call in
  `normalize()` logging the raw and normalized query. Previously invisible to logs: query
  normalization transformations (lowercase, hyphen→space, whitespace collapse) produced no
  output even when they silently dropped ASIC signal name tokens.

- **`RegexEntityExtractor` in `extraction/regex_extractor.py`** (iter 003) — Added
  `import logging` and `logger = logging.getLogger(__name__)` at module level. Added
  `logger.debug()` call in `extract()` logging source path, entity count, and triple count.
  Previously there was zero visibility into extraction yield per document — no way to
  distinguish "correctly extracted 0 entities" from "regex patterns never matched."

- **`GraphQueryExpander.expand()`** (iter 004) — Added `import time`. Inserted
  `_t0 = time.monotonic()` at the start of the try block and a `logger.debug()` call before
  `return result` logging query, effective depth, term count, and elapsed seconds. The
  expand() hot path (called on every RAG query) was previously completely opaque to latency
  profiling — no way to know whether slow queries were caused by the matcher or graph
  traversal.

- **`CommunityDetector.detect()`** (iter 005) — Added `import time`. Inserted
  `_t0 = time.monotonic()` after the `_LEIDEN_AVAILABLE` guard. Replaced the existing
  `logger.info(...)` terminal log with `logger.debug(...)` that includes community count,
  entity count, and elapsed seconds. Leiden detection on large ASIC graphs can be slow;
  timing was entirely missing, making it impossible to attribute latency to graph conversion
  vs. algorithm execution.

- **`EmbeddingResolver.find_candidates()`** (iter 006) — Added `import time`. Inserted
  `_t0 = time.monotonic()` after the `len(all_entities) < 2` early-return guard. Replaced
  `logger.info(...)` with `logger.debug(...)` logging candidate count and elapsed seconds.
  Pairwise embedding similarity on large entity sets is O(N²); the operation was entirely
  untracked.

---

### What didn't work

Nothing was discarded. Every iteration produced a strict score improvement. The strategy was
well-scoped: loggers-first (A), then timing (B), one file or method per iteration.

---

### Hypothesis mispredictions

None. All 6 hypotheses predicted the correct score delta before the change was made:

| Iter | Predicted | Actual |
|------|-----------|--------|
| 002  | +2/7      | +2/7   |
| 003  | +2/7      | +2/7   |
| 004  | +1/7      | +1/7   |
| 005  | +1/7      | +1/7   |
| 006  | +1/7      | +1/7   |

Iteration 004 required a second commit to fix an indentation error in the initial edit.
This is not a hypothesis misprediction — the hypothesis was correct; the implementation had
a mechanical error that was caught and fixed before the final score was recorded.

---

### Remaining gaps (out of scope by PROGRAM.md design)

1. **Alias scan in `entity_matcher._match_spacy()`** — O(N) scan per entity match; should
   be a pre-built lowercase dict. No timing added (out of scope for this run).
2. **`query/sanitizer.py:expand_aliases()`** — O(N×M) alias scan; no timing added.
3. **CRITICAL Graph-RAG gaps** — path-based retrieval (`violation → fixed_by → approach`),
   graph-context injection into LLM prompt, edge-type filtering in neighbor traversal — all
   out of scope; need a separate implementation session.
4. **`community/detector.py:_run_leiden()`** — Leiden algorithm execution time is not
   separately instrumented; detect() timing covers the full method including conversion.
   A future run could add per-phase timing.
