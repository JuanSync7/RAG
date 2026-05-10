# knowledge_graph — Observability Coverage Program

## Objective

Reach 100% observability coverage on 7 identified targets across 5 files.
Baseline: 0/7 (0%) — `query/sanitizer.py` and `extraction/regex_extractor.py`
have no logger at all; `query/expander.py:expand()`,
`community/detector.py:detect()`, and
`resolution/embedding_resolver.py:find_candidates()` have no timing
instrumentation.

Use case: ASIC design fix-policy knowledge graph. Visibility into query
normalization transformations, extraction counts, and per-operation latency
is needed to diagnose slow or incorrect RAG responses without adding
production overhead (all timing at DEBUG level).

## Scoring mode

Numerical

## Metric

Score = number of passing observability targets / 7. Higher is better.
A target passes when the AST check for that specific file and construct
succeeds (see scorer for exact criteria).
Improvement = strict increase in score. Target = 7/7 (100%).

## Scoring mechanism

```
python3 src/knowledge_graph/observability/score_observability.py
```

Run from the RagWeave project root. Prints `SCORE: N/7 (X%)` on success.
Exits with code 1 and prints `GUARD FAILED:` if the package is not importable
(correctness guard — see below).

Baseline verified: 0/7 (0%).

### Correctness guard

The scorer verifies that `src.knowledge_graph`, `src.knowledge_graph.backend`,
and `src.knowledge_graph.common` are importable before reporting any score.
An iteration that breaks imports is recorded as `crash` and reverted,
even if the AST check would have passed.

## Mutable files

Only modify these — everything else is locked:

- `query/sanitizer.py`
- `extraction/regex_extractor.py`
- `query/expander.py` (only `GraphQueryExpander.expand()`)
- `community/detector.py` (only `CommunityDetector.detect()` — do NOT touch `_to_igraph()` or any other method)
- `resolution/embedding_resolver.py` (only `EmbeddingResolver.find_candidates()`)
- `tests/knowledge_graph/` — new or existing test files may be added or
  modified to cover the new observability additions

## Immutable files (DO NOT MODIFY)

- `observability/score_observability.py` — the scorer; never modify during the loop
- `observability/PROGRAM.md` — this file
- `backend.py` — ABC contract
- `common/` — all files
- `extraction/` — all files except `extraction/regex_extractor.py`
- `backends/` — all files
- `community/summarizer.py`
- `community/detector.py` — except `detect()` method only
- `resolution/resolver.py`, `resolution/alias_resolver.py`, `resolution/schemas.py`
- `query/sanitizer.py` — immutable only if already passing T1+T2 (do not re-edit once done)
- `query/entity_matcher.py`
- `score_error_handling.py`
- `PROGRAM.md` (root-level error handling program)

## Exploration directions

Work through these groups in order. One group = one logical unit of work:

1. **Logger group (A)** — Add `import logging` and `logger = logging.getLogger(__name__)` to
   `query/sanitizer.py` and `extraction/regex_extractor.py`. Then add at least one
   `logger.debug(...)` call in a key method in each file.
   - `sanitizer.py`: log in `normalize()` or `expand_aliases()` — emit the
     normalized result or the number of aliases appended.
   - `regex_extractor.py`: log in `extract()` — emit entity and triple counts.
   This covers T1, T2, T3, T4 (one file per iteration, or both in one if identical
   structure makes it a single logical unit).

2. **Timing group (B)** — Add `import time` (if not already imported) and
   `time.monotonic()` timing to each of the three methods. Log elapsed time at
   DEBUG with operation context. One method per iteration.
   - `query/expander.py:GraphQueryExpander.expand()` — measure full method duration
     including matcher + graph traversal. Log at DEBUG with query and depth.
   - `community/detector.py:CommunityDetector.detect()` — measure from igraph
     conversion through Leiden and assignment. Log at DEBUG with entity and
     community counts.
   - `resolution/embedding_resolver.py:EmbeddingResolver.find_candidates()` —
     measure from entity load through candidate scan. Log at DEBUG with candidate count.

## Constraints

- Log timing at DEBUG level only — not INFO. Timing logs must never appear in
  production output at default log level.
- Log calls must include operation context (entity name, query string, counts)
  — not just elapsed time alone.
- Do NOT change function signatures or return types.
- Do NOT add error handling — that was covered in the previous run. These are
  purely additive logging/timing changes.
- Do NOT add `import time` if it is already imported.
- The public API surface (`__init__.py` exports) must remain unchanged.

## Stop conditions

- Score reaches 7/7 (100%) — success, stop
- 2 consecutive iterations with no improvement — stop, log conclusion in
  `research/changelog.md`
- Correctness guard fails 3 times in a row — scorer infrastructure is
  broken, stop and report to user
