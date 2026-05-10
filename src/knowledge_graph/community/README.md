<!-- @summary
Phase 2 community detection (Leiden algorithm) and LLM-based community summarization.
@end-summary -->

# community/ — Community Detection and Summarization (Phase 2)

Implements graph community analysis using the Leiden algorithm and LLM-powered summarization.

## Files

| File | Purpose |
|------|---------|
| `schemas.py` | `CommunitySummary` and `CommunityDiff` dataclasses |
| `detector.py` | `CommunityDetector` — Leiden via igraph+leidenalg, sidecar JSON persistence |
| `summarizer.py` | `CommunitySummarizer` — parallel LLM summarization with token budgets |

## Key Concepts

- **Leiden resolution**: `community_resolution` (default 1.0) controls cluster granularity
- **Min size filtering**: Communities smaller than `community_min_size` (default 3) merge to bucket -1
- **Sidecar persistence**: Detection results saved to `<graph_path>.communities.json`
- **Incremental refresh**: `summarizer.refresh()` re-summarizes only new/changed communities
- **Lifecycle**: `detector.is_ready` is True only after `detect()` + summaries are set

## Dependencies

Requires `igraph` and `leidenalg` packages. When unavailable, `detect()` raises `ImportError`
and `is_ready` stays False — community features degrade gracefully.
