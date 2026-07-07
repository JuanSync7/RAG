<!-- @summary
A/B eval harness comparing baseline retrieval vs deep_research mode on golden
multi-aspect queries. Measures recall@k, topic coverage, latency, and LLM-call cost.
@end-summary -->

# Deep Research Eval Suite

Compares `RAGChain.run(deep_research=False)` vs `RAGChain.run(deep_research=True)`
on a golden query set. Designed to answer two questions:

1. **Does deep research actually retrieve better evidence on multi-aspect queries?**
2. **Does it cost more without helping on coherent single-aspect queries?**

## Layout

```
deep_research/
├── README.md                      # this file
├── __init__.py
├── conftest.py                    # fixture loader + skip-if-unpopulated
├── harness.py                     # runs both modes, computes metrics
├── test_compare.py                # pytest entry (marker: eval_deep_research)
└── fixtures/
    └── asic/
        └── golden_queries.json    # starter set (TODO: populate expected chunks)
```

## Golden Query Schema

```json
{
  "version": 1,
  "domain": "asic",
  "queries": [
    {
      "id": "dr-001",
      "query": "Compare verification methodology and lint flow for the GPIO block",
      "category": "disjoint",
      "expected_topics": ["verification", "lint"],
      "expected_chunks": [
        {"source": "verif_guide.md", "heading_contains": "GPIO testbench"},
        {"source": "lint_flow.md",   "heading_contains": "GPIO lint rules"}
      ],
      "notes": "Multi-topic compound query — deep research should fan out."
    }
  ]
}
```

`category` is one of:
- `multi_aspect` — coherent question with several facets (deep research should help recall).
- `single_aspect` — focused question (deep research should match baseline, not hurt it).
- `disjoint` — compound query with unrelated topics (deep research should split into pools).

## Metrics

Per-query, the harness records for each mode:

| Metric | Description |
| --- | --- |
| `recall_at_k` | Fraction of `expected_chunks` matched in top-k results |
| `topic_coverage` | Fraction of `expected_topics` with ≥1 matching chunk |
| `latency_ms` | Wall-clock time end-to-end |
| `llm_calls` | LLM invocations (deep research only — from response metadata) |
| `chunk_count` | Number of returned chunks |

Aggregate report compares baseline vs deep_research across the suite.

## Running

```bash
# Skipped by default — eval marker keeps it out of normal CI runs.
pytest evals/retrieval/deep_research/ -m eval_deep_research -v

# Or directly:
python -m evals.retrieval.deep_research.harness \
    --fixtures evals/retrieval/deep_research/fixtures/asic/golden_queries.json \
    --output /tmp/dr_eval_report.json
```

The harness expects a fully initialized `RAGChain` (Weaviate populated, models loaded).
If init fails (e.g., no corpus), tests skip rather than fail.

## Corpus

The starter golden set targets the **OpenTitan** corpus (`opentitan_data/` at repo root).
Ingest it first so the indexed chunks have stable `source` paths the harness can match against:

```bash
python -m ingest --source opentitan_data --tenant asic
```

The matchers expect chunks whose `source` field carries the repo-relative path
(e.g. `doc/contributing/dv/README.md`, `hw/lint/README.md`). If your ingest pipeline
strips path prefixes and stores only the filename, shrink the `source` matchers to
bare filenames in `golden_queries.json`.

## Status

**Starter set populated** with 4 OpenTitan-grounded queries:
- 2 disjoint (verification vs lint, lint tools vs dvsim)
- 1 multi-aspect (DV coverage closure)
- 1 single-aspect (GPIO IP checklist)

Verify the matchers actually hit your indexed chunks before treating recall numbers
as signal — run the harness once with `--top-k 20` to see what's coming back.
