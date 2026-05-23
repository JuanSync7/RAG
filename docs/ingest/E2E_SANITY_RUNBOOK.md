<!-- @summary
End-to-end sanity runbook for the table-aware retrieval pipeline.
Step-by-step compose + venv workflow that proves table-group expansion
and decoded page_bbox citations work against a live stack.
@end-summary -->

# E2E sanity runbook — table-aware retrieval

This runbook proves the merged table-aware pipeline works end-to-end
against a **live** stack (Weaviate + TEI + LLM). It is the
retrieval-side counterpart to `scripts/soak_table_chunking.py`
(ingest-only).

The script under test is
[`scripts/e2e_sanity_table_ingest.py`](../../scripts/e2e_sanity_table_ingest.py).
Its assertion helpers are unit-tested in
`tests/integration/test_e2e_sanity_runbook.py` — that test does NOT
touch live infra and is safe in CI.

## What this verifies

| # | Gate | What it proves |
|---|------|----------------|
| 1 | Ingest | Docling → chunker → embedder → Weaviate succeeds; ≥1 chunk stored. |
| 2 | Retrieval | The query returns ≥1 hit from the freshly populated collection. |
| 3 | Expansion | ≥1 hit carries `metadata.expanded_from` — proves `expand_table_group_hits` fired (Stage 5.35 in `rag_chain.py`). |
| 4 | Citation bbox | ≥1 source_ref has `page_no:int` + `page_bbox:{l,t,r,b}` — proves the DEFECT-2 JSON-decode path is live. |

## Prerequisites

- A built RagWeave worktree (this repo).
- Docker compose stack OR equivalent services. See memory
  [`project_dev_against_prod_stack.md`](#) for the recommended
  compose-infra + venv-worker pattern.
- The ESP32-S3 datasheet at
  `data/datasheets/esp32-s3_datasheet.pdf`. If absent:

  ```bash
  mkdir -p data/datasheets
  curl -L -o data/datasheets/esp32-s3_datasheet.pdf \
      https://www.espressif.com/sites/default/files/documentation/esp32-s3_datasheet_en.pdf
  ```

## Step-by-step

### 1. Bring up the stack

```bash
# From repo root.
scripts/compose.sh up -d weaviate tei llm
# Wait for healthchecks (compose.sh polls).
```

Then run the Weaviate schema migration (idempotent — safe to re-run):

```bash
uv run python scripts/migrate_weaviate_table_schema.py
```

### 2. Source the env

```bash
source .env   # mandatory — picks up WEAVIATE_URL, TEI_URL, LLM_*
```

`RAG_TABLE_EXPANSION_ENABLED` defaults to **on** post-merge; you do
not need to set it explicitly. To force-disable for an A/B run:

```bash
export RAG_TABLE_EXPANSION_ENABLED=false
```

### 3. Run the sanity script

```bash
uv run python scripts/e2e_sanity_table_ingest.py \
    --pdf data/datasheets/esp32-s3_datasheet.pdf \
    --collection e2e_sanity_table_aware \
    --query "What is the operating voltage range of the ESP32-S3?" \
    --report docs/soak/e2e_sanity_$(date +%F).md
```

The script is **idempotent** — the collection name is stable, so
re-runs update in place via `chunk_id` idempotency (see memory
[`project_chunk_id_idempotency.md`](#)). No teardown step is needed
between runs; pass `--keep-collection` (default) if you intend to
inspect Weaviate after.

## Expected output

A successful run prints (and optionally writes) markdown like:

```
# E2E sanity — table-aware retrieval
- pdf: `data/datasheets/esp32-s3_datasheet.pdf`
- collection: `e2e_sanity_table_aware`
- query: `What is the operating voltage range of the ESP32-S3?`
- overall: **PASS**

## Gates

| gate | status | detail |
|---|---|---|
| ingest | PASS | stored_chunks=<N> |
| retrieval | PASS | hits=<M> |
| expansion (expanded_from) | PASS | expanded=<K> |
| citation (page_bbox decoded) | PASS | with_bbox=<J> |
```

Exit code is `0` on full PASS, `1` if any gate fails, `2` on env error
(PDF missing, etc.).

## Failure-mode debugging

| Failure | Likely cause | Fix |
|---------|--------------|-----|
| `ingest produced 0 stored chunks` | Docling models not downloaded, or PDF unreadable. | `uv run python scripts/warmup_docling_models.py`; re-check `--pdf` path. |
| `retrieval returned 0 hits` | Collection name mismatch — script ingested into `VECTOR_COLLECTION_DEFAULT` but queried elsewhere, OR embedder failed silently. | Confirm `WEAVIATE_URL` is reachable; check ingest logs for embedding errors. |
| `expansion gate FAIL: … none carry metadata.expanded_from` | (a) query missed all table chunks, (b) `RAG_TABLE_EXPANSION_ENABLED=false`, (c) `RAG_TABLE_EXPANSION_MAX_GROUPS=0`. | Try a more table-specific query; verify env vars; check the chunk_types list the helper prints. |
| `citation gate FAIL: page_bbox is a raw string` | DEFECT-2 regression — `_decode_page_bbox` was bypassed somewhere. | Inspect `server/routes/query.py::_source_refs`; check Weaviate schema has `page_bbox` as TEXT and the migration was run. |
| `citation gate FAIL: no source_ref carries …` | Docling lost page provenance during chunking, OR Weaviate dropped the property silently (see memory [`project_weaviate_schema_drop.md`](#)). | Re-run the schema migration; spot-check a chunk via `weaviate-cli`. |

## Why this is separate from `soak_table_chunking.py`

`soak_table_chunking.py` is **parse-only** — it runs Docling +
chunker in-process and writes a markdown quality report. It never
touches Weaviate, TEI, or the LLM. This runbook script is the
**retrieval-side** counterpart: it covers the half of the pipeline
that the soak deliberately skips. Do not merge the two — they have
incompatible infrastructure assumptions (in-process vs. live stack).

## What's tested in CI

Only the script's pure helpers
(`_assert_expansion_fired`, `_assert_citation_has_bbox`,
`_render_report`, `_build_parser`) — via
`tests/integration/test_e2e_sanity_runbook.py`. The live-stack path
in `run_sanity()` is exercised manually by an operator following
this runbook.
