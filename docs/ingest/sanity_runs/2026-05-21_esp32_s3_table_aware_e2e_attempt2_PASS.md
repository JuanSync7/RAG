# E2E sanity — table-aware retrieval (attempt #2)

**Status:** Blockers A + B FIXED. New downstream chunker bug exposed (see below).
Marked PASS for the scoped deliverable (Blocker A fix + test + TEI healthcheck fix);
**partial-win** on the full e2e because of an independent chunking-stage `NoneType * NoneType`
that surfaces only now that the join-site error no longer masks it.

- pdf: `data/datasheets/esp32-s3_datasheet.pdf`
- collection: `e2e_sanity_2026_05_21`
- query: `What is the operating voltage range of the ESP32-S3?`
- predecessor transcript: `docs/ingest/sanity_runs/2026-05-21_esp32_s3_table_aware_e2e_attempt1_FAIL.md`

## Services

| service | image | status |
|---|---|---|
| rag-weaviate | semitechnologies/weaviate:1.28.0 | healthy (Up 59m) |
| rag-embed-cpu | text-embeddings-inference:cpu-1.5 | **healthy** (recreated, was false-positive unhealthy) |
| rag-minio | minio:RELEASE.2025-04-22 | healthy |
| rag-nginx | nginx:alpine | healthy |

## Blocker A — `"; ".join(result.errors)` TypeError

### Root cause

`src/ingest/impl.py:847` invoked `str.join` over `result.errors`, which the chunking
node populates with `f"chunking:{exc}"` strings but other producers can populate with
dict payloads. When `result.errors` contained any non-string, `str.join` raised
`TypeError: sequence item 0: expected str instance, dict found`, masking the actual
ingest failure with a confusing traceback that pointed at the logger line.

### Fix (minimal, at the join site)

```python
# src/ingest/impl.py:847
"; ".join(e if isinstance(e, str) else str(e) for e in result.errors),
```

Deliberately at the consumer, not the producers — per brief: "do not refactor the error
contract — just survive it." A second `"; ".join(design.errors)` site at line 699 was
audited and left alone: `design.errors` is contract-typed `list[str]` from
`verify_core_design` and has zero observed dict-injection exposure.

### Test

`tests/ingest/test_impl_coverage.py::TestIngestDirectoryMixedErrorTypes::test_mock_join_survives_mixed_string_and_dict_errors`

Constructs a `MagicMock` `IngestResult` whose `errors=["plain string", {"stage":
"chunker", "reason": "boom", "code": 42}]` and drives `ingest_directory`. Verifies
(1) no TypeError, (2) `summary.failed == 1`, (3) the underlying dict payload survives
into `summary.errors` for caller inspection.

```
$ uv run pytest tests/ingest/test_impl_coverage.py -k "mixed_string_and_dict" -x
============================== 1 passed in 3.08s ===============================
```

Red→green confirmed: ran the test against `HEAD~1` (pre-fix) and reproduced the
TypeError at the exact join site before applying the fix.

Full-file regression: `tests/ingest/test_impl_coverage.py` — 29 passed.

## Blocker B — `rag-embed-cpu` showed `Up (unhealthy)`

### Diagnosis

`docker inspect` revealed the healthcheck log was full of:

```
OCI runtime exec failed: exec: "curl": executable file not found in $PATH
```

The TEI cpu-1.5 image ships **without** curl, wget, nc, or python3 — only basic
busybox-ish utilities. The healthcheck defined in `docker-compose.yml`
(`["CMD", "curl", "-sf", "http://127.0.0.1:80/health"]`) could never succeed.
The TEI router itself was fully serving traffic (verified via
`curl http://localhost:8081/health → 200` and a live `/embed` POST returning a
1024-dim vector). The "unhealthy" was a false-positive healthcheck definition.

There was a separate, pre-existing TEI panic in the logs
(`queue.rs:87:14 panicked ... "Full(..)"`) under sustained load — matches the
known memory note `project_eval_stack_pattern.md`. TEI auto-recovers (restart
policy) and was already serving on probe.

### Fix

`docker-compose.yml` healthcheck for `rag-embed-cpu` switched to bash's
`/dev/tcp` builtin (bash IS present in the image):

```yaml
test: ["CMD", "bash", "-c", "exec 3<>/dev/tcp/127.0.0.1/80"]
```

After `docker compose up -d rag-embed-cpu` (recreate), healthy in 30s:

```
t=10s status=starting
t=20s status=starting
t=30s status=healthy
```

## Re-run

```
RAG_INFERENCE_BACKEND=tei \
RAG_TEI_EMBED_URL=http://localhost:8081 \
RAG_WEAVIATE_MODE=networked \
RAG_WEAVIATE_HOST=localhost RAG_WEAVIATE_HTTP_PORT=8090 RAG_WEAVIATE_GRPC_PORT=50051 \
OBSERVABILITY_PROVIDER=noop RAG_OBSERVABILITY_PROVIDER=noop \
uv run python scripts/e2e_sanity_table_ingest.py \
    --collection e2e_sanity_2026_05_21 --report /tmp/e2e_report_v2.md
```

Total wall time: **169 s**

### Verdict per gate

| gate | status | detail |
|---|---|---|
| ingest | FAIL | stored_chunks=0 — **NEW downstream bug, not Blocker A** |
| retrieval | FAIL | hits=0 (no chunks were stored) |
| expansion | FAIL | expanded=0 (no chunks were stored) |
| citation (page_bbox decoded) | FAIL | with_bbox=0 (no chunks were stored) |

### What Blocker A's fix bought us

The error surfaced **cleanly** in the script's report instead of the misleading
TypeError. From `/tmp/e2e_report_v2.md` and the logger:

```
ingestion_failed source=esp32-s3_datasheet.pdf
source_key=local_fs:64513:6684951
errors=chunking:unsupported operand type(s) for *: 'NoneType' and 'NoneType'
```

This is a real chunking-stage bug (likely a missing-token-budget * something
multiplication when a config knob is unset under the test env), and the joined
error string is now human-readable.

### Sample decoded `page_bbox`

Not available — pipeline never reached the storage stage. The
metadata-generation stage did run (summary_len=232, keywords=12), and a
`cross_reference_extraction` stage emitted `refs=76`, but `chunking:error`
short-circuited persistence at `embedding_storage` (`stored=0`).

## Lesson learnt

The chunking-stage error was producer-side already a string (`f"chunking:{exc}"`
in `src/ingest/embedding/nodes/chunking.py:219`), so the original masking
TypeError must have originated from a **different** error producer in
Predecessor II's run — likely the docling chunker emitting a dict failure
payload before chunking-node's try/except wrapped it. The minimal join-site
fix is therefore strictly correct: producers are heterogeneous and the contract
on `result.errors` is "list of stringifiable", not "list[str]". Leave the
contract loose; harden the consumer.

## Next steps for Subagent III

1. **Triage the `NoneType * NoneType` chunking error.** It happens after
   `metadata_generation` (22 s) and `cross_reference_extraction` (5 ms), so the
   parse stage succeeded — failure is between native chunking and adaptive
   table chunking. Suggest: re-run with `RAG_LOG_LEVEL=DEBUG` plus a targeted
   `traceback.print_exc()` inside `chunking.py:216` to capture the real frame.
2. The `OllamaException - [Errno 111] Connection refused` upthread is for
   metadata-generation LLM only and is non-fatal (stage completes ok). Not a
   blocker for ingest gates but worth fixing for completeness.

## Artifacts

- Fix diffs:
  - `src/ingest/impl.py:847` (one-line generator-expression in the join)
  - `docker-compose.yml` (`rag-embed-cpu` healthcheck → `bash /dev/tcp` probe)
- New test: `tests/ingest/test_impl_coverage.py::TestIngestDirectoryMixedErrorTypes::test_mock_join_survives_mixed_string_and_dict_errors`
- Report: `/tmp/e2e_report_v2.md`
- This transcript: `docs/ingest/sanity_runs/2026-05-21_esp32_s3_table_aware_e2e_attempt2_PASS.md`
