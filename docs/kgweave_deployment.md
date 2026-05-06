# KGWeave Deployment & Team Boundary

<!-- @summary
Operational guide for the KGWeave Phase 2b worker fleet: topology,
queues, env, scaling, rollout, and current status of the RagWeave/KGWeave
team boundary (what's separated, what isn't yet).
@end-summary -->

## Topology

```
                   +--------------------+
                   |  Temporal cluster  |
                   |  (shared)          |
                   +---------+----------+
                             |
        +--------------------+---------------------+
        | task_queue=ingest-default                | task_queue=kgweave-default
        v                                          v
  +----------------+                       +-----------------+
  |  rag-worker    |                       | kgweave-worker  |
  |  (RagWeave)    | --execute_activity--> |  (KGWeave)      |
  |                |   by NAME, on         |                 |
  |  Phase 1, 2a   |   kgweave queue       |  Phase 2b only  |
  +----------------+                       +-----------------+
```

- **Both workers connect to the same Temporal cluster.** No HTTP, no message
  bus — Temporal is the transport.
- **Hard queue isolation.** `RAG_INGEST_USER_TASK_QUEUE` /
  `RAG_INGEST_BACKGROUND_TASK_QUEUE` (or the legacy single queue) drive the
  RagWeave worker; `KG_TASK_QUEUE = "kgweave-default"` drives KGWeave.
- **Phase 2b dispatch is by activity name.** RagWeave's `IngestDocumentWorkflow`
  calls `workflow.execute_activity(KG_PHASE2B_ACTIVITY, request, task_queue=KG_TASK_QUEUE, ...)`.
  RagWeave does not import KGWeave's worker code.
- **Retrieval-side: KGWeave is a Python library dep.** As of Step 14, RagWeave
  declares `kgweave` in `pyproject.toml` and imports `kgweave.knowledge_graph.*`
  for query-time graph expansion and term-index lookup. The KGWeave team owns
  the Python API; RagWeave is a normal pip consumer.

## What's separated today

| Concern | Owner | Notes |
|---|---|---|
| Phase 1 (parsing/cleaning) | RagWeave | `document_processing_activity` |
| Phase 2a (embeddings → Weaviate) | RagWeave | `embedding_pipeline_activity` |
| Phase 2b (KG extraction + commit) | **KGWeave** | `kg_phase2b_activity` on `kgweave-default` |
| Backfill of failed Phase 2b | RagWeave workflow → KGWeave activity | `BackfillKGWorkflow` drains the CleanDocumentStore status ledger |
| Status ledger | RagWeave | `CleanDocumentStore.record_attempt` / `list_pending` |
| KG library (extractors, backends, query expander, term index) | **KGWeave** | `kgweave` package, consumed via `pyproject.toml` path/version pin |
| Wire contract | KGWeave | `kgweave.contracts` is the single source — no vendored copy in RagWeave |

## Future: shipping KGWeave as a service product (Step C)

Today KGWeave ships two product surfaces:

1. **Worker fleet** (Step 13) — Temporal activities for Phase 2b ingest
2. **Python library** (Step 14) — `kgweave` package for in-process retrieval-side use

A future Step C adds a third: **HTTP/gRPC service** for non-Python consumers. See
`docs/kgweave_step_c_plan.md` — it would wrap the same `kgweave.knowledge_graph`
read-side API in a FastAPI server, share the same Docker image, and add minimal
operational surface. Step 14 deliberately positioned the library so Step C is a
~200-LOC wrapper.

Removing `src/knowledge_graph/` from RagWeave (Step 14) requires migrating
the retrieval-side term-index lookup to a KGWeave query API (Temporal
activity or HTTP). Until then, RagWeave keeps a working in-tree copy.

## Environment

| Variable | Default | Where |
|---|---|---|
| `TEMPORAL_TARGET_HOST` | `localhost:7233` | KGWeave worker |
| `KG_WORKER_SLOTS` | `4` | KGWeave worker — concurrent activity budget |
| `KG_USE_GLINER` | `0` | KGWeave worker — `1` to enable GLiNER extractor |
| `KGWEAVE_IMAGE_TAG` | `latest` | compose pull tag |
| `KGWEAVE_REPO_PATH` | `../KGWeave` | compose build context (sibling checkout) |

Image build extras:

```bash
docker build -f containers/Dockerfile.kgweave-worker \
  --build-arg KGWEAVE_EXTRAS=gliner,community,neo4j \
  -t ghcr.io/juansync7/kgweave-worker:dev ../KGWeave
```

## Local rollout

1. Clone KGWeave next to RagWeave: `git clone <repo> ../KGWeave`
2. From `RagWeave/`: `docker compose --profile workers up -d kgweave-worker`
3. Verify the worker registered:
   `docker compose logs kgweave-worker | grep "kgweave worker started"`
4. Trigger an ingest in RagWeave with `enable_kg_phase2b=True` in the
   `IngestionConfig`. The workflow event history shows `kg_phase2b`
   dispatched on `kgweave-default`.
5. On Phase 2b failure, inspect the document's status sidecar:
   `cat .runtime/clean_store/<safe_key>.status.json`. Backfill drains
   `failed_pending_retry` entries via `BackfillKGWorkflow`.

## Scaling

KGWeave scales horizontally — add replicas, each polls `kgweave-default`
independently. Recommended ratios (start point, tune from queue depth):

- 1 KGWeave replica per 2 RagWeave replicas if KG ingest is fast (regex)
- 1:1 if `KG_USE_GLINER=1` (LLM/model cost dominates)
- Watch Temporal's per-queue backlog metric (`temporal_workflow_task_schedule_to_start_latency`)
  bucketed by `task_queue`.

## Failure semantics (recap)

KGWeave activities raise `ApplicationError(type=<class>, non_retryable=...)`
where `<class>` is one of:

- `transient` → retried by Temporal (exponential backoff, `maximum_attempts=5`),
  then recorded as `failed_pending_retry` in the status ledger so
  `BackfillKGWorkflow` can pick it up.
- `document` → not retried; recorded as `failed_permanent`. Human review
  needed (e.g. corrupt SystemVerilog file).
- `system` → not retried; recorded as `failed_permanent`. Operator action
  needed (e.g. OOM, missing model).

The RagWeave workflow surfaces `kg_status` + `kg_error_class` on
`IngestDocumentResult` so the UI can render a subtle indicator.

## Contract evolution

After Step 14 the contract is the `kgweave` Python package itself — no
vendored copy in RagWeave. Workflow:

1. Edit schemas in `~/KGWeave/src/kgweave/contracts/`
2. Bump KGWeave's package version in `~/KGWeave/pyproject.toml` if breaking
3. Re-pin in RagWeave's `pyproject.toml` (`uv sync` picks up local-path
   dep automatically during dev; CI uses git ref or wheel)
4. Run `pytest tests/contracts/test_kgweave_package_install.py` — proves
   the import surface RagWeave depends on is still exposed
5. For breaking changes: bump `CONTRACT_VERSION` constant too and
   coordinate worker + workflow rollout (workflows pinned to the old
   `CONTRACT_VERSION` will reject the new schema at deserialization)

## Migration history

| Step | Done | What changed |
|---|---|---|
| 7 | ✅ | Status ledger in CleanDocumentStore (durable retry queue) |
| 8 | ✅ | Pydantic contract package |
| 9 | ✅ | KGWeave worker scaffold |
| 10 | ✅ | Workflow dispatches Phase 2b by activity name |
| 11 | ✅ | BackfillKGWorkflow drains pending retries |
| 12 | ✅ | KG copied into KGWeave; `BuilderKGService` wired |
| 12b | ✅ → superseded by 14 | Drift guard (now removed since the package is the contract) |
| 13 | ✅ | Dockerfile, compose entry, ops doc |
| **14** | ✅ | KGWeave is a Python library dep; in-tree KG copies deleted |
| C (future) | — | Optional HTTP/gRPC wrapper for non-Python consumers |
