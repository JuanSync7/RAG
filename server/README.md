<!-- @summary
Production server setup: FastAPI + Temporal orchestration for multi-user RAG queries.
Models load once at worker startup; users get inference-only latency (~100-500ms).
@end-summary -->

# RAG Server — Production Multi-User Setup

## Architecture

```
Users → FastAPI Server → Temporal Server → Temporal Worker(s)
          (HTTP API)      (orchestration)    (RAGChain singleton,
                                              models in GPU memory)
```

**Key idea**: The 22s model-loading cost happens **once** when the worker boots (or the container starts). After that, every query runs inference against models already in memory.

## Components

| Component | File | What it does |
|-----------|------|--------------|
| **API Server** | `api.py` | Accepts HTTP requests, dispatches Temporal workflows. No GPU needed. |
| **MCP Adapter** | `mcp_adapter.py` | Exposes API-backed `health` and `query` tools over MCP (`stdio`). |
| **Temporal Worker** | `worker.py` | Loads RAGChain at startup, processes query activities from the queue. |
| **Workflow** | `workflows.py` | Defines `RAGQueryWorkflow` with retry policy and timeouts. |
| **Activities** | `activities.py` | `execute_rag_query` — runs against the preloaded RAGChain singleton. |
| **CLI Client** | `cli_client.py` | Same REPL as `cli.py`, but talks to the API. Starts instantly. |
| **Web Console Module** | `console/` | Dedicated console package for web UX routes and console service helpers. |
| **User Console UI Asset** | `console/static/user/index.html` | Modern chat interface served at `/console`. |
| **Admin Console UI Asset** | `console/static/console.html` | Tabbed debug/ops interface served at `/console/admin`. |
| **Web Console TypeScript Source** | `console/web/` | TypeScript source/build config compiled to static assets served by the API. |
| **Schemas** | `schemas.py` | Pydantic models for API request/response validation. |
| **Route Modules** | `routes/` | Domain-split API routers (`query`, `admin`, `system`) included by `api.py`. |
| **Turn-Loop Runner** | `turn_loop_runner.py` | API-process assembly for the turn-level agentic loop: flag resolution, pre-loop sanitize/PII gate, `TurnLoopDeps` wiring (TurnRetrieveWorkflow bridge, MinIO doc fetch, LLM provider, SSE event sink) and result→payload helpers used by the query routes. |
| **Shared Server Common** | `common/` | Shared envelope schemas + request helper utilities reused by API surfaces. |
| **Server Utils Facade** | `utils.py` | Stable import facade for shared request/envelope helper functions. |

## Quick Start

### 1. Start Temporal (infrastructure)

```bash
# Option A: Compose (recommended — auto-detects Podman or Docker)
./scripts/compose.sh up -d

# Option B: Temporal CLI (lightweight, no containers)
temporal server start-dev
```

### 2. Start the Worker (loads models once)

```bash
python -m server.worker
# ⏳ ~22s first time (loading embeddings + reranker into GPU memory)
# After that, the worker is ready to process queries
```

### 3. Start the API Server

```bash
uvicorn server.api:app --host 0.0.0.0 --port 8000
```

### 4. Query (pick one)

```bash
# Interactive CLI (same experience as cli.py, but instant startup)
python -m server.cli_client

# Or direct HTTP
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is retrieval-augmented generation?"}'

# Or browse the API docs
open http://localhost:8000/docs

# Or use the web consoles
open http://localhost:8000/console        # User Console (chat)
open http://localhost:8000/console/admin  # Admin Console (ops/debug)
```

### 5. MCP Tooling Adapter (optional)

Run the MCP adapter for IDE/agent tooling integrations:

```bash
python -m server.mcp_adapter
```

Exposed MCP tools:

- `health`, `query`
- `admin_list_api_keys`, `admin_create_api_key`, `admin_revoke_api_key`
- `admin_list_quotas`, `admin_set_tenant_quota`, `admin_delete_tenant_quota`

Environment variables for adapter auth forwarding:

- `RAG_API_URL` (default `http://localhost:8000`)
- `RAG_MCP_API_KEY` (optional, forwarded as `x-api-key`)
- `RAG_MCP_BEARER_TOKEN` (optional, forwarded as `Authorization: Bearer ...`)
- `RAG_MCP_ENABLE_ADMIN_TOOLS` (default disabled; set to `1` to allow admin MCP tools)

## Managed Container Mode (Optional)

Run API and worker(s) as managed container services instead of hand-started terminals:

```bash
# Start Temporal + API + one worker container
./scripts/compose.sh --profile app --profile workers up -d

# Scale workers on the same Temporal task queue (rag-query)
./scripts/compose.sh --profile workers up -d --scale rag-worker=3
```

This stays true to production patterns:
- workers are long-running managed services/containers
- multiple replicas listen on the same `rag-query` queue
- Temporal load-balances activities across available workers
- each worker gets its own embedded Weaviate data directory inside the container

### Monitoring (Optional)

```bash
# Container log dashboard
./scripts/compose.sh --profile monitoring up -d
```

- **Dozzle logs UI**: http://localhost:9999
- **Temporal UI**: http://localhost:8080
- **API health**: http://localhost:8000/health

### Langfuse Observability UI (Optional)

Run self-hosted Langfuse under the `observability` profile:

```bash
./scripts/compose.sh --profile observability up -d
```

Open:
- **Langfuse UI**: http://localhost:3000
- **Prometheus**: http://localhost:9091
- **Grafana**: http://localhost:3001

Then point API/worker traces to this local Langfuse deployment:

```bash
export RAG_OBSERVABILITY_PROVIDER=langfuse
export LANGFUSE_HOST=http://localhost:3000
export LANGFUSE_PUBLIC_KEY=pk-local-dev
export LANGFUSE_SECRET_KEY=sk-local-dev
```

If you run `rag-api` and `rag-worker` via containers, pass the same env vars
into compose (for example in a local `.env` file) and start with:

```bash
./scripts/compose.sh --profile app --profile workers --profile observability up -d
```

## Current Setup

Your current local flow still works unchanged:

```bash
./scripts/compose.sh up -d
.venv/bin/python -m server.worker
.venv/bin/uvicorn server.api:app --host 0.0.0.0 --port 8000
.venv/bin/python -m server.cli_client
```

### Notes for Containerized Workers

- `rag-worker` expects Ollama on the host (`http://host.docker.internal:11434`).
- It seeds vector data from local `./.weaviate_data` into a per-container directory.
- If you re-run ingestion locally, restart worker containers so they pick up fresh seeded data.

## Scaling

```
                    ┌──── Worker 1 (GPU 0) ────┐
Users → LB → API →  │──── Worker 2 (GPU 1) ────│ ← Temporal routes to available workers
                    └──── Worker 3 (GPU 2) ────┘
```

- **Queue requests**: Single worker handles moderate load (~4 concurrent queries via thread pool)
- **Multiple replicas**: Run N workers on the same Temporal task queue for N× throughput
- **Batch inference**: Worker concurrency is configurable via `RAG_WORKER_CONCURRENCY`

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_TEMPORAL_TARGET_HOST` | `localhost:7233` | Temporal server address |
| `RAG_API_PORT` | `8000` | API server port |
| `RAG_API_URL` | `http://localhost:8000` | CLI client target (for `cli_client.py`) |
| `RAG_API_MAX_INFLIGHT_REQUESTS` | `64` | Max concurrent in-flight API requests before overload protection applies |
| `RAG_API_OVERLOAD_QUEUE_TIMEOUT_MS` | `250` | Max time to wait for in-flight slot before returning 503 |
| `RAG_WORKER_CONCURRENCY` | `4` | Max concurrent activities per worker |
| `RAG_AUTH_API_KEYS_REQUIRED` | `false` | Enforce API-key/JWT auth at API boundary |
| `RAG_AUTH_API_KEYS_JSON` | `{}` | API-key map (`token_id -> {key, tenant_id, roles}`) |
| `RAG_AUTH_OIDC_ENABLED` | `false` | Enable OIDC bearer token validation (issuer/audience/JWKS) |
| `RAG_AUTH_OIDC_ISSUER` | `""` | Expected OIDC token issuer |
| `RAG_AUTH_OIDC_AUDIENCE` | `""` | Expected OIDC token audience |
| `RAG_AUTH_OIDC_JWKS_URL` | `""` | JWKS URL used to verify JWT signatures |
| `RAG_RATE_LIMIT_REQUESTS_PER_MINUTE` | `60` | Per-principal fixed-window rate limit |
| `RAG_CACHE_PROVIDER` | `memory` | Cache backend (`memory` or `redis`) |
| `RAG_MEMORY_PROVIDER` | `redis` | Conversation memory backend (canonical: Redis) |
| `RAG_MEMORY_REDIS_URL` | `redis://localhost:6379/0` | Redis connection for conversation memory |
| `RAG_MEMORY_MAX_RECENT_TURNS` | `8` | Sliding window size for recent turns |
| `RAG_MEMORY_SUMMARY_TRIGGER_TURNS` | `12` | Trigger threshold for rolling summary updates |

## Admin API

These endpoints require an `admin` role:

- `GET /admin/api-keys`
- `POST /admin/api-keys`
- `DELETE /admin/api-keys/{key_id}`
- `GET /admin/quotas`
- `PUT /admin/quotas/{tenant_id}`
- `DELETE /admin/quotas/{tenant_id}`

## API Schema Contract

The API enforces a standard schema contract:

- request payloads use strict Pydantic models (`extra="forbid"` for query requests),
- stage budget overrides are validated against known stage keys,
- non-2xx responses are normalized to:
  - `ok: false`
  - `error: { code, message, details? }`
  - `request_id`
- each response includes `x-request-id` for correlation across logs and clients.

## Conversation Memory

Conversation memory is tenant-aware and persistent (Redis canonical backend).

- request fields: `conversation_id`, `memory_enabled`, `memory_turn_window`, `compact_now`,
- response includes `conversation_id`,
- context management uses rolling summary + recent-turn window (bounded token estimate),
- `compact_now=true` or `/compact` forces summary refresh.

Conversation APIs:

- `GET /conversations`
- `POST /conversations/new`
- `GET /conversations/{conversation_id}/history`
- `POST /conversations/{conversation_id}/compact`
- `DELETE /conversations/{conversation_id}`

Server CLI memory commands:

- `/new-chat`
- `/conversations`
- `/switch <conversation_id>`
- `/history [limit]`
- `/compact`
- `/delete [conversation_id]`

## Monitoring

- **Temporal UI**: http://localhost:8080 — see all workflows, query history, latencies
- **FastAPI docs**: http://localhost:8000/docs — interactive API documentation
- **Health check**: `GET /health` — reports Temporal connectivity and worker status
- **Overload signals**:
  - `rag_api_inflight_requests`
  - `rag_api_overload_rejections_total{endpoint=...}`

### Practical 100-user load check

Run a simple concurrency test with built-in SLO checks:

```bash
python scripts/load_test_api.py \
  --url http://localhost:8000 \
  --total-requests 1000 \
  --concurrency 100 \
  --max-error-rate 2.0 \
  --max-p95-ms 2500
```

Operational plan: `docs/operations/RAG_100_USER_EXECUTION_PLAN.md`

### Tuning Watch Script

Use the tuning watcher to continuously evaluate scaling/concurrency signals and emit actionable recommendations:

```bash
# One-time snapshot
python scripts/watch_tuning_signals.py --once

# Continuous watch (every 30s)
python scripts/watch_tuning_signals.py --interval-seconds 30

# Send alerts to a webhook (Slack/Teams/etc.)
python scripts/watch_tuning_signals.py --webhook-url "https://example/webhook"
```

What it checks:

- Temporal schedule-to-start latency (query overridable via CLI arg)
- Temporal queue backlog (query overridable via CLI arg)
- API p95 latency from Prometheus (`rag_api_request_latency_ms`)
- Retrieval/generation p95 stage latencies (`rag_pipeline_stage_ms`)
- Worker CPU/memory pressure from container stats
- GPU memory pressure from `nvidia-smi` (when available)

### Auto-Scale Worker Replicas

Use the autoscaler script to scale `rag-worker` replicas up when saturation is sustained and scale down after sustained idle periods.

```bash
# Preview decisions without changing replica count
python scripts/auto_scale_workers.py --dry-run

# Enable live scaling loop (conservative defaults)
python scripts/auto_scale_workers.py \
  --min-replicas 1 \
  --max-replicas 6 \
  --interval-seconds 30
```

Default behavior:

- scale up when backlog/schedule-to-start/API p95 signals stay hot for 2 intervals,
- scale down only after 5 cool intervals and a longer cooldown window,
- keep scale-down thresholds lower than scale-up thresholds to prevent oscillation.
