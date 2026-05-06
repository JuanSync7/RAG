# KGWeave Step C — Service Product (HTTP / gRPC)

<!-- @summary
Forward-looking plan for shipping KGWeave as a network service in addition
to its current Python-library + Temporal-worker product surfaces. Not
implemented; this is the design + scoping doc.
@end-summary -->

## Why a third surface

After Step 14, KGWeave ships:

1. **Worker fleet** (Step 13) — Phase 2b ingest activities on a Temporal queue
2. **Python library** (Step 14) — `kgweave` package for in-process retrieval

Both require the consumer to be a Python service connected to the same
Temporal cluster (#1) or capable of `pip install kgweave` (#2). Step C
unblocks:

- **Non-Python consumers** (Java, Go, Rust services that want graph queries)
- **External / customer-facing access** — the KG as an isolated SaaS
- **Multi-tenant deployments** — one KGWeave instance, many isolated graphs

## What Step C is NOT

- Not a replacement for Step 14. The Python library stays for in-process
  consumers (zero RPC latency, deepest API).
- Not a Phase 2b ingest endpoint. Ingest still flows through Temporal
  for durability + retry semantics.
- Not a replacement for direct Neo4j/NetworkX backend access. The HTTP
  layer is read-side query API only.

## Surface

Minimal v1 endpoints (sized at ~200 LOC FastAPI on top of `kgweave.knowledge_graph`):

| Endpoint | Purpose | Underlying call |
|---|---|---|
| `GET /v1/health` | Liveness + backend stats | `get_graph_backend().stats()` |
| `POST /v1/expand` | Graph query expansion | `get_query_expander().expand(query, depth)` |
| `POST /v1/term-index/match` | Vocabulary lookup | `get_term_index().suggest_for_words(words)` |
| `GET /v1/entities/{key}` | Single entity probe | `backend.get_entity(key)` |
| `GET /v1/paths` | Shortest-path query | `PathMatcher.find(...)` |

Request/response schemas live in `kgweave.contracts.http` (new sibling to
`kgweave.contracts.constants` / `kgweave.contracts.schemas`). Reuse Pydantic
models — same validation rules as the Python API.

## Transport options

**Option C-1: FastAPI / HTTP+JSON** (recommended start)
- ~200 LOC, fastest to ship
- Browser/CLI debuggable
- Industry default; OpenAPI schema generated for free

**Option C-2: gRPC**
- ~400 LOC, protobuf schemas duplicated from Pydantic
- Lower per-call overhead (matters at >1k QPS)
- Better for service-mesh deployments

Start with C-1; add gRPC only if a consumer hits scale.

## Container packaging

KGWeave already has `containers/Dockerfile.kgweave-worker`. Add
`containers/Dockerfile.kgweave-api` that builds the same Python env but
runs `uvicorn kgweave.api.main:app` instead of the worker entry point.
Same image base, different `CMD`. Compose:

```yaml
kgweave-api:
  profiles: ["kgweave"]
  image: ghcr.io/juansync7/kgweave-api:${KGWEAVE_IMAGE_TAG:-latest}
  build:
    context: ${KGWEAVE_REPO_PATH:-../KGWeave}
    dockerfile: containers/Dockerfile.kgweave-api
  ports:
    - "8090:8080"
  environment:
    - KG_GRAPH_PATH=/data/graph.json
  volumes:
    - kgweave-graph:/data:ro
```

## Operational posture

- **Read-only** — no mutating endpoints. Writes still go through Temporal
  Phase 2b. This dramatically simplifies caching, rate limiting, and
  multi-replica scaling.
- **Stateless replicas** — backend state is the graph file; replicas can
  scale horizontally with no coordination. Snapshot freshness is the only
  staleness window (controlled by Phase 2b commit cadence).
- **Auth at the edge** — bearer token validated by an FastAPI dependency.
  No per-route auth logic; treat the whole API as a single trust boundary.
- **Observability** — reuse OpenTelemetry stack; emit per-endpoint
  histograms + a single `kgweave_api_query_total` counter labeled by
  endpoint and status.

## Migration path for RagWeave

Once C ships, RagWeave can *optionally* swap its in-process
`kgweave.knowledge_graph` calls for HTTP calls — useful if:

- RagWeave wants to deploy without the GLiNER / spaCy / leidenalg deps
  (the read-side ones are still ~200MB)
- KGWeave needs to evolve its read-side faster than RagWeave can re-pin

The swap is mechanical: introduce a `kgweave.client` thin façade that
either calls in-process (default) or HTTP (when `KGWEAVE_API_URL` is
set). RagWeave imports `kgweave.client.*` instead of
`kgweave.knowledge_graph.*`. Latency hit per query: 1-5 ms intra-cluster.

## Effort

- v1 (HTTP, read-only, 5 endpoints): ~1 week including tests, Dockerfile,
  compose entry, observability wiring
- gRPC variant: +3 days
- RagWeave thin façade swap: ~2 days

## Non-goals

- Schema CRUD endpoints (KG schema lives in YAML; reload-on-write is fine)
- Streaming / subscription endpoints (no consumer asked yet)
- Multi-graph per process (use multiple replicas with different volume mounts)
