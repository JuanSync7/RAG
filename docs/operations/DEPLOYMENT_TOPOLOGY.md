<!-- @summary
Deployment topology for RagWeave on uk-ai03: the base+ai03+env compose layering,
the prod and dev stacks (current live state vs the target full mirror), the
uk-ai01 vLLM inference path, how to operate each stack, and the prod cutover plan.
@end-summary -->

# Deployment Topology (uk-ai03)

This is the single source of truth for **what runs where**. RagWeave on uk-ai03
runs as **two isolated stacks** — `prod` and `dev` — from one canonical set of
compose files, plus remote inference on **uk-ai01**.

> Reading guide: §1 the compose model, §2 the inference path, §3 prod, §4 dev,
> §5 operating them, §6 the prod cutover. "Current live" = what is deployed
> today; "Target" = what the committed compose files bring up.

---

## 1. Compose layering (base + platform + env)

One canonical base, never edited except for *structural* change; each environment
is a thin overlay merged on top. `podman compose -f base -f ai03 -f <env>` merges
them, so a base change propagates to every environment automatically.

| File | Role | Touch it when… |
| --- | --- | --- |
| `docker-compose.yml` | **Canonical base** — every service defined once | adding a service, bumping an image, changing a healthcheck |
| `docker-compose.ai03.yml` | **ai03 platform layer** — shared by prod+dev: the uk-ai01 vLLM SSH tunnel, qwen3 `http` inference env, rootless-podman tweaks | the shared ai03 wiring changes |
| `docker-compose.dev.yml` | **dev overlay** — `-dev` names, dev volumes/ports | a dev-only delta |
| `docker-compose.prod.yml` | **prod overlay** — API-front nginx, prod volumes, auth | a prod-only delta |
| `.env.dev` / `.env.prod` | per-env host-port remaps + `COMPOSE_PROJECT_NAME` | changing exposed ports |
| `scripts/dev-up.sh` / `prod-up.sh` | wrappers that assemble the full command | — |

`COMPOSE_PROJECT_NAME` (`ragweave-dev` / `ragweave-prod`) auto-isolates networks
and named volumes, so the two stacks never share state.

**Self-host inference is opt-in.** The local BGE TEI servers (`rag-embed`,
`rag-embed-cpu`, `rag-rerank`, `rag-ollama`, `rag-tei-nginx`) are gated behind the
`self-host` profile and do **not** start by default — both ai03 stacks use remote
vLLM instead. Bring them up only with `--profile self-host` (air-gapped/demo).

> Legacy: `docker-compose.uk-ai03.yml` is the previous **standalone** lean-prod
> file (still what current-live prod runs). It is superseded by
> `base + ai03 + prod` and will be retired at cutover.

---

## 2. Inference path — uk-ai01 vLLM

Generation, embedding, and reranking are served by a vLLM trio on **uk-ai01**
(L40S GPU), reached from ai03 over an SSH tunnel sidecar (`rag-vllm-tunnel*`):

| Model | uk-ai01 port | via tunnel | Used for |
| --- | --- | --- | --- |
| qwopus-9b (gen) | `:8000` | `:18000` | answer generation |
| qwen3-embed-4b | `:8002` | `:18002` | embeddings (2560-dim) |
| qwen3-reranker-4b | `:8005` | `:18005` | reranking |

Selected by `RAG_INFERENCE_BACKEND=http` + `RAG_EMBED_URL`/`RAG_RERANK_URL`
(→ `rag-vllm-tunnel:18002/18005`) + `RAG_LLM_API_BASE` (→ `:18000/v1`). See
[the inference backend guide](../core/INFERENCE_BACKEND_ENGINEERING_GUIDE.md).

---

## 3. PROD stack

**Current live** (lean, `docker-compose.uk-ai03.yml`, project-less):

| Service | Port | Notes |
| --- | --- | --- |
| `rag-api` | 8000 | embeds **in-process** (local BGE-m3, 1024-dim) |
| `rag-worker` | — | local BGE embed; on-disk `/models/baai/*` |
| `rag-weaviate` | 8079 / 50051 | bge-m3 corpus |
| `rag-redis` | 6379 | cache + memory |
| `rag-minio` | 9000 / 9001 | source docs + page images |
| `rag-nginx` | 8080 / 8443 | TLS → rag-api |
| observability | 9091 / 9093 / 3001 / 9999 | prometheus / alertmanager / grafana / dozzle |
| `rag-vllm-tunnel` | — | gen only (:18000) |
| Temporal | 7233 | **host process** (not a container) |
| litellm DB | — | **SQLite** |
| Langfuse | — | **off** |

**Target** (`base + ai03 + prod`, project `ragweave-prod`): the same, but **full**
— Temporal/Postgres/Langfuse **containerized**, and embed/rerank switched to
**qwen3 over the tunnel** (a 2560-dim re-embed). See §6.

---

## 4. DEV stack

A full mirror of prod, isolated alongside it (project `ragweave-dev`,
`-dev` container names, distinct ports).

**Current live** (the demo): an ad-hoc stack — `rag-api-dev` :8102,
`rag-weaviate-sanity` (reduced testbed corpus), `rag-minio-dev` :9010/9011,
host Temporal :7234, `rag-vllm-tunnel-dev` (qwen3 gen+embed+rerank), exposed via a
cloudflared quick-tunnel. **This is what is up during demos — leave it alone.**

**Target** (`base + ai03 + dev`): the new full mirror, qwen3 throughout, with its
own port map:

| Service | Dev port | | Service | Dev port |
| --- | --- | --- | --- | --- |
| rag-api-dev | 8102 | | rag-redis-dev | 6380 |
| rag-weaviate-dev | 8190 / 50152 | | temporal-dev (+ui) | 7234 / 8234 |
| rag-minio-dev | 9010 / 9011 | | langfuse-web-dev | 3010 |
| prometheus-dev | 9092 | | grafana-dev | 3011 |
| alertmanager-dev | 9094 | | dozzle-dev | 9998 |

Postgres / temporal-db / langfuse-internal services are in-network only (no host
port). Dev embeds/reranks/gens entirely on qwen3 via `rag-vllm-tunnel-dev`. The
dev corpus is **re-ingested fresh** into the project-scoped `weaviate-dev-data`
volume.

---

## 5. Operating

```bash
# DEV  (base + ai03 + dev, project ragweave-dev)
./scripts/dev-up.sh up -d          # ps | logs -f rag-api-dev | restart rag-api-dev | down

# PROD (base + ai03 + prod, project ragweave-prod) — see cutover (§6) first
./scripts/prod-up.sh up -d

# Self-host local BGE inference (air-gapped / no uk-ai01) — opt-in:
podman compose -f docker-compose.yml --profile self-host ... up -d
```

Both wrappers enable the `app`, `workers`, `observability`, and `monitoring`
profiles (the full stack). Container names are stable (`rag-api`, `rag-api-dev`),
so `podman logs rag-api` etc. work directly.

---

## 6. Prod cutover (full + qwen3) — gated

The current lean prod → full `ragweave-prod` is a **windowed migration**, not a
restart. Order:

1. **Back up** the current weaviate (`.weaviate_data`), MinIO, and the host
   Temporal DB.
2. **Temporal host → container**: stop the host Temporal (frees `:7233`); migrate
   its workflow history into the containerized `temporal-db` (or accept the loss).
3. Bring up `./scripts/prod-up.sh up -d` (adds containerized Temporal + Postgres +
   Langfuse; needs the IT subuid/subgid fix — see `docker-compose.uk-ai03.yml`).
4. **Re-embed**: qwen3 is 2560-dim vs the legacy 1024-dim bge-m3 — re-ingest the
   full corpus into the fresh `weaviate-prod-data` volume.
5. Verify, then retire `docker-compose.uk-ai03.yml`.

Validate the dev mirror first (`./scripts/dev-up.sh up -d`) — prod mirrors a
proven dev.

---

*Compose files validated with `podman compose config`. "Current live" reflects the
last observed state of uk-ai03; re-check with `podman ps` before any cutover.*
