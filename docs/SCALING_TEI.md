# Scaling TEI Inference (Embed + Rerank)

`rag-embed` and `rag-rerank` are TEI containers fronted by `rag-nginx` so they
can be scaled horizontally without client reconfiguration.

## Architecture

```
rag-api (queries) ─┐                     ┌──▶ rag-embed       (GPU, 1..N)
                   ├─▶ rag-nginx :8081 ──┤
rag-worker(ingest)─┘    (tier-aware LB)  └──▶ rag-embed-cpu   (CPU, 1..M)
                          │  (5xx fallback ↑)
                          └─ :8082 ───────────▶ rag-rerank   (CPU default)
```

### Tier routing

The embed lane is **tier-aware** via the `X-RagWeave-Tier` request header:

| Header value         | Primary pool       | Fallback on 5xx     | Use case                  |
| -------------------- | ------------------ | ------------------- | ------------------------- |
| *(none)* — default   | `rag-embed` (GPU)  | `rag-embed-cpu`     | Latency-sensitive queries |
| `ingest`             | `rag-embed-cpu`    | *(none — explicit)* | Ingest backfill           |

Set automatically by the Python client: `get_embedding_provider(tier="ingest")`
in `src/ingest/temporal/activities.py` and `src/ingest/impl.py` pins all
ingest-side embedding to the CPU pool. Query-path callers omit the tier
arg → GPU primary with CPU as automatic fallback on overload.

- Clients hit `http://rag-nginx:8081` (embed) and `http://rag-nginx:8082` (rerank).
- nginx uses Docker's embedded DNS (`127.0.0.11`, `valid=10s`) to re-resolve
  service names without reload, so new replicas are picked up automatically.
- Health is **passive** (nginx OSS — no active probes). TEI's own `/health`
  plus compose `service_healthy` gates startup.

## Scaling on a single host (compose)

```bash
make scale-tei EMBED=2 EMBED_CPU=4 RERANK=2
# or directly:
docker compose up -d --scale rag-embed=2 --scale rag-embed-cpu=4 --scale rag-rerank=2
```

**GPU caveat**: every replica binds to *all* available GPUs via the deploy
reservation. On a single-GPU host, `EMBED=3` means three TEI processes contend
for one device — you get OOM risk and serialised inference, not parallelism.
For real horizontal scale, run **one replica per GPU node**.

## Scaling on AWS (target topology)

The compose layout is the dev/staging template; prod replaces in-cluster nginx
with managed L7 load balancing:

| Compose         | AWS equivalent                                |
| --------------- | --------------------------------------------- |
| `rag-nginx`     | ALB with two listener rules / target groups  |
| `rag-embed:80`  | ECS service, 1 task per `g4dn.xlarge` (1 GPU) |
| `rag-rerank:80` | ECS service, 1 task per CPU instance          |
| Docker DNS      | ECS service discovery / target group health   |

Client URLs (`RAG_TEI_EMBED_URL`, `RAG_TEI_RERANK_URL`) flip from the in-cluster
`http://rag-nginx:8081` to the ALB DNS name. No code change required.

## Verifying

```bash
# After scaling:
docker compose ps rag-embed rag-rerank          # see replica count
./scripts/stack.sh status                        # human-readable status
curl http://localhost:8081/health                # via nginx → embed lane
curl http://localhost:8082/health                # via nginx → rerank lane
```

`scripts/stack.sh status` reports replica counts when scale > 1
(`● 3 replica(s) running`).

## Tradeoffs vs alternatives

- **Why not direct service DNS** (`http://rag-embed:80`)? Docker DNS round-robins
  but Python HTTP clients cache the resolved IP for the connection's lifetime,
  so a long-running worker pins to one replica. nginx re-resolves on each
  request batch.
- **Why not `keepalive` in upstream?** `keepalive` requires a static `upstream`
  block, which conflicts with `resolver`-based dynamic resolution. For TEI
  workloads where backend latency dominates, the connection setup cost is
  negligible.
- **Why not put nginx in front of LiteLLM/Ollama for generation?** LiteLLM is
  itself a routing/LB proxy with retries and fallbacks. Adding nginx in front
  is redundant unless you need to scale LiteLLM replicas, which is rarely the
  bottleneck in a RAG pipeline.

## Autoscaling (AWS target architecture)

Compose `--scale` is manual. For traffic-driven autoscaling on AWS:

| Layer            | Mechanism                                                   |
| ---------------- | ----------------------------------------------------------- |
| GPU pool        | ECS service on GPU capacity provider (`g4dn.*`), 1 task/GPU |
| CPU pool        | ECS service on CPU capacity provider, fast-scaling          |
| Routing          | ALB with two target groups (replaces nginx 8081 lanes)      |
| Scale signal    | CloudWatch on (a) ALB `RequestCountPerTarget` for queries, (b) Temporal queue depth metric for ingest |
| Scale policy    | Step scaling — aggressive scale-out, conservative scale-in |
| Predictive      | EC2 Auto Scaling Predictive Scaling for GPU ASG (lead time covers ~3-5 min instance + model-load cold start) |

**Cold-start strategy** — GPU instance + TEI model load is 3-5 min, so reactive
scaling lags spikes. The recommended pattern:

1. **Query traffic spike**: ALB target tracking on `RequestCountPerTarget` → CPU
   pool scales out within seconds (no GPU wait), absorbs the burst as fallback
   via the nginx error_page rule (or ALB weighted forwarding in prod).
2. **Sustained load**: CloudWatch alarm fires → GPU service scales out; new
   GPU tasks come up over the next 3-5 min and take over the primary pool.
3. **Decay**: query traffic drops → conservative scale-in policy retains GPU
   for next spike, CPU pool drains first.

**Ingest queue scaling** — Temporal worker emits `ingest_queue_depth` to
Prometheus → CloudWatch → triggers `rag-embed-cpu` task scaling. Ingest never
touches GPU pool (pinned via `X-RagWeave-Tier: ingest` header).

**What NOT to build**: a Python autoscaler in the worker. ECS Service Auto
Scaling + EC2 Predictive Scaling cover the use case natively. Application code
should only emit metrics and respect tier-routing — orchestration is platform.
