<!-- @summary
Multi-modal RAG platform with pipeline-first ingestion, visual+text embeddings,
and graph-based orchestration. Includes engineering docs, onboarding guides, and
operations tooling for observability, backup/restore, and scaling.
@end-summary -->

<p align="center">
  <img src="assets/banners/04-woven-hexagon-mesh-animated.svg" alt="RagWeave — Multi-Modal RAG Platform" width="100%"/>
</p>

**RagWeave** is a production-grade, multi-modal Retrieval-Augmented Generation platform. It ingests documents of any format, builds dual-track text and visual embeddings, and serves grounded answers through a full retrieval-reranking-generation pipeline — with guardrails, observability, and confidence routing built in.

### What It Does

- **Ingests anything** — PDFs, DOCX, PPTX, HTML, Markdown, images, tables, and code. A LangGraph pipeline handles parsing (via Docling), structure detection, VLM figure captioning, text cleaning, semantic chunking, metadata extraction, and quality validation. Knowledge-graph ingest is owned by **KGWeave** and dispatched out-of-process via the Temporal Phase 2b handoff (`KG_PHASE2B_ACTIVITY` on `KG_TASK_QUEUE`).
- **Dual-track embeddings** — Text chunks are embedded with BGE-M3 (1024-dim dense vectors). Document pages are visually embedded with ColQwen2 (128-dim patch vectors via a 4-bit quantized Qwen2-VL backbone). Both tracks are stored in Weaviate and searched simultaneously at query time.
- **Hybrid retrieval + reranking** — Combines BM25 keyword search with dense vector search (configurable alpha blend), expands queries with knowledge graph terms, reranks with a BGE cross-encoder, and merges visual page results via ColQwen2 MaxSim scoring.
- **Confidence-aware generation** — A 3-signal composite score (retrieval confidence, LLM self-assessment, citation coverage) routes each answer to RETURN, RE_RETRIEVE, FLAG, or BLOCK — no silent hallucinations.
- **Full safety rails** — Input guardrails (intent classification, injection/jailbreak detection, PII redaction, toxicity filtering, topic safety) and output guardrails (faithfulness checking, hallucination detection) run in parallel with per-rail timeouts.
- **Provider-agnostic LLMs** — LiteLLM Router with named aliases (`default`, `vision`, `query`, `smart`, `fast`). Swap between Ollama, OpenAI, Anthropic, or any OpenAI-compatible endpoint via config alone.
- **Temporal orchestration** — Both ingestion and query serving run as durable Temporal workflows with independent retry policies. Workers scale horizontally.
- **Observability built in** — Langfuse LLM tracing, Prometheus metrics, Grafana dashboards, per-stage timing budgets, and token budget tracking per request.

### Key Strengths

| Strength | Detail |
|----------|--------|
| **True multi-modal** | Not just text — visual page embeddings let you search diagrams, charts, and layouts that text extraction misses |
| **Pipeline-first** | Every stage is a discrete LangGraph node with its own config toggle — add, skip, or replace any stage without touching the rest |
| **Swappable backends** | Abstract base classes for vector store, document store, guardrails, observability, and retry — implement the ABC, add one config branch |
| **Runs anywhere** | Local with Ollama + embedded Weaviate, or fully containerized with Docker/Podman profiles for app, workers, monitoring, and HTTPS gateway |
| **Battle-tested safety** | Defense-in-depth: regex + NeMo + LLM semantic classification for injection detection; Presidio + GLiNER for PII; claim-level hallucination scoring |
| **Multi-tenant ready** | JWT + API key auth, per-tenant Redis conversation memory with sliding window + rolling summary, rate limiting and quotas |

### Architecture & Layout

```text
Users/CLI -> FastAPI (server/api.py) -> Temporal workflow -> Worker activity
                                                    |
                                                    v
                                          RAGChain singleton
                                  (retrieval, reranking, optional generation)
```

Ingestion runs as a separate Temporal workflow that writes content + embeddings consumed by retrieval.

| Directory | Purpose |
| --- | --- |
| `src/ingest/` | LangGraph ingestion pipeline (node-per-file + shared helpers); KG ingest delegated to KGWeave via Temporal |
| `src/retrieval/` | Query processing, retrieval orchestration, reranking, generation |
| `src/platform/` | Cross-cutting services: auth, quotas/rate limits, cache, metrics, observability |
| `src/common/` | Deterministic helpers shared across ingestion/retrieval |
| `server/` | FastAPI/Temporal runtime: API, workflows, activities, worker, schemas, web console |
| `config/` | Environment-driven settings (`config/settings.py`) |
| `docs/` | Engineering guides, specs, operations runbooks |
| `tests/` | Unit + integration tests (ingestion in `tests/ingest/`) |
| `scripts/` | Ops helpers (stack control, backup/restore, DR drill, smoke test) |
| `prompts/` | Prompt templates for retrieval query processing |

---

## Quick Start

The fastest path from clone to working query is **7 steps**. Containers-only users can skip steps 2–4.

### Prerequisites

- **Python 3.10+** (3.12 recommended)
- **[uv](https://docs.astral.sh/uv/)** — fast Python package manager (`pip` is not supported)
- **Docker** + **Docker Compose v2** (or **Podman** + **podman-compose** — see [Podman Setup](#podman-setup))
- **Node.js 18+** + **npm** — only for editing the web console (`make setup` builds it for you)
- **(Optional) `cloudflared`** — for `make tunnel` public URLs. Not in default Ubuntu repos; see [COLD_START_GUIDE §0.5](docs/operations/COLD_START_GUIDE.md).

> **First time on a clean Linux box?** Follow [`docs/operations/COLD_START_GUIDE.md`](docs/operations/COLD_START_GUIDE.md) — it walks every prerequisite install command and known gaps (cloudflared install, nvm/PATH, profile combinations).

### Choose a run mode

| Mode | When to use | What you need |
| --- | --- | --- |
| **Containers-only** (fastest start) | Just trying it out, or running without modifying Python/TS code | Docker only — `rag-api` / `rag-worker` pull from `ghcr.io/juansync7/ragweave-{api,worker}:latest` |
| **Local dev** (this guide) | Iterating on Python or TypeScript code | All prerequisites above |

For containers-only, jump to [Step 5](#step-5--start-the-stack) after copying `.env`. To rebuild app images locally instead of pulling: `./scripts/compose.sh build rag-api rag-worker`. Pin to a specific image tag via `RAG_API_IMAGE_TAG` / `RAG_WORKER_IMAGE_TAG` in `.env`.

### Step 1 — Clone & copy environment

```bash
git clone <repo-url> RagWeave && cd RagWeave
cp .env.example .env
./scripts/bootstrap_kgweave.sh   # clones the KGWeave sibling repo to ../KGWeave
```

You will edit `.env` in steps 3–4. Defaults work for local dev with the bundled containerised LLM and storage.

> **Why bootstrap?** The knowledge-graph subsystem lives in the separate [KGWeave](https://github.com/JuanSync7/KGWeave) repo. Both the editable Python install (`pyproject.toml` -> `[tool.uv.sources]`) and the Docker builds (compose `additional_contexts`) resolve it from `../KGWeave`. Override the path with `KGWEAVE_REPO_PATH=/elsewhere` if you keep it somewhere else.

### Step 2 — Install Python + web console

```bash
make setup
```

Creates `.venv/`, installs all deps via `uv sync --extra dev` (respects `uv.lock`), runs `npm install`, and compiles the TypeScript console. Run **once per clone**.

> Prefer explicit steps? `make install` (Python only) + `make console-install && make console-build` (frontend). Or skip `make` entirely with `uv sync --extra dev`. **Never use `pip` directly** — it bypasses the lock file.

**Optional dependency groups** (not installed by default):

```bash
uv sync --extra pii          # PII detection (presidio, spacy)
uv sync --extra gliner       # GLiNER entity extraction
uv sync --extra all          # Everything
```

> **Vector store note:** Weaviate is the default and currently the only fully supported backend. ChromaDB / Pinecone / Qdrant extras install client libs but the adapters are not yet implemented.

### Step 3 — Choose your LLM

**Option A — Containerised Ollama (default, no host install).**

The `rag-ollama` container starts automatically as part of the always-on stack. Once Step 5 brings the stack up, pull the model into it:

```bash
docker exec rag-ollama ollama pull qwen2.5:3b        # generation (required)
# Optional vision model (only if you ingest images/figures):
# docker exec rag-ollama ollama pull qwen2.5vl:3b
```

The container publishes `127.0.0.1:11434` to the host, so `RAG_OLLAMA_URL=http://localhost:11434` (the `.env.example` default) works as-is. Models are cached in `./.ollama_data/` and survive recreation. Stop with `docker compose stop rag-ollama` to free GPU/RAM (retrieval still works; generation fails fast with `ECONNREFUSED`).

**Option B — Cloud provider (OpenRouter, OpenAI, Anthropic, …).** Edit `.env`:

```bash
RAG_LLM_MODEL=openrouter/anthropic/claude-3-haiku    # LiteLLM model string
RAG_LLM_API_BASE=https://openrouter.ai/api/v1
RAG_LLM_API_KEY=sk-or-v1-...
```

Model strings follow `<provider>/<model-name>`. See [LiteLLM docs](https://docs.litellm.ai/docs/providers) for the full list.

### Step 4 — Embedding & reranker models

You have two choices. Pick one.

**Choice 1 — Local model files (default for dev).** Requires the `local-embed` extra: `uv sync --extra local-embed`. By default the loader resolves models by HuggingFace repo ID through the local HF cache (`~/.cache/huggingface/`) — first run downloads automatically, no env var needed.

If you prefer to pre-download (offline machines, controlled mirrors) or pin to a specific directory:

```bash
uv run --with huggingface-hub huggingface-cli download BAAI/bge-m3             --local-dir ~/models/baai/bge-m3
uv run --with huggingface-hub huggingface-cli download BAAI/bge-reranker-v2-m3 --local-dir ~/models/baai/bge-reranker-v2-m3

# In .env (only needed if pinning to a directory):
RAG_MODEL_ROOT=/home/you/models
```

Each model is ~570 MB (~1.2 GB total). For `git-lfs` or `./models` symlink layouts, see [COLD_START_GUIDE §3](docs/operations/COLD_START_GUIDE.md).

**Choice 2 — TEI containers (no manual download).** Set `RAG_INFERENCE_BACKEND=tei` in `.env`. The `rag-embed` / `rag-rerank` containers are always-on and download weights into `./.tei_cache/` on first start. Skip the manual download.

### Step 5 — Start the stack

```bash
./scripts/compose.sh --profile app --profile workers up -d
```

The wrapper auto-detects Docker or Podman. The `app` profile starts the API + Redis; the `workers` profile starts the Temporal worker(s). **Many other services start automatically without any profile flag** — Postgres, MinIO, Weaviate, Temporal, Ollama, TEI embed/rerank, nginx. See [Container Profiles](#container-profiles) for the full breakdown.

> First boot pulls images and downloads model weights — expect 5–15 minutes depending on bandwidth. Subsequent boots are seconds.

### Step 6 — Verify

```bash
# All containers should be healthy:
./scripts/compose.sh ps

# API health:
curl -s http://localhost:8000/health

# Temporal UI (workflow runs):
open http://localhost:8080

# MinIO console (default creds: minioadmin / minioadmin):
open http://localhost:9001

# Web console:
open http://localhost:8000/console
```

If any container is `unhealthy`, check `docker logs <container-name>` — first-boot 502s on `rag-nginx` are normal until upstreams finish warming.

### Step 7 — Ingest and query

```bash
python -m src.ingest.cli --dir ./documents     # ingest a folder
python query.py "What is RAG?"                 # one-shot query
python cli.py                                  # interactive REPL
```

### Optional — Tune behaviour

These have working defaults but are worth reviewing before production:

| Variable | Default | Notes |
|----------|---------|-------|
| `RAG_LLM_TEMPERATURE` | `0.3` | Generation temperature |
| `RAG_LLM_MAX_TOKENS` | `1024` | Max tokens per response |
| `RAG_CACHE_TTL_SECONDS` | `120` | Query result cache lifetime |
| `RAG_RATE_LIMIT_REQUESTS_PER_MINUTE` | `60` | Per-tenant rate limit |
| `RAG_MEMORY_MAX_RECENT_TURNS` | `8` | Conversation history window |
| `RAG_RETRIEVAL_TIMEOUT_MS` | `30000` | End-to-end query timeout |

See [`.env.example`](.env.example) for all available settings.

> **After changing `.env`:** most settings are read at startup. For changes to take effect in the containerised stack, run `make restart-worker` (worker config) or restart the API container. Generation model changes require a worker restart only. Embedding/reranker path changes require `make restart-worker` plus confirming new model files are mounted.

---

## Storage Backends

RagWeave ships with four stateful services. **All start automatically with any compose invocation** — they're profile-less, so you do not need a flag for them.

| Service | Container | Default port(s) | Default credentials | Data location | What it stores |
| --- | --- | --- | --- | --- | --- |
| **Weaviate** | `rag-weaviate` | 8090 (HTTP), 50051 (gRPC) | anonymous access enabled | `./.weaviate_data/` (bind mount) | Dual-track text + visual embeddings, BM25 indexes |
| **MinIO** | `rag-minio` | 9000 (S3 API), 9001 (console) | `minioadmin` / `minioadmin` | volume `rag-minio-data` | Document artifacts, intermediate ingest blobs |
| **Postgres** | `rag-postgres` | 5432 | configured via env | volume `rag-postgres-data` | API metadata, tenancy, audit logs |
| **Redis** | `rag-redis` | 6379 | none | volume `rag-redis-data` | Conversation memory, query cache, rate-limit counters |

**No setup required.** Buckets are created by workers on first ingest; Weaviate collections are created on first ingest; Postgres schemas are managed by the API on startup.

**Important defaults to change before any non-local deployment:**
- `RAG_MINIO_ACCESS_KEY` / `RAG_MINIO_SECRET_KEY` — currently `minioadmin / minioadmin`.
- Weaviate runs with `AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true`. Disable and configure API keys before exposing it.
- Postgres credentials are set in `.env`; the defaults are dev-only.

**Resetting state:**

```bash
./scripts/compose.sh down                      # stop containers, keep data
./scripts/compose.sh down -v                   # stop AND delete all volumes (DESTRUCTIVE)
rm -rf .weaviate_data/                         # nuke Weaviate only
docker volume rm ragweave_rag-minio-data       # nuke MinIO only
```

**Backup / restore / DR drill:** see [`scripts/backup_all.sh`](scripts/backup_all.sh), [`scripts/restore_all.sh`](scripts/restore_all.sh), [`scripts/dr_drill.sh`](scripts/dr_drill.sh), and [`docs/operations/`](docs/operations/).

---

## Container Profiles

Compose uses [Docker profiles](https://docs.docker.com/compose/profiles/) to gate optional services. **Services without a profile start on every `compose up`**, regardless of which `--profile` flags you pass.

### Always-on (no profile flag needed)

| Container | Image | Why it's here |
| --- | --- | --- |
| `rag-postgres` | `postgres:16-alpine` | Application metadata |
| `rag-minio` | `minio/minio` | Artifact storage |
| `rag-weaviate` | `semitechnologies/weaviate:1.28.0` | Vector store |
| `temporal-db` | `postgres:16-alpine` | Temporal backing store |
| `temporal` | `temporalio/auto-setup` | Workflow engine |
| `temporal-ui` | `temporalio/ui` | Workflow inspector (port 8080) |
| `rag-embed` / `rag-embed-cpu` | `text-embeddings-inference` | TEI embed pool (GPU + CPU fallback) |
| `rag-rerank` | `text-embeddings-inference` | TEI reranker |
| `rag-ollama` | `ollama/ollama` | Local LLM serving |
| `rag-nginx` | `nginx:alpine` | Fronts API + load-balances TEI replicas |
| `dozzle` | `amir20/dozzle` | Container log viewer |

### Profile-gated

| Profile | Activates | Approx. additional pull |
|---------|-----------|--------------------------|
| `app` | `rag-api`, `rag-redis`, `pg-maintenance` | +544 MB |
| `workers` | `rag-worker`, `rag-redis` | +5.79 GB |
| `monitoring` | Prometheus, Alertmanager, Grafana | +1.74 GB |
| `observability` | Langfuse stack (6 containers — Postgres, Redis, ClickHouse, MinIO, web, worker) | +5.93 GB |

```bash
# Typical local dev:
./scripts/compose.sh --profile app --profile workers up -d

# Full production-ish stack with metrics + LLM tracing:
./scripts/compose.sh --profile app --profile workers --profile monitoring --profile observability up -d
```

> **Note:** older docs referenced `--profile temporal` and `--profile gateway`. Those flags are no-ops today — Temporal and nginx are always-on. The flag namespace is reserved for future use.

---

## Container Images

The stack uses two custom images with strict dependency isolation:

| Image | Size | Contents |
|---|---|---|
| `rag-api` | 389 MB | FastAPI, Temporal client, Weaviate client — no torch, no docling, no ML stack |
| `rag-worker` | 5.79 GB | Full ML stack (torch, sentence-transformers, docling, langchain, nemoguardrails) |

Container deps live in `containers/requirements-api.txt` and `containers/requirements-worker.txt` — **not** in `pyproject.toml`. This is deliberate: `pip install .` would pull every dep listed under `[project.dependencies]`, undoing the isolation. Local dev uses `pyproject.toml` via `make install`; containers bypass it.

**Adding a new dependency:**
- API server imports it → add to `pyproject.toml` AND `containers/requirements-api.txt`
- Worker-only → add to `pyproject.toml` AND `containers/requirements-worker.txt`
- Dev-only (pytest, deptry, …) → `pyproject.toml` only

### Build the images

```bash
make container-build           # docker (BuildKit) — builds both
make container-build-podman    # podman (--format docker) — builds both
make container-probe           # API import probe — catches transitive ML leakage
make container-sizes           # show current image sizes
make container-clean           # remove local rag-api / rag-worker images
```

**Manual:**

```bash
DOCKER_BUILDKIT=1 docker build -t rag-api    -f containers/Dockerfile.api     .
DOCKER_BUILDKIT=1 docker build -t rag-worker -f containers/Dockerfile.runtime .
```

Multi-stage builds, BuildKit pip-cache mounts, `.dockerignore`, compose-level healthchecks (podman-friendly), `PYTHONPATH=/app` (no `pip install .`), GPU support in the worker — see [`docs/operations/DOCKER_OPTIMIZATION.md`](docs/operations/DOCKER_OPTIMIZATION.md) for the full design.

---

## Running the API Server

### Option A — Local dev (fast iteration, no Docker rebuild)

```bash
make start    # Terminal 1: infrastructure + containerised workers
make dev      # Terminal 2: API server with hot-reload
make worker   # Terminal 3: Temporal worker (needed for ingest/query workflows)
```

> **WSL2 users:** if inter-container networking breaks after a restart, run `sudo ./scripts/fix-docker-networking.sh` once, or wire it into `/etc/wsl.conf` — see [WSL2 Setup](#wsl2-setup).

### Option B — Fully containerised stack

```bash
make restart                  # rebuild + restart app + workers
make restart-all              # all profiles (monitoring, observability, …)
make scale-workers N=3        # scale workers horizontally
```

Then use the CLI client or web console:

```bash
python -m server.cli_client                       # CLI targeting the API
# User console (chat):  http://localhost:8000/console
# Admin console (ops):  http://localhost:8000/console/admin
```

### Expose publicly via Cloudflare Tunnel (no account required)

```bash
make tunnel   # prints a public https://*.trycloudflare.com URL — Ctrl+C to kill
```

Requires `cloudflared` (system binary, not in default Ubuntu repos — see [COLD_START_GUIDE §0.5](docs/operations/COLD_START_GUIDE.md)).

For demos on a different network, you can also tunnel the nginx gateway:

```bash
cloudflared tunnel --url https://localhost:443 --no-tls-verify
```

---

## HTTPS Gateway (nginx)

`rag-nginx` runs always-on, but TLS only activates when you provide certs. One-time setup:

```bash
sudo apt install mkcert            # Debian/Ubuntu (or `brew install mkcert`)
./scripts/generate-certs.sh        # locally-trusted certs in ./certs/
echo "127.0.0.1  aion.local" | sudo tee -a /etc/hosts
```

Then browse `https://aion.local`. See [`certs/README.md`](certs/README.md) for details.

> **Security note:** port 8000 stays directly accessible (bypassing TLS). For LAN demos, set `RAG_API_HOST_PORT=127.0.0.1:8000` in `.env` to restrict direct access to localhost.

---

## Running Tests

```bash
make test                          # full pytest suite (uv run pytest)

# Fast static gates (no test execution):
make precommit-check               # L1+L2+L3+L4+TS on git-tracked files (skips WIP)
make all-check                     # same, but over the full tree including untracked

# Individual layers:
make py-compile-check              # L1: compileall across source tree
make import-check-tracked          # L2 (tracked files only)
make import-check                  # L2 (full tree)
make dep-check                     # L3: deptry
make container-dep-check           # L4: requirements-*.txt in sync with pyproject.toml

pytest tests/ingest/ -v            # targeted runs still work directly
```

> Neither `precommit-check` nor `all-check` runs pytest — they're fast static gates. Run `make test` separately. Use `precommit-check` before every `git commit` (skips WIP); use `all-check` before releases or as a periodic hygiene sweep.

---

## WSL2 Setup

Docker bridge networking on WSL2 needs a one-time fix per session (iptables FORWARD rules reset on WSL2 restart). Make it automatic via `/etc/wsl.conf`:

```ini
# /etc/wsl.conf  (create if missing)
[boot]
command = "service docker start && iptables -P FORWARD ACCEPT"
```

Then from PowerShell: `wsl --shutdown`.

**Manual fix for the current session:**

```bash
sudo ./scripts/fix-docker-networking.sh   # WSL2-aware; no-ops on Linux/macOS
```

---

## Podman Setup

Podman works as a rootless, daemonless drop-in for Docker. One-time:

```bash
sudo apt-get install -y podman podman-compose                   # 1. install
systemctl --user enable --now podman.socket                     # 2. user socket (Dozzle)
podman info | grep -i rootless                                  # 3. verify rootless: true
echo "CONTAINER_SOCK=\$XDG_RUNTIME_DIR/podman/podman.sock" >> .env   # 4. tell compose
./scripts/compose.sh --profile app --profile workers up -d      # 5. go
```

For internal design notes (rootless networking, socket detection, image-format trade-offs), see [`docs/operations/PODMAN_SPEC.md`](docs/operations/PODMAN_SPEC.md).

---

## Entry Points

| Command | Description |
|---------|-------------|
| `python -m src.ingest.cli --dir ./documents` | CLI for ingestion runs |
| `python query.py "question"` | Local retrieval query CLI |
| `python cli.py` | Unified interactive REPL |
| `python -m server.worker` | Temporal worker process |
| `uvicorn server.api:app --host 0.0.0.0 --port 8000` | API server (use `make dev` for hot reload) |
| `python -m server.cli_client` | Interactive client targeting the API server |
| `python -m server.mcp_adapter` | MCP tooling adapter over the API (`stdio` transport) |

---

## Make Targets

Run `make help` for this list in the terminal. All targets are also documented in comments in the [Makefile](Makefile).

| Target | Purpose |
|---|---|
| **Setup & install** | |
| `make setup` | **First-time setup.** Creates venv, installs Python deps, runs `npm install`, builds the web console |
| `make install` | (Re)install Python deps into the active env (`uv sync --extra dev`) |
| **Web console (TypeScript)** | |
| `make console-install` | `npm install` for the web console |
| `make console-check` | TypeScript type-check (no emit) |
| `make console-build` | Compile TS → `static/main.js` |
| `make console-watch` | Watch mode — rebuild on TS change |
| **Checks & tests** | |
| `make test` | Run the pytest suite |
| `make py-compile-check` | L1 syntax: `compileall` across `src/`, `server/`, `config/`, `import_check/` |
| `make import-check` | L2 internal: resolve imports + encapsulation, **whole tree** (includes untracked) |
| `make import-check-tracked` | L2 internal but only for **git-tracked** files (for `precommit-check`) |
| `make dep-check` | L3 external: `deptry` — `pyproject.toml` vs actual imports |
| `make container-dep-check` | L4 container: `requirements-*.txt` in sync with `pyproject.toml` |
| `make precommit-check` | **Compound gate for `git commit`**: L1 + L2(tracked) + L3 + L4 + `npm ci` + console-check. Excludes untracked WIP. |
| `make all-check` | **Compound gate for release**: same checks but over the entire tree including untracked. |
| **Container images** (see [Container Images](#container-images) for details) | |
| `make container-build` | Compile frontend + build `rag-api` + `rag-worker` with docker (BuildKit) |
| `make container-build-api` | Build only `rag-api` |
| `make container-build-worker` | Build only `rag-worker` |
| `make container-build-podman` | Compile frontend + build both with podman (`--format docker`) |
| `make container-probe` | Run the API import probe inside `rag-api` — catches transitive ML leakage |
| `make container-sizes` | Print current `rag-api` / `rag-worker` image sizes |
| `make container-clean` | Remove local `rag-api` / `rag-worker` images + dangling images |
| `make smoke-test` | Full integration check: build + stack + cloudflared tunnel + API checks + teardown |
| `make container-build-and-test` | Build images then immediately run smoke test (`SKIP_BUILD=1`) |
| **Stack control** (uses `scripts/stack.sh` — auto-detects docker/podman) | |
| `make start` | Bring up base + workers (no rebuild) |
| `make start-all` | Bring up every profile (no rebuild) |
| `make restart` | Frontend rebuild + recreate base + workers (mirrors `start`, with rebuild) |
| `make restart-all` | Frontend rebuild + recreate every profile (mirrors `start-all`, with rebuild) |
| `make tunnel` | Cloudflare tunnel for local API (port 8000) |

---

## Engineering Docs

| Directory | Contents |
|-----------|----------|
| `docs/ingestion/` | Ingestion pipeline spec (split: pipeline nodes + platform/cross-cutting), implementation guide, engineering guide, onboarding checklist |
| `docs/retrieval/` | Retrieval pipeline specs (split: query/ranking + generation/safety), NeMo Guardrails, engineering guide, onboarding checklist |
| `docs/server/` | Server API spec + implementation, platform services spec (auth, tenancy, rate limits, caching) |
| `docs/ui/` | CLI spec + implementation, web console spec + implementation, token budget spec + implementation |
| `docs/performance/` | Retrieval performance spec (runtime controls, benchmarking, load testing) |
| `docs/operations/` | Operations platform spec (deployment, scaling, monitoring, DR, CI/CD), 100-user plan, Podman migration |
| `docs/llm/` | LiteLLM SDK integration guide |

Key starting points:
- Ingestion: `docs/ingestion/INGESTION_PIPELINE_ENGINEERING_GUIDE.md`
- Retrieval: `docs/retrieval/RETRIEVAL_ENGINEERING_GUIDE.md`
- Server/runtime: `server/README.md`

---

## License

See [LICENSE](LICENSE) for details.
