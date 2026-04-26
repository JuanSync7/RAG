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

- **Ingests anything** — PDFs, DOCX, PPTX, HTML, Markdown, images, tables, and code. A 13-node LangGraph pipeline handles parsing (via Docling), structure detection, VLM figure captioning, text cleaning, semantic chunking, metadata extraction, knowledge graph triples, and quality validation.
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
| `src/ingest/` | 13-node LangGraph ingestion pipeline (node-per-file + shared helpers) |
| `src/retrieval/` | Query processing, retrieval orchestration, reranking, generation |
| `src/platform/` | Cross-cutting services: auth, quotas/rate limits, cache, metrics, observability |
| `src/common/` | Deterministic helpers shared across ingestion/retrieval |
| `server/` | FastAPI/Temporal runtime: API, workflows, activities, worker, schemas, web console |
| `config/` | Environment-driven settings (`config/settings.py`) |
| `docs/` | Engineering guides, specs, operations runbooks |
| `tests/` | Unit + integration tests (ingestion in `tests/ingest/`) |
| `scripts/` | Ops helpers (stack control, backup/restore, DR drill, smoke test) |
| `prompts/` | Prompt templates for retrieval query processing |

## Quick Start

### Prerequisites

- **Python 3.10+** (3.12 recommended)
- **[uv](https://docs.astral.sh/uv/)** — fast Python package manager
- **Node.js 18+** and **npm** — for the web console TypeScript build
- **Docker** and **Docker Compose v2** (or **Podman** and **podman-compose**) — for the full container stack (Temporal, Redis, Ollama, Weaviate, …)

> **No host Ollama needed.** The LLM runs in the bundled `rag-ollama`
> container. Cloud LLMs are also supported — see [Step B](#step-b--set-up-your-llm).

> **Podman users**: Podman is supported as a drop-in replacement for Docker.
> See [Podman Setup](#podman-setup) below for one-time configuration.

> **First time on a clean Linux box?** Follow
> [docs/operations/COLD_START_GUIDE.md](docs/operations/COLD_START_GUIDE.md) —
> it walks every prerequisite install command and the exact gaps in this
> Quick Start (cloudflared install, nvm/PATH issues, profile combinations).

### 1. Clone and set up the project

```bash
git clone <repo-url> RagWeave && cd RagWeave
make setup
```

`make setup` runs the full one-shot: creates `.venv/`, installs all runtime + dev dependencies via `uv`, installs web-console npm deps, and compiles the TypeScript console. Run once per clone.

> Prefer explicit steps? `make install` does just the Python deps; `make console-install && make console-build` handles the console. Or skip `make` entirely:
> `uv sync --extra dev` (auto-creates `.venv/`, respects `uv.lock`). Use `uv` everywhere — plain `pip` is not supported.

#### Optional dependency groups

Some features require extra packages that are not installed by default:

```bash
uv sync --extra pii          # PII detection (presidio, spacy)
uv sync --extra gliner       # GLiNER entity extraction
uv sync --extra all          # All optional dependencies
```

> **Vector store:** Weaviate is the default and currently the only fully supported backend.
> ChromaDB, Pinecone, and Qdrant extras (`.[chromadb]`, `.[pinecone]`, `.[qdrant]`) install
> the client libraries but the backend adapters are not yet implemented — they are planned.

### 2. Web console (already built by `make setup`)

`make setup` already installs and compiles the web console. You only need these targets when iterating on the TypeScript source:

```bash
make console-watch   # rebuild on change (live dev)
make console-check   # type-check only, no emit
make console-build   # one-shot production build
```

### 3. Start infrastructure services

```bash
./scripts/compose.sh --profile temporal up -d
```

This starts the core services: **Temporal** (orchestration) + **Temporal UI** (port 8080).
The `compose.sh` wrapper auto-detects Docker or Podman — no configuration needed.

Redis starts automatically when you use the `app` or `workers` profiles (see below).

### 4. Configure environment

```bash
cp .env.example .env
```

RagWeave will not start correctly without the three steps below. Everything else has a working default.

---

#### Step A — Download embedding and reranker models

The worker loads BGE models from your local filesystem — they are not bundled in the image.

```bash
uv run --with huggingface-hub huggingface-cli download BAAI/bge-m3             --local-dir ~/models/baai/bge-m3
uv run --with huggingface-hub huggingface-cli download BAAI/bge-reranker-v2-m3 --local-dir ~/models/baai/bge-reranker-v2-m3

# Tell RagWeave where they live (in .env):
RAG_MODEL_ROOT=/home/you/models
```

Each model is ~570 MB (~1.2 GB total). For alternative download methods (`git-lfs`) or path layouts (`./models` symlink), see [COLD_START_GUIDE.md §3](docs/operations/COLD_START_GUIDE.md).

> **TEI mode?** Set `RAG_INFERENCE_BACKEND=tei` to delegate embedding + reranking to the `rag-embed` / `rag-rerank` containers (always-on; TEI downloads weights into `./.tei_cache/` on first start — Step A is then optional).

---

#### Step B — Set up your LLM

**Option 1 — Containerised Ollama (default):**

The `rag-ollama` container is part of the always-on compose stack — no
host-side Ollama install. Once `make start` brings the stack up, pull the
generation model into it:

```bash
docker exec rag-ollama ollama pull qwen2.5:3b        # generation model (required)
# Optional vision model (only if you ingest images/figures):
# docker exec rag-ollama ollama pull qwen2.5vl:3b
```

The container publishes `127.0.0.1:11434` to the host, so the default
`RAG_OLLAMA_URL=http://localhost:11434` in `.env` works as-is. Models are
cached in the repo-local `./.ollama_data/` bind mount and survive
container recreation.

**Disabling generation** (e.g. retrieval-only mode, or to free GPU/RAM): stop
the container with `docker compose stop rag-ollama`. Retrieval keeps working;
generation calls fail fast with `ECONNREFUSED`. Bring it back with
`docker compose start rag-ollama`.

**Option 2 — Cloud provider (OpenRouter, OpenAI, Anthropic, etc.):**

```bash
# In .env:
RAG_LLM_MODEL=openrouter/anthropic/claude-3-haiku   # LiteLLM model string
RAG_LLM_API_BASE=https://openrouter.ai/api/v1
RAG_LLM_API_KEY=sk-or-v1-...
```

LiteLLM model strings follow the pattern `<provider>/<model-name>`. See [LiteLLM docs](https://docs.litellm.ai/docs/providers) for the full list.

---

#### Step C — (Optional) tune behaviour

These have working defaults but are worth reviewing before production use:

| Variable | Default | Notes |
|----------|---------|-------|
| `RAG_LLM_TEMPERATURE` | `0.3` | Generation temperature |
| `RAG_LLM_MAX_TOKENS` | `1024` | Max tokens per response |
| `RAG_CACHE_TTL_SECONDS` | `120` | Query result cache lifetime |
| `RAG_RATE_LIMIT_REQUESTS_PER_MINUTE` | `60` | Per-tenant rate limit |
| `RAG_MEMORY_MAX_RECENT_TURNS` | `8` | Conversation history window |
| `RAG_RETRIEVAL_TIMEOUT_MS` | `30000` | End-to-end query timeout |

See [.env.example](.env.example) for all available settings.

---

> **After changing `.env`:** most settings are read at startup. For changes to take effect in the containerised stack, run `make restart-worker` (worker config) or restart the API container. Generation model changes (e.g. switching Ollama model) only require a worker restart. Embedding or reranker model path changes require `make restart-worker` and confirming the new model files are mounted.

### 6. Run

```bash
# Ingest documents
python -m src.ingest.cli --dir ./documents

# Query locally (no server needed)
python query.py "What is RAG?"

# Or use the interactive CLI
python cli.py
```

## Running the API Server

### Option A — Local dev (fast iteration, no Docker rebuild)

```bash
make start    # Terminal 1: infrastructure + containerised workers
make dev      # Terminal 2: API server with hot-reload
make worker   # Terminal 3: Temporal worker (needed for ingestion/query workflows)
```

> **WSL2 users:** if inter-container networking is broken after a WSL2 restart, run
> `sudo ./scripts/fix-docker-networking.sh` once, or set up the automatic fix in
> `/etc/wsl.conf` — see [WSL2 Setup](#wsl2-setup) below.

### Option B — Fully containerised stack

```bash
make restart                  # start (or rebuild + restart) app + workers in containers
make restart-all              # all profiles (monitoring, gateway, etc.)
make scale-workers N=3        # scale workers horizontally
```

Then use the CLI client or web console:

```bash
# CLI client (targets the API server)
python -m server.cli_client

# User Console (chat):  open http://localhost:8000/console
# Admin Console (ops):  open http://localhost:8000/console/admin
```

### Expose publicly via Cloudflare Tunnel (no account needed)

```bash
make tunnel   # prints a public https://*.trycloudflare.com URL — kill with Ctrl+C
```

> Requires `cloudflared` system binary — see [Internet access](#internet-access-cloudflare-tunnel) for install instructions.

## Running Tests

```bash
make test                          # full suite (uv run pytest)

# Fast static gates (no test execution). Pick one depending on scope:
make precommit-check               # runs L1+L2+L3+L4+TS on git-tracked files (skips WIP)
make all-check                     # same, but over the full tree including untracked

# Individual layers:
make py-compile-check              # L1: compileall across source tree
make import-check-tracked          # L2 (tracked files only)
make import-check                  # L2 (full tree)
make dep-check                     # L3: deptry
make container-dep-check           # L4: requirements-*.txt in sync with pyproject.toml

# Targeted pytest invocations still work directly:
pytest tests/ingest/ -v            # ingestion tests only
```

> Neither `precommit-check` nor `all-check` runs the pytest suite — they're fast static gates. Run `make test` separately. **Use `precommit-check` before every `git commit`** so in-progress untracked work doesn't block your commit. **Use `all-check` before releases or as a periodic hygiene sweep** to catch issues in WIP code you haven't committed yet. L4 (`container-dep-check`) catches missing or misplaced deps across `pyproject.toml` and the two container requirements files.

## Container Profiles

Two services have no profile and start whenever compose is invoked:

| Container | Image | Size |
|-----------|-------|------|
| `rag-postgres` | `postgres:16-alpine` | 395 MB |
| `rag-minio` | `minio/minio:latest` | 241 MB |

All other services are profile-gated:

| Profile | Use Case | Key containers | Third-party images | Approx. pull |
|---------|----------|----------------|--------------------|--------------|
| `temporal` | Temporal orchestration engine + UI | `rag-temporal-db`, `rag-temporal`, `rag-temporal-ui` | `postgres:16-alpine`¹, `temporalio/auto-setup`, `temporalio/ui` | +1.24 GB |
| `app` | Containerized API server | `rag-api`², `rag-nginx`, `rag-redis`, `pg-maintenance` | `nginx:alpine`, `redis:7-alpine`, `postgres:16-alpine`¹ | +544 MB |
| `workers` | Containerized ingest/query workers | `rag-worker`² | `redis:7-alpine`¹ | +5.79 GB |
| `monitoring` | Prometheus + Grafana + Dozzle | `rag-prometheus`, `rag-alertmanager`, `rag-grafana`, `rag-monitor` | `prom/prometheus`, `prom/alertmanager`, `grafana/grafana`, `amir20/dozzle` | +1.74 GB |
| `observability` | Langfuse LLM tracing (full stack) | `rag-langfuse-*` (6 containers) | `postgres:17`, `redis:7`, `clickhouse/clickhouse-server`, `cgr.dev/chainguard/minio`, `langfuse/langfuse-worker:3`, `langfuse/langfuse:3` | +5.93 GB |
| `gateway` | nginx HTTPS reverse proxy | `rag-nginx` | `nginx:alpine`¹ | +94 MB |

¹ Shared image — no additional pull if already present from another profile.  
² Custom-built locally — see [Container Images](#container-images).

```bash
# Example: full production stack with monitoring
./scripts/compose.sh --profile temporal --profile app --profile workers --profile monitoring up -d
```

## Container Images

The stack uses two images with strict dependency isolation:

| Image | Size | Contents |
|---|---|---|
| `rag-api` | 389 MB | FastAPI, Temporal client, Weaviate client — no torch, no docling, no ML stack |
| `rag-worker` | 5.79 GB | Full ML stack (torch, sentence-transformers, docling, langchain, nemoguardrails) |

Dependencies live in `containers/requirements-api.txt` and `containers/requirements-worker.txt` — **not** in `pyproject.toml`. This is deliberate: `pip install .` would install every dep listed under `[project.dependencies]`, which undoes the isolation. Local dev still uses `pyproject.toml` via `make install`; containers bypass it.

**Adding a new dependency:**
- If the API server imports it → add to `pyproject.toml` AND `containers/requirements-api.txt`
- If only the worker uses it → add to `pyproject.toml` AND `containers/requirements-worker.txt`
- Dev-only (pytest, deptry, etc.) → `pyproject.toml` only

### Build the images

**With make (recommended):**

```bash
make container-build          # build both with docker (BuildKit)
make container-build-podman   # build both with podman (preferred for production)
make container-probe          # run API import probe — catches transitive ML leakage
make container-sizes          # show current image sizes
make container-clean          # remove local rag-api / rag-worker images
```

**Manual Docker:**

```bash
DOCKER_BUILDKIT=1 docker build -t rag-api    -f containers/Dockerfile.api .
DOCKER_BUILDKIT=1 docker build -t rag-worker -f containers/Dockerfile.runtime .
```

**Manual Podman:**

```bash
# --format docker preserves HEALTHCHECK directives if ever re-added to the Dockerfile
podman build --format docker -t rag-api    -f containers/Dockerfile.api .
podman build --format docker -t rag-worker -f containers/Dockerfile.runtime .
```

### Architecture notes

Multi-stage builds, BuildKit pip-cache mounts, `.dockerignore`, compose-level healthchecks (podman-friendly), `PYTHONPATH=/app` (no `pip install .`), and GPU support in the worker image — see [`docs/operations/DOCKER_OPTIMIZATION.md`](docs/operations/DOCKER_OPTIMIZATION.md) for the full design and 9-iteration optimization history.

## HTTPS Gateway (nginx)

Add HTTPS support in front of the API server using nginx:

### One-time setup

```bash
# 1. Install mkcert
sudo apt install mkcert          # Debian/Ubuntu
# brew install mkcert             # macOS

# 2. Generate locally-trusted certs
./scripts/generate-certs.sh

# 3. Add hostname to /etc/hosts
echo "127.0.0.1  aion.local" | sudo tee -a /etc/hosts
```

### Start with HTTPS

```bash
./scripts/compose.sh --profile temporal --profile app --profile gateway up -d
# Browse: https://aion.local
```

The `gateway` profile requires the `app` profile. See `certs/README.md` for details.

> **Security note:** When the gateway is active, port 8000 remains directly accessible (bypassing TLS). For LAN demos, set `RAG_API_HOST_PORT=127.0.0.1:8000` in `.env` to restrict direct access to localhost only.

### Internet access (Cloudflare Tunnel)

For demos on a different network, use [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/) (free, no account needed) to get a public HTTPS URL.

Install `cloudflared` once (it's a system binary, not in default Ubuntu repos) — see [COLD_START_GUIDE.md §0.5](docs/operations/COLD_START_GUIDE.md) for the apt-repo install commands. Then:

```bash
make tunnel                                                        # tunnel local dev API (port 8000)
cloudflared tunnel --url https://localhost:443 --no-tls-verify     # or tunnel the nginx gateway
```

Prints a `https://*.trycloudflare.com` URL. Kill with Ctrl+C when done.

## WSL2 Setup

Docker bridge networking on WSL2 requires a one-time fix per WSL2 session (iptables FORWARD rules are reset when WSL2 restarts). To make it automatic, add a boot command to `/etc/wsl.conf`:

```ini
# /etc/wsl.conf  (create if it doesn't exist)
[boot]
command = "service docker start && iptables -P FORWARD ACCEPT"
```

After saving, restart WSL2 from PowerShell:

```powershell
wsl --shutdown
```

From that point on, Docker inter-container networking will work automatically on every WSL2 startup. No manual steps needed when cloning the repo on a new WSL2 machine.

**Manual fix (current session only):**

```bash
sudo ./scripts/fix-docker-networking.sh
```

The script is WSL2-aware — it no-ops on Linux and macOS.

## Podman Setup

Podman is supported as a rootless, daemonless alternative to Docker. One-time setup:

```bash
# 1. Install Podman
sudo apt-get install -y podman podman-compose   # Debian/Ubuntu

# 2. Enable user socket (needed for Dozzle log viewer)
systemctl --user enable --now podman.socket

# 3. Verify rootless mode
podman info | grep -i rootless   # should show: rootless: true

# 4. Set the container socket in .env
echo "CONTAINER_SOCK=\$XDG_RUNTIME_DIR/podman/podman.sock" >> .env

# 5. Use compose.sh as normal — it auto-detects Podman
./scripts/compose.sh --profile temporal --profile app up -d
```

For internal design notes (rootless networking, socket detection, image-format trade-offs), see [`docs/operations/PODMAN_SPEC.md`](docs/operations/PODMAN_SPEC.md) — that doc is the implementation spec, not the setup guide. The five steps above are the setup.


## Entry Points

| Command | Description |
|---------|-------------|
| `python ingest.py --dir ./documents` | CLI for ingestion runs |
| `python query.py "question"` | Local retrieval query CLI |
| `python cli.py [query\|ingest]` | Unified interactive CLI |
| `python -m server.worker` | Temporal worker process |
| `uvicorn server.api:app --host 0.0.0.0 --port 8000` | API server |
| `python -m server.cli_client` | Interactive client targeting the API server |
| `python -m server.mcp_adapter` | MCP tooling adapter over the API (`stdio` transport) |

## Make Targets

Run `make help` for this list in the terminal. All targets are also documented in comments in the [Makefile](Makefile) itself.

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

## License

See [LICENSE](LICENSE) for details.
