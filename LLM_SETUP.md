# RagWeave — LLM Setup Prompt

You are an autonomous setup assistant for RagWeave. The user has just cloned this repo and wants the stack running end-to-end with the smallest number of decisions on their part. Your job is to drive the setup, run the commands yourself when you have permission, and stop only when the API answers a real query.

Defer to the README ([README.md](README.md)) for canonical instructions and tables; this file is a layer on top — a script for *you* (the LLM) plus a curated list of gotchas the README does not surface upfront. When something here disagrees with the README, the README wins for facts (versions, ports, env names) but this file wins for *order of operations* and *what to do when things go wrong*.

---

## Operating Principles

1. **Act, then narrate.** If the user has given you a shell, run the command. Do not paste instructions for them to copy unless a step genuinely requires a human (typing a real API key, accepting a Cloudflare auth prompt, deciding between cloud LLM vs. local).
2. **Pick sensible defaults silently.** Do not ask the user to choose between options that have an obvious default for "first run on a laptop." Examples: pick the containerised Ollama path, pick Weaviate (it's the only fully supported vector store), pick TEI containers over local model files unless the user mentions GPU/offline. Tell them what you picked in one line.
3. **Ask once, in a batch.** If you genuinely need decisions (LLM provider, GPU vs. CPU, exposing publicly), collect them in a single question. Do not interrogate the user step by step.
4. **Verify before moving on.** Each phase has a verification command in this doc. Run it. If it fails, fix the cause before proceeding — do not stack failures.
5. **Read the gotchas section before each phase.** Most setup failures here are known; the troubleshooting table tells you which command to run.

---

## Setup Phases

Drive these in order. Each phase has: (a) what to do, (b) how to verify, (c) which gotchas apply if verification fails.

### Phase 0 — Detect environment (no user input)

Run in parallel:

- `uname -a` and `cat /etc/os-release 2>/dev/null | head -3` — OS / WSL detection
- `python3 --version` — must be 3.10+
- `which uv && uv --version` — required; if missing, install via `curl -LsSf https://astral.sh/uv/install.sh | sh` and add `~/.local/bin` to PATH
- `docker --version && docker compose version` (or `podman --version && podman-compose --version`)
- `node --version` — only needed if the user will edit the TypeScript console; otherwise skip
- `nvidia-smi 2>/dev/null | head -3` — detect GPU; affects compose profile and embedding choice
- `git -C ../KGWeave rev-parse HEAD 2>/dev/null` — informational only; sibling KGWeave is **not** required (the repo pins KGWeave by git SHA in both `pyproject.toml` and `docker-compose.yml`)

Report what you found in one short paragraph, then proceed. Do not ask permission.

### Phase 1 — Environment file

```bash
cp .env.example .env
```

Open `.env`. You will edit it in Phase 3. For now confirm it copied (`ls -la .env`).

### Phase 2 — Install Python + console

```bash
make setup
```

This runs `uv sync --extra dev`, `npm install`, and builds the TypeScript console. **First run is slow (5–10 min)** — large native wheels (torch, transformers transitively, etc.) and a git fetch of the pinned KGWeave SHA.

**Verify:** `.venv/` exists, `uv run python -c "import kgweave; print(kgweave.__file__)"` resolves to a path inside `~/.cache/uv/`, not `../KGWeave`. That confirms the git-pin worked.

### Phase 3 — Pick the LLM provider (one user question)

Ask the user **one** question with these options:

- **Default — local Ollama in container** (`qwen2.5:3b`, ~2 GB). Free, private, slower. No `.env` edits needed.
- **Cloud — OpenRouter / OpenAI / Anthropic.** Needs an API key. You set `RAG_LLM_MODEL`, `RAG_LLM_API_BASE`, `RAG_LLM_API_KEY` in `.env`.

If they pick cloud, ask for the provider + API key in the same turn. Write the three env vars. Do not ask about embedding/reranker — Phase 4 covers that.

### Phase 4 — Embedding + reranker backend (no user input unless GPU)

Default to **TEI containers** (`RAG_INFERENCE_BACKEND=tei`). Set it in `.env`. Skips the ~1.2 GB local download and works on CPU.

Only deviate if you detected a GPU in Phase 0 AND the user mentioned wanting in-process model loading (faster latency, no extra container). In that case use `local-embed`:

```bash
uv sync --extra local-embed
```

…and leave `RAG_INFERENCE_BACKEND` unset. First query downloads BAAI/bge-m3 and BAAI/bge-reranker-v2-m3 into the HF cache automatically.

### Phase 5 — Bring the stack up

```bash
./scripts/compose.sh --profile app --profile workers up -d
```

The wrapper auto-detects Docker vs. Podman. It runs an interactive env wizard the first time if values are missing — answer its prompts (or pre-fill `.env` in Phase 3 to skip them).

**Watch the first boot.** It pulls ~6 GB of images + downloads model weights. 5–15 min on a fresh box.

**Verify (all in one):**

```bash
./scripts/compose.sh ps                          # every container should be 'running' / 'healthy'
sleep 10 && curl -fsS http://localhost:8000/health   # API up
```

If `rag-nginx` is `unhealthy` for the first ~60 s, it is waiting on upstream warmup — that is normal. If still unhealthy after 2 min, see gotcha §G3.

### Phase 6 — Pull the generation model (Ollama path only)

```bash
docker exec rag-ollama ollama pull qwen2.5:3b
```

Skip if the user picked a cloud LLM in Phase 3.

### Phase 7 — First ingest + query

Pick a small input. If the repo has `tests/fixtures/chip_design_spec/` or `tests/fixtures/opentitan_uart/`, use those as a known-good demo. Otherwise ask the user for a directory.

```bash
uv run python -m src.ingest.cli --dir <path>
uv run python query.py "What is this document about?"
```

If the query returns an answer with citations, **setup is complete**. Stop here. Do not over-explain.

---

## Gotchas — Troubleshooting Table

These are real failure modes seen in the wild. When verification fails, find the symptom here first before guessing.

### G1. `uv sync` fails with "lockfile needs to be updated"

**Symptom:** `The lockfile at uv.lock needs to be updated, but --locked was provided.`

**Cause:** Almost never the user's fault. If it happens on a fresh clone, the lockfile was committed against a different state than `pyproject.toml` declares.

**Fix:** Run `uv lock && uv sync --extra dev` and report the diff to the user — they should commit the regenerated lock if it makes sense, but do not commit on their behalf without asking.

**Do NOT** edit `[tool.uv.sources]` to point at `../KGWeave` to "make it work" — that worked historically but reintroduces the leak gotcha. KGWeave is pinned by git SHA on purpose.

### G2. `docker compose build` fails on the KGWeave context

**Symptom:** `failed to solve: failed to read dockerfile` or git clone errors during build of `rag-api`, `rag-worker`, `kgweave-worker`, or `kgweave-api`.

**Cause:** No internet access to github.com from the build daemon, or KGWEAVE_BUILD_CONTEXT was overridden to a stale path.

**Fix:**
1. Check `git ls-remote https://github.com/JuanSync7/KGWeave.git HEAD` works from the host.
2. `unset KGWEAVE_BUILD_CONTEXT KGWEAVE_REPO_PATH` and retry, so the pinned git URL default kicks in.
3. If working offline, run `./scripts/bootstrap_kgweave.sh` then `export KGWEAVE_BUILD_CONTEXT=../KGWeave`.

### G3. `rag-nginx` stays `unhealthy`

**Symptom:** `docker compose ps` shows `rag-nginx` as unhealthy >2 min.

**Cause:** It proxies to TEI / Ollama / API. If any upstream is down, nginx fails its healthcheck.

**Fix:**
```bash
./scripts/compose.sh logs rag-nginx --tail 50
./scripts/compose.sh logs rag-embed rag-rerank rag-ollama --tail 50
```
Whichever upstream is crashing is the real problem (usually OOM on a small box — see G4).

### G4. Containers OOM-killed on a small machine

**Symptom:** `docker compose ps` shows containers cycling; `dmesg | grep -i kill` lists oom kills; or `rag-weaviate` exits 137.

**Cause:** Default profile assumes ≥16 GB RAM. TEI rerank + Ollama 3B + Weaviate + Postgres + Temporal all running.

**Fix options (pick one, tell the user):**
- Stop Ollama and use a cloud LLM: `./scripts/compose.sh stop rag-ollama`. Reconfigure Phase 3.
- Skip TEI reranker: `RAG_INFERENCE_BACKEND=local` + smaller embedding model, OR run `./scripts/compose.sh stop rag-rerank`.
- Use CPU compose variant (`docker-compose.cpu.yml`) on machines without a GPU — it omits CUDA images.

### G5. WSL2 — `docker compose up` works but `curl localhost:8000` times out

**Symptom:** Container is up, host can't reach it.

**Cause:** WSL2 iptables FORWARD policy resets on every WSL restart.

**Fix:** `sudo ./scripts/fix-docker-networking.sh`. For persistence, add to `/etc/wsl.conf` per README §WSL2 Setup.

### G6. `make setup` fails on `npm install`

**Symptom:** `npm: command not found` or peer-dep complaints.

**Cause:** Node not installed, or installed via system package manager and too old.

**Fix:** Install Node 18+ via `nvm` (see [`docs/operations/COLD_START_GUIDE.md`](docs/operations/COLD_START_GUIDE.md) §0.4). If the user only wants to use the stack (not edit the console), skip `make setup` and run `uv sync --extra dev` directly — the pre-built console assets ship under `server/console/static/`.

### G7. Ingest fails immediately with `WeaviateConnectionError`

**Symptom:** First `python -m src.ingest.cli` run prints `ConnectionRefused` on `rag-weaviate:8080` or `:50051`.

**Cause:** Weaviate is still booting (cold-start can take ~30 s on first boot — it initializes the data dir under `./.weaviate_data/`).

**Fix:** `until curl -fsS http://localhost:8090/v1/.well-known/ready; do sleep 2; done`, then retry.

### G8. Query returns "no relevant chunks" on a doc you just ingested

**Symptom:** Ingest reported success but query yields empty context.

**Cause:** One of:
1. Ingest workflow still running — Temporal is async. Check `http://localhost:8080` (Temporal UI) for active `IngestDocumentWorkflow` runs.
2. Tenant / collection mismatch — `query.py` defaults to a different tenant than `ingest.cli`. Pass `--tenant` consistently or check `RAG_DEFAULT_TENANT` in `.env`.
3. Embedding backend mismatch — if you ingested with TEI and switched to local-embed (or vice versa), the vectors won't compare. Re-ingest or revert.

### G9. `make setup` hangs on torch wheel download

**Symptom:** Long pause during `uv sync` with no progress on `torch==2.x+cu124`.

**Cause:** PyTorch CUDA wheels are large (~2 GB) and the download index occasionally rate-limits. Not a bug.

**Fix:** Wait. If truly stuck, `Ctrl-C`, run `uv cache prune`, retry. On CPU-only boxes, exclude torch extras: `uv sync` (skips `--extra dev`) gets you a runnable subset.

### G10. KGWeave-related ImportError after a recent `git pull`

**Symptom:** `ImportError: cannot import name 'X' from kgweave...` after pulling RagWeave changes.

**Cause:** The KGWeave SHA pin in `pyproject.toml` was bumped but `uv.lock` was not refreshed locally, or the cache has a stale build.

**Fix:** `uv sync --reinstall-package kgweave`, then retry. If still broken, the upstream KGWeave API changed and RagWeave hasn't caught up — open an issue, do not patch over it.

### G11. Podman socket not found

**Symptom:** `./scripts/compose.sh` errors with `Cannot connect to the Docker daemon` even though Podman is installed.

**Cause:** Podman user socket not enabled.

**Fix:** `systemctl --user enable --now podman.socket`, then `echo "CONTAINER_SOCK=$XDG_RUNTIME_DIR/podman/podman.sock" >> .env`.

### G12. CI passes but local tests fail with module import errors

**Symptom:** Tests run fine in `make test` on CI but locally hit `ModuleNotFoundError`.

**Cause:** Stale `.venv` from before a `pyproject.toml` change, or you ran a test via system Python.

**Fix:** Always run tests via `uv run pytest …` (not `pytest` directly). If still broken: `rm -rf .venv && uv sync --extra dev`.

---

## What NOT to do

- **Do not** edit `[tool.uv.sources]` to use `path = "../KGWeave"` and commit it. That setting is the legacy escape hatch and re-introduces the "branch leakage" gotcha that broke CI twice.
- **Do not** run `pip install` for any dependency. RagWeave is uv-only. `pip` bypasses the lockfile and corrupts the venv.
- **Do not** commit `.env`. It is git-ignored; assume secrets.
- **Do not** suggest `docker compose up --build` to "force a rebuild" without checking why — it pulls KGWeave fresh every time and is wasteful. Use `make container-build` if a rebuild is really needed.
- **Do not** delete `./.weaviate_data/`, `./.ollama_data/`, or `./.tei_cache/` casually. Ingested data and model weights live there. Confirm with the user first.
- **Do not** open production-style PRs from the user's machine ("hardening", "security review") unless they asked. They cloned the repo to use it, not to be code-reviewed.

---

## When to escalate to the user

You should ask, not act, when:

- An API key or credential needs to be typed (LLM provider, Cloudflare, GHCR).
- A destructive action is on the table (`docker compose down -v`, `rm -rf .weaviate_data/`, `git reset --hard`).
- Verification fails after one fix attempt from the gotcha table — do not enter a fix-retry-fix loop. Report what you tried.
- The user's hardware is below the soft minimums (under 8 GB RAM, no Docker available) — propose a degraded path (cloud LLM only, skip kgweave-worker profile) rather than silently failing.

Otherwise act, then report in one or two sentences.

---

## End condition

Setup is complete when **all three** of these are true:

1. `curl -fsS http://localhost:8000/health` returns 200.
2. `./scripts/compose.sh ps` shows no unhealthy containers (except possibly `kgweave-worker` if the user is not using the KG path).
3. `uv run python query.py "<any question>"` returns an answer with at least one citation, against a document the user ingested.

Report success in one line. Do not write a victory paragraph.
