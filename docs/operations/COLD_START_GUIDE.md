<!-- @summary
End-to-end cold-start setup: from a freshly-installed Linux box to a working
RagWeave instance reachable over Cloudflare Tunnel. Documents prereq install
commands, every manual step, and the gaps the top-level README glosses over.
@end-summary -->

# Cold-Start Setup Guide

> **Audience.** Someone who just `git clone`'d this repo onto a fresh Linux box
> (Ubuntu 22.04 / 24.04, with WSL2 callouts where relevant) and wants the API
> running locally and exposed via Cloudflare Tunnel.
>
> The top-level [README](../../README.md) assumes most prerequisites are
> present; this doc assumes nothing.

Companion script: [`scripts/cold_start_setup.sh`](../../scripts/cold_start_setup.sh)
codifies the working path as an automated verifier (`bash scripts/cold_start_setup.sh`).

---

## What's tracked vs. what you install

The repo tracks **source code** and **Python packages**
(`pyproject.toml` + `uv.lock`, plus `containers/requirements-*.txt` for
container builds). It does **not** ship the system-level tools they depend
on — those must exist on the host before `uv sync` or `docker build` can run. The full prereq list:

| Tool | Why it's needed | Where it lives |
|---|---|---|
| `git`, `curl`, `make`, `build-essential` | basic toolchain | apt |
| `python3` ≥ 3.10 | runtime + venv host | apt |
| `uv` | installs Python deps from `pyproject.toml` | https://astral.sh/uv |
| `docker` + Compose v2 | runs the rag-* container stack | https://get.docker.com |
| Node.js 20 + npm | builds the TypeScript web console | NodeSource apt or nvm |
| `cloudflared` | optional — `make tunnel` exposes the API publicly | Cloudflare apt repo |

**Notably *not* in this list:** Ollama. It ships as the `rag-ollama` container
in `docker-compose.yml`. **Do not install host-side Ollama** — the stack is
container-only by design.

---

## 0. Install prerequisites (Ubuntu/Debian)

### 0.1 System packages

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential ca-certificates curl git make \
    python3 python3-venv python3-pip
```

### 0.2 `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"   # add this line to ~/.bashrc and ~/.zshrc
uv --version
```

### 0.3 Node.js 20 + npm

> **🟡 README gap — PATH propagation.** If you install Node via `nvm`, its
> shims are usually only sourced by interactive shells (zsh/bash login). But
> `make setup` runs `npm` from a non-login `sh` subshell, where nvm shims
> are **not** present. Result: `make setup` fails at `console-install` with
> `npm: command not found` even though `npm --version` works in your shell.

Pick **one** of:

**(A) System Node — recommended for setup.** Real `/usr/bin/node`, available
to every subshell:

```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs
node --version    # v20.x
```

**(B) `nvm` — but make PATH leak to non-interactive shells:**

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
source "$HOME/.nvm/nvm.sh"
nvm install --lts
# Append to ~/.profile (NOT just ~/.bashrc / ~/.zshrc) so make's sh subshell sees it:
echo 'export PATH="$HOME/.nvm/versions/node/$(nvm version)/bin:$PATH"' >> ~/.profile
```

Verify it works in a `sh` subshell (this is what `make` uses):

```bash
sh -c 'command -v npm && npm --version'   # must print a path + version
```

### 0.4 Docker + Compose v2

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"
newgrp docker
docker --version
docker compose version    # v2.x — the legacy `docker-compose` v1 is not enough
```

WSL2 + Windows: install **Docker Desktop** instead and enable WSL2 integration
in Settings → Resources → WSL Integration.

### 0.5 cloudflared (optional — only if you'll use `make tunnel`)

> **🟡 README gap.** README says `sudo apt install cloudflared`, but the
> package is **not in the default Ubuntu/Debian repos**. Pick one of:

**(A) Cloudflare apt repo — preferred (signed updates):**

```bash
sudo mkdir -p --mode=0755 /usr/share/keyrings
curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg \
    | sudo tee /usr/share/keyrings/cloudflare-main.gpg >/dev/null

echo 'deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared any main' \
    | sudo tee /etc/apt/sources.list.d/cloudflared.list

sudo apt-get update
sudo apt-get install -y cloudflared
```

**(B) Direct `.deb` download:**

```bash
ARCH=$(dpkg --print-architecture)        # amd64 or arm64
curl -L --output /tmp/cloudflared.deb \
    "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-${ARCH}.deb"
sudo dpkg -i /tmp/cloudflared.deb
rm /tmp/cloudflared.deb
```

The `.deb` is **not** tracked in the repo — it's a 30 MB versioned binary
maintained upstream. Use the apt repo for automatic updates.

---

## 1. Clone and run setup

```bash
git clone <repo-url> RagWeave        # README §1 says `cd RAG`; the actual dir is RagWeave
cd RagWeave
make setup
```

`make setup` does, in order:

1. `uv sync --extra dev` — auto-creates `.venv/` and installs Python deps from `uv.lock`
2. `npm --prefix server/console/web install` — frontend deps
3. `npm --prefix server/console/web run build` — compile TS → `static/main.js`

**Optional Python extras** (if you need them):

```bash
uv sync --extra pii       # Presidio + spaCy
uv sync --extra gliner    # GLiNER NER
uv sync --extra all       # everything
```

> **Always use `uv`, never plain `pip`.** The repo has a `uv.lock` for
> reproducible installs; plain `pip` ignores it and can corrupt resolution
> in a uv-managed venv. `uv sync` is preferred for project deps;
> `uv pip install <pkg>` only for ad-hoc packages not declared in
> `pyproject.toml`; `uv run <cmd>` for one-off scripts without activating
> the venv.

---

## 2. Configure `.env`

```bash
cp .env.example .env
```

The defaults work as-is for local dev with the bundled container stack. The
only knobs you might need:

| Variable | Change when… | Set to |
|---|---|---|
| `RAG_MODEL_ROOT` | You keep BGE models outside the repo | absolute path to parent dir of `baai/` |
| `RAG_LLM_MODEL` | You want a cloud LLM instead of container Ollama | e.g. `openrouter/anthropic/claude-3-haiku` |
| `RAG_INFERENCE_BACKEND` | You want TEI containers instead of in-process BGE | `tei` |

`RAG_OLLAMA_URL` and `RAG_LLM_API_BASE` default to `http://localhost:11434`
and reach the `rag-ollama` container via its host port publish
(`127.0.0.1:11434:11434` in `docker-compose.yml`). Leave them alone unless
you're routing through a different LLM provider.

---

## 3. Download embedding & reranker models (BGE)

Skip this if `RAG_INFERENCE_BACKEND=tei` — the `rag-embed` and `rag-rerank`
containers download BGE weights into `./.tei_cache/` on first start.

For the default `local` backend (BGE in-process inside the venv):

```bash
uv pip install huggingface-hub             # ad-hoc install — not a pyproject extra
mkdir -p ~/models/baai
uv run huggingface-cli download BAAI/bge-m3             --local-dir ~/models/baai/bge-m3
uv run huggingface-cli download BAAI/bge-reranker-v2-m3 --local-dir ~/models/baai/bge-reranker-v2-m3

# Either set in .env:
echo "RAG_MODEL_ROOT=$HOME/models" >> .env
# Or symlink into the repo (default RAG_MODEL_ROOT=./models then resolves):
ln -s ~/models models
```

Each model is ~570 MB; ~1.2 GB total.

---

## 4. Bring up the stack

`rag-ollama`, `rag-embed`, and `rag-rerank` are always-on (no profile).
Temporal lives under its own profile. Bring them all up:

```bash
./scripts/compose.sh --profile temporal up -d   # Temporal + UI
make start                                       # base (incl. ollama + TEI) + workers
```

> **🟡 README gap — `make start` doesn't include `--profile temporal`.**
> Without it, Temporal workflows are unavailable. Run the temporal-profile
> compose command above before `make start`.

### 4.1 Pull the generation model into the container

```bash
docker exec rag-ollama ollama pull qwen2.5:3b
# Optional vision model (only if you ingest images/figures):
docker exec rag-ollama ollama pull qwen2.5vl:3b
```

Models live in the repo-local bind mount (`./.ollama_data`) so they survive
container recreation but stay out of git (`.gitignore`d).

### 4.2 Verify

```bash
docker ps --format 'table {{.Names}}\t{{.Status}}' | grep '^rag-'
curl -sf http://localhost:11434/api/tags | head -c 200    # Ollama via container
curl -sf http://localhost:8080/                            # Temporal UI
```

---

## 5. Run the API and worker

Three terminals (or three `tmux` panes):

```bash
# Terminal 1 — API server (hot-reload)
make dev

# Terminal 2 — Temporal worker (required for query/ingest workflows)
make worker

# Terminal 3 — Cloudflare Tunnel (public HTTPS) — optional
make tunnel
```

`make tunnel` runs `cloudflared tunnel --url http://localhost:8000` and
prints a `https://<random>.trycloudflare.com` URL. Kill with Ctrl+C.

### 5.1 Smoke test

```bash
curl -sf http://localhost:8000/health
python query.py "What is RAG?"      # exercises retrieval + generation
```

---

## 6. Common cold-start failures

| Symptom | Cause | Fix |
|---|---|---|
| `make setup` fails at `npm install` with `npm: command not found` | nvm shims not on PATH for `sh` subshells | §0.3 Option A or fix nvm PATH for non-interactive shells |
| `make tunnel`: `cloudflared: command not found` after `apt install cloudflared` | Package not in default Ubuntu repos | §0.5 |
| `make dev` errors with `ECONNREFUSED localhost:11434` | `rag-ollama` stopped or port not published | `docker compose up -d rag-ollama`; verify `docker port rag-ollama` shows `127.0.0.1:11434` |
| Worker errors with Temporal connection refused | `make start` doesn't include Temporal profile | `./scripts/compose.sh --profile temporal up -d` |
| API up but `/health` reports BGE missing | `RAG_MODEL_ROOT` unset / models not downloaded | §3 |
| Inter-container DNS broken on WSL2 after restart | iptables FORWARD rules reset | `sudo ./scripts/fix-docker-networking.sh` or `/etc/wsl.conf` boot fix (README §WSL2 Setup) |
| `python query.py` import errors | venv not active and `uv run` not used | `source .venv/bin/activate` or `uv run python query.py …` |

---

## 7. Reproduce automatically

```bash
bash scripts/cold_start_setup.sh                 # check + setup + smoke (default)
bash scripts/cold_start_setup.sh --check-only    # validate prereqs only, no mutations
bash scripts/cold_start_setup.sh --no-smoke      # check + setup, skip smoke test
```

The script logs every command before running it and exits non-zero on the
first hard failure.
