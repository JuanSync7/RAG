#!/usr/bin/env bash
# cold_start_setup.sh — codified cold-start path for RagWeave.
#
# Companion to docs/operations/COLD_START_GUIDE.md. Walks the same steps a
# new user would after `git clone`: verifies prereqs, configures .env, brings
# up infra, pulls models into the rag-ollama container, and smoke-tests the
# API. Designed to run cleanly on a fresh Ubuntu 22.04/24.04 box (WSL2 OK).
#
# Modes (all separate phases — pick one):
#   --check     Read-only: validate prereqs + repo state. No mutations.
#   --run       Idempotent setup: skip already-done steps, do the rest.
#   --smoke     Skip setup; run the end-to-end smoke test against a live API.
#   --all       Full pipeline: --check, then --run, then --smoke. Stops on
#               the first hard failure but always prints the gap list.
#   -h|--help   This help.
#
# Always uv, never plain pip. Echoes every command before running it.

set -u   # no -e: full per-step reporting wins over fail-fast

MODE="${1:---all}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ── Output helpers ─────────────────────────────────────────────────────
if [[ -t 1 ]]; then
    G=$'\e[32m'; R=$'\e[31m'; Y=$'\e[33m'; B=$'\e[34m'; D=$'\e[2m'; N=$'\e[0m'
else
    G=""; R=""; Y=""; B=""; D=""; N=""
fi

PASS=0; FAIL=0; WARN=0
GAPS=()

step()  { printf '\n%s── %s%s\n' "$B" "$1" "$N"; }
ok()    { printf '  %s✓%s %s\n' "$G" "$N" "$1"; PASS=$((PASS+1)); }
bad()   { printf '  %s✗%s %s\n' "$R" "$N" "$1"; FAIL=$((FAIL+1)); GAPS+=("FAIL: $1"); }
warn()  { printf '  %s⚠%s %s\n' "$Y" "$N" "$1"; WARN=$((WARN+1)); GAPS+=("WARN: $1"); }
info()  { printf '  %s%s%s\n' "$D" "$1" "$N"; }
run()   { printf '  %s$%s %s\n' "$D" "$N" "$*"; "$@"; }

# ─────────────────────────────────────────────────────────────────────
# Phase: CHECK — prereq + repo validation. No mutations.
# ─────────────────────────────────────────────────────────────────────
do_check() {
    step "1. Prerequisite binaries"
    local cmd
    for cmd in git curl make python3 uv docker ollama cloudflared; do
        if command -v "$cmd" >/dev/null 2>&1; then
            local v; v=$("$cmd" --version 2>/dev/null | head -n1 || echo '?')
            ok "$cmd  $D[$v]$N"
        else
            case "$cmd" in
                cloudflared) bad "$cmd missing — see COLD_START_GUIDE.md §0.5 (apt repo or .deb)";;
                ollama)      info "$cmd missing on host — that's fine, the stack uses rag-ollama container";;
                uv)          bad "$cmd missing — install: curl -LsSf https://astral.sh/uv/install.sh | sh";;
                *)           bad "$cmd missing — see COLD_START_GUIDE.md §0";;
            esac
        fi
    done

    step "2. Make-compatible toolchain (sh subshell)"
    # `make` runs recipes in /bin/sh, not your login shell. nvm shims
    # often only load in interactive zsh/bash, so make can't see them.
    local sub
    for sub in npm node uv; do
        if sh -c "command -v $sub" >/dev/null 2>&1; then
            ok "$sub reachable from sh subshell (make-compatible)"
        else
            bad "$sub NOT reachable from sh subshell — make targets will fail. See guide §0.3."
        fi
    done

    step "3. Compose v2"
    if docker compose version >/dev/null 2>&1; then
        ok "docker compose v2 present  $D[$(docker compose version | head -1)]$N"
    else
        bad "docker compose v2 missing (legacy v1 docker-compose is not enough)"
    fi

    step "4. Repo state"
    [[ -f pyproject.toml ]]     && ok "pyproject.toml present"     || bad "not at repo root"
    [[ -f uv.lock ]]            && ok "uv.lock present"            || warn "uv.lock missing — uv sync will resolve from scratch"
    [[ -f Makefile ]]           && ok "Makefile present"           || bad "Makefile missing"
    [[ -f docker-compose.yml ]] && ok "docker-compose.yml present" || bad "docker-compose.yml missing"
    [[ -f .env.example ]]       && ok ".env.example present"       || bad ".env.example missing"
}

# ─────────────────────────────────────────────────────────────────────
# Phase: RUN — idempotent setup. Skip what's already done.
# ─────────────────────────────────────────────────────────────────────
do_run() {
    step "5. Python venv + deps (idempotent, uv sync)"
    if [[ -d .venv && -x .venv/bin/python3 ]]; then
        ok ".venv already present"
    else
        run uv venv || { bad "uv venv failed"; return 1; }
        ok ".venv created"
    fi
    if .venv/bin/python -c 'import fastapi' 2>/dev/null; then
        ok "Python deps already installed (fastapi importable)"
    else
        info "Installing project deps via uv sync --extra dev (~2-5 min on first run)"
        if run uv sync --extra dev; then
            ok "Python deps installed"
        else
            warn "uv sync failed — falling back to uv pip install -e .[dev]"
            run uv pip install -e ".[dev]" || { bad "Python dep install failed"; return 1; }
            ok "Python deps installed (fallback)"
        fi
    fi

    step "6. Web console"
    if [[ -d server/console/web/node_modules ]]; then
        ok "node_modules present"
    else
        run npm --prefix server/console/web install || { bad "npm install failed"; return 1; }
        ok "npm install done"
    fi
    if [[ -f static/main.js || -f server/console/web/static/main.js ]]; then
        ok "console build artefact present"
    else
        run npm --prefix server/console/web run build || { bad "console build failed"; return 1; }
        ok "console built"
    fi

    step "7. .env"
    if [[ -f .env ]]; then
        ok ".env already present — not overwriting"
    else
        run cp .env.example .env || { bad ".env copy failed"; return 1; }
        ok ".env created from .env.example"
    fi

    step "8. Container stack (compose up — base + workers + temporal)"
    run ./scripts/compose.sh up -d                       || warn "compose base up returned non-zero (rag-ollama, rag-embed, rag-rerank are always-on)"
    run ./scripts/compose.sh --profile workers  up -d    || warn "compose workers up returned non-zero"
    run ./scripts/compose.sh --profile temporal up -d    || warn "compose temporal up returned non-zero"

    step "9. Pull qwen2.5:3b into rag-ollama (~2 GB, skip if cached)"
    if docker exec rag-ollama ollama list 2>/dev/null | grep -q '^qwen2.5:3b'; then
        ok "qwen2.5:3b already pulled into rag-ollama"
    else
        info "Pulling qwen2.5:3b into rag-ollama — first run takes several minutes"
        run docker exec rag-ollama ollama pull qwen2.5:3b || warn "ollama pull failed inside rag-ollama"
    fi

    step "10. BGE model presence (local backend only; tei pulls into rag-embed/rag-rerank automatically)"
    local backend; backend="$(grep -E '^RAG_INFERENCE_BACKEND=' .env 2>/dev/null | cut -d= -f2)"
    backend="${backend:-local}"
    if [[ "$backend" == "tei" ]]; then
        info "RAG_INFERENCE_BACKEND=tei — BGE downloaded by rag-embed/rag-rerank on first start"
    else
        local root; root="$(grep -E '^RAG_MODEL_ROOT=' .env 2>/dev/null | cut -d= -f2)"
        root="${root:-./models}"
        if [[ -d "$root/baai/bge-m3" && -d "$root/baai/bge-reranker-v2-m3" ]]; then
            ok "BGE models present at $root"
        else
            warn "BGE models missing at $root — guide §3 has download commands (~1.2 GB)"
        fi
    fi
}

# ─────────────────────────────────────────────────────────────────────
# Phase: SMOKE — end-to-end. Requires API+worker running.
# ─────────────────────────────────────────────────────────────────────
do_smoke() {
    step "S1. Ollama reachable (via rag-ollama container)"
    if curl -sf --max-time 5 http://localhost:11434/api/tags >/dev/null; then
        ok "rag-ollama answers on localhost:11434"
    else
        bad "rag-ollama not reachable — check 'docker port rag-ollama' and that the inference profile is up"
        return 1
    fi

    step "S2. API server health"
    local api_url="${API_URL:-http://localhost:8000/health}"
    if curl -sf --max-time 5 "$api_url" >/dev/null; then
        ok "API answers on $api_url"
    else
        bad "API not up at $api_url — start it with 'make dev' (and 'make worker' for workflows)"
        return 1
    fi

    step "S3. Cloudflare tunnel dry-run (15s)"
    if ! command -v cloudflared >/dev/null 2>&1; then
        bad "cloudflared missing — see guide §0.5"
        return 1
    fi
    local tlog; tlog="$(mktemp)"
    cloudflared tunnel --url http://localhost:8000 >"$tlog" 2>&1 &
    local tpid=$!
    local url=""
    for _ in $(seq 1 15); do
        sleep 1
        url=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$tlog" | head -1)
        [[ -n "$url" ]] && break
    done
    if [[ -n "$url" ]]; then
        ok "Tunnel up: $url"
        if curl -sf --max-time 10 "$url/health" >/dev/null; then
            ok "Tunnel forwards /health correctly"
        else
            warn "Tunnel up but /health did not respond through it"
        fi
    else
        bad "cloudflared did not produce a URL within 15s — see $tlog"
    fi
    kill "$tpid" 2>/dev/null || true
    rm -f "$tlog"
}

# ─────────────────────────────────────────────────────────────────────
# Mode dispatch
# ─────────────────────────────────────────────────────────────────────
case "$MODE" in
    --check)  do_check ;;
    --run)    do_check; [[ $FAIL -eq 0 ]] && do_run ;;
    --smoke)  do_smoke ;;
    --all)
        do_check
        if [[ $FAIL -gt 0 ]]; then
            warn "Check phase found hard failures — skipping run/smoke. Fix the gaps above and re-run."
        else
            do_run
            do_smoke
        fi
        ;;
    -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "Unknown mode: $MODE (try --check | --run | --smoke | --all)"; exit 2 ;;
esac

# ── Summary + gap list ─────────────────────────────────────────────────
printf '\n%s── Summary ──%s\n' "$B" "$N"
printf '  %s%d pass%s  %s%d warn%s  %s%d fail%s\n' \
    "$G" "$PASS" "$N" "$Y" "$WARN" "$N" "$R" "$FAIL" "$N"
if (( ${#GAPS[@]} > 0 )); then
    printf '\n%sGaps:%s\n' "$Y" "$N"
    for g in "${GAPS[@]}"; do printf '  - %s\n' "$g"; done
fi
[[ $FAIL -gt 0 ]] && exit 1 || exit 0
