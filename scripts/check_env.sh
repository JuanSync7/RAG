#!/usr/bin/env bash
# check_env.sh — single-command verification that the RagWeave dev/prod
# environment is ready for the TEI-based inference backend.
#
# What it checks:
#   1. GPU arch → correct TEI image tag mapping
#   2. Docker daemon + compose v2 are reachable
#   3. TEI image for the detected GPU arch is pullable
#   4. Each infra service responds on its host-side port (Weaviate, TEI
#      embed/rerank, Temporal, Redis) — best-effort; failures are expected
#      if the compose stack is not up.
#   5. Dev venv has the right extras installed (local-embed, core httpx)
#   6. Required env vars are set when RAG_INFERENCE_BACKEND=tei
#
# Prints a ✓ / ✗ checklist. Exits non-zero if anything fails.
#
# Usage:
#   bash scripts/check_env.sh
#
# Optional env overrides:
#   RAG_WEAVIATE_HTTP_HOST_PORT (default 8090)
#   RAG_TEI_EMBED_PORT          (default 8081)
#   RAG_TEI_RERANK_PORT         (default 8082)

set -u  # no -e: we want to run every check, not bail on the first failure

# ── Colors (tty only) ──────────────────────────────────────────────────
if [[ -t 1 ]]; then
    _GREEN=$'\e[32m'; _RED=$'\e[31m'; _YELLOW=$'\e[33m'; _DIM=$'\e[2m'; _RESET=$'\e[0m'
else
    _GREEN=""; _RED=""; _YELLOW=""; _DIM=""; _RESET=""
fi

FAIL=0

_check() {
    # Usage: _check "<label>" <cmd> [args...]
    local label="$1"; shift
    printf "  %-55s " "$label"
    if "$@" >/dev/null 2>&1; then
        printf "%s✓%s\n" "$_GREEN" "$_RESET"
    else
        printf "%s✗%s\n" "$_RED" "$_RESET"
        FAIL=1
    fi
}

_info() {
    printf "  %s%s%s\n" "$_DIM" "$1" "$_RESET"
}

# ── 1. GPU arch → TEI image tag ────────────────────────────────────────
echo ""
echo "GPU / TEI image"
if command -v nvidia-smi >/dev/null 2>&1; then
    SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    case "$SM" in
        7.5)      TEI_TAG="turing-1.5" ;;
        8.0|8.6)  TEI_TAG="86-1.5"     ;;
        8.9)      TEI_TAG="89-1.5"     ;;
        9.0)      TEI_TAG="latest"     ;;
        "")       TEI_TAG="cpu-1.5"    ;;
        *)        TEI_TAG="cpu-1.5"    ;;
    esac
    _info "Detected GPU: ${GPU_NAME:-unknown} (SM=${SM:-unknown}) → tag=${TEI_TAG}"
else
    TEI_TAG="cpu-1.5"
    _info "No nvidia-smi found → tag=${TEI_TAG} (CPU-only)"
fi

# Compare to RAG_TEI_IMAGE_TAG if set — drift means the .env is stale.
if [[ -n "${RAG_TEI_IMAGE_TAG:-}" && "${RAG_TEI_IMAGE_TAG}" != "$TEI_TAG" ]]; then
    printf "  %s⚠ RAG_TEI_IMAGE_TAG=%s but detected %s — update .env to match.%s\n" \
        "$_YELLOW" "$RAG_TEI_IMAGE_TAG" "$TEI_TAG" "$_RESET"
fi

# ── 2. Docker + compose ────────────────────────────────────────────────
echo ""
echo "Container runtime"
_check "Docker daemon reachable"          docker info
_check "Docker Compose v2 available"      docker compose version

# ── 3. TEI image pullable ──────────────────────────────────────────────
echo ""
echo "TEI image availability"
_check "Pullable: text-embeddings-inference:${TEI_TAG}" \
    docker pull -q "ghcr.io/huggingface/text-embeddings-inference:${TEI_TAG}"

# ── 4. Infra services (best-effort) ────────────────────────────────────
echo ""
echo "Infra services (best-effort — expected failures when stack is down)"
_WV_PORT="${RAG_WEAVIATE_HTTP_HOST_PORT:-8090}"
_EMBED_PORT="${RAG_TEI_EMBED_PORT:-8081}"
_RERANK_PORT="${RAG_TEI_RERANK_PORT:-8082}"

_check "Weaviate   http://localhost:${_WV_PORT}/v1/.well-known/ready" \
    curl -sf "http://localhost:${_WV_PORT}/v1/.well-known/ready"
_check "TEI embed  http://localhost:${_EMBED_PORT}/health" \
    curl -sf "http://localhost:${_EMBED_PORT}/health"
_check "TEI rerank http://localhost:${_RERANK_PORT}/health" \
    curl -sf "http://localhost:${_RERANK_PORT}/health"
if command -v nc >/dev/null 2>&1; then
    _check "Temporal  localhost:7233"     nc -z -w 2 localhost 7233
    _check "Redis     localhost:6379"     nc -z -w 2 localhost 6379
else
    _info "nc not installed — skipping Temporal/Redis TCP probes"
fi

# ── 5. Dev venv extras ─────────────────────────────────────────────────
echo ""
echo "Dev venv"
if [[ -x .venv/bin/python ]]; then
    _check "httpx available (core dep)" \
        .venv/bin/python -c "import httpx"
    _check "sentence_transformers (local-embed extra)" \
        .venv/bin/python -c "import sentence_transformers"
    _check "transformers (local-embed extra)" \
        .venv/bin/python -c "import transformers"
    _check "torch (local-embed extra)" \
        .venv/bin/python -c "import torch"
else
    _info "No .venv/ found — skipping venv checks (run: uv sync --extra local-embed)"
fi

# ── 6. Env vars when backend=tei ───────────────────────────────────────
echo ""
echo "Env vars"
_BACKEND="${RAG_INFERENCE_BACKEND:-local}"
_info "RAG_INFERENCE_BACKEND=${_BACKEND}"
if [[ "$_BACKEND" == "tei" ]]; then
    _check "RAG_TEI_EMBED_URL set"  test -n "${RAG_TEI_EMBED_URL:-}"
    _check "RAG_TEI_RERANK_URL set" test -n "${RAG_TEI_RERANK_URL:-}"
fi

# ── Summary ────────────────────────────────────────────────────────────
echo ""
if [[ "$FAIL" == "0" ]]; then
    printf "%sAll checks passed.%s\n" "$_GREEN" "$_RESET"
else
    printf "%sSome checks failed — see ✗ above.%s\n" "$_RED" "$_RESET"
    echo ""
    echo "Common remediations:"
    echo "  - Start infra:         docker compose up -d rag-embed rag-rerank rag-weaviate"
    echo "  - Install dev extras:  uv sync --extra local-embed"
    echo "  - Set TEI image tag:   export RAG_TEI_IMAGE_TAG=${TEI_TAG}"
fi

exit $FAIL
