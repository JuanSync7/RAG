#!/usr/bin/env bash
# Pre-download TEI models into .tei_cache/ from the host.
#
# Why this exists:
#   TEI's first container boot downloads ~3GB from HuggingFace inside the
#   container. On networks where huggingface.co serves IPv6-heavy CDN records,
#   Docker's IPv4-only default bridge causes that download to hang for ~9min
#   per file before timing out. Even with the IPv4-DNS sysctl fix in
#   docker-compose.yml, the in-container download is fragile: no resume on
#   restart, ties up TEI startup for 5-15min, and re-runs on every fresh
#   container that hits an empty cache.
#
#   Running huggingface-cli on the host sidesteps all of that. The cache
#   directory is bind-mounted into the container (./.tei_cache:/data), so
#   once it's populated here, every container start is offline-fast (~30s).
#
# Usage:
#   ./scripts/seed-tei-cache.sh                 # download both models
#   ./scripts/seed-tei-cache.sh embed           # embedder only (BGE-M3)
#   ./scripts/seed-tei-cache.sh rerank          # reranker only (BGE-reranker-v2-m3)
#
# Required: uv (https://github.com/astral-sh/uv) — installs huggingface-hub
# in an ephemeral environment.
set -euo pipefail

cd "$(dirname "$0")/.."
CACHE_DIR="$(pwd)/.tei_cache"
mkdir -p "$CACHE_DIR"

EMBED_MODEL="${RAG_TEI_EMBEDDING_MODEL:-BAAI/bge-m3}"
RERANK_MODEL="${RAG_TEI_RERANKER_MODEL:-BAAI/bge-reranker-v2-m3}"

target="${1:-both}"

download() {
    local model="$1"
    echo "→ Downloading ${model} into ${CACHE_DIR}"
    uv tool run --from huggingface-hub huggingface-cli download \
        "${model}" \
        --cache-dir "${CACHE_DIR}"
}

case "${target}" in
    embed)  download "${EMBED_MODEL}" ;;
    rerank) download "${RERANK_MODEL}" ;;
    both)
        download "${EMBED_MODEL}"
        download "${RERANK_MODEL}"
        ;;
    *)
        echo "Usage: $0 [embed|rerank|both]" >&2
        exit 2
        ;;
esac

echo
echo "Done. Cache contents:"
du -sh "${CACHE_DIR}"/* 2>/dev/null || true
echo
echo "Next: 'docker compose up -d rag-rerank rag-embed' will load from cache (~30s startup)."
