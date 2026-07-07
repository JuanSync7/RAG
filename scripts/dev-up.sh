#!/usr/bin/env bash
# Control the RagWeave DEV stack on uk-ai03 — full mirror of prod, isolated as
# the `ragweave-dev` compose project (own networks/volumes/-dev container names).
#
# Thin overlay (docker-compose.dev.yml) on the canonical base (docker-compose.yml)
# + per-env values (.env.dev). Embed/rerank/gen are remote on uk-ai01 vLLM via
# the rag-vllm-tunnel-dev sidecar (qwen3-embed / qwen3-reranker / qwopus).
#
# Usage:
#   ./scripts/dev-up.sh up -d
#   ./scripts/dev-up.sh ps
#   ./scripts/dev-up.sh logs -f rag-api-dev
#   ./scripts/dev-up.sh restart rag-api-dev rag-worker-dev
#   ./scripts/dev-up.sh down
set -euo pipefail
cd "$(dirname "$0")/.."

exec podman compose \
  -f docker-compose.yml \
  -f docker-compose.ai03.yml \
  -f docker-compose.dev.yml \
  --env-file .env.dev \
  -p ragweave-dev \
  --profile app \
  --profile workers \
  --profile observability \
  --profile monitoring \
  "$@"
