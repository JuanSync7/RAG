#!/usr/bin/env bash
# Control the RagWeave PROD stack on uk-ai03 — full containerized production stack
# as the `ragweave-prod` compose project (canonical base container names).
#
# Layered: base (docker-compose.yml) + ai03 platform (docker-compose.ai03.yml)
# + prod overlay (docker-compose.prod.yml) + per-env values (.env.prod). Embed/
# rerank/gen are remote on uk-ai01 vLLM via the rag-vllm-tunnel sidecar (qwen3).
#
# WARNING: this is the FULL prod stack and REPLACES the current lean prod. Read
# the cutover notes in docker-compose.prod.yml before bringing it up (host->
# container Temporal history migration; qwen3 re-embed). Do it in a window.
#
# Usage:
#   ./scripts/prod-up.sh up -d
#   ./scripts/prod-up.sh ps
#   ./scripts/prod-up.sh logs -f rag-api
#   ./scripts/prod-up.sh down
set -euo pipefail
cd "$(dirname "$0")/.."

exec podman compose \
  -f docker-compose.yml \
  -f docker-compose.ai03.yml \
  -f docker-compose.prod.yml \
  --env-file .env.prod \
  -p ragweave-prod \
  --profile app \
  --profile workers \
  --profile observability \
  --profile monitoring \
  "$@"
