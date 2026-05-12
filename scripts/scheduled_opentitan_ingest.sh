#!/usr/bin/env bash
# Scheduled OpenTitan ingestion wrapper.
#
# Pre-flight checks the docker stack, restarts unhealthy services that the
# ingest pipeline depends on (weaviate, ollama, embed), waits for them to
# settle, then runs the full pipeline:
#
#     manifest -> stage -> ingest --update
#
# Reranker health is reported but NOT blocking — ingestion only uses
# embed/weaviate/ollama. The reranker matters for the post-ingest A/B
# harness, which is a separate step.
#
# Usage (manual):
#     bash scripts/scheduled_opentitan_ingest.sh
#
# Scheduled (e.g., 01:30 launch via nohup+sleep):
#     nohup bash -c 'sleep 12000 && bash scripts/scheduled_opentitan_ingest.sh' \
#         > /tmp/opentitan_ingest_$(date +%Y%m%d_%H%M%S).launcher.log 2>&1 &

set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/home/juansync7/RagWeave}"
WORKTREE="${WORKTREE:-${REPO_ROOT}/.claude/worktrees/feat-deep-research}"
LOG_DIR="${LOG_DIR:-/tmp}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/opentitan_ingest_${TIMESTAMP}.log"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-300}"   # max seconds to wait for stack
INGEST_TIMEOUT="${INGEST_TIMEOUT:-21600}" # max ingest wall (6h)

# Required-for-ingest services (must reach healthy / responsive)
REQUIRED_SERVICES=(rag-weaviate rag-ollama rag-embed)
# Best-effort services (warn-only)
OPTIONAL_SERVICES=(rag-rerank rag-worker)

WEAVIATE_HEALTH_URL="http://localhost:8090/v1/meta"
OLLAMA_HEALTH_URL="http://localhost:11434/api/tags"

log() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }
fail() { log "FATAL: $*"; exit 1; }

# Tee everything into a log file for post-mortem.
exec > >(tee -a "$LOG_FILE") 2>&1

log "=== OpenTitan ingestion wrapper started ==="
log "log file: $LOG_FILE"
log "repo:     $REPO_ROOT"
log "worktree: $WORKTREE"

cd "$REPO_ROOT" || fail "cannot cd to $REPO_ROOT"

# ---------------------------------------------------------------------------
# Phase 1: docker stack health
# ---------------------------------------------------------------------------
log "--- phase 1: docker stack health ---"

if ! docker info >/dev/null 2>&1; then
    fail "docker daemon not reachable"
fi

# Check whether each required service is running; if not, bring it up.
needs_up=()
for svc in "${REQUIRED_SERVICES[@]}"; do
    state="$(docker compose ps --status running --services 2>/dev/null | grep -Fx "$svc" || true)"
    if [[ -z "$state" ]]; then
        log "service $svc is NOT running — will start it"
        needs_up+=("$svc")
    else
        log "service $svc is running"
    fi
done

if (( ${#needs_up[@]} > 0 )); then
    log "starting: ${needs_up[*]}"
    if ! docker compose up -d "${needs_up[@]}"; then
        fail "docker compose up -d ${needs_up[*]} failed"
    fi
fi

# Health probes with timeout.
poll_until() {
    local label="$1"; shift
    local deadline=$(( $(date +%s) + HEALTH_TIMEOUT ))
    while (( $(date +%s) < deadline )); do
        if "$@"; then
            log "$label: healthy"
            return 0
        fi
        sleep 5
    done
    log "$label: NOT healthy after ${HEALTH_TIMEOUT}s"
    return 1
}

probe_weaviate() { curl -sf -o /dev/null "$WEAVIATE_HEALTH_URL"; }
probe_ollama()   { curl -sf -o /dev/null "$OLLAMA_HEALTH_URL"; }
probe_embed_healthy() {
    local h
    h="$(docker compose ps --format json rag-embed 2>/dev/null | python3 -c \
        'import json,sys; d=json.loads(sys.stdin.read() or "{}"); print(d.get("Health","none"))' 2>/dev/null)"
    [[ "$h" == "healthy" ]]
}

poll_until "weaviate"     probe_weaviate     || fail "weaviate never came up"
poll_until "ollama"       probe_ollama       || fail "ollama never came up"
poll_until "rag-embed"    probe_embed_healthy || fail "rag-embed never reached healthy"

for svc in "${OPTIONAL_SERVICES[@]}"; do
    state="$(docker compose ps --status running --services 2>/dev/null | grep -Fx "$svc" || true)"
    if [[ -z "$state" ]]; then
        log "WARN: optional service $svc is not running (post-ingest A/B may be affected)"
    fi
done

log "stack pre-flight: PASS"

# ---------------------------------------------------------------------------
# Phase 2: manifest + staging
# ---------------------------------------------------------------------------
log "--- phase 2: manifest + staging ---"

cd "$WORKTREE" || fail "cannot cd to $WORKTREE"

python3 scripts/build_opentitan_manifest.py || fail "manifest build failed"
python3 scripts/stage_opentitan_corpus.py   || fail "corpus staging failed"

staged_count="$(find documents/opentitan -type l 2>/dev/null | wc -l)"
log "staged $staged_count symlinks under documents/opentitan/"
if (( staged_count < 100 )); then
    fail "staged < 100 files; aborting"
fi

# ---------------------------------------------------------------------------
# Phase 3: ingestion
# ---------------------------------------------------------------------------
log "--- phase 3: ingestion ---"
log "running: uv run --no-sync python -m src.ingest.cli --dir documents/opentitan --update"

cd "$REPO_ROOT" || fail "cannot cd to $REPO_ROOT"

ingest_start="$(date +%s)"
RAG_WEAVIATE_MODE=networked timeout "$INGEST_TIMEOUT" \
    uv run --no-sync python -m src.ingest.cli \
        --dir "$WORKTREE/documents/opentitan" \
        --update
ingest_rc=$?
ingest_dur=$(( $(date +%s) - ingest_start ))

log "ingest exit_code=$ingest_rc duration=${ingest_dur}s"

# ---------------------------------------------------------------------------
# Phase 4: post-ingest verification
# ---------------------------------------------------------------------------
log "--- phase 4: post-ingest verification ---"

total_after="$(curl -sf "http://localhost:8090/v1/graphql" \
    -H 'Content-Type: application/json' \
    -d '{"query":"{ Aggregate { RAGDocuments { meta { count } } } }"}' \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"]["Aggregate"]["RAGDocuments"][0]["meta"]["count"])' 2>/dev/null \
    || echo "?")"

log "Weaviate RAGDocuments total after ingest: $total_after"
log "=== OpenTitan ingestion wrapper finished (rc=$ingest_rc) ==="

exit "$ingest_rc"
