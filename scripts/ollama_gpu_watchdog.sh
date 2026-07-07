#!/usr/bin/env bash
# ollama_gpu_watchdog.sh — detect and recover when rag-ollama silently
# falls back to CPU.
#
# Failure mode this addresses:
#   When the rag-ollama container loads a model while VRAM is tight, it
#   pins to CPU. Even after VRAM frees up, NVML inside the container can
#   stay stuck on the old reading, so subsequent reloads also pick CPU.
#   Symptom: `ollama ps` shows `100% CPU`, queries are 10–20× slower,
#   nothing in the logs flags it as a problem.
#
# What this script does:
#   1. Runs `ollama ps`. If no model is loaded, exits 0 (nothing to check).
#   2. If a loaded model is on CPU while the host has enough free VRAM
#      to fit it, restarts the container so NVML re-probes.
#   3. After restart, fires a one-token warmup so the model claims VRAM
#      before TEI replicas (or anything else) take it.
#
# Usage:
#   scripts/ollama_gpu_watchdog.sh             # one-shot check + heal
#   scripts/ollama_gpu_watchdog.sh --watch     # loop every 60s
#   scripts/ollama_gpu_watchdog.sh --dry-run   # report only, never restart
#
# Safe to invoke from cron, a systemd timer, or a pre-query hook.

set -euo pipefail

CONTAINER="${OLLAMA_CONTAINER:-rag-ollama}"
INTERVAL="${WATCHDOG_INTERVAL:-60}"
# Min host-side free VRAM (MiB) before we attempt a heal. Below this, a
# restart wouldn't help — the GPU genuinely doesn't have room.
MIN_FREE_VRAM_MIB="${MIN_FREE_VRAM_MIB:-1500}"
DRY_RUN=0
WATCH=0

for arg in "$@"; do
    case "$arg" in
        --watch)   WATCH=1 ;;
        --dry-run) DRY_RUN=1 ;;
        -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

log() { printf '[%s] %s\n' "$(date -Iseconds)" "$*"; }

check_once() {
    if ! docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -q true; then
        log "container $CONTAINER not running — skipping"
        return 0
    fi

    local ps_out
    ps_out=$(docker exec "$CONTAINER" ollama ps 2>/dev/null || true)

    # Header-only output means no model is loaded; nothing to heal.
    if ! printf '%s\n' "$ps_out" | tail -n +2 | grep -q .; then
        log "no model loaded — ok"
        return 0
    fi

    if ! printf '%s\n' "$ps_out" | grep -qE '[[:space:]]+CPU[[:space:]]'; then
        log "model on GPU — ok"
        return 0
    fi

    local model
    model=$(printf '%s\n' "$ps_out" | awk 'NR==2 {print $1}')
    log "DEGRADED: model=$model is on CPU"

    local free_mib
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || echo 0)
    log "host VRAM free=${free_mib} MiB (threshold ${MIN_FREE_VRAM_MIB})"

    if [ "${free_mib:-0}" -lt "$MIN_FREE_VRAM_MIB" ]; then
        log "not enough free VRAM to relocate — leaving on CPU"
        return 0
    fi

    if [ "$DRY_RUN" -eq 1 ]; then
        log "[dry-run] would restart $CONTAINER and warm $model"
        return 0
    fi

    log "restarting $CONTAINER to force NVML re-probe"
    docker restart "$CONTAINER" >/dev/null

    # Wait for healthy.
    for _ in $(seq 1 60); do
        local s
        s=$(docker inspect -f '{{.State.Health.Status}}' "$CONTAINER" 2>/dev/null || echo unknown)
        [ "$s" = "healthy" ] && break
        sleep 2
    done

    log "warming $model so it claims VRAM before competitors"
    docker exec "$CONTAINER" sh -c "printf '' | ollama run $model >/dev/null" || true

    local after
    after=$(docker exec "$CONTAINER" ollama ps 2>/dev/null | awk 'NR==2 {for (i=4;i<=NF;i++) printf $i" "; print ""}')
    log "post-heal ollama ps: $after"
}

if [ "$WATCH" -eq 1 ]; then
    log "watchdog loop started (interval=${INTERVAL}s)"
    while true; do
        check_once || true
        sleep "$INTERVAL"
    done
else
    check_once
fi
