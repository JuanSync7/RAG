#!/usr/bin/env bash
# @summary
# Single entry point for `docker compose` / `podman compose` in RagWeave.
# Auto-detects the container runtime AND the host GPU, then picks the right
# compose overlays and TEI image tags so `up`, `down`, `ps`, etc. "just work"
# on NVIDIA, AMD, Intel, Apple Silicon, and WSL2 hosts without manual flags.
#
# Usage:
#   ./scripts/compose.sh up -d                  # full stack, auto GPU/CPU
#   ./scripts/compose.sh --profile workers up   # with a compose profile
#   ./scripts/compose.sh --explain up -d        # print plan, do not execute
#   ./scripts/compose.sh logs -f rag-embed
#
# Knobs (env vars — set before running, or export in your shell):
#   RAGWEAVE_FORCE_CPU=1        Force CPU overlay even if NVIDIA is present.
#   RAGWEAVE_FORCE_GPU=1        Skip CPU overlay even if no GPU detected
#                               (will fail at runtime — for debugging).
#   RAGWEAVE_COMPOSE_EXTRA="-f docker-compose.extra.yml"
#                               Extra files appended after auto-detected ones.
#   RAGWEAVE_RUNTIME=docker|podman
#                               Override runtime auto-detect.
#   RAGWEAVE_VERBOSE=1          Print resolved command before exec.
#
# Exports: (none — exec's into the chosen compose binary)
# Deps: docker compose v2 | podman compose | podman-compose; optionally
#       nvidia-smi for GPU detection.
# @end-summary

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ── Colors (tty only) ──────────────────────────────────────────────────
if [[ -t 1 ]]; then
    _G=$'\e[32m'; _Y=$'\e[33m'; _R=$'\e[31m'; _D=$'\e[2m'; _B=$'\e[1m'; _N=$'\e[0m'
else
    _G=""; _Y=""; _R=""; _D=""; _B=""; _N=""
fi
_log()  { printf '%s[compose]%s %s\n' "$_D" "$_N" "$*" >&2; }
_warn() { printf '%s[compose]%s %s%s%s\n' "$_D" "$_N" "$_Y" "$*" "$_N" >&2; }
_err()  { printf '%s[compose]%s %s%s%s\n' "$_D" "$_N" "$_R" "$*" "$_N" >&2; }

# ── Optional flags handled by the wrapper (everything else passes through) ──
EXPLAIN=0
PASSTHRU=()
for arg in "$@"; do
    case "$arg" in
        --explain|--dry-run) EXPLAIN=1 ;;
        --help-wrapper)
            sed -n '2,28p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) PASSTHRU+=("$arg") ;;
    esac
done

# ── 1. Container runtime ───────────────────────────────────────────────
choose_runtime() {
    if [[ -n "${RAGWEAVE_RUNTIME:-}" ]]; then
        case "$RAGWEAVE_RUNTIME" in
            docker)
                command -v docker >/dev/null && docker compose version >/dev/null 2>&1 \
                    && { echo "docker compose"; return; }
                ;;
            podman)
                if command -v podman-compose >/dev/null 2>&1; then
                    echo "podman-compose"; return
                elif podman compose version >/dev/null 2>&1; then
                    echo "podman compose"; return
                fi
                ;;
        esac
        _err "RAGWEAVE_RUNTIME=$RAGWEAVE_RUNTIME requested but not usable."
        exit 1
    fi
    if command -v podman-compose >/dev/null 2>&1; then echo "podman-compose"; return; fi
    if command -v podman >/dev/null 2>&1 && podman compose version >/dev/null 2>&1; then
        echo "podman compose"; return
    fi
    if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
        echo "docker compose"; return
    fi
    _err "Neither docker compose v2 nor podman compose found. Install one and retry."
    exit 1
}
COMPOSE_CMD="$(choose_runtime)"

# ── 2. GPU detection → TEI image tags + overlay choice ────────────────
# Mirrors scripts/check_env.sh. SM = "compute capability"; TEI publishes one
# image per arch family. Mapping is deliberately conservative — bumping to a
# newer minor (1.6/1.7/1.8) per arch is opt-in via .env overrides.
detect_gpu_and_tei_tags() {
    local sm="" gpu_name="" wsl=""
    if [[ -r /proc/version ]] && grep -qiE "microsoft|wsl" /proc/version; then
        wsl="yes"
    fi

    if [[ "${RAGWEAVE_FORCE_CPU:-0}" == "1" ]]; then
        echo "cpu|cpu-1.5|cpu-1.8||$wsl"; return
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        # On WSL2, if nvidia-smi returns a compute_cap the CUDA passthrough
        # is wired and Docker will see the GPU.
        sm=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
                | head -1 | tr -d ' ')
        gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    fi

    local embed_tag rerank_tag mode
    case "$sm" in
        7.5)      embed_tag="turing-1.5"; rerank_tag="turing-1.8"; mode="gpu" ;;
        8.0|8.6)  embed_tag="86-1.5";     rerank_tag="86-1.8";     mode="gpu" ;;
        8.9)      embed_tag="89-1.5";     rerank_tag="89-1.8";     mode="gpu" ;;
        9.0|10.0|12.0)
                  embed_tag="latest";     rerank_tag="latest";     mode="gpu" ;;
        "")       embed_tag="cpu-1.5";    rerank_tag="cpu-1.8";    mode="cpu" ;;
        *)        # Unknown / future arch — try `latest`, but warn.
                  embed_tag="latest";     rerank_tag="latest";     mode="gpu"
                  _warn "Unknown GPU compute capability '$sm' (${gpu_name:-?}); using TEI 'latest'."
                  ;;
    esac

    # Force-GPU escape hatch even when no nvidia-smi (testing / mis-detection).
    if [[ "$mode" == "cpu" && "${RAGWEAVE_FORCE_GPU:-0}" == "1" ]]; then
        _warn "RAGWEAVE_FORCE_GPU=1 set but no NVIDIA detected — using turing-1.5 anyway."
        embed_tag="turing-1.5"; rerank_tag="turing-1.8"; mode="gpu"
    fi

    echo "${mode}|${embed_tag}|${rerank_tag}|${gpu_name}|${wsl}"
}

IFS='|' read -r GPU_MODE EMBED_TAG RERANK_TAG GPU_NAME IS_WSL \
    <<<"$(detect_gpu_and_tei_tags)"

# Respect any user-set values in the environment — env wins over auto-detect.
EMBED_TAG="${RAG_TEI_IMAGE_TAG:-$EMBED_TAG}"
RERANK_TAG="${RAG_TEI_RERANK_IMAGE_TAG:-$RERANK_TAG}"
export RAG_TEI_IMAGE_TAG="$EMBED_TAG"
export RAG_TEI_RERANK_IMAGE_TAG="$RERANK_TAG"

# ── 3. Overlay assembly ───────────────────────────────────────────────
# Compose auto-loads docker-compose.override.yml, so we do NOT pass it via
# -f. The CPU overlay is appended only when no GPU was detected; user extras
# come last so they can override either layer.
FILES=(-f docker-compose.yml)
if [[ "$GPU_MODE" == "cpu" ]]; then
    FILES+=(-f docker-compose.cpu.yml)
fi
# shellcheck disable=SC2206  # intentional word-split of caller-supplied flags
EXTRA=( ${RAGWEAVE_COMPOSE_EXTRA:-} )
if (( ${#EXTRA[@]} )); then
    FILES+=("${EXTRA[@]}")
fi

# ── 4. Banner ─────────────────────────────────────────────────────────
print_banner() {
    printf '%s[compose]%s runtime  %s%s%s\n' "$_D" "$_N" "$_B" "$COMPOSE_CMD" "$_N"
    if [[ "$GPU_MODE" == "gpu" ]]; then
        printf '%s[compose]%s mode     %sGPU%s  (%s)\n' \
            "$_D" "$_N" "$_G" "$_N" "${GPU_NAME:-NVIDIA detected}"
    else
        local forced=""
        [[ "${RAGWEAVE_FORCE_CPU:-0}" == "1" ]] && forced=', forced by RAGWEAVE_FORCE_CPU=1'
        printf '%s[compose]%s mode     %sCPU%s  (no NVIDIA GPU detected%s)\n' \
            "$_D" "$_N" "$_Y" "$_N" "$forced"
    fi
    printf '%s[compose]%s embed    %s\n' "$_D" "$_N" "$EMBED_TAG"
    printf '%s[compose]%s rerank   %s\n' "$_D" "$_N" "$RERANK_TAG"
    if [[ -n "$IS_WSL" ]]; then
        printf '%s[compose]%s host     WSL2  %s(ensure NVIDIA Container Toolkit + WSL CUDA driver)%s\n' \
            "$_D" "$_N" "$_D" "$_N"
    fi
    printf '%s[compose]%s files    %s\n' "$_D" "$_N" "${FILES[*]}"
}
print_banner

if (( EXPLAIN )); then
    printf '%s[compose]%s --explain: not executing.\n' "$_D" "$_N"
    printf '%s[compose]%s would run: %s %s %s\n' \
        "$_D" "$_N" "$COMPOSE_CMD" "${FILES[*]}" "${PASSTHRU[*]:-}"
    exit 0
fi

# ── 5. Exec ───────────────────────────────────────────────────────────
[[ "${RAGWEAVE_VERBOSE:-0}" == "1" ]] && \
    _log "exec: $COMPOSE_CMD ${FILES[*]} ${PASSTHRU[*]:-}"

exec $COMPOSE_CMD "${FILES[@]}" "${PASSTHRU[@]}"
