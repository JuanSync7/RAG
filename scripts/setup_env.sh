#!/usr/bin/env bash
# @summary
# Interactive first-run env walkthrough for RagWeave. Copies .env.example to
# .env, generates strong secrets in place of `replace_me` placeholders, auto-
# detects the host GPU to pick TEI image tags, and prompts for the handful of
# values a user actually needs to think about (inference backend, HF token).
#
# Idempotent: safe to re-run. Existing .env values are preserved unless the
# user explicitly opts to reset them. Generated secrets are written only when
# the current value still equals the upstream placeholder.
#
# Usage:
#   ./scripts/setup_env.sh                  # interactive walkthrough
#   ./scripts/setup_env.sh --non-interactive   # accept all defaults
#   ./scripts/setup_env.sh --reset          # overwrite existing .env
#
# Exports: (none — mutates .env in place)
# Deps: openssl (for secret generation); optional nvidia-smi (GPU detect).
# @end-summary

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV_FILE=".env"
EXAMPLE_FILE=".env.example"

NON_INTERACTIVE=0
RESET=0
for arg in "$@"; do
    case "$arg" in
        --non-interactive|-y) NON_INTERACTIVE=1 ;;
        --reset)              RESET=1 ;;
        -h|--help)
            sed -n '2,21p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "Unknown flag: $arg" >&2; exit 1 ;;
    esac
done

# ── Output helpers ─────────────────────────────────────────────────────
if [[ -t 1 ]]; then
    _G=$'\e[32m'; _Y=$'\e[33m'; _R=$'\e[31m'; _B=$'\e[1m'; _D=$'\e[2m'; _N=$'\e[0m'
else
    _G=""; _Y=""; _R=""; _B=""; _D=""; _N=""
fi
step() { printf '\n%s── %s%s\n' "$_B" "$1" "$_N"; }
ok()   { printf '  %s✓%s %s\n' "$_G" "$_N" "$1"; }
warn() { printf '  %s⚠%s %s\n' "$_Y" "$_N" "$1"; }
info() { printf '  %s%s%s\n' "$_D" "$1" "$_N"; }

# ── Helpers ────────────────────────────────────────────────────────────
# Set KEY=VALUE in .env, replacing the existing line if present, appending
# otherwise. Values are passed raw — caller is responsible for quoting if
# the value contains spaces or shell metacharacters (none of ours do).
set_env() {
    local key="$1" value="$2"
    if grep -qE "^${key}=" "$ENV_FILE"; then
        # Use a delimiter unlikely to appear in secrets/URLs.
        local esc; esc=$(printf '%s' "$value" | sed 's/[\&|]/\\&/g')
        sed -i "s|^${key}=.*|${key}=${esc}|" "$ENV_FILE"
    else
        printf '%s=%s\n' "$key" "$value" >> "$ENV_FILE"
    fi
}

get_env() {
    # `|| true` keeps `set -o pipefail` from tripping when the key is absent.
    grep -E "^$1=" "$ENV_FILE" 2>/dev/null | tail -1 | cut -d= -f2- || true
}

prompt() {
    # prompt VAR "Question text" "default" [hidden=0|1]
    local var="$1" question="$2" default="${3:-}" hidden="${4:-0}" reply
    if (( NON_INTERACTIVE )); then
        printf -v "$var" '%s' "$default"
        return
    fi
    if [[ "$hidden" == "1" ]]; then
        read -rsp "  $question [${default:-leave blank to skip}]: " reply
        echo
    else
        read -rp "  $question [${default:-leave blank to skip}]: " reply
    fi
    printf -v "$var" '%s' "${reply:-$default}"
}

gen_secret() { openssl rand -hex 32; }

# ── 1. Bootstrap .env ─────────────────────────────────────────────────
step "1. Bootstrap .env"
if [[ ! -f "$EXAMPLE_FILE" ]]; then
    echo "Missing $EXAMPLE_FILE — are you in the repo root?" >&2
    exit 1
fi

if [[ -f "$ENV_FILE" && "$RESET" == "0" ]]; then
    ok ".env exists — keeping current values; only filling placeholders & missing keys"
elif [[ -f "$ENV_FILE" && "$RESET" == "1" ]]; then
    cp "$ENV_FILE" "${ENV_FILE}.bak.$(date +%s)"
    cp "$EXAMPLE_FILE" "$ENV_FILE"
    warn "Reset: backed up old .env to .env.bak.<timestamp> and copied from .env.example"
else
    cp "$EXAMPLE_FILE" "$ENV_FILE"
    ok "Created .env from .env.example"
fi

# Add any new keys from .env.example that are missing from current .env.
# Lets users re-run after pulling a repo update without losing local edits.
ADDED=0
while IFS= read -r line; do
    [[ "$line" =~ ^[A-Z_]+= ]] || continue
    key="${line%%=*}"
    if ! grep -qE "^${key}=" "$ENV_FILE"; then
        echo "$line" >> "$ENV_FILE"
        ADDED=$((ADDED+1))
    fi
done < "$EXAMPLE_FILE"
(( ADDED > 0 )) && ok "Appended $ADDED new key(s) from .env.example"

# ── 2. Generate secrets where placeholders remain ─────────────────────
step "2. Generate secrets"
if ! command -v openssl >/dev/null 2>&1; then
    warn "openssl not found — skipping secret generation. Install openssl and re-run."
else
    # Each secret is independent; never reuse one across slots.
    SECRET_KEYS=(
        LANGFUSE_PUBLIC_KEY
        LANGFUSE_SECRET_KEY
        LANGFUSE_NEXTAUTH_SECRET
        LANGFUSE_SALT
        LANGFUSE_ENCRYPTION_KEY
        RAG_AUTH_JWT_HS256_SECRET
    )
    GENERATED=0
    for k in "${SECRET_KEYS[@]}"; do
        current="$(get_env "$k")"
        if [[ -z "$current" || "$current" == *replace_me* ]]; then
            # Langfuse public/secret keys conventionally prefix pk_/sk_.
            case "$k" in
                LANGFUSE_PUBLIC_KEY) val="pk_$(gen_secret)" ;;
                LANGFUSE_SECRET_KEY) val="sk_$(gen_secret)" ;;
                *)                   val="$(gen_secret)"    ;;
            esac
            set_env "$k" "$val"
            GENERATED=$((GENERATED+1))
        fi
    done
    if (( GENERATED > 0 )); then
        ok "Generated $GENERATED secret(s) — Langfuse + JWT keys"
    else
        info "All secrets already set — nothing to generate"
    fi
fi

# ── 3. GPU auto-detect → TEI tags ─────────────────────────────────────
step "3. GPU detection → TEI image tags"
SM=""; GPU_NAME=""
if command -v nvidia-smi >/dev/null 2>&1; then
    SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
            | head -1 | tr -d ' ')
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
fi
case "$SM" in
    7.5)      EMBED_TAG="turing-1.5"; RERANK_TAG="turing-1.8" ;;
    8.0|8.6)  EMBED_TAG="86-1.5";     RERANK_TAG="86-1.8"    ;;
    8.9)      EMBED_TAG="89-1.5";     RERANK_TAG="89-1.8"    ;;
    9.0|10.0|12.0)
              EMBED_TAG="latest";     RERANK_TAG="latest"    ;;
    "")       EMBED_TAG="cpu-1.5";    RERANK_TAG="cpu-1.8"   ;;
    *)        EMBED_TAG="latest";     RERANK_TAG="latest"    ;;
esac
if [[ -n "$SM" ]]; then
    ok "Detected: ${GPU_NAME:-unknown} (SM=$SM) → embed=$EMBED_TAG, rerank=$RERANK_TAG"
else
    info "No NVIDIA GPU detected → embed=$EMBED_TAG, rerank=$RERANK_TAG (CPU)"
    info "Run scripts/compose.sh and it will auto-load docker-compose.cpu.yml"
fi
set_env RAG_TEI_IMAGE_TAG "$EMBED_TAG"
set_env RAG_TEI_RERANK_IMAGE_TAG "$RERANK_TAG"

# ── 4. User-facing prompts (only things they actually need to choose) ─
step "4. Inference backend"
info "tei    = TEI containers (rag-embed/rag-rerank) — recommended, scales with GPU"
info "local  = in-process torch + transformers — needs ~/models/baai/bge-m3 on disk"
current_backend="$(get_env RAG_INFERENCE_BACKEND)"
prompt BACKEND "Inference backend (tei|local)" "${current_backend:-tei}"
set_env RAG_INFERENCE_BACKEND "$BACKEND"
ok "RAG_INFERENCE_BACKEND=$BACKEND"

step "5. Hugging Face Hub token (optional)"
info "Needed only for gated/private models. BGE-M3 and bge-reranker-v2-m3 are public."
current_hf="$(get_env HUGGING_FACE_HUB_TOKEN)"
prompt HF "HF token (hf_...) — leave blank to skip" "$current_hf" 1
if [[ -n "$HF" ]]; then
    set_env HUGGING_FACE_HUB_TOKEN "$HF"
    ok "HUGGING_FACE_HUB_TOKEN set"
else
    info "Skipped — public models will still work"
fi

# ── 6. Sanity checks (non-fatal) ──────────────────────────────────────
step "6. Sanity checks"
command -v docker  >/dev/null 2>&1 && ok "docker present"           || warn "docker missing — install Docker Engine or Podman"
docker compose version >/dev/null 2>&1 && ok "docker compose v2 present" || warn "docker compose v2 missing"
if [[ -n "$SM" ]]; then
    if docker info 2>/dev/null | grep -qi nvidia; then
        ok "Docker has the nvidia runtime registered"
    else
        warn "NVIDIA GPU detected but Docker has no 'nvidia' runtime — install nvidia-container-toolkit:"
        info "  sudo apt install nvidia-container-toolkit && sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
    fi
fi

# ── 7. Summary ────────────────────────────────────────────────────────
step "Done"
cat <<EOF
  Env file: $_B$ENV_FILE$_N
  Mode:     $([[ -n "$SM" ]] && echo "${_G}GPU${_N} (${GPU_NAME})" || echo "${_Y}CPU${_N}")
  Backend:  $BACKEND
  TEI:      embed=$EMBED_TAG  rerank=$RERANK_TAG

  ${_D}config/settings.py reads everything from environment variables, so the
  values above flow through to the Python side automatically — no separate
  Python config step is needed.${_N}

  Next:
    ${_B}./scripts/compose.sh up -d${_N}              # start the stack (auto GPU/CPU)
    ${_B}./scripts/compose.sh --explain up -d${_N}    # preview without running
    ${_B}./scripts/check_env.sh${_N}                  # deeper verification

EOF
