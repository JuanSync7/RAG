#!/usr/bin/env bash
# Optional: clone a local KGWeave sibling checkout for co-development.
#
# RagWeave normally consumes KGWeave from a pinned git ref — Python deps via
# `[tool.uv.sources]` in pyproject.toml, container builds via the
# `KGWEAVE_BUILD_CONTEXT` default in docker-compose.yml. A fresh clone of
# RagWeave does NOT require this script.
#
# Run this only if you want to edit KGWeave alongside RagWeave:
#   ./scripts/bootstrap_kgweave.sh
# Then point compose at the local checkout:
#   echo 'KGWEAVE_BUILD_CONTEXT=../KGWeave' >> .env
# And switch the uv source line locally (do NOT commit) to:
#   kgweave = { path = "../KGWeave", editable = true }
#
# Override the remote URL, ref, or local path with env vars if needed:
#   KGWEAVE_REMOTE=git@github.com:JuanSync7/KGWeave.git \
#   KGWEAVE_REF=v0.1.0 \
#   KGWEAVE_REPO_PATH=../KGWeave \
#   ./scripts/bootstrap_kgweave.sh

set -euo pipefail

KGWEAVE_REMOTE="${KGWEAVE_REMOTE:-https://github.com/JuanSync7/KGWeave.git}"
KGWEAVE_REF="${KGWEAVE_REF:-main}"
KGWEAVE_REPO_PATH="${KGWEAVE_REPO_PATH:-../KGWeave}"

if [ -d "$KGWEAVE_REPO_PATH/.git" ]; then
    if git -C "$KGWEAVE_REPO_PATH" remote get-url origin >/dev/null 2>&1; then
        echo "[bootstrap] KGWeave already present at $KGWEAVE_REPO_PATH; updating to $KGWEAVE_REF"
        git -C "$KGWEAVE_REPO_PATH" fetch --tags origin
        git -C "$KGWEAVE_REPO_PATH" checkout "$KGWEAVE_REF"
        # Fast-forward only if we're on a branch (not a detached tag/sha checkout).
        if git -C "$KGWEAVE_REPO_PATH" symbolic-ref -q HEAD >/dev/null; then
            git -C "$KGWEAVE_REPO_PATH" pull --ff-only
        fi
    else
        echo "[bootstrap] KGWeave checkout has no 'origin' remote — leaving it as-is."
        echo "[bootstrap] To enable updates, add one: git -C $KGWEAVE_REPO_PATH remote add origin <url>"
    fi
else
    if [ -e "$KGWEAVE_REPO_PATH" ]; then
        echo "[bootstrap] $KGWEAVE_REPO_PATH exists but is not a git checkout — refusing to overwrite." >&2
        exit 1
    fi
    echo "[bootstrap] cloning $KGWEAVE_REMOTE -> $KGWEAVE_REPO_PATH (ref: $KGWEAVE_REF)"
    git clone "$KGWEAVE_REMOTE" "$KGWEAVE_REPO_PATH"
    git -C "$KGWEAVE_REPO_PATH" checkout "$KGWEAVE_REF"
fi

echo "[bootstrap] KGWeave ready at $(cd "$KGWEAVE_REPO_PATH" && pwd) ($(git -C "$KGWEAVE_REPO_PATH" rev-parse --short HEAD))"
