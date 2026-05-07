#!/usr/bin/env bash
# Ensure the KGWeave sibling checkout exists at $KGWEAVE_REPO_PATH.
#
# RagWeave depends on the `kgweave` Python package, which lives in a separate
# repo (extracted in commits a662cd7 / f0191c5). Both the editable dev install
# (`pyproject.toml` -> [tool.uv.sources] kgweave = { path = "../KGWeave" }) and
# the Docker builds (compose `additional_contexts: kgweave: ../KGWeave`)
# resolve KGWeave from a sibling directory by default.
#
# Run this once after cloning RagWeave on a new machine:
#   ./scripts/bootstrap_kgweave.sh
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
