#!/usr/bin/env bash
# Register a host directory as a console-viewable corpus root.
#
# Idempotently extends docker-compose.override.yml (at the repo root) so that:
#   1. <dir> is bind-mounted into rag-api at the same host path
#   2. <dir> is appended to RAG_CONSOLE_SOURCE_ROOTS for rag-api
# Then recreates rag-api so the change takes effect.
#
# Usage:
#   scripts/register_corpus.sh <absolute-or-relative-dir>
#
# Example:
#   scripts/register_corpus.sh ~/RagWeave/opentitan_data
#   scripts/register_corpus.sh ./vendor_docs
#
# Notes:
#   - Only applies to local-filesystem corpora. API-mounted sources (SharePoint,
#     Confluence, Drive) need a different mechanism — see docs/console/
#     SOURCE_VIEW_CONTRACT.md.
#   - The override file is local-only (gitignored at the worktree level).

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/juansync7/RagWeave}"
OVERRIDE="${REPO_ROOT}/docker-compose.override.yml"

[[ $# -eq 1 ]] || { echo "usage: $0 <dir>" >&2; exit 2; }

DIR="$(realpath "$1")"
[[ -d "$DIR" ]] || { echo "FATAL: $DIR is not a directory" >&2; exit 1; }

# Path inside the container — same as host path so metadata.source resolves
# without translation. Requires PWD == REPO_ROOT when compose is invoked.
MOUNT_LINE="      - ${DIR}:${DIR}:ro"

if [[ ! -f "$OVERRIDE" ]]; then
    cat > "$OVERRIDE" <<EOF
services:
  rag-api:
    environment:
      - RAG_CONSOLE_SOURCE_ROOTS=${DIR}
    volumes:
${MOUNT_LINE}
EOF
    echo "Created $OVERRIDE with $DIR registered."
else
    if grep -qF "$MOUNT_LINE" "$OVERRIDE"; then
        echo "Already registered: $DIR"
    else
        echo "Registering $DIR — please review the diff before restart:"
        echo
        echo "  Add to rag-api volumes:    $MOUNT_LINE"
        echo "  Append to RAG_CONSOLE_SOURCE_ROOTS: ${DIR}"
        echo
        echo "Edit $OVERRIDE manually, then run:"
        echo "  cd $REPO_ROOT && docker compose up -d rag-api"
        exit 0
    fi
fi

cd "$REPO_ROOT"
docker compose up -d rag-api
echo "rag-api recreated. Try the View button now."
