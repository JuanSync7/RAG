<!-- @summary
Operations and tooling scripts: container compose wrapper, backup/restore, DR drills, load testing, auto-scaling, tuning signals, and developer utilities.
@end-summary -->

# scripts

## Overview

This directory contains operations and development tooling scripts for running, scaling, and maintaining the RAG stack.

## Quick Verification (LLM agents: read this first)

When making changes — especially to deps, imports, or containers — run these checks in order:

| Make target | What it catches | When to run |
| --- | --- | --- |
| `make py-compile-check` | Syntax errors in all `.py` files | After any Python edit |
| `make import-check` | Broken internal imports, encapsulation violations | After any import change or refactor |
| `make dep-check` | Missing/unused deps in `pyproject.toml` | After adding/removing a package |
| `make container-dep-check` | `pyproject.toml` out of sync with `containers/requirements-*.txt` | After editing requirements files |
| `make container-probe` | Worker import failures inside the built `rag-api` image | After rebuilding `rag-api` |
| `make container-probe-worker` | Worker import failures inside the built `rag-worker` image | After rebuilding `rag-worker` |
| `make test` | Full pytest suite | Before committing or after functional changes |
| `make precommit-check` | L1+L2+L3+L4+TypeScript (tracked files only) | Before every commit |

**Key rules for `containers/requirements-*.txt`:**
- `requirements-api.txt` — deps for the FastAPI server (lean ~390 MB image; no torch/ML)
- `requirements-worker.txt` — deps for the Temporal worker (heavy ~9 GB image; includes torch, docling, transformers)
- If a dep is used by code that runs in the worker (anything under `src/`, `server/activities.py`, `server/worker.py`), it must be in `requirements-worker.txt` — not just `requirements-api.txt`.
- `make container-dep-check` verifies pyproject.toml sync but does NOT validate which requirements file contains each dep. Use `make container-probe-worker` to catch missing worker deps without a full end-to-end run.
- Adding deps to the worker image increases its size (~9 GB already); small packages (litellm, prometheus-client) add ~50–200 MB each.

**Underlying scripts:**
- `scripts/check_container_deps.py` — backing script for `make container-dep-check`
- `scripts/import_check_tracked.py` — backing script for `make import-check-tracked`

---

## Container Management

| Script | Purpose |
| --- | --- |
| `compose.sh` | Docker/Podman compose wrapper with auto-runtime detection. **Use this instead of `docker-compose` directly.** Reads `docker-compose.yml` at repo root. |
| `container-runtime.sh` | Detects and exports the active container runtime (`docker` or `podman`). Sourced by `compose.sh`. |
| `stack.sh` | Stack control: status table + per-service start/stop/restart/logs + full up/down. Run `./scripts/stack.sh` for an interactive menu, or `stack.sh status\|start <svc>\|stop <svc>\|up\|down\|ui`. CLI companion to Dozzle (web UI at http://localhost:9999, always-on, start/stop buttons enabled). |

## TLS / Certificates

| Script | Purpose |
| --- | --- |
| `generate-certs.sh` | Generates mkcert locally-trusted TLS certificates for the nginx HTTPS gateway. Writes to `certs/`. |

## Backup and Recovery

| Script | Purpose |
| --- | --- |
| `backup_all.sh` | Backs up Weaviate vector data and runtime state to a timestamped archive. |
| `restore_all.sh` | Restores Weaviate data and runtime state from a backup archive. |
| `dr_drill.sh` | Disaster recovery drill: runs backup → restore → smoke test cycle. |

## Testing and Load

| Script | Purpose |
| --- | --- |
| `smoke_test.sh` | Basic smoke test against the running API (`/health`, `/query`). |
| `load_test_api.py` | Concurrency load test with configurable SLO assertions (max error rate, max p95 latency). |

## Scaling and Tuning

| Script | Purpose |
| --- | --- |
| `auto_scale_workers.py` | Auto-scales `rag-worker` replicas based on Temporal queue and API p95 saturation signals. |
| `watch_tuning_signals.py` | Continuously watches Temporal/Prometheus/GPU signals and emits scaling recommendations. |

## Developer Utilities

| Script | Purpose |
| --- | --- |
| `check_container_deps.py` | Verifies `containers/requirements-*.txt` stay in sync with `pyproject.toml`. Backing script for `make container-dep-check`. |
| `import_check_tracked.py` | Checks internal imports for git-tracked files only. Backing script for `make import-check-tracked`. |
| `warmup_docling_models.py` | Pre-downloads Docling layout and TableFormer models before ingestion runs. |
| `temporal_worker.py` | Convenience script to start a Temporal worker (alternative to `python -m server.worker`). |
| `format_spec_fr_blocks.py` | Developer utility: normalizes FR block formatting in spec Markdown documents. |
| `fix-docker-networking.sh` | Fixes Docker bridge networking on WSL2 (run via `/etc/wsl.conf` on boot — not needed manually). |

## Usage Examples

```bash
# Full verification sweep before committing
make precommit-check

# Check if container images have all required imports
make container-probe        # API image
make container-probe-worker # Worker image (catches missing ML/LLM deps)

# Start infrastructure (auto-detects Docker vs Podman)
./scripts/compose.sh --profile temporal up -d

# Full stack with monitoring
./scripts/compose.sh --profile temporal --profile app --profile workers --profile monitoring up -d

# Run load test with SLO checks
python scripts/load_test_api.py --url http://localhost:8000 --total-requests 1000 --concurrency 100

# Preview auto-scale decisions without applying
python scripts/auto_scale_workers.py --dry-run
```
