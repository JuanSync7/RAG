# @summary
# Root developer shortcuts for setup, console TypeScript, Python sanity checks, and testing.
# Exports: make targets
# Deps: npm, uv, server/console/web package scripts
# @end-summary

.PHONY: help install console-install console-check console-build console-watch \
        py-compile-check test dep-check import-check import-check-tracked \
        all-check precommit-check setup restart restart-all \
        start start-all dev worker tunnel restart-worker scale-workers \
        venv-doctor _venv-auto-heal \
        container-build container-build-api container-build-worker \
        container-build-podman container-probe container-probe-worker \
        container-sizes container-clean \
        smoke-test container-build-and-test

# Default target: print the help menu when `make` is run with no arguments.
.DEFAULT_GOAL := help

# ---------------------------------------------------------------------------
# Help
#
# Grouped listing of targets. Keep this in sync with README.md's
# "Make Targets" section — both derive from the same Makefile.
# ---------------------------------------------------------------------------
help:
	@echo ""
	@echo "RagWeave — developer make targets"
	@echo ""
	@echo "Setup & install"
	@echo "  setup              Full one-shot setup: venv + deps + console install + build"
	@echo "  install            (Re)install Python deps into the active env (editable + dev extras)"
	@echo "  venv-doctor        Check .venv health (detects stale shebangs after dir rename)"
	@echo ""
	@echo "Web console (TypeScript)"
	@echo "  console-install    npm install for the web console"
	@echo "  console-check      TypeScript type-check (no emit)"
	@echo "  console-build      Compile TS -> static/main.js"
	@echo "  console-watch      Watch mode — rebuild on TS change"
	@echo ""
	@echo "Static checks (4 layers — each catches a distinct bug class)"
	@echo "  py-compile-check       L1 syntax: compileall across src/ server/ config/ import_check/"
	@echo "  import-check           L2 internal: resolve imports + encapsulation (whole tree)"
	@echo "  import-check-tracked   L2 internal but only for git-tracked files (for precommit)"
	@echo "  dep-check              L3 external: pyproject.toml vs imports (deptry)"
	@echo "  container-dep-check    L4 container: requirements-*.txt in sync with pyproject.toml"
	@echo ""
	@echo "Compound gates"
	@echo "  precommit-check    L1 + L2(tracked) + L3 + L4 + npm ci + console-check (excludes WIP)"
	@echo "  all-check          L1 + L2(all)     + L3 + L4 + npm ci + console-check (full sweep)"
	@echo ""
	@echo "Tests"
	@echo "  test               Run the pytest suite (not included in all-check)"
	@echo ""
	@echo "Container lifecycle (docker or podman — auto-detected)"
	@echo "  container-build         Build rag-api + rag-worker with docker (BuildKit)"
	@echo "  container-build-api     Build only rag-api"
	@echo "  container-build-worker  Build only rag-worker"
	@echo "  container-build-podman  Build both with podman (--format docker)"
	@echo "  container-probe             Run the API import probe inside rag-api"
	@echo "  container-probe-worker      Run the worker import probe inside rag-worker (catches missing deps)"
	@echo "  container-sizes             Print current rag-api / rag-worker image sizes"
	@echo "  container-clean             Remove local rag-api / rag-worker images + dangling images"
	@echo "  smoke-test                  Full integration check: build + stack + tunnel + checks"
	@echo "  container-build-and-test    Build images then immediately run smoke-test (SKIP_BUILD=1)"
	@echo ""
	@echo "Daily dev startup (cold start after WSL2 restart)"
	@echo "  restart-worker     Rebuild + restart rag-worker (use after .env or code changes)"
	@echo "  start              Bring up infra + workers (no rebuild)"
	@echo "  start-all          Bring up FULL stack (all profiles incl. monitoring/langfuse)"
	@echo "  dev                Start uvicorn with hot-reload (run in its own terminal)"
	@echo "  worker             Start Temporal worker locally (run in its own terminal)"
	@echo "  scale-workers      Scale containerised workers: make scale-workers N=3"
	@echo "  tunnel             Start Cloudflare trycloudflare.com tunnel (run in its own terminal)"
	@echo ""
	@echo "Stack restart (uses scripts/stack.sh — auto-detects docker/podman)"
	@echo "  restart            Restart app + workers with rebuild"
	@echo "  restart-all        Restart all profiles with rebuild"
	@echo ""

# ---------------------------------------------------------------------------
# Virtual env health
#
# pip/uv install console scripts (deptry, pytest, uvicorn, ...) with an
# ABSOLUTE-path shebang pointing at the venv's python. If the project
# directory is renamed after setup, every console script in .venv/bin/
# silently breaks with "Permission denied" or "No such file or directory".
# The python interpreter itself (.venv/bin/python3 -> /usr/bin/python3 or
# similar) is a symlink and keeps working, which makes the failure mode
# confusing — `python ...` works but `pytest` / `deptry` / `uvicorn` don't.
#
# `venv-doctor` detects the mismatch by comparing a known console script's
# shebang to the current project path. `_venv-auto-heal` is the silent
# version used as an implicit dep by `setup` and `install` so repair is
# automatic on re-run. Neither touches .venv unless the mismatch is real.
# ---------------------------------------------------------------------------

# Verbose health check — prints either "ok" or a diagnosis.
venv-doctor:
	@if [ ! -d .venv ]; then \
		echo "venv-doctor: no .venv/ present — run 'make setup'"; exit 1; \
	elif [ ! -x .venv/bin/python3 ]; then \
		echo "venv-doctor: .venv/ exists but has no python3 — run 'rm -rf .venv && make setup'"; exit 1; \
	else \
		expected="$$(pwd)/.venv/bin/python3"; \
		for script in .venv/bin/pytest .venv/bin/pip .venv/bin/python3; do \
			if [ -f "$$script" ]; then \
				first="$$(head -c 2 $$script)"; \
				if [ "$$first" = "#!" ]; then \
					actual="$$(head -n 1 $$script | sed 's|^#!||')"; \
					if [ "$$actual" != "$$expected" ]; then \
						echo "venv-doctor: STALE venv detected"; \
						echo "  $$script shebang:   $$actual"; \
						echo "  expected:          $$expected"; \
						echo "  cause:             project directory was renamed after 'make setup'"; \
						echo "  fix:               rm -rf .venv && make setup"; \
						echo "                     (or: make install — auto-heals before reinstalling)"; \
						exit 1; \
					fi; \
					break; \
				fi; \
			fi; \
		done; \
		echo "venv-doctor: .venv/ healthy"; \
	fi

# Silent version used as a hidden dep of setup/install. Nukes and rebuilds
# .venv ONLY when a stale shebang is detected — no-op on a healthy venv.
# Detects via the python3 wrapper script (uv sync may not install pip), with
# pip as a secondary check for legacy venvs.
_venv-auto-heal:
	@if [ -d .venv ]; then \
		expected="$$(pwd)/.venv/bin/python3"; \
		probe=""; \
		for f in .venv/bin/pip .venv/bin/pytest; do \
			if [ -f "$$f" ] && [ "$$(head -c 2 $$f)" = "#!" ]; then probe="$$f"; break; fi; \
		done; \
		if [ -n "$$probe" ]; then \
			actual="$$(head -n 1 $$probe | sed 's|^#!||')"; \
			if [ "$$actual" != "$$expected" ]; then \
				echo ">>> Stale venv detected ($$probe shebang $$actual)"; \
				echo ">>> Removing .venv/ so it can be recreated at $$expected"; \
				rm -rf .venv; \
			fi; \
		fi; \
	fi

# Full project setup (Python + TypeScript). Auto-heals a stale venv first.
# Uses `uv sync --extra dev` so installs respect uv.lock (reproducible) and
# auto-create the venv. Plain pip is banned — the project is uv-only.
setup: _venv-auto-heal
	uv sync --extra dev
	$(MAKE) console-install
	$(MAKE) console-build
	@echo "\n✓ Setup complete. Activate with: source .venv/bin/activate"

# (Re)install Python deps into the active env. Auto-heals a stale venv
# first so `make install` after a directory rename Just Works.
install: _venv-auto-heal
	uv sync --extra dev

console-install:
	npm --prefix server/console/web install

console-check:
	npm --prefix server/console/web run check

console-build:
	npm --prefix server/console/web run build

console-watch:
	npm --prefix server/console/web run watch

# ---------------------------------------------------------------------------
# Static checks & tests
#
# Three Python checks that stack in layers — each catches a distinct bug class:
#
#   py-compile-check  Layer 1: does every .py parse as valid Python syntax?
#                     (python -m compileall, fast, skips unchanged files)
#   import-check      Layer 2: do project-internal `from X import Y` statements
#                     resolve to real symbols? Catches stale imports after
#                     refactors and encapsulation violations.
#   dep-check         Layer 3: is pyproject.toml in sync with actual third-party
#                     imports? (deptry — unused/missing/misplaced deps)
#
# Two compound gates use these layers in different scopes:
#
#   precommit-check   Runs L1+L2+L3+TS over files git ALREADY knows about.
#                     L2 uses the `import-check-tracked` variant which reads
#                     `git ls-files` so untracked WIP (e.g. feature branches
#                     with new modules you haven't committed yet) doesn't
#                     block the commit gate. Use before every commit.
#
#   all-check         Runs the same checks over the ENTIRE tree, including
#                     untracked files. Use before a release, on CI, or as
#                     a periodic hygiene sweep. This is the one that will
#                     catch issues in the work-in-progress you're about to
#                     commit.
#
# Neither gate runs the pytest suite — run `make test` separately. Keeping
# the gates fast is intentional: devs skip slow gates.
# ---------------------------------------------------------------------------

# Layer 1: syntax-only check on the whole source tree. compileall is fast
# because it skips .py files whose cached .pyc is already current.
py-compile-check:
	uv run python -m compileall -q src server config import_check

test:
	uv run pytest

# Layer 3: deptry verifies pyproject.toml matches actual imports.
# Invoked via `python -m deptry` instead of the deptry console script so it
# survives venv-path rot (e.g. directory renames that break cached shebangs).
dep-check:
	uv run python -m deptry .

# Layer 4: verify containers/requirements-*.txt stay in sync with pyproject.toml.
# Catches deps added to a requirements file but missing from pyproject.toml, and
# pyproject.toml deps absent from both container files without being allowlisted.
container-dep-check:
	uv run python scripts/check_container_deps.py

# Layer 2 (all files): internal import resolution + encapsulation violations.
# Scans every .py file in the source directories, including untracked WIP.
import-check:
	uv run python -m import_check

# Layer 2 (tracked only): same check but limited to git-tracked files.
# Used by `precommit-check` so untracked WIP doesn't block commits.
import-check-tracked:
	uv run python scripts/import_check_tracked.py

# Pre-commit gate. Runs the fast static checks against TRACKED files only.
# Order: cheap -> expensive so failures surface the lowest-level issue first.
precommit-check:
	$(MAKE) py-compile-check
	$(MAKE) import-check-tracked
	$(MAKE) dep-check
	$(MAKE) container-dep-check
	npm --prefix server/console/web ci
	$(MAKE) console-check

# Full hygiene sweep. Same checks but over the entire tree including
# untracked WIP. Use before releases, in CI, or periodically.
all-check:
	$(MAKE) py-compile-check
	$(MAKE) import-check
	$(MAKE) dep-check
	$(MAKE) container-dep-check
	npm --prefix server/console/web ci
	$(MAKE) console-check

# ---------------------------------------------------------------------------
# Daily dev startup
#
# Cold-start sequence after a WSL2 restart:
#   1. (WSL2 only, auto via /etc/wsl.conf) fix-docker-networking.sh
#   2. make start          — infra + workers in background
#   3. make dev            — uvicorn in a dedicated terminal (hot-reload)
#   4. make tunnel         — cloudflared in a dedicated terminal
#
# `make start` is idempotent — safe to re-run if containers are already up.
# ---------------------------------------------------------------------------

restart-worker:
	./scripts/compose.sh --profile workers build rag-worker
	./scripts/compose.sh --profile workers up -d --force-recreate rag-worker

# Scale rag-worker horizontally. Usage: make scale-workers N=3
scale-workers:
	./scripts/compose.sh --profile workers up -d --scale rag-worker=$${N:-2}

start:
	./scripts/compose.sh up -d
	./scripts/compose.sh --profile workers up -d

# Bring up the full stack: every profile (app, workers, monitoring,
# observability, gateway, temporal). Equivalent to `./scripts/stack.sh up`.
start-all:
	./scripts/stack.sh up

dev:
	uv run uvicorn server.api:app --host 0.0.0.0 --port 8000 --reload

worker:
	uv run python -m server.worker

tunnel:
	cloudflared tunnel --url http://localhost:8000

# Mirrors `make start` (base + workers) but rebuilds images and force-recreates.
restart: console-build
	./scripts/stack.sh restart

# Mirrors `make start-all` (every profile) but rebuilds images and force-recreates.
restart-all: console-build
	./scripts/stack.sh restart-all

# ---------------------------------------------------------------------------
# Container images
#
# Two images are built from separate dependency lists (see containers/):
#   - rag-api      ~390 MB  (fastapi, temporalio, weaviate — no ML stack)
#   - rag-worker    ~5.8 GB  (torch, docling, transformers — full ML stack)
#
# Dependencies are tracked in containers/requirements-{api,worker}.txt, NOT
# in pyproject.toml. Local dev uses pyproject.toml via `make install`;
# containers bypass it for strict dep isolation. When adding a new dep:
#   - API-side import → add to pyproject.toml AND containers/requirements-api.txt
#   - Worker-side    → add to pyproject.toml AND containers/requirements-worker.txt
# ---------------------------------------------------------------------------

# Default: build both images with Docker (BuildKit enabled).
# console-build runs first so the compiled frontend JS is baked into rag-api.
container-build: console-build container-build-api container-build-worker container-sizes

container-build-api:
	DOCKER_BUILDKIT=1 docker build -t rag-api -f containers/Dockerfile.api .

container-build-worker:
	DOCKER_BUILDKIT=1 docker build -t rag-worker -f containers/Dockerfile.runtime .

# Build with Podman instead of Docker. Passes --format docker so HEALTHCHECK
# directives are preserved if ever re-added (the compose-level healthcheck
# works regardless of image format).
# console-build runs first so the compiled frontend JS is baked into rag-api.
container-build-podman: console-build
	podman build --format docker -t rag-api    -f containers/Dockerfile.api .
	podman build --format docker -t rag-worker -f containers/Dockerfile.runtime .
	@$(MAKE) container-sizes

# Run the import probe inside the built API image. Catches any transitive
# ML/torch dep leakage after dep-split changes. Detects docker vs podman.
container-probe:
	@if command -v docker >/dev/null 2>&1; then \
		docker run --rm --entrypoint python rag-api \
			-c "from server.api import app; print('[probe] API import OK')"; \
	elif command -v podman >/dev/null 2>&1; then \
		podman run --rm --entrypoint python rag-api \
			-c "from server.api import app; print('[probe] API import OK')"; \
	else \
		echo "neither docker nor podman found"; exit 1; \
	fi

# Run the import probe inside the built worker image. Catches missing deps in
# requirements-worker.txt (e.g. litellm, prometheus-client) that don't surface
# until the worker tries to serve a real activity. Detects docker vs podman.
container-probe-worker:
	@if command -v docker >/dev/null 2>&1; then \
		docker run --rm --entrypoint "" ragweave-rag-worker \
			sh -c 'cd /app && python -c "from server.activities import execute_rag_query, init_rag_chain; print(\"[probe] worker import OK\")"'; \
	elif command -v podman >/dev/null 2>&1; then \
		podman run --rm --entrypoint "" ragweave-rag-worker \
			sh -c 'cd /app && python -c "from server.activities import execute_rag_query, init_rag_chain; print(\"[probe] worker import OK\")"'; \
	else \
		echo "neither docker nor podman found"; exit 1; \
	fi

# Print current image sizes (podman reports actual disk usage, docker
# underreports via .Size but shows true sum via 'images').
container-sizes:
	@echo "=== Image sizes ==="
	@if command -v podman >/dev/null 2>&1; then \
		podman images --format "  {{.Repository}}:{{.Tag}}\t{{.Size}}" | grep -E "rag-(api|worker)" || true; \
	fi
	@if command -v docker >/dev/null 2>&1; then \
		docker images --format "  {{.Repository}}:{{.Tag}}\t{{.Size}}" | grep -E "rag-(api|worker)" || true; \
	fi

# Remove locally-built rag-api / rag-worker images and dangling (<none>) images from both engines
container-clean:
	-docker rmi rag-api rag-worker 2>/dev/null || true
	-docker image prune -f 2>/dev/null || true
	-podman rmi rag-api rag-worker 2>/dev/null || true
	-podman image prune -f 2>/dev/null || true

# Run the full integration smoke test (build + stack + cloudflare tunnel + checks + teardown).
# Pass SKIP_BUILD=1 to reuse already-built images.
smoke-test:
	SKIP_BUILD=$${SKIP_BUILD:-0} bash scripts/smoke_test.sh

# Build images then immediately run the smoke test. The smoke test skips its
# own build step since container-build already produced fresh images.
container-build-and-test: container-build
	SKIP_BUILD=1 bash scripts/smoke_test.sh
