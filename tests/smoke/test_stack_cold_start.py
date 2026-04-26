# @summary
# Smoke test asserting the stack reaches a usable state within a time budget.
# Regression guard for PR 38 review: a hard service_healthy dep on a slow-
# warming service must not gate the whole stack via the nginx healthcheck
# chain (rag-api → rag-nginx → rag-embed-cpu).
# Deps: pytest, docker (CLI)
# @end-summary
"""Cold-start budget for the RagWeave compose stack.

Brings up the stack from a clean state and asserts that rag-nginx becomes
healthy within a budget that excludes a full TEI model download. Run sparsely
(nightly CI, before-PR-merge), not on every PR.

Run with: `pytest -m smoke tests/smoke/test_stack_cold_start.py`
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Budget for rag-nginx to become healthy. Real failures we want to catch:
# - rag-nginx hard-deps on a service with a 180s start_period (PR 38 issue).
# - rag-nginx healthcheck depends on an upstream that may not be up yet.
# Generous budget so we don't flake on slow disks; fails clearly on a 3-min
# regression.
NGINX_HEALTHY_BUDGET_SECONDS = 90

pytestmark = pytest.mark.smoke


def _docker_available() -> bool:
    return shutil.which("docker") is not None and (
        subprocess.run(["docker", "info"], capture_output=True).returncode == 0
    )


def _container_health(name: str) -> str | None:
    """Return Docker health status string, or None if container missing."""
    proc = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Health.Status}}", name],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


@pytest.fixture(scope="module")
def warm_tei_cache() -> None:
    """The first cold start downloads ~2.3 GB BGE-M3. We want this test to
    measure dependency-graph wait time, not download time. Skip if the cache
    is empty (run cold-start once locally to populate, then re-run)."""
    cache = REPO_ROOT / ".tei_cache"
    if not cache.exists() or not any(cache.iterdir()):
        pytest.skip(
            f"TEI model cache empty at {cache}. Run `docker compose up -d "
            "rag-embed` once to populate, then re-run this test."
        )


def test_rag_nginx_healthy_within_budget(warm_tei_cache) -> None:
    """rag-nginx should reach healthy state within NGINX_HEALTHY_BUDGET_SECONDS
    seconds of `compose up`. Regression guard for: PR 38's CPU pool dep
    (180s start_period) being attached as service_healthy instead of
    service_started, which serialised the whole stack through CPU model load.
    """
    if not _docker_available():
        pytest.skip("docker not available")

    # Stop the stack first to ensure a deterministic start (no leftover state).
    subprocess.run(
        ["docker", "compose", "down", "--remove-orphans"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )

    start = time.time()
    up = subprocess.run(
        ["docker", "compose", "up", "-d", "rag-nginx"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if up.returncode != 0:
        pytest.fail(f"compose up failed: {up.stderr}")

    deadline = start + NGINX_HEALTHY_BUDGET_SECONDS + 30  # grace
    while time.time() < deadline:
        status = _container_health("rag-nginx")
        if status == "healthy":
            elapsed = time.time() - start
            assert elapsed < NGINX_HEALTHY_BUDGET_SECONDS, (
                f"rag-nginx took {elapsed:.0f}s to reach healthy "
                f"(budget {NGINX_HEALTHY_BUDGET_SECONDS}s). Likely a hard "
                "service_healthy dep on a slow-warming TEI service."
            )
            return
        if status == "unhealthy":
            pytest.fail("rag-nginx reached unhealthy state during startup")
        time.sleep(2)

    pytest.fail(
        f"rag-nginx did not become healthy within "
        f"{NGINX_HEALTHY_BUDGET_SECONDS + 30}s. Inspect dependency chain — "
        "`docker compose ps` and `docker inspect rag-nginx` will show what's "
        "blocking."
    )
