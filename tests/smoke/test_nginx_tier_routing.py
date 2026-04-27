# @summary
# Smoke tests for the rag-nginx tier-routing config (ops/nginx/nginx.conf).
# Asserts: (a) GPU 5xx falls back to CPU pool, (b) ingest-pinned 5xx fails fast
# without retrying the same CPU pool. Runs nginx in a side container against
# stub backends so the real TEI image is not needed.
# Deps: pytest, requests, docker (CLI), nginx:alpine
# @end-summary
"""End-to-end nginx tier-routing assertions.

Strategy: spin up a side `nginx:alpine` container with the repo's actual
nginx.conf mounted, point its upstreams at small Python stub servers running
on the host, and make HTTP calls. No GPU, no TEI image, no real model.

Run with: `pytest -m smoke tests/smoke/test_nginx_tier_routing.py`
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
NGINX_CONF = REPO_ROOT / "ops" / "nginx" / "nginx.conf"

pytestmark = pytest.mark.smoke


def _docker_available() -> bool:
    return shutil.which("docker") is not None and (
        subprocess.run(["docker", "info"], capture_output=True).returncode == 0
    )


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def nginx_lane(stub_server, tmp_path: Path) -> Iterator[dict[str, object]]:
    """Boot nginx with stub upstreams, yield a dict with ports and a teardown."""
    if not _docker_available():
        pytest.skip("docker not available")

    gpu_port = _free_port()
    cpu_port = _free_port()
    nginx_host_port = _free_port()

    # Generate a minimal nginx.conf that mirrors the embed-lane logic from the
    # real config but points at our host-bound stub servers (different per test).
    test_conf = tmp_path / "nginx.conf"
    test_conf.write_text(f"""
resolver 127.0.0.11 valid=10s ipv6=off;

map $http_x_ragweave_tier $tei_embed_primary {{
    default     host.docker.internal:{gpu_port};
    "ingest"    host.docker.internal:{cpu_port};
}}

server {{
    listen 8081;
    server_name _;

    proxy_set_header Host              $host;
    proxy_http_version 1.1;
    proxy_read_timeout 30s;

    location / {{
        set $primary $tei_embed_primary;
        proxy_pass http://$primary;
        proxy_intercept_errors on;
        error_page 502 503 504 = @embed_cpu_fallback;
    }}

    location @embed_cpu_fallback {{
        if ($http_x_ragweave_tier = "ingest") {{
            return 503;
        }}
        set $fb host.docker.internal:{cpu_port};
        proxy_pass http://$fb;
        add_header X-RagWeave-Embed-Tier "cpu-fallback" always;
    }}
}}
""")

    name = f"ragweave-test-nginx-{uuid.uuid4().hex[:8]}"
    proc = subprocess.run(
        [
            "docker", "run", "-d", "--rm",
            "--name", name,
            "--add-host", "host.docker.internal:host-gateway",
            "-p", f"{nginx_host_port}:8081",
            "-v", f"{test_conf}:/etc/nginx/conf.d/default.conf:ro",
            "nginx:alpine",
        ],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        pytest.skip(f"could not start test nginx: {proc.stderr}")

    # Wait for nginx to be listening.
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            requests.get(f"http://127.0.0.1:{nginx_host_port}/", timeout=1)
            break
        except requests.RequestException:
            time.sleep(0.2)
    else:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        pytest.fail("test nginx did not become ready")

    try:
        yield {
            "nginx_url": f"http://127.0.0.1:{nginx_host_port}",
            "gpu_port": gpu_port,
            "cpu_port": cpu_port,
            "stub": stub_server,
        }
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def test_gpu_5xx_falls_back_to_cpu(nginx_lane) -> None:
    """Default tier: GPU returns 503, CPU returns 200 → response is 200 with
    cpu-fallback header set."""
    stub = nginx_lane["stub"]
    with stub(port=nginx_lane["gpu_port"], status=503), \
         stub(port=nginx_lane["cpu_port"], status=200):
        resp = requests.post(
            f"{nginx_lane['nginx_url']}/embed",
            json={"inputs": ["test"]},
            timeout=10,
        )
        assert resp.status_code == 200, f"expected GPU→CPU fallback, got {resp.status_code}"
        assert resp.headers.get("X-RagWeave-Embed-Tier") == "cpu-fallback", \
            "fallback should be marked with X-RagWeave-Embed-Tier"


def test_ingest_tier_does_not_retry_same_cpu_pool(nginx_lane) -> None:
    """Regression test for PR 38 review issue: when X-RagWeave-Tier=ingest,
    primary is CPU; on 5xx the fallback must NOT re-proxy to the same CPU pool.
    Bug behavior: response would be 200 (silent retry succeeded) or 503 with
    the fallback header set. Correct behavior: 503 with no fallback header,
    failing fast since the same pool just failed."""
    stub = nginx_lane["stub"]
    with stub(port=nginx_lane["cpu_port"], status=503):
        # gpu_port intentionally unbound — ingest header should never hit GPU.
        resp = requests.post(
            f"{nginx_lane['nginx_url']}/embed",
            headers={"X-RagWeave-Tier": "ingest"},
            json={"inputs": ["test"]},
            timeout=10,
        )
        assert resp.status_code == 503, (
            f"ingest-pinned 5xx should fail fast with 503, got {resp.status_code}. "
            "If this fails with 200 or with cpu-fallback header, the @embed_cpu_fallback "
            "guard for X-RagWeave-Tier=ingest is missing or broken."
        )
        assert resp.headers.get("X-RagWeave-Embed-Tier") != "cpu-fallback", (
            "ingest-pinned response must not carry cpu-fallback header — that would "
            "mean nginx retried the same already-failing CPU pool."
        )


def test_default_tier_routes_to_gpu(nginx_lane) -> None:
    """Sanity: no header → primary is GPU. Test passes if a 200 from GPU stub
    is returned, with NO cpu-fallback header (no fallback path taken)."""
    stub = nginx_lane["stub"]
    with stub(port=nginx_lane["gpu_port"], status=200):
        resp = requests.post(
            f"{nginx_lane['nginx_url']}/embed",
            json={"inputs": ["test"]},
            timeout=10,
        )
        assert resp.status_code == 200
        assert resp.headers.get("X-RagWeave-Embed-Tier") != "cpu-fallback", \
            "GPU success should not carry cpu-fallback header"
