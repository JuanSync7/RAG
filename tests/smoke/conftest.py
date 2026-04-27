# @summary
# Smoke-test fixtures: small HTTP stub servers used to simulate failing TEI
# replicas without booting the real BGE-M3 image. Used by the nginx tier-
# routing tests to assert behavior on 5xx, fallback, and ingest-pinning.
# Exports: stub_server (factory fixture)
# Deps: pytest, http.server, threading
# @end-summary
"""Fixtures for tests/smoke/ — stub HTTP servers replacing TEI replicas."""

from __future__ import annotations

import threading
from collections.abc import Callable
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Iterator

import pytest


def _make_handler(status: int, body: bytes = b"{}") -> type[BaseHTTPRequestHandler]:
    """Build a request handler that always returns ``status``."""

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802
            self.do_GET()

        def log_message(self, *_args: object) -> None:  # silence
            return

    return _Handler


@contextmanager
def _serve(port: int, status: int, body: bytes = b"{}") -> Iterator[HTTPServer]:
    server = HTTPServer(("127.0.0.1", port), _make_handler(status, body))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def stub_server() -> Callable[..., object]:
    """Factory yielding a stub HTTP server on a chosen port returning a fixed status.

    Usage:
        with stub_server(port=18080, status=503) as srv:
            ...  # any request to 127.0.0.1:18080 returns 503
    """
    return _serve
