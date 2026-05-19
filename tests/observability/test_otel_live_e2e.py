"""Live end-to-end tests for the OTel observability span trees.

These tests exercise the *public* observability API (``get_tracer``,
``backend.trace``, ``trace.span``, ``trace.generation``,
``backend.start_trace_from_carrier``) the same way the real RAG /
ingest / API call sites do, and verify the resulting traces land in
the running Langfuse stack with the expected span-name set.

Unlike :mod:`tests.observability.test_otel_live`, the two tests here
explicitly mirror real callers:

* :class:`TestRAGChainSpanTreeLands` — replays the ``retrieval.rag``
  root + ``retrieval.rerank.local`` + ``llm.generate`` shape emitted
  by :class:`src.retrieval.pipeline.rag_chain.RAGChain.run`. We
  hand-build the span tree rather than spinning up real LLM/vector
  backends so the assertion stays focused on OTLP delivery + span
  shape, not on retrieval correctness.

* :class:`TestApiQueryPostCarrierRoundTrip` — replays the
  ``api.query.post`` Starlette-middleware path that calls
  :meth:`OTelBackend.start_trace_from_carrier` with a W3C
  ``traceparent`` carrier, then verifies the inbound trace_id
  survives the round-trip through Langfuse.

Skip behavior matches ``test_otel_live.py``: the whole module skips
when ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` are absent or
the stack is unreachable. Run with::

    set -a; source .env; set +a
    uv run --extra dev pytest tests/observability/test_otel_live_e2e.py \\
        -m integration -v
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Any, Optional

import httpx
import pytest

pytestmark = pytest.mark.integration


INGEST_POLL_INTERVAL_S = 0.5
INGEST_POLL_TIMEOUT_S = 30.0
HEALTH_TIMEOUT_S = 2.0


def _langfuse_host() -> str:
    return os.environ.get("LANGFUSE_HOST", "http://localhost:3000").rstrip("/")


def _credentials() -> tuple[Optional[str], Optional[str]]:
    return (
        os.environ.get("LANGFUSE_PUBLIC_KEY"),
        os.environ.get("LANGFUSE_SECRET_KEY"),
    )


def _stack_reachable(host: str) -> bool:
    try:
        resp = httpx.get(f"{host}/api/public/health", timeout=HEALTH_TIMEOUT_S)
        return resp.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Fixtures — mirror test_otel_live.py so behavior stays consistent.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def _require_live_stack() -> None:
    """Skip the whole module unless creds + a reachable Langfuse stack exist."""
    host = _langfuse_host()
    public_key, secret_key = _credentials()
    if not public_key or not secret_key:
        pytest.skip(
            "LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set — "
            "source .env or export the keys to enable this test"
        )
    if not _stack_reachable(host):
        pytest.skip(
            f"Langfuse stack not reachable at {host} — "
            "run 'docker compose --profile observability up -d' to enable this test"
        )


@pytest.fixture(autouse=True)
def _reset_otel_global_state(monkeypatch):
    """Reset OTel global TracerProvider + observability singleton per-test.

    Mirrors the reset logic in ``test_otel_live.py`` so the conftest's
    "force noop" autouse doesn't mask exporter wiring.
    """
    import src.platform.observability as _obs_module
    import config.settings as _settings

    from opentelemetry import trace as _otel_trace

    _otel_trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]
    if hasattr(_otel_trace, "_TRACER_PROVIDER_SET_ONCE"):
        try:
            from opentelemetry.util._once import Once  # type: ignore[import-not-found]
            _otel_trace._TRACER_PROVIDER_SET_ONCE = Once()  # type: ignore[attr-defined]
        except Exception:
            pass

    monkeypatch.setattr(_settings, "OBSERVABILITY_PROVIDER", "otel", raising=False)
    monkeypatch.setenv("RAG_OBSERVABILITY_PROVIDER", "otel")
    monkeypatch.setenv("OBSERVABILITY_PROVIDER", "otel")
    _obs_module._backend = None

    yield

    backend = _obs_module._backend
    if backend is not None:
        try:
            backend.flush()
            backend.shutdown()
        except Exception:
            pass
    _obs_module._backend = None
    _otel_trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _poll_for_trace(
    host: str,
    auth: tuple[str, str],
    run_id: str,
    timeout_s: float = INGEST_POLL_TIMEOUT_S,
    interval_s: float = INGEST_POLL_INTERVAL_S,
) -> tuple[dict[str, Any], float]:
    """Poll ``/api/public/traces`` until one tagged ``e2e_run_id=run_id`` appears."""
    start = time.monotonic()
    deadline = start + timeout_s
    last_response_text = ""

    while time.monotonic() < deadline:
        try:
            resp = httpx.get(
                f"{host}/api/public/traces",
                auth=auth,
                params={"limit": 50},
                timeout=5.0,
            )
            last_response_text = resp.text
            if resp.status_code == 200:
                traces = resp.json().get("data", []) or []
                for t in traces:
                    md = t.get("metadata") or {}
                    if not isinstance(md, dict):
                        continue
                    flat_id = md.get("e2e_run_id")
                    nested_id = (md.get("attributes") or {}).get("e2e_run_id")
                    if flat_id == run_id or nested_id == run_id:
                        return t, time.monotonic() - start
        except Exception as exc:  # pragma: no cover — diagnostic only
            last_response_text = f"<exception during poll: {exc!r}>"
        time.sleep(interval_s)

    pytest.fail(
        f"Trace with e2e_run_id={run_id} never appeared after "
        f"{timeout_s:.0f}s. Last response body (truncated): "
        f"{last_response_text[:500]}"
    )


def _poll_for_observations(
    host: str,
    auth: tuple[str, str],
    trace_id: str,
    minimum: int,
    timeout_s: float = INGEST_POLL_TIMEOUT_S,
    interval_s: float = INGEST_POLL_INTERVAL_S,
) -> list[dict]:
    """Poll ``/api/public/observations`` until at least ``minimum`` exist."""
    deadline = time.monotonic() + timeout_s
    last: list[dict] = []
    while time.monotonic() < deadline:
        resp = httpx.get(
            f"{host}/api/public/observations",
            auth=auth,
            params={"traceId": trace_id, "limit": 100},
            timeout=5.0,
        )
        if resp.status_code == 200:
            last = resp.json().get("data", []) or []
            if len(last) >= minimum:
                return last
        time.sleep(interval_s)
    pytest.fail(
        f"Trace {trace_id} only had {len(last)} observations after "
        f"{timeout_s:.0f}s; expected >= {minimum}. Names so far: "
        f"{[o.get('name') for o in last]!r}"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRAGChainSpanTreeLands:
    """Replay the RAGChain.run-style span tree and assert delivery to Langfuse."""

    def test_rag_root_with_rerank_and_llm_children_land(self) -> None:
        """Emit retrieval.rag → retrieval.rerank.local + llm.generate; verify all land.

        Mirrors the shape produced by
        :meth:`src.retrieval.pipeline.rag_chain.RAGChain.run`:

            retrieval.rag (root trace)
              ├─ retrieval.rag.process_query
              ├─ retrieval.search.hybrid
              ├─ retrieval.rerank.local
              └─ retrieval.generation
                   └─ llm.generate  (generation, gen_ai.* attributes)

        Hand-built rather than driven through real backends — the
        contract under test is OTLP delivery + span-name shape, not
        retrieval correctness.
        """
        host = _langfuse_host()
        public_key, secret_key = _credentials()
        assert public_key and secret_key
        auth = (public_key, secret_key)
        run_id = str(uuid.uuid4())

        from src.platform.observability import get_tracer
        from src.platform.observability.otel.backend import OTelBackend

        backend = get_tracer()
        assert isinstance(backend, OTelBackend)

        with backend.trace(
            "retrieval.rag",
            metadata={
                "e2e_run_id": run_id,
                "category": "e2e_rag_chain",
                "query": "what is RAG?",
            },
        ) as t:
            # process_query
            pq = t.span("retrieval.rag.process_query", attributes={"chars": 12})
            pq.end(status="ok")

            # hybrid search
            sh = t.span(
                "retrieval.search.hybrid",
                attributes={"top_k": 10, "alpha": 0.5},
            )
            sh.end(status="ok")

            # local rerank — the canonical span the assertion targets
            rr = t.span(
                "retrieval.rerank.local",
                attributes={"candidates": 10, "top_n": 3},
            )
            rr.end(status="ok")

            # generation — uses Trace.generation to get gen_ai.* attrs
            gen = t.generation(
                "llm.generate",
                model="claude-test-model",
                input="answer the question using the context",
                metadata={"system": "anthropic"},
            )
            gen.set_output("RAG combines retrieval with generation.")
            gen.set_token_counts(42, 11)
            gen.end(status="ok")

        backend.flush()

        matched, elapsed = _poll_for_trace(host, auth, run_id)
        logging.getLogger(__name__).info(
            "RAGChain trace appeared after %.2fs (run_id=%s, trace_id=%s)",
            elapsed,
            run_id,
            matched.get("id"),
        )

        # Trace-level shape.
        assert matched.get("name") == "retrieval.rag"
        trace_id = matched.get("id")
        assert trace_id

        # All four children present.
        observations = _poll_for_observations(host, auth, trace_id, minimum=4)
        names = {o.get("name") for o in observations if o.get("name")}

        expected_names = {
            "retrieval.rag.process_query",
            "retrieval.search.hybrid",
            "retrieval.rerank.local",
            "llm.generate",
        }
        missing = expected_names - names
        assert not missing, (
            f"Missing expected spans: {missing!r}. Got: {sorted(names)!r}"
        )

        # The llm.generate observation must be a GENERATION with model attached.
        gen_obs = next(o for o in observations if o.get("name") == "llm.generate")
        assert gen_obs.get("type", "").upper() == "GENERATION", (
            f"llm.generate observation should be GENERATION; got "
            f"{gen_obs.get('type')!r}"
        )
        assert gen_obs.get("model") == "claude-test-model", (
            f"Expected model 'claude-test-model'; got {gen_obs.get('model')!r}"
        )


class TestApiQueryPostCarrierRoundTrip:
    """Replay the api.query.post path with an inbound W3C traceparent carrier."""

    def test_inbound_traceparent_roundtrips_to_langfuse(self) -> None:
        """Drive ``start_trace_from_carrier`` with a synthetic traceparent and
        verify Langfuse exposes a trace whose ``id`` matches the carrier's
        trace-id (W3C trace-context round-trip).

        Mirrors the shape produced by the Starlette OTel middleware on
        :http:post:`/query`:

            api.query.post (root, parented to inbound traceparent)
              ├─ retrieval.rag.process_query
              ├─ retrieval.rerank.local
              └─ llm.generate
        """
        host = _langfuse_host()
        public_key, secret_key = _credentials()
        assert public_key and secret_key
        auth = (public_key, secret_key)
        run_id = str(uuid.uuid4())

        # --- Build a synthetic W3C traceparent. ---
        # version=00, trace-id=32 hex, parent-id=16 hex, flags=01 (sampled).
        # The Langfuse trace id should match this 32-hex string after round-trip.
        carrier_trace_hex = uuid.uuid4().hex + uuid.uuid4().hex[:0]  # 32 hex
        # uuid4().hex is 32 hex already — use it directly for clarity.
        carrier_trace_hex = uuid.uuid4().hex
        assert len(carrier_trace_hex) == 32
        parent_span_hex = uuid.uuid4().hex[:16]
        traceparent = f"00-{carrier_trace_hex}-{parent_span_hex}-01"

        from src.platform.observability import get_tracer
        from src.platform.observability.otel.backend import OTelBackend

        backend = get_tracer()
        assert isinstance(backend, OTelBackend)

        carrier = {"traceparent": traceparent}

        with backend.start_trace_from_carrier(
            "api.query.post",
            carrier=carrier,
            metadata={
                "e2e_run_id": run_id,
                "category": "e2e_api_carrier",
                "http.method": "POST",
                "http.route": "/query",
            },
        ) as t:
            t.span("retrieval.rag.process_query", attributes={"chars": 12}).end(
                status="ok"
            )
            t.span(
                "retrieval.rerank.local",
                attributes={"candidates": 5, "top_n": 3},
            ).end(status="ok")
            gen = t.generation(
                "llm.generate",
                model="claude-test-model",
                input="api query body",
                metadata={"system": "anthropic"},
            )
            gen.set_output("api answer")
            gen.set_token_counts(8, 4)
            gen.end(status="ok")

        backend.flush()

        # CRITICAL: when the trace root is parented to an inbound
        # traceparent (cross-service propagation), Langfuse creates a
        # synthetic *placeholder* trace record at the carrier's
        # trace-id — because from Langfuse's perspective the "real
        # root" lives in the upstream service. Our locally-emitted
        # ``api.query.post`` span lands as a child observation under
        # that placeholder trace, with all the attributes
        # (``e2e_run_id``, ``http.method``, ...) intact.
        #
        # We therefore poll observations by ``traceId == carrier hex``
        # and look for our root span by name there, rather than
        # expecting a top-level trace record tagged with
        # ``e2e_run_id``. This is the correct W3C trace-context
        # contract for distributed propagation.
        observations = _poll_for_observations(
            host, auth, carrier_trace_hex, minimum=4
        )
        logging.getLogger(__name__).info(
            "API-carrier observations under trace_id=%s: %s",
            carrier_trace_hex,
            [o.get("name") for o in observations],
        )

        by_name = {o.get("name"): o for o in observations if o.get("name")}
        expected = {
            "api.query.post",
            "retrieval.rag.process_query",
            "retrieval.rerank.local",
            "llm.generate",
        }
        missing = expected - set(by_name)
        assert not missing, (
            f"Missing spans under carrier trace_id={carrier_trace_hex}: "
            f"{missing!r}. Got: {sorted(by_name)!r}"
        )

        # The api.query.post observation should carry our run_id —
        # this is the load-bearing assertion that the W3C carrier
        # round-tripped to the right trace_id AND our attributes
        # survived.
        root_obs = by_name["api.query.post"]
        root_md = root_obs.get("metadata") or {}
        root_attrs = (
            root_md.get("attributes")
            if isinstance(root_md.get("attributes"), dict)
            else {}
        )
        found_run_id = root_md.get("e2e_run_id") or (root_attrs or {}).get(
            "e2e_run_id"
        )
        assert found_run_id == run_id, (
            f"api.query.post observation did not carry e2e_run_id; "
            f"metadata={root_md!r}"
        )
