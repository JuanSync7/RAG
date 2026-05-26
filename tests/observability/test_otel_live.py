"""Live end-to-end test for the OTel observability backend against Langfuse.

This exercises the full path:

    Python -> OTel SDK -> OTLP/HTTP exporter -> Langfuse OTLP ingest
           -> Langfuse worker -> Langfuse REST API.

Each test emits a trace tagged with a per-run UUID, then polls the Langfuse
REST trace listing until the trace is visible. Observations on the trace are
then asserted against the gen_ai.* semantic conventions emitted by
:class:`src.platform.observability.otel.backend.OTelBackend`.

Skip behavior:
    The whole module skips uniformly when either
        - the Langfuse stack is unreachable at ``LANGFUSE_HOST``, or
        - ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` env vars are absent.

    Skips are emitted with an actionable message pointing operators at
    ``docker compose --profile observability up -d``. Default
    ``pytest tests/`` runs without the stack therefore succeed (collect + skip),
    and the test is selectable explicitly via ``-m integration``.

Cleanup:
    Each test uses a fresh UUID, so live runs accumulate dev traces. We do
    NOT delete traces afterward — Langfuse dev volumes are cheap, and the
    audit trail is useful when debugging the integration.

Concurrency/state:
    The OTel global ``TracerProvider`` is process-global. We reset it before
    each test (and re-build the backend) so a stale exporter from another
    test cannot mask wiring bugs.
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Any, Optional

import httpx
import pytest

# ---------------------------------------------------------------------------
# Module-wide marker — all tests here are live integration tests.
# ---------------------------------------------------------------------------
pytestmark = [pytest.mark.slow, pytest.mark.integration]


# ---------------------------------------------------------------------------
# Constants — chosen so we can fail loudly on slow ingest while still
# tolerating typical Langfuse worker latency (~200ms-2s).
# ---------------------------------------------------------------------------
INGEST_POLL_INTERVAL_S = 0.5
INGEST_POLL_TIMEOUT_S = 30.0
HEALTH_TIMEOUT_S = 2.0


def _langfuse_host() -> str:
    """Return the configured Langfuse host URL (no trailing slash)."""
    return os.environ.get("LANGFUSE_HOST", "http://localhost:3000").rstrip("/")


def _credentials() -> tuple[Optional[str], Optional[str]]:
    """Return the (public_key, secret_key) tuple from the environment."""
    return (
        os.environ.get("LANGFUSE_PUBLIC_KEY"),
        os.environ.get("LANGFUSE_SECRET_KEY"),
    )


def _stack_reachable(host: str) -> bool:
    """Return True iff ``GET <host>/api/public/health`` answers 200 within 2s."""
    try:
        resp = httpx.get(f"{host}/api/public/health", timeout=HEALTH_TIMEOUT_S)
        return resp.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def _require_live_stack() -> None:
    """Skip the whole module unless creds + a reachable Langfuse stack exist.

    Runs once per module — skip is uniform across all tests here.
    """
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

    The OTel SDK lazily caches the global ``TracerProvider`` and (more
    importantly) any ``BatchSpanProcessor`` attached to it. Tests in other
    files may have installed a provider with no exporter; if we don't reset
    it, ``OTelBackend.__init__`` will see "provider already configured" and
    skip exporter installation, silently dropping our traces.

    We override the autouse ``reset_observability_singleton`` fixture's
    "force noop" behavior here by setting RAG_OBSERVABILITY_PROVIDER=otel
    explicitly. (That fixture still runs first and pre-sets noop; we
    override afterward in this fixture, which runs second within the
    same test setup.)
    """
    import src.platform.observability as _obs_module
    import config.settings as _settings

    # Reset the OTel global state so OTelBackend.__init__ installs a fresh
    # provider + exporter for this test. There is no public OTel API for
    # this; the SDK exposes _TRACER_PROVIDER_SET_ONCE as the gate. We
    # reach into the module to clear it. This is brittle by design — it
    # documents our dependency on a stable provider lifecycle.
    from opentelemetry import trace as _otel_trace
    from opentelemetry.trace import ProxyTracerProvider

    # Replace the global provider with a fresh proxy so the next
    # set_tracer_provider() call succeeds. The "set once" flag must be
    # cleared so set_tracer_provider() in OTelBackend.__init__ actually
    # installs the SDK provider.
    _otel_trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]
    if hasattr(_otel_trace, "_TRACER_PROVIDER_SET_ONCE"):
        # opentelemetry-api uses an internal Once flag; reset by replacing
        # with a fresh instance of the same class.
        try:
            from opentelemetry.util._once import Once  # type: ignore[import-not-found]
            _otel_trace._TRACER_PROVIDER_SET_ONCE = Once()  # type: ignore[attr-defined]
        except Exception:
            # Older OTel versions store the flag differently; best-effort.
            pass

    # Force RAG_OBSERVABILITY_PROVIDER=otel so get_tracer() builds OTelBackend.
    # (The module-level autouse conftest set it to "noop" — we override here.)
    monkeypatch.setattr(_settings, "OBSERVABILITY_PROVIDER", "otel", raising=False)
    monkeypatch.setenv("RAG_OBSERVABILITY_PROVIDER", "otel")
    monkeypatch.setenv("OBSERVABILITY_PROVIDER", "otel")
    _obs_module._backend = None

    yield

    # Best-effort teardown — flush any pending spans so a leaked test
    # doesn't bleed exports into a later test's poll window.
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
    """Poll the trace list endpoint until a trace with the given run_id appears.

    Args:
        host: Base URL of the Langfuse stack.
        auth: ``(public_key, secret_key)``.
        run_id: Unique per-test UUID stamped into trace metadata.
        timeout_s: Max wall-clock seconds to poll.
        interval_s: Sleep between polls.

    Returns:
        ``(matched_trace_dict, elapsed_seconds)``.

    Raises:
        ``pytest.fail`` if no trace appears within ``timeout_s``.
    """
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
                    # Langfuse nests OTel span attributes under
                    # metadata.attributes (not flat metadata). We accept
                    # either shape for forward compatibility.
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


def _get_observations(host: str, auth: tuple[str, str], trace_id: str) -> list[dict]:
    """List observations for a Langfuse trace by id."""
    resp = httpx.get(
        f"{host}/api/public/observations",
        auth=auth,
        params={"traceId": trace_id, "limit": 50},
        timeout=5.0,
    )
    resp.raise_for_status()
    return resp.json().get("data", []) or []


def _poll_for_observations(
    host: str,
    auth: tuple[str, str],
    trace_id: str,
    minimum: int,
    timeout_s: float = INGEST_POLL_TIMEOUT_S,
    interval_s: float = INGEST_POLL_INTERVAL_S,
) -> list[dict]:
    """Poll observations endpoint until at least ``minimum`` observations exist."""
    deadline = time.monotonic() + timeout_s
    last: list[dict] = []
    while time.monotonic() < deadline:
        last = _get_observations(host, auth, trace_id)
        if len(last) >= minimum:
            return last
        time.sleep(interval_s)
    pytest.fail(
        f"Trace {trace_id} only had {len(last)} observations after "
        f"{timeout_s:.0f}s; expected >= {minimum}. Last: {last!r}"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOTelLiveTraceRoundTrip:
    """End-to-end live test: emit traces, verify they land in Langfuse.

    These tests prove that the wiring from :class:`OTelBackend` through the
    OTLP/HTTP exporter to the Langfuse OTLP ingest is correct, including
    the Basic-auth header derivation in
    :func:`src.platform.observability.otel.auth.ensure_otlp_headers_from_env`.
    """

    def test_happy_path_trace_with_span_and_generation_lands(self) -> None:
        """Emit trace + child span + generation; verify Langfuse exposes them."""
        host = _langfuse_host()
        public_key, secret_key = _credentials()
        assert public_key and secret_key  # guarded by module fixture
        auth = (public_key, secret_key)
        run_id = str(uuid.uuid4())

        from src.platform.observability import get_tracer

        backend = get_tracer()
        # Sanity: the factory must have given us the OTel backend, not noop.
        from src.platform.observability.otel.backend import OTelBackend
        assert isinstance(backend, OTelBackend), (
            f"Expected OTelBackend, got {type(backend).__name__}. "
            "Check RAG_OBSERVABILITY_PROVIDER and config.settings."
        )

        # Emit the trace tree.
        with backend.trace(
            "e2e.live.smoke",
            metadata={"e2e_run_id": run_id, "category": "e2e"},
        ) as t:
            child = t.span("e2e.child", attributes={"step": 1})
            child.end(status="ok")
            gen = t.generation(
                "e2e.llm",
                model="claude-test-model",
                input="hello world",
                metadata={"system": "anthropic"},
            )
            gen.set_output("bye world")
            gen.set_token_counts(5, 7)
            gen.end(status="ok")

        # CRITICAL: BatchSpanProcessor buffers; without flush, the test may
        # finish before the exporter posts anything. This is the #1 thing
        # that bites people writing OTel integration tests.
        backend.flush()

        matched, elapsed = _poll_for_trace(host, auth, run_id)
        logging.getLogger(__name__).info(
            "Trace appeared after %.2fs (run_id=%s)", elapsed, run_id
        )

        # --- Trace-level assertions ---
        assert matched.get("name") == "e2e.live.smoke", (
            f"Unexpected trace name: {matched.get('name')!r}; full: {matched!r}"
        )
        metadata = matched.get("metadata") or {}
        assert isinstance(metadata, dict)
        # Langfuse OTLP ingest nests span attributes under metadata.attributes.
        attrs = (metadata.get("attributes") or {}) if isinstance(metadata.get("attributes"), dict) else {}
        assert (metadata.get("e2e_run_id") or attrs.get("e2e_run_id")) == run_id, (
            f"e2e_run_id missing from metadata; full metadata={metadata!r}"
        )
        assert (metadata.get("category") or attrs.get("category")) == "e2e", (
            f"category missing from metadata; full metadata={metadata!r}"
        )

        trace_id = matched.get("id")
        assert trace_id, f"Matched trace has no id: {matched!r}"

        # --- Observations ---
        observations = _poll_for_observations(host, auth, trace_id, minimum=2)
        # Log shape for future maintainers (visible with -s).
        logging.getLogger(__name__).info(
            "Observations JSON shape (one example): %r",
            observations[0] if observations else None,
        )

        by_name: dict[str, dict] = {}
        for o in observations:
            name = o.get("name")
            if name:
                by_name[name] = o

        assert "e2e.child" in by_name, (
            f"Expected an observation named 'e2e.child'; got names "
            f"{[o.get('name') for o in observations]!r}"
        )
        assert "e2e.llm" in by_name, (
            f"Expected an observation named 'e2e.llm'; got names "
            f"{[o.get('name') for o in observations]!r}"
        )

        # Child span: 'step' attribute should round-trip via metadata or
        # attributes. Langfuse stores OTel span attributes under 'metadata'.
        child_obs = by_name["e2e.child"]
        child_md = child_obs.get("metadata") or {}
        step_val: Any = None
        if isinstance(child_md, dict):
            child_attrs = child_md.get("attributes") if isinstance(child_md.get("attributes"), dict) else {}
            step_val = child_md.get("step") or (child_attrs or {}).get("step")
        # Some Langfuse versions stringify attribute values.
        assert step_val in (1, "1"), (
            f"Expected child span 'step' == 1; got {step_val!r}. "
            f"Full observation: {child_obs!r}"
        )

        # Generation: model + output + tokens.
        gen_obs = by_name["e2e.llm"]
        assert gen_obs.get("type", "").upper() == "GENERATION", (
            f"Expected observation type GENERATION; got {gen_obs.get('type')!r}. "
            f"Full: {gen_obs!r}"
        )
        assert gen_obs.get("model") == "claude-test-model", (
            f"Expected model 'claude-test-model'; got {gen_obs.get('model')!r}"
        )
        # The output is mapped from gen_ai.completion to a top-level
        # ``output`` string on the Langfuse generation observation.
        output_field = gen_obs.get("output")
        if isinstance(output_field, dict):
            # Forward-compat: some Langfuse versions may wrap as a dict.
            output_str = (
                output_field.get("output")
                or output_field.get("gen_ai.completion")
                or output_field.get("value")
            )
        else:
            output_str = output_field
        assert output_str == "bye world", (
            f"Expected output 'bye world'; got {output_field!r}"
        )

        # Token usage — Langfuse v3 derives these from
        # gen_ai.usage.input_tokens / output_tokens and exposes them as
        # top-level promptTokens/completionTokens, the legacy
        # ``usage`` block (with input/output), and the newer
        # ``usageDetails`` mapping. Accept any of the three.
        usage = gen_obs.get("usage") or {}
        usage_details = gen_obs.get("usageDetails") or {}
        input_tokens = (
            gen_obs.get("promptTokens")
            or usage.get("input")
            or usage.get("promptTokens")
            or usage_details.get("input")
        )
        output_tokens = (
            gen_obs.get("completionTokens")
            or usage.get("output")
            or usage.get("completionTokens")
            or usage_details.get("output")
        )
        assert input_tokens in (5, "5"), (
            f"Expected 5 input tokens; got {input_tokens!r} "
            f"(usage={usage!r}, usageDetails={usage_details!r}, "
            f"promptTokens={gen_obs.get('promptTokens')!r})"
        )
        assert output_tokens in (7, "7"), (
            f"Expected 7 output tokens; got {output_tokens!r} "
            f"(usage={usage!r}, usageDetails={usage_details!r}, "
            f"completionTokens={gen_obs.get('completionTokens')!r})"
        )

    def test_trace_with_error_status_lands(self) -> None:
        """Emit a trace whose generation ends with an error; verify level=ERROR."""
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
            "e2e.live.error",
            metadata={"e2e_run_id": run_id, "category": "e2e_error"},
        ) as t:
            gen = t.generation(
                "e2e.llm.err",
                model="claude-test-model",
                input="trigger failure",
                metadata={"system": "anthropic"},
            )
            gen.set_output("partial output before failure")
            gen.end(status="error", error=RuntimeError("boom"))

        backend.flush()

        matched, elapsed = _poll_for_trace(host, auth, run_id)
        logging.getLogger(__name__).info(
            "Error trace appeared after %.2fs (run_id=%s)", elapsed, run_id
        )

        trace_id = matched.get("id")
        assert trace_id

        observations = _poll_for_observations(host, auth, trace_id, minimum=1)
        gen_obs = next(
            (o for o in observations if o.get("name") == "e2e.llm.err"),
            None,
        )
        assert gen_obs is not None, (
            f"Expected a generation named 'e2e.llm.err'; observations: "
            f"{[o.get('name') for o in observations]!r}"
        )

        # Langfuse maps OTel Status(ERROR) to observation.level == "ERROR".
        level = gen_obs.get("level")
        status_msg = gen_obs.get("statusMessage")
        assert level == "ERROR", (
            f"Expected level 'ERROR'; got {level!r}. "
            f"statusMessage={status_msg!r}. Full: {gen_obs!r}"
        )

    def test_skip_message_is_actionable(self) -> None:
        """Smoke check: when this test runs, the module fixture passed.

        This is intentionally trivial — its purpose is to give a concrete
        green test when the stack IS up, so CI dashboards distinguish
        "skipped because no stack" from "no tests collected at all".
        """
        host = _langfuse_host()
        assert _stack_reachable(host), (
            f"Module fixture should have skipped if {host} were unreachable"
        )
