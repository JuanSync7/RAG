"""Endpoint + ``_run_workflow`` tests for ``server/routes/ingest.py``.

This slice (``server-ingest-endpoints``) covers the parts NOT exercised by
``tests/server/test_ingest_jobstore.py`` (which owns the registry, sweeper,
and pure helpers):

- ``_run_workflow`` — the async orchestrator, driven directly via
  ``asyncio.run`` with fake Temporal client/handle objects. Covers the happy
  path, result-errors-failed, no-client/missing-staging-file failures, the
  three cancellation shapes (direct ``CancelledError``/``TemporalCancelledError``,
  Temporal-wrapped ``.cause``), generic-exception formatting, and the
  ``_owns_path`` staging-cleanup branch (both directions).
- The FastAPI endpoints from ``create_ingest_router`` driven through a
  ``TestClient`` with ``authenticate_request`` overridden and
  ``_run_workflow`` monkeypatched to an async recording no-op so no real
  workflow runs: ``upload``, ``check-path``, ``url``, ``directory``,
  ``GET /jobs``, ``GET /jobs/{id}``, ``GET /jobs/{id}/stream``,
  ``POST /jobs/{id}/cancel``.

The module-global ``_registry`` is reset around every test (autouse fixture)
so jobs never leak between tests. ``_STAGING_DIR`` is redirected to ``tmp_path``
in the upload test so no files land under ``documents/_uploads``.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from temporalio.exceptions import CancelledError as TemporalCancelledError
from temporalio.service import RPCError, RPCStatusCode

from server.routes import ingest
from server.routes.ingest import Job, JobRegistry, create_ingest_router, _run_workflow
from src.ingest.temporal.workflows import IngestDocumentResult
from src.platform.security.auth import authenticate_request, Principal


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset the module-global registry around every test (no job leakage)."""
    ingest._registry._jobs.clear()
    yield
    ingest._registry._jobs.clear()


def _principal() -> Principal:
    return Principal(
        subject="u",
        tenant_id="t",
        roles=["ingest"],
        auth_type="none",
        project_id="p",
    )


def _result(
    *,
    stored_count: int = 0,
    errors: list | None = None,
    processing_log: list | None = None,
    source_key: str = "upload:k",
) -> IngestDocumentResult:
    return IngestDocumentResult(
        source_key=source_key,
        errors=list(errors or []),
        stored_count=stored_count,
        processing_log=list(processing_log or []),
    )


class _FakeHandle:
    """Async handle: ``await handle.result()`` returns a value or raises."""

    def __init__(self, result=None, raises: BaseException | None = None):
        self._result = result
        self._raises = raises
        self.cancel_called = False

    async def result(self):
        if self._raises is not None:
            raise self._raises
        return self._result

    async def cancel(self):
        self.cancel_called = True


class _FakeClient:
    """Async ``start_workflow`` returning a fixed handle; records the call."""

    def __init__(self, handle: _FakeHandle | None = None, *, cancel_exc=None):
        self._handle = handle or _FakeHandle(result=_result())
        self.start_args = None
        self.start_kwargs = None
        self._cancel_handle = _FakeHandle()
        self._cancel_exc = cancel_exc

    async def start_workflow(self, *args, **kwargs):
        self.start_args = args
        self.start_kwargs = kwargs
        return self._handle

    def get_workflow_handle(self, workflow_id):
        if self._cancel_exc is not None:
            raise self._cancel_exc
        return self._cancel_handle


def _job(tmp_path: Path | None = None, *, owns_path: bool = True, write: bool = True) -> Job:
    """Build a Job for direct ``_run_workflow`` tests with a real staging file."""
    staging = None
    if tmp_path is not None:
        staging = tmp_path / "staged.pdf"
        if write:
            staging.write_bytes(b"PDFDATA")
    return Job(
        job_id="jobx",
        filename="doc.pdf",
        size_bytes=7,
        _staging_path=staging,
        _owns_path=owns_path,
    )


def _kinds(job: Job) -> list[str]:
    return [e.kind for e in job.events]


# ===========================================================================
# _run_workflow — driven directly
# ===========================================================================


def test_run_workflow_happy_marks_done(tmp_path):
    """Successful workflow: status done, stored_chunks set, done event, ids set."""
    job = _job(tmp_path)
    client = _FakeClient(_FakeHandle(result=_result(stored_count=5)))

    asyncio.run(_run_workflow(job, {}, lambda: client))

    assert job.status == "done"
    assert job.stored_chunks == 5
    assert job.error is None
    assert "done" in _kinds(job)
    done = [e for e in job.events if e.kind == "done"][0]
    assert done.detail == {"stored_chunks": 5}
    assert job.workflow_id is not None
    assert job.workflow_id.startswith("ingest-")
    assert job.started_at is not None
    assert job.finished_at is not None


def test_run_workflow_result_errors_marks_failed(tmp_path):
    """result.errors non-empty: status failed, error joins them, error event."""
    job = _job(tmp_path)
    client = _FakeClient(_FakeHandle(result=_result(stored_count=2, errors=["boom", "bad"])))

    asyncio.run(_run_workflow(job, {}, lambda: client))

    assert job.status == "failed"
    assert job.error == "boom; bad"
    assert "error" in _kinds(job)
    # stored_chunks stays 0 because the failure branch is taken, not done.
    assert job.stored_chunks == 0


def test_run_workflow_processing_log_emitted_as_stages(tmp_path):
    """Each processing_log line is emitted as a stage event."""
    job = _job(tmp_path)
    client = _FakeClient(_FakeHandle(result=_result(stored_count=1, processing_log=["L1", "L2"])))

    asyncio.run(_run_workflow(job, {}, lambda: client))

    messages = [e.message for e in job.events if e.kind == "stage"]
    assert "L1" in messages and "L2" in messages


def test_run_workflow_no_client_marks_failed():
    """get_temporal_client returning None -> RuntimeError -> failed."""
    job = _job(None)

    asyncio.run(_run_workflow(job, {}, lambda: None))

    assert job.status == "failed"
    assert job.error is not None
    assert job.error.startswith("RuntimeError:")


def test_run_workflow_missing_staging_file_marks_failed(tmp_path):
    """Staging path that does not exist -> FileNotFoundError -> failed."""
    job = _job(tmp_path, write=False)  # path set but file absent
    client = _FakeClient()

    asyncio.run(_run_workflow(job, {}, lambda: client))

    assert job.status == "failed"
    assert job.error is not None
    assert job.error.startswith("FileNotFoundError:")


def test_run_workflow_staging_path_none_marks_failed():
    """_staging_path None -> FileNotFoundError -> failed."""
    job = Job(job_id="j", filename="d.pdf", size_bytes=1, _staging_path=None)
    client = _FakeClient()

    asyncio.run(_run_workflow(job, {}, lambda: client))

    assert job.status == "failed"
    assert job.error.startswith("FileNotFoundError:")


def test_run_workflow_temporal_cancelled_marks_cancelled(tmp_path):
    """handle.result raising TemporalCancelledError -> cancelled, error None."""
    job = _job(tmp_path)
    client = _FakeClient(_FakeHandle(raises=TemporalCancelledError("stopped")))

    asyncio.run(_run_workflow(job, {}, lambda: client))

    assert job.status == "cancelled"
    assert job.error is None
    done = [e for e in job.events if e.kind == "done"][0]
    assert done.detail == {"cancelled": True}


def test_run_workflow_asyncio_cancelled_marks_cancelled(tmp_path):
    """handle.result raising asyncio.CancelledError -> cancelled."""
    job = _job(tmp_path)
    client = _FakeClient(_FakeHandle(raises=asyncio.CancelledError()))

    asyncio.run(_run_workflow(job, {}, lambda: client))

    assert job.status == "cancelled"
    assert job.error is None


def test_run_workflow_wrapped_cancel_via_cause_marks_cancelled(tmp_path):
    """A generic Exception whose .cause is TemporalCancelledError -> cancelled."""
    job = _job(tmp_path)
    wrapper = RuntimeError("workflow failed")
    wrapper.cause = TemporalCancelledError("inner cancel")
    client = _FakeClient(_FakeHandle(raises=wrapper))

    asyncio.run(_run_workflow(job, {}, lambda: client))

    assert job.status == "cancelled"
    assert job.error is None


def test_run_workflow_generic_exception_marks_failed(tmp_path):
    """A non-cancel Exception -> failed with 'Type: msg' error formatting."""
    job = _job(tmp_path)
    client = _FakeClient(_FakeHandle(raises=ValueError("kaboom")))

    asyncio.run(_run_workflow(job, {}, lambda: client))

    assert job.status == "failed"
    assert job.error == "ValueError: kaboom"
    assert "error" in _kinds(job)


def test_run_workflow_cleans_staging_when_owns_path(tmp_path):
    """_owns_path True: the staging file is unlinked in the finally block."""
    job = _job(tmp_path, owns_path=True)
    assert job._staging_path.exists()
    client = _FakeClient(_FakeHandle(result=_result(stored_count=1)))

    asyncio.run(_run_workflow(job, {}, lambda: client))

    assert not job._staging_path.exists()


def test_run_workflow_keeps_staging_when_not_owned(tmp_path):
    """_owns_path False (directory ingestion): the file is NOT unlinked."""
    job = _job(tmp_path, owns_path=False)
    assert job._staging_path.exists()
    client = _FakeClient(_FakeHandle(result=_result(stored_count=1)))

    asyncio.run(_run_workflow(job, {}, lambda: client))

    assert job._staging_path.exists()


# ===========================================================================
# Endpoints — TestClient with _run_workflow patched to a recording no-op
# ===========================================================================


class _Recorder:
    """Async recording stand-in for _run_workflow scheduled as a task."""

    def __init__(self):
        self.calls: list[tuple] = []

    async def __call__(self, job, options, get_temporal_client):
        self.calls.append((job, options))


@pytest.fixture
def recorder(monkeypatch) -> _Recorder:
    rec = _Recorder()
    monkeypatch.setattr(ingest, "_run_workflow", rec)
    return rec


def _make_client(get_temporal_client) -> TestClient:
    app = FastAPI()
    app.include_router(create_ingest_router(get_temporal_client))
    app.dependency_overrides[authenticate_request] = _principal
    return TestClient(app)


@pytest.fixture
def client_with_temporal(recorder) -> TestClient:
    return _make_client(lambda: object())


@pytest.fixture
def client_no_temporal(recorder) -> TestClient:
    return _make_client(lambda: None)


# --- upload ----------------------------------------------------------------


def test_upload_happy(monkeypatch, tmp_path, recorder):
    """Upload writes the staged bytes, registers a pending job, schedules run."""
    staging_dir = tmp_path / "uploads"
    staging_dir.mkdir()
    monkeypatch.setattr(ingest, "_STAGING_DIR", staging_dir)
    client = _make_client(lambda: object())

    payload = b"hello world bytes"
    resp = client.post(
        "/api/v1/ingest/upload",
        files={"file": ("report.pdf", payload, "application/pdf")},
        data={"options": "{}"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["filename"] == "report.pdf"
    assert body["size_bytes"] == len(payload)
    assert body["status"] == "pending"
    job_id = body["job_id"]
    # Job registered.
    job = ingest._registry.get(job_id)
    assert job is not None
    # Staged file written with the uploaded bytes.
    staged = list(staging_dir.iterdir())
    assert len(staged) == 1
    assert staged[0].read_bytes() == payload
    # _run_workflow scheduled for that job (give the event loop a beat).
    assert recorder.calls and recorder.calls[0][0].job_id == job_id


def test_upload_invalid_options_json_400(monkeypatch, tmp_path, recorder):
    """options that is not valid JSON -> 400."""
    monkeypatch.setattr(ingest, "_STAGING_DIR", tmp_path)
    client = _make_client(lambda: object())

    resp = client.post(
        "/api/v1/ingest/upload",
        files={"file": ("a.pdf", b"x", "application/pdf")},
        data={"options": "{not json"},
    )

    assert resp.status_code == 400
    assert resp.json()["detail"] == "Invalid options JSON"


def test_upload_options_not_a_dict_400(monkeypatch, tmp_path, recorder):
    """options that parses to a non-dict (a list) -> 400."""
    monkeypatch.setattr(ingest, "_STAGING_DIR", tmp_path)
    client = _make_client(lambda: object())

    resp = client.post(
        "/api/v1/ingest/upload",
        files={"file": ("a.pdf", b"x", "application/pdf")},
        data={"options": "[]"},
    )

    assert resp.status_code == 400
    assert resp.json()["detail"] == "options must be a JSON object"


def test_upload_temporal_unavailable_503(monkeypatch, tmp_path, recorder):
    """No Temporal client -> 503."""
    monkeypatch.setattr(ingest, "_STAGING_DIR", tmp_path)
    client = _make_client(lambda: None)

    resp = client.post(
        "/api/v1/ingest/upload",
        files={"file": ("a.pdf", b"x", "application/pdf")},
        data={"options": "{}"},
    )

    assert resp.status_code == 503


def _upload_endpoint(get_temporal_client):
    """Pull the raw ``upload`` route handler out of the router (bypasses the
    multipart layer, which rejects a truly-empty filename with 422 before the
    handler runs — the ``not file.filename`` guard is only reachable for a
    ``None`` filename)."""
    router = create_ingest_router(get_temporal_client)
    for route in router.routes:
        if route.name == "upload":
            return route.endpoint
    raise AssertionError("upload route not found")


def test_upload_missing_filename_400(monkeypatch, tmp_path, recorder):
    """A falsy (None) filename -> 400 (Missing filename) via the guard."""
    from io import BytesIO

    from fastapi import HTTPException, UploadFile

    monkeypatch.setattr(ingest, "_STAGING_DIR", tmp_path)
    endpoint = _upload_endpoint(lambda: object())
    upload = UploadFile(filename=None, file=BytesIO(b"x"))

    with pytest.raises(HTTPException) as ei:
        asyncio.run(endpoint(file=upload, options="{}", principal=_principal()))

    assert ei.value.status_code == 400
    assert ei.value.detail == "Missing filename"
    # The guard fires before any work: no job registered, no run scheduled.
    assert ingest._registry.list() == []
    assert recorder.calls == []


# --- check-path ------------------------------------------------------------


def test_check_path_missing_400(client_with_temporal):
    resp = client_with_temporal.post("/api/v1/ingest/check-path", json={"path": "  "})
    assert resp.status_code == 400
    assert resp.json()["detail"] == "path required"


def test_check_path_nonexistent_unreachable(client_with_temporal, tmp_path):
    missing = tmp_path / "nope"
    resp = client_with_temporal.post("/api/v1/ingest/check-path", json={"path": str(missing)})
    assert resp.status_code == 200
    assert resp.json()["reachable"] is False


def test_check_path_real_file(client_with_temporal, tmp_path):
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"12345")
    resp = client_with_temporal.post("/api/v1/ingest/check-path", json={"path": str(f)})
    body = resp.json()
    assert body["reachable"] is True
    assert body["is_file"] is True
    assert body["is_dir"] is False
    assert body["size_bytes"] == 5


def test_check_path_real_dir_counts_only_supported(client_with_temporal, tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"x")
    (tmp_path / "b.txt").write_bytes(b"y")
    (tmp_path / "c.bin").write_bytes(b"z")  # unsupported ext
    resp = client_with_temporal.post("/api/v1/ingest/check-path", json={"path": str(tmp_path)})
    body = resp.json()
    assert body["reachable"] is True
    assert body["is_dir"] is True
    assert body["file_count"] == 2  # only .pdf + .txt counted


# --- ingest_url ------------------------------------------------------------


def test_ingest_url_empty_400(client_with_temporal):
    resp = client_with_temporal.post("/api/v1/ingest/url", json={"url": "   "})
    assert resp.status_code == 400
    assert resp.json()["detail"] == "url required"


def test_ingest_url_non_http_scheme_400(client_with_temporal):
    resp = client_with_temporal.post("/api/v1/ingest/url", json={"url": "ftp://host/x.pdf"})
    assert resp.status_code == 400
    assert "http(s)" in resp.json()["detail"]


def test_ingest_url_temporal_unavailable_503(client_no_temporal):
    resp = client_no_temporal.post("/api/v1/ingest/url", json={"url": "https://h/x.pdf"})
    assert resp.status_code == 503


class _FakeStreamResp:
    def __init__(self, status_code: int, chunks: list[bytes]):
        self.status_code = status_code
        self._chunks = chunks

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def aiter_bytes(self):
        for c in self._chunks:
            yield c


class _FakeHttpxClient:
    def __init__(self, status_code: int, chunks: list[bytes]):
        self._status = status_code
        self._chunks = chunks

    def __init_subclass__(cls, **kw):  # pragma: no cover - defensive
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def stream(self, method, url):
        return _FakeStreamResp(self._status, self._chunks)


def _patch_httpx(monkeypatch, status_code: int, chunks: list[bytes]):
    def _factory(*args, **kwargs):
        return _FakeHttpxClient(status_code, chunks)

    monkeypatch.setattr(ingest.httpx, "AsyncClient", _factory)


def test_ingest_url_happy(monkeypatch, tmp_path, recorder):
    """A 200 stream downloads bytes to staging and returns a pending job."""
    monkeypatch.setattr(ingest, "_STAGING_DIR", tmp_path)
    _patch_httpx(monkeypatch, 200, [b"abc", b"def"])
    client = _make_client(lambda: object())

    resp = client.post("/api/v1/ingest/url", json={"url": "https://h/report.pdf"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["filename"] == "report.pdf"
    assert body["size_bytes"] == 6  # len(b"abc"+b"def")
    assert body["status"] == "pending"
    assert ingest._registry.get(body["job_id"]) is not None


def test_ingest_url_http_error_400(monkeypatch, tmp_path, recorder):
    """A >=400 response status maps to a 400 fetch-failed error."""
    monkeypatch.setattr(ingest, "_STAGING_DIR", tmp_path)
    _patch_httpx(monkeypatch, 404, [])
    client = _make_client(lambda: object())

    resp = client.post("/api/v1/ingest/url", json={"url": "https://h/missing.pdf"})

    assert resp.status_code == 400
    assert "HTTP 404" in resp.json()["detail"]


# --- ingest_directory ------------------------------------------------------


def test_ingest_directory_missing_path_400(client_with_temporal):
    resp = client_with_temporal.post("/api/v1/ingest/directory", json={"path": ""})
    assert resp.status_code == 400
    assert resp.json()["detail"] == "path required"


def test_ingest_directory_temporal_unavailable_503(client_no_temporal, tmp_path):
    resp = client_no_temporal.post("/api/v1/ingest/directory", json={"path": str(tmp_path)})
    assert resp.status_code == 503


def test_ingest_directory_not_a_directory_400(client_with_temporal, tmp_path):
    f = tmp_path / "file.pdf"
    f.write_bytes(b"x")
    resp = client_with_temporal.post("/api/v1/ingest/directory", json={"path": str(f)})
    assert resp.status_code == 400
    assert "not a directory" in resp.json()["detail"]


def test_ingest_directory_empty_400(client_with_temporal, tmp_path):
    """A directory with no supported files -> 400."""
    (tmp_path / "x.bin").write_bytes(b"z")  # unsupported only
    resp = client_with_temporal.post("/api/v1/ingest/directory", json={"path": str(tmp_path)})
    assert resp.status_code == 400
    assert resp.json()["detail"] == "no supported files in directory"


def test_ingest_directory_happy_owns_path_false(client_with_temporal, tmp_path):
    """One job per supported file; created jobs have _owns_path False."""
    (tmp_path / "a.pdf").write_bytes(b"x")
    (tmp_path / "b.txt").write_bytes(b"yy")
    (tmp_path / "c.bin").write_bytes(b"z")  # unsupported

    resp = client_with_temporal.post("/api/v1/ingest/directory", json={"path": str(tmp_path)})

    assert resp.status_code == 200
    body = resp.json()
    assert body["submitted"] == 2
    assert len(body["jobs"]) == 2
    for j in body["jobs"]:
        job = ingest._registry.get(j["job_id"])
        assert job is not None
        assert job._owns_path is False


# --- GET /jobs -------------------------------------------------------------


def test_list_jobs_returns_newest_first(client_with_temporal):
    old = asyncio.run(ingest._registry.create("old.pdf", 1, {}, Path("/tmp/old.pdf")))
    new = asyncio.run(ingest._registry.create("new.pdf", 2, {}, Path("/tmp/new.pdf")))
    old.created_at = 100.0
    new.created_at = 200.0

    resp = client_with_temporal.get("/api/v1/ingest/jobs")

    assert resp.status_code == 200
    ids = [j["job_id"] for j in resp.json()["jobs"]]
    assert ids == [new.job_id, old.job_id]


# --- GET /jobs/{id} --------------------------------------------------------


def test_get_job_returns_events(client_with_temporal):
    job = asyncio.run(ingest._registry.create("a.pdf", 1, {}, Path("/tmp/a.pdf")))
    job.emit("stage", "parsing", {"phase": "parse"})

    resp = client_with_temporal.get(f"/api/v1/ingest/jobs/{job.job_id}")

    assert resp.status_code == 200
    body = resp.json()
    assert body["job_id"] == job.job_id
    assert "events" in body
    assert body["events"][0]["message"] == "parsing"


def test_get_job_unknown_404(client_with_temporal):
    resp = client_with_temporal.get("/api/v1/ingest/jobs/does-not-exist")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Job not found"


# --- cancel_job ------------------------------------------------------------


def test_cancel_unknown_404(client_with_temporal):
    resp = client_with_temporal.post("/api/v1/ingest/jobs/nope/cancel")
    assert resp.status_code == 404


def test_cancel_already_terminal_returns_false(client_with_temporal):
    """A done job is not re-cancelled: {cancelled: False}."""
    job = asyncio.run(ingest._registry.create("a.pdf", 1, {}, Path("/tmp/a.pdf")))
    job.status = "done"

    resp = client_with_temporal.post(f"/api/v1/ingest/jobs/{job.job_id}/cancel")

    assert resp.status_code == 200
    body = resp.json()
    assert body["cancelled"] is False
    assert body["status"] == "done"


def test_cancel_local_when_no_client_or_workflow():
    """Pending job, no client/workflow_id: local cancel flips status."""
    app = FastAPI()
    app.include_router(create_ingest_router(lambda: None))
    app.dependency_overrides[authenticate_request] = _principal
    client = TestClient(app)

    job = asyncio.run(ingest._registry.create("a.pdf", 1, {}, Path("/tmp/a.pdf")))
    assert job.status == "pending"

    resp = client.post(f"/api/v1/ingest/jobs/{job.job_id}/cancel")

    assert resp.status_code == 200
    body = resp.json()
    assert body["cancelled"] is True
    assert body["status"] == "cancelled"
    assert ingest._registry.get(job.job_id).status == "cancelled"


def test_cancel_remote_when_client_and_workflow():
    """Pending job with client + workflow_id, handle.cancel() succeeds."""
    fake_client = _FakeClient()
    app = FastAPI()
    app.include_router(create_ingest_router(lambda: fake_client))
    app.dependency_overrides[authenticate_request] = _principal
    client = TestClient(app)

    job = asyncio.run(ingest._registry.create("a.pdf", 1, {}, Path("/tmp/a.pdf")))
    job.workflow_id = "ingest-123"

    resp = client.post(f"/api/v1/ingest/jobs/{job.job_id}/cancel")

    assert resp.status_code == 200
    body = resp.json()
    assert body["cancelled"] is True
    assert body["status"] == "cancelling"
    assert fake_client._cancel_handle.cancel_called is True
    # Local status not flipped (worker will converge).
    assert ingest._registry.get(job.job_id).status == "pending"


def test_cancel_remote_rpc_error_falls_back_to_local():
    """If handle lookup raises RPCError, fall back to local cancel."""
    rpc_exc = RPCError("boom", RPCStatusCode.UNAVAILABLE, b"")
    fake_client = _FakeClient(cancel_exc=rpc_exc)
    app = FastAPI()
    app.include_router(create_ingest_router(lambda: fake_client))
    app.dependency_overrides[authenticate_request] = _principal
    client = TestClient(app)

    job = asyncio.run(ingest._registry.create("a.pdf", 1, {}, Path("/tmp/a.pdf")))
    job.workflow_id = "ingest-123"

    resp = client.post(f"/api/v1/ingest/jobs/{job.job_id}/cancel")

    assert resp.status_code == 200
    body = resp.json()
    assert body["cancelled"] is True
    assert body["status"] == "cancelled"


# --- stream_job ------------------------------------------------------------


def test_stream_unknown_404(client_with_temporal):
    resp = client_with_temporal.get("/api/v1/ingest/jobs/nope/stream")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Job not found"


def test_stream_terminal_job_yields_one_frame(client_with_temporal):
    """A terminal job pre-seeded with a done event streams at least one frame."""
    job = asyncio.run(ingest._registry.create("a.pdf", 1, {}, Path("/tmp/a.pdf")))
    job.status = "done"
    job.emit("done", "stored 3 chunks", {"stored_chunks": 3})

    with client_with_temporal.stream("GET", f"/api/v1/ingest/jobs/{job.job_id}/stream") as resp:
        assert resp.status_code == 200
        first = next(resp.iter_lines())

    assert "event: done" in first or first.startswith("event:")
