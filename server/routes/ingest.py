# @summary
# Ingest API routes: upload, list jobs, get job, SSE stream, cancel.
# Submits IngestDocumentWorkflow to Temporal (workers run embedded Weaviate);
# the API process tracks job lifecycle via an in-memory registry.
# Exports: create_ingest_router
# Deps: fastapi, temporalio, src.ingest.temporal, src.platform.security.auth
# @end-summary

"""Document ingestion API: multipart upload + Temporal-backed jobs."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import httpx
from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from temporalio.client import Client  # pyright: ignore[reportMissingImports]
from temporalio.exceptions import CancelledError as TemporalCancelledError  # pyright: ignore[reportMissingImports]
from temporalio.service import RPCError  # pyright: ignore[reportMissingImports]
from urllib.parse import urlparse

from config.settings import PROJECT_ROOT
from src.ingest import IngestionConfig
from src.ingest.temporal.activities import SourceArgs
from src.ingest.temporal.constants import TRIGGER_SINGLE, trigger_to_queue
from src.ingest.temporal.workflows import IngestDocumentArgs, IngestDocumentResult, IngestDocumentWorkflow
from src.platform.security.auth import authenticate_request, Principal

logger = logging.getLogger("rag.server.ingest")

_STAGING_DIR = PROJECT_ROOT / "documents" / "_uploads"
_STAGING_DIR.mkdir(parents=True, exist_ok=True)

_MAX_RECENT_JOBS = 50
_JOB_RETENTION_SECONDS = 3600
_SUPPORTED_EXTS = {".pdf", ".docx", ".pptx", ".xlsx", ".txt", ".md", ".html", ".htm"}


@dataclass
class JobEvent:
    seq: int
    timestamp: float
    kind: str  # "stage" | "error" | "done"
    message: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class Job:
    job_id: str
    filename: str
    size_bytes: int
    status: str = "pending"  # pending | running | done | failed | cancelled
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    stored_chunks: int = 0
    error: Optional[str] = None
    config_summary: dict[str, Any] = field(default_factory=dict)
    events: list[JobEvent] = field(default_factory=list)
    workflow_id: Optional[str] = None
    _seq: int = 0
    _subscribers: list[asyncio.Queue[JobEvent]] = field(default_factory=list)
    _staging_path: Optional[Path] = None
    # When true the path is in our staging dir and we own its lifecycle.
    # When false (directory ingestion) the path is the user's; never delete.
    _owns_path: bool = True

    def emit(self, kind: str, message: str, detail: Optional[dict[str, Any]] = None) -> JobEvent:
        self._seq += 1
        evt = JobEvent(seq=self._seq, timestamp=time.time(), kind=kind, message=message, detail=detail or {})
        self.events.append(evt)
        for q in list(self._subscribers):
            try:
                q.put_nowait(evt)
            except asyncio.QueueFull:
                pass
        return evt


class JobRegistry:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = asyncio.Lock()

    async def create(self, filename: str, size_bytes: int, config_summary: dict[str, Any], staging_path: Path, owns_path: bool = True) -> Job:
        async with self._lock:
            self._evict_old()
            job = Job(
                job_id=uuid.uuid4().hex[:12],
                filename=filename,
                size_bytes=size_bytes,
                config_summary=config_summary,
                _staging_path=staging_path,
                _owns_path=owns_path,
            )
            self._jobs[job.job_id] = job
            return job

    def _evict_old(self) -> None:
        now = time.time()
        finished = [
            j for j in self._jobs.values()
            if j.status in {"done", "failed", "cancelled"}
            and j.finished_at and (now - j.finished_at) > _JOB_RETENTION_SECONDS
        ]
        for j in finished:
            self._jobs.pop(j.job_id, None)
        if len(self._jobs) > _MAX_RECENT_JOBS:
            ordered = sorted(self._jobs.values(), key=lambda j: j.created_at)
            for j in ordered[:len(self._jobs) - _MAX_RECENT_JOBS]:
                if j.status in {"done", "failed", "cancelled"}:
                    self._jobs.pop(j.job_id, None)

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    async def subscribe(self, job: Job) -> asyncio.Queue[JobEvent]:
        q: asyncio.Queue[JobEvent] = asyncio.Queue(maxsize=256)
        for evt in job.events:
            try:
                q.put_nowait(evt)
            except asyncio.QueueFull:
                break
        job._subscribers.append(q)
        return q

    def unsubscribe(self, job: Job, q: asyncio.Queue[JobEvent]) -> None:
        try:
            job._subscribers.remove(q)
        except ValueError:
            pass


_registry = JobRegistry()


def _config_from_options(opts: dict[str, Any]) -> IngestionConfig:
    """Build IngestionConfig — UI surface intentionally minimal; defaults
    (sourced from env config) carry the full processing toggles."""
    return IngestionConfig()


def _build_source_args(staging_path: Path, filename: str) -> SourceArgs:
    """Build SourceArgs for an uploaded file. The worker reads this path off
    the shared filesystem; in single-host dev the staging dir is enough."""
    stat = staging_path.stat()
    source_id = f"upload:{stat.st_dev}:{stat.st_ino}"
    source_key = f"upload:{source_id}"
    return SourceArgs(
        source_path=str(staging_path.resolve()),
        source_name=filename,
        source_uri=staging_path.resolve().as_uri(),
        source_key=source_key,
        source_id=source_id,
        connector="upload",
        source_version=str(stat.st_mtime_ns),
    )


async def _run_workflow(job: Job, options: dict[str, Any], get_temporal_client: Callable[[], Optional[Client]]) -> None:
    """Submit the ingest workflow and watch its result."""
    job.status = "running"
    job.started_at = time.time()
    job.emit("stage", "submitting to Temporal", {"phase": "submit"})

    try:
        client = get_temporal_client()
        if client is None:
            raise RuntimeError("Temporal client unavailable — worker may not be running")

        path = job._staging_path
        if path is None or not path.exists():
            raise FileNotFoundError("Staging file missing")

        config = _config_from_options(options)
        source = _build_source_args(path, job.filename)
        args = IngestDocumentArgs(
            source=source,
            config=dataclasses.asdict(config),
            trigger_type=TRIGGER_SINGLE,
        )
        workflow_id = f"ingest-{job.job_id}-{int(time.time())}"
        job.workflow_id = workflow_id

        handle = await client.start_workflow(
            IngestDocumentWorkflow.run,
            args,
            id=workflow_id,
            task_queue=trigger_to_queue(TRIGGER_SINGLE),
        )
        job.emit("stage", f"workflow {workflow_id} started", {"workflow_id": workflow_id})

        result: IngestDocumentResult = await handle.result()

        for line in result.processing_log or []:
            job.emit("stage", str(line))

        if result.errors:
            job.status = "failed"
            job.error = "; ".join(str(e) for e in result.errors)
            job.emit("error", job.error)
        else:
            job.stored_chunks = result.stored_count
            job.status = "done"
            job.emit("done", f"stored {result.stored_count} chunks", {"stored_chunks": result.stored_count})
    except (asyncio.CancelledError, TemporalCancelledError):
        # Workflow was cancelled (either via cancel_job endpoint or task shutdown).
        logger.info("Ingest job %s cancelled", job.job_id)
        job.status = "cancelled"
        job.error = None
        job.emit("done", "cancelled by user", {"cancelled": True})
        return
    except Exception as exc:
        # Temporal wraps workflow CancelledError in WorkflowFailureError; detect it.
        cause = getattr(exc, "cause", None)
        if isinstance(cause, TemporalCancelledError) or isinstance(exc, TemporalCancelledError):
            logger.info("Ingest job %s cancelled", job.job_id)
            job.status = "cancelled"
            job.error = None
            job.emit("done", "cancelled by user", {"cancelled": True})
            return
        logger.exception("Ingest job %s failed", job.job_id)
        job.status = "failed"
        job.error = f"{type(exc).__name__}: {exc}"
        job.emit("error", job.error, {"traceback": traceback.format_exc()})
    finally:
        job.finished_at = time.time()
        if job._owns_path:
            try:
                if job._staging_path and job._staging_path.exists():
                    job._staging_path.unlink()
            except Exception:
                logger.debug("Staging cleanup failed for %s", job.job_id, exc_info=True)


def create_ingest_router(get_temporal_client: Callable[[], Optional[Client]]) -> APIRouter:
    """Build the FastAPI router for /api/v1/ingest/* endpoints.

    Args:
        get_temporal_client: Callable returning the active Temporal client, or
            ``None`` if Temporal is unavailable. Wired from server.api.
    """
    router = APIRouter()

    @router.post("/api/v1/ingest/upload")
    async def upload(
        file: UploadFile = File(...),
        options: str = Form("{}"),
        principal: Principal = Depends(authenticate_request),
    ) -> dict[str, Any]:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Missing filename")
        try:
            opts = json.loads(options) if options else {}
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid options JSON")
        if not isinstance(opts, dict):
            raise HTTPException(status_code=400, detail="options must be a JSON object")
        if get_temporal_client() is None:
            raise HTTPException(status_code=503, detail="Ingestion worker unavailable (Temporal not connected)")

        safe_name = Path(file.filename).name
        staging = _STAGING_DIR / f"{uuid.uuid4().hex[:8]}_{safe_name}"
        size = 0
        with staging.open("wb") as fh:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                fh.write(chunk)

        job = await _registry.create(safe_name, size, opts, staging)
        asyncio.create_task(_run_workflow(job, opts, get_temporal_client))
        return {
            "job_id": job.job_id,
            "filename": job.filename,
            "size_bytes": job.size_bytes,
            "status": job.status,
        }

    @router.post("/api/v1/ingest/check-path")
    async def check_path(
        body: dict[str, Any] = Body(...),
        principal: Principal = Depends(authenticate_request),
    ) -> dict[str, Any]:
        """Probe whether the API process can read a directory or file path.

        The API and worker share the same filesystem mounts in dev. In production
        deployments where they don't, the user sees an "unreachable" response and
        knows to upload via file/URL instead.
        """
        raw = str(body.get("path", "")).strip()
        if not raw:
            raise HTTPException(status_code=400, detail="path required")
        path = Path(raw).expanduser()
        try:
            resolved = path.resolve()
        except OSError as exc:
            return {"path": raw, "reachable": False, "reason": f"resolve failed: {exc}"}
        if not resolved.exists():
            return {"path": str(resolved), "reachable": False, "reason": "path does not exist on the ingestion host"}
        if resolved.is_file():
            return {
                "path": str(resolved),
                "reachable": True,
                "is_dir": False,
                "is_file": True,
                "size_bytes": resolved.stat().st_size,
                "files": [resolved.name],
            }
        if resolved.is_dir():
            try:
                files = [
                    p for p in resolved.rglob("*")
                    if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTS
                ]
            except PermissionError as exc:
                return {"path": str(resolved), "reachable": False, "reason": f"permission denied: {exc}"}
            return {
                "path": str(resolved),
                "reachable": True,
                "is_dir": True,
                "is_file": False,
                "file_count": len(files),
                "files": [str(p.relative_to(resolved)) for p in files[:50]],
                "truncated": len(files) > 50,
            }
        return {"path": str(resolved), "reachable": False, "reason": "not a regular file or directory"}

    @router.post("/api/v1/ingest/url")
    async def ingest_url(
        body: dict[str, Any] = Body(...),
        principal: Principal = Depends(authenticate_request),
    ) -> dict[str, Any]:
        """Download a URL into the staging area and submit an ingestion job."""
        url = str(body.get("url", "")).strip()
        if not url:
            raise HTTPException(status_code=400, detail="url required")
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise HTTPException(status_code=400, detail="only http(s) URLs are supported")
        if get_temporal_client() is None:
            raise HTTPException(status_code=503, detail="Ingestion worker unavailable (Temporal not connected)")

        # Derive a filename from the URL path; fall back to the host.
        leaf = Path(parsed.path).name or parsed.netloc.replace(":", "_") or "download"
        safe_name = Path(leaf).name
        staging = _STAGING_DIR / f"{uuid.uuid4().hex[:8]}_{safe_name}"
        size = 0
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
                async with client.stream("GET", url) as resp:
                    if resp.status_code >= 400:
                        raise HTTPException(status_code=400, detail=f"fetch failed: HTTP {resp.status_code}")
                    with staging.open("wb") as fh:
                        async for chunk in resp.aiter_bytes():
                            size += len(chunk)
                            fh.write(chunk)
        except httpx.RequestError as exc:
            raise HTTPException(status_code=400, detail=f"fetch error: {exc}") from exc

        job = await _registry.create(safe_name, size, {"url": url}, staging)
        asyncio.create_task(_run_workflow(job, {}, get_temporal_client))
        return {"job_id": job.job_id, "filename": job.filename, "size_bytes": job.size_bytes, "status": job.status}

    @router.post("/api/v1/ingest/directory")
    async def ingest_directory(
        body: dict[str, Any] = Body(...),
        principal: Principal = Depends(authenticate_request),
    ) -> dict[str, Any]:
        """Enumerate a server-side directory and submit one job per supported file.

        Files are *not* copied into the staging area — the worker reads from
        the original path. The path must be reachable from both API and worker.
        """
        raw = str(body.get("path", "")).strip()
        if not raw:
            raise HTTPException(status_code=400, detail="path required")
        if get_temporal_client() is None:
            raise HTTPException(status_code=503, detail="Ingestion worker unavailable (Temporal not connected)")
        root = Path(raw).expanduser().resolve()
        if not root.is_dir():
            raise HTTPException(status_code=400, detail=f"not a directory: {root}")
        try:
            files = [
                p for p in root.rglob("*")
                if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTS
            ]
        except PermissionError as exc:
            raise HTTPException(status_code=400, detail=f"permission denied: {exc}")
        if not files:
            raise HTTPException(status_code=400, detail="no supported files in directory")

        jobs: list[dict[str, Any]] = []
        for path in files:
            display = str(path.relative_to(root))
            size = path.stat().st_size
            job = await _registry.create(display, size, {"directory": str(root)}, path, owns_path=False)
            asyncio.create_task(_run_workflow(job, {}, get_temporal_client))
            jobs.append({
                "job_id": job.job_id,
                "filename": job.filename,
                "size_bytes": job.size_bytes,
                "status": job.status,
            })
        return {"directory": str(root), "submitted": len(jobs), "jobs": jobs}

    @router.get("/api/v1/ingest/jobs")
    async def list_jobs(principal: Principal = Depends(authenticate_request)) -> dict[str, Any]:
        return {"jobs": [_summarize(j) for j in _registry.list()]}

    @router.get("/api/v1/ingest/jobs/{job_id}")
    async def get_job(job_id: str, principal: Principal = Depends(authenticate_request)) -> dict[str, Any]:
        job = _registry.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return _summarize(job, include_events=True)

    @router.get("/api/v1/ingest/jobs/{job_id}/stream")
    async def stream_job(job_id: str, principal: Principal = Depends(authenticate_request)) -> StreamingResponse:
        job = _registry.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")

        async def gen():
            q = await _registry.subscribe(job)
            try:
                terminal = {"done", "failed", "cancelled"}
                while True:
                    try:
                        evt = await asyncio.wait_for(q.get(), timeout=30.0)
                    except asyncio.TimeoutError:
                        yield ": ping\n\n"
                        if job.status in terminal:
                            return
                        continue
                    payload = {
                        "seq": evt.seq,
                        "kind": evt.kind,
                        "message": evt.message,
                        "detail": evt.detail,
                        "timestamp": evt.timestamp,
                        "status": job.status,
                    }
                    yield f"event: {evt.kind}\ndata: {json.dumps(payload)}\n\n"
                    if evt.kind in {"done", "error"} and job.status in terminal:
                        return
            finally:
                _registry.unsubscribe(job, q)

        return StreamingResponse(gen(), media_type="text/event-stream")

    @router.post("/api/v1/ingest/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str, principal: Principal = Depends(authenticate_request)) -> dict[str, Any]:
        job = _registry.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.status not in {"pending", "running"}:
            return {"job_id": job_id, "status": job.status, "cancelled": False}

        client = get_temporal_client()
        cancel_sent = False
        if client and job.workflow_id:
            try:
                handle = client.get_workflow_handle(job.workflow_id)
                await handle.cancel()
                cancel_sent = True
            except RPCError as exc:
                logger.warning("Workflow cancel RPC failed for %s: %s", job.workflow_id, exc)
            except Exception as exc:
                logger.warning("Workflow cancel failed for %s: %s", job.workflow_id, exc)

        # If no worker exists to act on the cancellation (or the call failed),
        # mark the job as cancelled locally so the UI doesn't lie. The worker,
        # if it ever wakes up, will also raise CancelledError and converge.
        if not cancel_sent:
            job.status = "cancelled"
            job.finished_at = time.time()
            job.emit("done", "cancelled by user (no worker)", {"cancelled": True, "no_worker": True})
            return {"job_id": job_id, "status": "cancelled", "cancelled": True}

        return {"job_id": job_id, "status": "cancelling", "cancelled": True}

    return router


def _summarize(job: Job, include_events: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {
        "job_id": job.job_id,
        "filename": job.filename,
        "size_bytes": job.size_bytes,
        "status": job.status,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "stored_chunks": job.stored_chunks,
        "error": job.error,
        "config_summary": job.config_summary,
        "workflow_id": job.workflow_id,
    }
    if include_events:
        out["events"] = [
            {"seq": e.seq, "kind": e.kind, "message": e.message, "detail": e.detail, "timestamp": e.timestamp}
            for e in job.events
        ]
    return out


__all__ = ["create_ingest_router"]
