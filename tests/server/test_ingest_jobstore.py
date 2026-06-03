"""Unit tests for the infra-free logic in ``server/routes/ingest.py``.

This module covers the in-memory job registry and the pure helper functions
that do not require a running Temporal worker, FastAPI TestClient, or any
network/filesystem service:

- ``Job.emit``: appends a ``JobEvent`` with a monotonically increasing ``seq``,
  the supplied ``kind``/``message``/``detail`` (defaulting ``detail`` to ``{}``),
  returns the event, and fans it out to subscriber queues.
- ``JobRegistry.create``/``get``/``list``: id allocation, field population,
  initial status, lookup hit/miss, and newest-first listing.
- ``JobRegistry.subscribe``/``unsubscribe``: queue creation + backfill of prior
  events, live delivery on ``emit``, and removal so later events are not
  delivered.
- ``JobRegistry._evict_old``: the retention-by-age policy and the cap-based
  eviction policy (only *finished* jobs are evicted at/over the cap), with the
  at-cap vs one-over boundary given teeth.
- ``_last_activity_ts``: returns the max of created/started/last-event ts.
- ``_sweep_stale_jobs``: flips stale pending/running jobs to ``failed`` (strict
  ``> _STALE_JOB_SECONDS`` boundary), leaves fresh and already-terminal jobs
  untouched, and returns the flip count.
- ``_config_from_options``: returns a default ``IngestionConfig`` (the UI
  surface is intentionally minimal; options are ignored).
- ``_build_source_args``: builds a ``SourceArgs`` from a staging path + filename.
- ``_summarize``: exact key set and the ``include_events`` toggle.
- ``start_job_sweeper``: returns an ``asyncio.Task``.

Out of scope (deferred slice): the FastAPI endpoints (``upload``/``check_path``/
``ingest_url``/``ingest_directory``/``list_jobs``/``get_job``/``stream_job``/
``cancel_job``) and ``_run_workflow`` — those need a TestClient plus Temporal
mocking and are covered separately.
"""
from __future__ import annotations

import asyncio
import dataclasses
from pathlib import Path

import pytest

from server.routes import ingest
from server.routes.ingest import (
    Job,
    JobEvent,
    JobRegistry,
    _build_source_args,
    _config_from_options,
    _last_activity_ts,
    _summarize,
    start_job_sweeper,
)
from src.ingest import IngestionConfig
from src.ingest.temporal.activities import SourceArgs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_job(**overrides) -> Job:
    """Construct a Job directly with sensible defaults for unit testing."""
    base = {
        "job_id": "job-1",
        "filename": "doc.pdf",
        "size_bytes": 123,
    }
    base.update(overrides)
    return Job(**base)


# ---------------------------------------------------------------------------
# Job.emit
# ---------------------------------------------------------------------------


def test_emit_appends_event_with_supplied_fields():
    """emit records kind/message/detail on a new JobEvent and appends it."""
    job = _make_job()

    evt = job.emit("stage", "parsing", {"phase": "parse"})

    assert isinstance(evt, JobEvent)
    assert job.events == [evt]
    assert evt.kind == "stage"
    assert evt.message == "parsing"
    assert evt.detail == {"phase": "parse"}


def test_emit_returns_the_appended_event():
    """emit returns the same JobEvent object it appended to job.events."""
    job = _make_job()

    evt = job.emit("done", "finished")

    assert job.events[-1] is evt


def test_emit_defaults_detail_to_empty_dict():
    """Omitting detail yields an empty dict, not None."""
    job = _make_job()

    evt = job.emit("stage", "no detail")

    assert evt.detail == {}


def test_emit_increments_seq_across_calls():
    """Each emit bumps the sequence counter starting at 1."""
    job = _make_job()

    first = job.emit("stage", "one")
    second = job.emit("stage", "two")
    third = job.emit("stage", "three")

    assert [first.seq, second.seq, third.seq] == [1, 2, 3]


def test_emit_fans_out_to_subscriber_queue():
    """A queue registered on _subscribers receives the emitted event."""
    job = _make_job()
    q: asyncio.Queue = asyncio.Queue(maxsize=8)
    job._subscribers.append(q)

    evt = job.emit("stage", "broadcast")

    assert q.get_nowait() is evt


# ---------------------------------------------------------------------------
# JobRegistry.create / get / list
# ---------------------------------------------------------------------------


def test_create_populates_fields_from_args():
    """create stores the filename/size/config_summary and a 12-char id."""
    reg = JobRegistry()
    staging = Path("/tmp/whatever.pdf")

    job = asyncio.run(
        reg.create("report.pdf", 4096, {"k": "v"}, staging)
    )

    assert job.filename == "report.pdf"
    assert job.size_bytes == 4096
    assert job.config_summary == {"k": "v"}
    assert job._staging_path == staging
    assert job._owns_path is True
    assert len(job.job_id) == 12


def test_create_initial_status_is_pending():
    """A freshly created job starts in the pending state."""
    reg = JobRegistry()

    job = asyncio.run(reg.create("a.pdf", 1, {}, Path("/tmp/a.pdf")))

    assert job.status == "pending"


def test_create_allocates_unique_ids():
    """Two creates yield two distinct job ids."""
    reg = JobRegistry()

    j1 = asyncio.run(reg.create("a.pdf", 1, {}, Path("/tmp/a.pdf")))
    j2 = asyncio.run(reg.create("b.pdf", 2, {}, Path("/tmp/b.pdf")))

    assert j1.job_id != j2.job_id


def test_create_honors_owns_path_false():
    """owns_path=False (directory ingestion) is recorded on the job."""
    reg = JobRegistry()

    job = asyncio.run(
        reg.create("a.pdf", 1, {}, Path("/data/a.pdf"), owns_path=False)
    )

    assert job._owns_path is False


def test_get_returns_created_job():
    """get(id) returns the exact job object that create produced."""
    reg = JobRegistry()
    job = asyncio.run(reg.create("a.pdf", 1, {}, Path("/tmp/a.pdf")))

    assert reg.get(job.job_id) is job


def test_get_unknown_returns_none():
    """get with an unknown id returns None rather than raising."""
    reg = JobRegistry()

    assert reg.get("does-not-exist") is None


def test_list_orders_newest_first():
    """list() returns jobs sorted by created_at descending (newest first)."""
    reg = JobRegistry()
    old = asyncio.run(reg.create("old.pdf", 1, {}, Path("/tmp/old.pdf")))
    new = asyncio.run(reg.create("new.pdf", 2, {}, Path("/tmp/new.pdf")))
    old.created_at = 100.0
    new.created_at = 200.0

    listed = reg.list()

    assert [j.job_id for j in listed] == [new.job_id, old.job_id]


# ---------------------------------------------------------------------------
# JobRegistry.subscribe / unsubscribe
# ---------------------------------------------------------------------------


def test_subscribe_returns_queue_backfilling_prior_events():
    """subscribe returns a queue pre-loaded with the job's existing events."""
    reg = JobRegistry()
    job = asyncio.run(reg.create("a.pdf", 1, {}, Path("/tmp/a.pdf")))
    prior = job.emit("stage", "already happened")

    q = asyncio.run(reg.subscribe(job))

    assert q.get_nowait() is prior


def test_subscribe_then_emit_delivers_live_event():
    """An event emitted after subscribe is delivered to the queue."""
    reg = JobRegistry()
    job = asyncio.run(reg.create("a.pdf", 1, {}, Path("/tmp/a.pdf")))
    q = asyncio.run(reg.subscribe(job))

    evt = job.emit("stage", "live")

    assert q.get_nowait() is evt


def test_unsubscribe_stops_delivery():
    """After unsubscribe a later event is not delivered to the queue."""
    reg = JobRegistry()
    job = asyncio.run(reg.create("a.pdf", 1, {}, Path("/tmp/a.pdf")))
    q = asyncio.run(reg.subscribe(job))

    reg.unsubscribe(job, q)
    job.emit("stage", "after unsubscribe")

    assert q not in job._subscribers
    with pytest.raises(asyncio.QueueEmpty):
        q.get_nowait()


def test_unsubscribe_unknown_queue_is_noop():
    """Unsubscribing a queue that was never registered does not raise."""
    reg = JobRegistry()
    job = asyncio.run(reg.create("a.pdf", 1, {}, Path("/tmp/a.pdf")))
    stray: asyncio.Queue = asyncio.Queue()

    reg.unsubscribe(job, stray)  # must not raise


# ---------------------------------------------------------------------------
# JobRegistry._evict_old
# ---------------------------------------------------------------------------


def _seed_finished_jobs(reg: JobRegistry, count: int, *, status: str = "done") -> list[Job]:
    """Insert `count` finished jobs with strictly increasing created_at."""
    jobs = []
    for i in range(count):
        job = Job(
            job_id=f"job-{i:03d}",
            filename=f"f{i}.pdf",
            size_bytes=i,
            status=status,
        )
        job.created_at = float(i)
        job.finished_at = float(i)  # recently finished -> not age-evicted
        reg._jobs[job.job_id] = job
        jobs.append(job)
    return jobs


def test_evict_old_at_cap_evicts_nothing(monkeypatch):
    """Exactly _MAX_RECENT_JOBS finished jobs: none are evicted (boundary)."""
    monkeypatch.setattr(ingest.time, "time", lambda: 0.0)
    reg = JobRegistry()
    cap = ingest._MAX_RECENT_JOBS
    _seed_finished_jobs(reg, cap)

    reg._evict_old()

    assert len(reg._jobs) == cap


def test_evict_old_one_over_cap_evicts_exactly_one(monkeypatch):
    """One over the cap: exactly the oldest finished job is evicted."""
    monkeypatch.setattr(ingest.time, "time", lambda: 0.0)
    reg = JobRegistry()
    cap = ingest._MAX_RECENT_JOBS
    jobs = _seed_finished_jobs(reg, cap + 1)

    reg._evict_old()

    assert len(reg._jobs) == cap
    # The oldest (created_at == 0.0) is the one removed; newest survive.
    assert jobs[0].job_id not in reg._jobs
    assert jobs[-1].job_id in reg._jobs


def test_evict_old_keeps_unfinished_jobs_over_cap(monkeypatch):
    """Over-cap eviction only removes finished jobs; pending ones survive."""
    monkeypatch.setattr(ingest.time, "time", lambda: 0.0)
    reg = JobRegistry()
    cap = ingest._MAX_RECENT_JOBS
    # All pending and one over cap -> nothing is eligible for cap eviction.
    _seed_finished_jobs(reg, cap + 1, status="pending")

    reg._evict_old()

    assert len(reg._jobs) == cap + 1


def test_evict_old_removes_aged_finished_job(monkeypatch):
    """A finished job older than the retention window is age-evicted."""
    now = 10_000.0
    monkeypatch.setattr(ingest.time, "time", lambda: now)
    reg = JobRegistry()
    stale = Job(job_id="old", filename="o.pdf", size_bytes=1, status="done")
    stale.finished_at = now - ingest._JOB_RETENTION_SECONDS - 1
    fresh = Job(job_id="new", filename="n.pdf", size_bytes=1, status="done")
    fresh.finished_at = now  # well within retention
    reg._jobs["old"] = stale
    reg._jobs["new"] = fresh

    reg._evict_old()

    assert "old" not in reg._jobs
    assert "new" in reg._jobs


# ---------------------------------------------------------------------------
# _last_activity_ts
# ---------------------------------------------------------------------------


def test_last_activity_ts_uses_created_when_only_field():
    """With no start/events, the created_at timestamp is the activity ts."""
    job = _make_job(created_at=500.0)

    assert _last_activity_ts(job) == 500.0


def test_last_activity_ts_prefers_latest_event(monkeypatch):
    """The most recent event timestamp dominates created/started."""
    monkeypatch.setattr(ingest.time, "time", lambda: 900.0)
    job = _make_job(created_at=100.0, started_at=200.0)
    job.emit("stage", "newest")  # event ts == 900.0

    assert _last_activity_ts(job) == 900.0


def test_last_activity_ts_uses_started_when_later_than_created():
    """started_at wins over created_at when there are no events."""
    job = _make_job(created_at=100.0, started_at=300.0)

    assert _last_activity_ts(job) == 300.0


# ---------------------------------------------------------------------------
# _sweep_stale_jobs
# ---------------------------------------------------------------------------


def _isolated_registry(monkeypatch) -> JobRegistry:
    """Swap in a clean module-level registry for sweeper tests."""
    reg = JobRegistry()
    monkeypatch.setattr(ingest, "_registry", reg)
    return reg


def test_sweep_flips_stale_running_job(monkeypatch):
    """A running job past the stale window is flipped to failed and counted."""
    now = 100_000.0
    monkeypatch.setattr(ingest.time, "time", lambda: now)
    reg = _isolated_registry(monkeypatch)
    job = Job(job_id="j", filename="f.pdf", size_bytes=1, status="running")
    job.created_at = now - ingest._STALE_JOB_SECONDS - 1  # strictly older
    reg._jobs["j"] = job

    flipped = asyncio.run(ingest._sweep_stale_jobs())

    assert flipped == 1
    assert job.status == "failed"
    assert job.error == "job stalled — no worker activity"
    assert job.finished_at == now


def test_sweep_leaves_fresh_job(monkeypatch):
    """A running job within the stale window is left untouched and uncounted."""
    now = 100_000.0
    monkeypatch.setattr(ingest.time, "time", lambda: now)
    reg = _isolated_registry(monkeypatch)
    job = Job(job_id="j", filename="f.pdf", size_bytes=1, status="running")
    job.created_at = now - ingest._STALE_JOB_SECONDS + 1  # still fresh
    reg._jobs["j"] = job

    flipped = asyncio.run(ingest._sweep_stale_jobs())

    assert flipped == 0
    assert job.status == "running"


def test_sweep_boundary_at_exact_timeout_is_not_stale(monkeypatch):
    """age == _STALE_JOB_SECONDS is NOT stale (strict > comparison)."""
    now = 100_000.0
    monkeypatch.setattr(ingest.time, "time", lambda: now)
    reg = _isolated_registry(monkeypatch)
    job = Job(job_id="j", filename="f.pdf", size_bytes=1, status="running")
    job.created_at = now - ingest._STALE_JOB_SECONDS  # exactly at the window
    reg._jobs["j"] = job

    flipped = asyncio.run(ingest._sweep_stale_jobs())

    assert flipped == 0
    assert job.status == "running"


def test_sweep_ignores_terminal_jobs(monkeypatch):
    """A done job, however old, is never swept."""
    now = 100_000.0
    monkeypatch.setattr(ingest.time, "time", lambda: now)
    reg = _isolated_registry(monkeypatch)
    job = Job(job_id="j", filename="f.pdf", size_bytes=1, status="done")
    job.created_at = 0.0  # ancient
    reg._jobs["j"] = job

    flipped = asyncio.run(ingest._sweep_stale_jobs())

    assert flipped == 0
    assert job.status == "done"


# ---------------------------------------------------------------------------
# _config_from_options
# ---------------------------------------------------------------------------


def test_config_from_options_returns_ingestion_config():
    """_config_from_options yields an IngestionConfig instance."""
    cfg = _config_from_options({"anything": "ignored"})

    assert isinstance(cfg, IngestionConfig)


def test_config_from_options_ignores_input_and_uses_defaults():
    """The UI surface is minimal: options are ignored; defaults are returned."""
    from_empty = _config_from_options({})
    from_populated = _config_from_options({"llm_temperature": 0.99, "max_keywords": 999})
    default = IngestionConfig()

    assert from_empty.llm_temperature == default.llm_temperature
    assert from_populated.llm_temperature == default.llm_temperature
    assert from_populated.max_keywords == default.max_keywords


# ---------------------------------------------------------------------------
# _build_source_args
# ---------------------------------------------------------------------------


def test_build_source_args_from_path_and_filename(tmp_path):
    """SourceArgs carries the resolved path, filename, upload connector, ids."""
    f = tmp_path / "real.pdf"
    f.write_bytes(b"hello")
    stat = f.stat()

    args = _build_source_args(f, "real.pdf")

    assert isinstance(args, SourceArgs)
    assert args.source_name == "real.pdf"
    assert args.source_path == str(f.resolve())
    assert args.source_uri == f.resolve().as_uri()
    assert args.connector == "upload"
    assert args.source_id == f"upload:{stat.st_dev}:{stat.st_ino}"
    assert args.source_key == f"upload:upload:{stat.st_dev}:{stat.st_ino}"
    assert args.source_version == str(stat.st_mtime_ns)


# ---------------------------------------------------------------------------
# _summarize
# ---------------------------------------------------------------------------


_SUMMARY_KEYS = {
    "job_id",
    "filename",
    "size_bytes",
    "status",
    "created_at",
    "started_at",
    "finished_at",
    "stored_chunks",
    "error",
    "config_summary",
    "workflow_id",
}


def test_summarize_exact_key_set_without_events():
    """include_events=False (default): the fixed key set, no events list."""
    job = _make_job(job_id="j-9", filename="x.pdf", size_bytes=7)
    job.emit("stage", "ignored in summary")

    summary = _summarize(job)

    assert set(summary.keys()) == _SUMMARY_KEYS
    assert "events" not in summary
    assert summary["job_id"] == "j-9"
    assert summary["filename"] == "x.pdf"
    assert summary["size_bytes"] == 7
    assert summary["status"] == "pending"


def test_summarize_include_events_adds_event_dicts():
    """include_events=True appends serialized event dicts under 'events'."""
    job = _make_job()
    evt = job.emit("stage", "parsed", {"phase": "parse"})

    summary = _summarize(job, include_events=True)

    assert set(summary.keys()) == _SUMMARY_KEYS | {"events"}
    assert summary["events"] == [
        {
            "seq": evt.seq,
            "kind": "stage",
            "message": "parsed",
            "detail": {"phase": "parse"},
            "timestamp": evt.timestamp,
        }
    ]


# ---------------------------------------------------------------------------
# start_job_sweeper
# ---------------------------------------------------------------------------


def test_start_job_sweeper_returns_task():
    """start_job_sweeper schedules the loop and returns an asyncio.Task."""

    async def _scenario():
        task = start_job_sweeper()
        assert isinstance(task, asyncio.Task)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return task

    asyncio.run(_scenario())
