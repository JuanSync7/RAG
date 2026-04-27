# @summary
# Temporal worker entry point for the ingestion pipeline — dual-queue architecture.
# In dual-queue mode runs two Worker instances (user + background queues) with
# independent concurrency slot budgets.  Falls back to a single legacy worker
# when dual-queue env vars are unset (FR-3553).
# Registers DeleteSourceWorkflow and delete_source_activity (Issue #42, PR-B).
# Exports: main, run_worker
# Deps: temporalio, config.settings, src.ingest.temporal.activities,
#       src.ingest.temporal.workflows, src.ingest.temporal.constants
# @end-summary
"""Ingestion pipeline Temporal worker — dual-queue architecture.

Run with:
    python -m src.ingest.temporal.worker

Two workers run concurrently in the same process, each polling a different
task queue with its own slot budget.  Hard slot isolation is achieved
structurally: each ``Worker`` instance has its own ``max_concurrent_activities``
limit, so background tasks can never consume user-queue slots (FR-3562).

When ``RAG_INGEST_USER_TASK_QUEUE`` and ``RAG_INGEST_BACKGROUND_TASK_QUEUE``
are both unset the worker falls back to the legacy single-queue mode on
``TEMPORAL_TASK_QUEUE`` (FR-3553) so deployments can roll out gradually.

Scale by adding replicas — each replica loads its own embedding model and
handles the full pipeline end-to-end.
"""

from __future__ import annotations

import asyncio
import logging
import os

from temporalio.client import Client
from temporalio.worker import Worker

from config.settings import TEMPORAL_TARGET_HOST, TEMPORAL_TASK_QUEUE
from src.ingest.temporal.activities import (
    delete_source_activity,
    document_processing_activity,
    embedding_pipeline_activity,
    prewarm_worker_resources,
)
from src.ingest.temporal.workflows import (
    DeleteSourceWorkflow,
    IngestDirectoryWorkflow,
    IngestDocumentWorkflow,
)

logger = logging.getLogger("rag.ingest.temporal.worker")

_WORKFLOWS = [IngestDirectoryWorkflow, IngestDocumentWorkflow, DeleteSourceWorkflow]
_ACTIVITIES = [document_processing_activity, embedding_pipeline_activity, delete_source_activity]

# ---------------------------------------------------------------------------
# Slot allocation helpers (mirrors config/settings.py precedence — until the
# manager lands those symbols, we derive them locally from the same env vars)
# ---------------------------------------------------------------------------

def _resolve_slots() -> tuple[int, int]:
    """Compute (user_slots, background_slots) from environment variables.

    Precedence (FR-3571):
      1. RAG_INGEST_USER_SLOTS / RAG_INGEST_BACKGROUND_SLOTS (explicit)
      2. RAG_INGEST_WORKER_CONCURRENCY (legacy — 75 %% user / 25 %% background)
      3. Hardcoded defaults: user=3, background=1
    """
    user_raw = os.environ.get("RAG_INGEST_USER_SLOTS")
    bg_raw = os.environ.get("RAG_INGEST_BACKGROUND_SLOTS")
    legacy_raw = os.environ.get("RAG_INGEST_WORKER_CONCURRENCY")

    if user_raw or bg_raw:
        user_slots = int(user_raw) if user_raw else 3
        bg_slots = int(bg_raw) if bg_raw else 1
        if legacy_raw:
            logger.warning(
                "RAG_INGEST_WORKER_CONCURRENCY is set alongside slot-specific "
                "variables (RAG_INGEST_USER_SLOTS / RAG_INGEST_BACKGROUND_SLOTS); "
                "ignoring legacy concurrency value."
            )
        return user_slots, bg_slots

    if legacy_raw:
        total = int(legacy_raw)
        bg_slots = max(1, total // 4)
        user_slots = max(1, total - bg_slots)
        return user_slots, bg_slots

    return 3, 1


def _resolve_queues() -> tuple[bool, str, str]:
    """Return (dual_enabled, user_queue, background_queue).

    When both RAG_INGEST_USER_TASK_QUEUE and RAG_INGEST_BACKGROUND_TASK_QUEUE
    are set, dual-queue mode is active (FR-3553).
    """
    user_q = os.environ.get("RAG_INGEST_USER_TASK_QUEUE", "")
    bg_q = os.environ.get("RAG_INGEST_BACKGROUND_TASK_QUEUE", "")
    dual = bool(user_q and bg_q)
    return dual, user_q, bg_q


def _validate_queues(user_queue: str, bg_queue: str) -> None:
    """Validate queue name strings at worker startup (FR-3570 AC-3)."""
    for name, value in [
        ("TEMPORAL_USER_TASK_QUEUE", user_queue),
        ("TEMPORAL_BACKGROUND_TASK_QUEUE", bg_queue),
    ]:
        if not value or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
        if len(value) > 200:
            raise ValueError(f"{name} exceeds maximum length of 200 characters")
        if any(c.isspace() for c in value):
            raise ValueError(f"{name} must not contain whitespace: {value!r}")


def _validate_slots(user_slots: int, bg_slots: int) -> None:
    """Validate slot allocation values at worker startup (FR-3561 AC-4, AC-5)."""
    if user_slots < 1:
        raise ValueError(f"RAG_INGEST_USER_SLOTS must be >= 1, got {user_slots}")
    if bg_slots < 1:
        raise ValueError(f"RAG_INGEST_BACKGROUND_SLOTS must be >= 1, got {bg_slots}")
    if user_slots + bg_slots < 2:
        raise ValueError("Total slot count (user + background) must be >= 2")


# ---------------------------------------------------------------------------
# Main worker coroutine
# ---------------------------------------------------------------------------

async def run_worker() -> None:
    """Connect to Temporal, prewarm resources, and start worker(s).

    In dual-queue mode (FR-3560): creates two ``Worker`` instances in the
    same process, each polling a different queue with its own slot limit.
    Hard slot isolation is achieved by separate ``Worker`` instances (FR-3562).

    In legacy mode (FR-3553): creates a single ``Worker`` on the legacy
    queue.  Logs a warning so operators know dual-queue is not active.
    """
    dual_enabled, user_queue, bg_queue = _resolve_queues()
    user_slots, bg_slots = _resolve_slots()

    # Validate slots in both modes.
    _validate_slots(user_slots, bg_slots)

    if dual_enabled:
        # Full queue-name validation only when dual mode is active.
        _validate_queues(user_queue, bg_queue)

    logger.info("prewarming worker resources (loading embedding model)...")
    prewarm_worker_resources()
    logger.info("worker resources ready")

    client = await Client.connect(TEMPORAL_TARGET_HOST)

    if dual_enabled:
        # --- Dual-queue mode (FR-3560) ---
        user_worker = Worker(
            client,
            task_queue=user_queue,
            max_concurrent_activities=user_slots,
            workflows=_WORKFLOWS,
            activities=_ACTIVITIES,
        )
        bg_worker = Worker(
            client,
            task_queue=bg_queue,
            max_concurrent_activities=bg_slots,
            workflows=_WORKFLOWS,
            activities=_ACTIVITIES,
        )
        logger.info(
            "worker started mode=dual-queue "
            "user_queue=%s user_slots=%d "
            "bg_queue=%s bg_slots=%d",
            user_queue,
            user_slots,
            bg_queue,
            bg_slots,
        )
        # Both workers run concurrently, sharing prewarmed resources (FR-3562).
        await asyncio.gather(user_worker.run(), bg_worker.run())

    else:
        # --- Legacy single-queue mode (FR-3553) ---
        total_slots = user_slots + bg_slots
        worker = Worker(
            client,
            task_queue=TEMPORAL_TASK_QUEUE,
            max_concurrent_activities=total_slots,
            workflows=_WORKFLOWS,
            activities=_ACTIVITIES,
        )
        logger.warning(
            "Running in legacy single-queue mode. "
            "Set RAG_INGEST_USER_TASK_QUEUE and RAG_INGEST_BACKGROUND_TASK_QUEUE "
            "to enable dual-queue topology. "
            "task_queue=%s max_concurrent_activities=%d",
            TEMPORAL_TASK_QUEUE,
            total_slots,
        )
        await worker.run()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for the ingestion worker process."""
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
