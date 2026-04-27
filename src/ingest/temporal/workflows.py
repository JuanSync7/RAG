# @summary
# Temporal workflow definitions for per-document and per-directory ingestion.
# Workflows now accept a trigger_type argument (FR-3565, FR-3566) that is
# propagated to child workflows and used for structured log enrichment (FR-3575).
# Exports: IngestDocumentWorkflow, IngestDirectoryWorkflow,
#          IngestDocumentArgs, IngestDocumentResult,
#          IngestDirectoryArgs, IngestDirectoryResult
# Deps: temporalio, src.ingest.temporal.activities, src.ingest.temporal.constants,
#       config.settings
# @end-summary
"""Temporal workflows for the two-phase ingestion pipeline.

IngestDocumentWorkflow   — one per document, chains Phase 1 → Phase 2.
IngestDirectoryWorkflow  — fans out to N IngestDocumentWorkflow children.

Workflow ID = source_key so re-submitting the same document is idempotent.

Trigger types (FR-3565, FR-3566)
---------------------------------
Both workflow args dataclasses carry a ``trigger_type`` field.  At workflow
entry the type is mapped to the corresponding queue and priority via
``src.ingest.temporal.constants``, and the mapping is emitted as a
structured log entry (FR-3575).

Backward compatibility
-----------------------
``IngestDocumentArgs.trigger_type`` defaults to ``TRIGGER_BATCH`` and
``IngestDirectoryArgs.trigger_type`` defaults to ``TRIGGER_BATCH`` so
existing callers that do not pass the field continue to work without
change (they are treated as batch items at medium priority).
"""

from __future__ import annotations

import asyncio
import dataclasses
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Optional

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from config.settings import (
        RAG_INGEST_TEMPORAL_DOC_TIMEOUT_MIN,
        RAG_INGEST_TEMPORAL_EMB_TIMEOUT_MIN,
        RAG_INGEST_TEMPORAL_RETRY_INTERVAL_S,
        RAG_INGEST_TEMPORAL_RETRY_MAX,
    )
    from src.ingest.temporal.activities import (
        ActivityArgs,
        DocProcessingResult,
        EmbeddingResult,
        SourceArgs,
        document_processing_activity,
        embedding_pipeline_activity,
    )
    from src.ingest.temporal.constants import (
        TRIGGER_BATCH,
        TRIGGER_SINGLE,
        trigger_to_priority,
        trigger_to_queue,
    )

import logging

logger = logging.getLogger("rag.ingest.temporal.workflows")


# ---------------------------------------------------------------------------
# Workflow input / output contracts
# ---------------------------------------------------------------------------

@dataclass
class IngestDocumentArgs:
    """Arguments for a single-document ingestion workflow.

    Args:
        source: Source document descriptor.
        config: ``IngestionConfig`` serialised via ``dataclasses.asdict()``.
        trigger_type: Routing hint — ``"single"``, ``"batch"``, or
            ``"background"``.  Defaults to ``"batch"`` for backward
            compatibility with callers that pre-date dual-queue routing.
    """

    source: SourceArgs
    config: dict  # IngestionConfig serialised via dataclasses.asdict()
    trigger_type: str = TRIGGER_BATCH   # backward-compat default (FR-3565)


@dataclass
class IngestDocumentResult:
    source_key: str
    errors: list
    stored_count: int
    processing_log: list


@dataclass
class IngestDirectoryArgs:
    """Arguments for a directory-level fan-out ingestion workflow.

    Args:
        sources: List of ``SourceArgs`` (or dicts for Temporal serialisation).
        config: ``IngestionConfig`` serialised via ``dataclasses.asdict()``.
        trigger_type: Routing hint — ``"batch"`` or ``"background"``.
            Defaults to ``"batch"`` for backward compatibility.
    """

    sources: list           # list[SourceArgs]
    config: dict
    trigger_type: str = TRIGGER_BATCH   # backward-compat default (FR-3565)


@dataclass
class IngestDirectoryResult:
    processed: int
    failed: int
    stored_chunks: int
    errors: list


# ---------------------------------------------------------------------------
# Per-document workflow
# ---------------------------------------------------------------------------

_DOC_PROCESSING_TIMEOUT = timedelta(minutes=RAG_INGEST_TEMPORAL_DOC_TIMEOUT_MIN)
_EMBEDDING_TIMEOUT = timedelta(minutes=RAG_INGEST_TEMPORAL_EMB_TIMEOUT_MIN)
_RETRY_POLICY = RetryPolicy(
    maximum_attempts=RAG_INGEST_TEMPORAL_RETRY_MAX,
    initial_interval=timedelta(seconds=RAG_INGEST_TEMPORAL_RETRY_INTERVAL_S),
)


@workflow.defn
class IngestDocumentWorkflow:
    """Two-phase ingestion for a single document.

    Phase 1 (document_processing_activity) runs first and saves output to
    CleanDocumentStore.  If it succeeds, Phase 2 (embedding_pipeline_activity)
    reads from there — no large data is passed through Temporal.

    Either phase can be retried independently without re-running the other.

    The ``trigger_type`` field in ``IngestDocumentArgs`` is logged at workflow
    entry for structured observability (FR-3575) but does not change the
    execution path — queue and priority routing happen at the submission
    call site, not inside the workflow body.
    """

    @workflow.run
    async def run(self, args: IngestDocumentArgs) -> IngestDocumentResult:
        # --- Priority / queue mapping log (FR-3575) ---
        _priority = trigger_to_priority(args.trigger_type)
        _queue = trigger_to_queue(args.trigger_type)
        workflow.logger.info(
            "workflow entry source_key=%s trigger_type=%s queue=%s priority=%d",
            args.source.source_key,
            args.trigger_type,
            _queue,
            _priority,
        )

        # Mint a per-document UUID at workflow entry (Issue #42). The same
        # id is passed to Phase 1 and Phase 2 so the commit step can roll
        # back any partial writes by filtering chunks on this property.
        # workflow.uuid4() is deterministic across replay.
        staging_batch_id = str(workflow.uuid4())
        activity_args = ActivityArgs(
            source=args.source,
            config=args.config,
            staging_batch_id=staging_batch_id,
        )

        # ── Phase 1 ──────────────────────────────────────────────────────
        phase1: DocProcessingResult = await workflow.execute_activity(
            document_processing_activity,
            activity_args,
            schedule_to_close_timeout=_DOC_PROCESSING_TIMEOUT,
            retry_policy=_RETRY_POLICY,
        )
        if phase1.errors:
            return IngestDocumentResult(
                source_key=args.source.source_key,
                errors=phase1.errors,
                stored_count=0,
                processing_log=phase1.processing_log,
            )

        # Idempotent short-circuit: source bytes unchanged since last ingest.
        # Phase 2 has nothing to do — chunks for this source_key + source_hash
        # are already in Weaviate (that's how Phase 1 detected the match).
        if getattr(phase1, "skipped", False):
            return IngestDocumentResult(
                source_key=args.source.source_key,
                errors=[],
                stored_count=0,
                processing_log=phase1.processing_log,
            )

        # ── Phase 2 ──────────────────────────────────────────────────────
        phase2: EmbeddingResult = await workflow.execute_activity(
            embedding_pipeline_activity,
            activity_args,
            schedule_to_close_timeout=_EMBEDDING_TIMEOUT,
            retry_policy=_RETRY_POLICY,
        )
        return IngestDocumentResult(
            source_key=args.source.source_key,
            errors=phase2.errors,
            stored_count=phase2.stored_count,
            processing_log=phase1.processing_log + phase2.processing_log,
        )


# ---------------------------------------------------------------------------
# Directory-level workflow (fan-out)
# ---------------------------------------------------------------------------

@workflow.defn
class IngestDirectoryWorkflow:
    """Submit one IngestDocumentWorkflow child per discovered source file.

    All children run concurrently up to Temporal's concurrency limits.
    Results are collected and summarised.

    The ``trigger_type`` from ``IngestDirectoryArgs`` is propagated to every
    child workflow so that batch children inherit medium priority and background
    directory scans inherit low priority (FR-3556 AC-2, FR-3557).
    """

    @workflow.run
    async def run(self, args: IngestDirectoryArgs) -> IngestDirectoryResult:
        # --- Priority / queue mapping log (FR-3575) ---
        _priority = trigger_to_priority(args.trigger_type)
        _queue = trigger_to_queue(args.trigger_type)
        workflow.logger.info(
            "workflow entry source_count=%d trigger_type=%s queue=%s priority=%d",
            len(args.sources),
            args.trigger_type,
            _queue,
            _priority,
        )

        child_handles = []
        for source_dict in args.sources:
            source = SourceArgs(**source_dict) if isinstance(source_dict, dict) else source_dict
            handle = await workflow.start_child_workflow(
                IngestDocumentWorkflow,
                IngestDocumentArgs(
                    source=source,
                    config=args.config,
                    trigger_type=args.trigger_type,   # propagate trigger (FR-3556 AC-2)
                ),
                id=f"ingest-doc-{source.source_key}",
                task_queue=workflow.info().task_queue,  # inherit parent queue
                retry_policy=RetryPolicy(maximum_attempts=1),  # retries handled inside child
            )
            child_handles.append(handle)

        results = await asyncio.gather(*child_handles, return_exceptions=True)

        processed = failed = stored_chunks = 0
        errors: list = []
        for r in results:
            if isinstance(r, Exception):
                failed += 1
                errors.append(str(r))
            elif r.errors:
                failed += 1
                errors.extend(r.errors)
            else:
                processed += 1
                stored_chunks += r.stored_count

        return IngestDirectoryResult(
            processed=processed,
            failed=failed,
            stored_chunks=stored_chunks,
            errors=errors,
        )
