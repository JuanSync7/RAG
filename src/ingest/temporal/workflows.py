# @summary
# Temporal workflow definitions for per-document and per-directory ingestion,
# plus DeleteSourceWorkflow for durable post-workflow source cleanup.
# Exports: IngestDocumentWorkflow, IngestDirectoryWorkflow,
#          IngestDocumentArgs, IngestDocumentResult,
#          IngestDirectoryArgs, IngestDirectoryResult,
#          DeleteSourceWorkflow
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
        RAG_INGEST_TEMPORAL_KG_TIMEOUT_MIN,
        RAG_INGEST_TEMPORAL_KG_RETRY_MAX,
        RAG_INGEST_TEMPORAL_KG_RETRY_INTERVAL_S,
    )
    from kgweave.contracts import (
        KG_PHASE2B_ACTIVITY,
        KG_TASK_QUEUE,
        KGIngestRequest,
        KGIngestResult,
    )
    from src.ingest.temporal.activities import (
        ActivityArgs,
        DeleteSourceArgs,
        DeleteSourceResult,
        DocProcessingResult,
        EmbeddingResult,
        ListPendingPhase2bArgs,
        RecordPhaseStatusArgs,
        SourceArgs,
        delete_source_activity,
        document_processing_activity,
        embedding_pipeline_activity,
        list_pending_phase2b_activity,
        record_phase_status_activity,
    )
    from src.ingest.temporal.constants import (
        TRIGGER_BATCH,
        TRIGGER_SINGLE,
        trigger_to_priority,
        trigger_to_queue,
    )
    from temporalio.exceptions import ActivityError, ApplicationError

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
    # Phase 2b (KG) outcome — populated only when enable_kg_phase2b=True.
    # Values: "skipped" | "ok" | "failed_pending_retry" | "failed_permanent".
    kg_status: str = "skipped"
    kg_error_class: str = ""  # "transient" | "document" | "system" | ""
    kg_entities_added: int = 0
    kg_triples_added: int = 0


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

# KG phase 2b: bounded retries, transient-only. document/system error classes
# are non_retryable on the worker side, so Temporal won't retry them.
_KG_PHASE2B_TIMEOUT = timedelta(minutes=RAG_INGEST_TEMPORAL_KG_TIMEOUT_MIN)
_KG_PHASE2B_RETRY = RetryPolicy(
    maximum_attempts=RAG_INGEST_TEMPORAL_KG_RETRY_MAX,
    initial_interval=timedelta(seconds=RAG_INGEST_TEMPORAL_KG_RETRY_INTERVAL_S),
    maximum_interval=timedelta(minutes=5),
    non_retryable_error_types=["document", "system"],
)
_LEDGER_RETRY = RetryPolicy(maximum_attempts=3)


def _extract_error_class(exc: BaseException) -> str:
    """Walk an ActivityError chain to find the underlying ApplicationError type."""
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, ApplicationError) and cur.type:
            return cur.type
        if isinstance(cur, ActivityError) and cur.cause is not None:
            cur = cur.cause
            continue
        cur = cur.__cause__ or cur.__context__
    return "transient"


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

        # ── Phase 2a (embedding) ─────────────────────────────────────────
        phase2: EmbeddingResult = await workflow.execute_activity(
            embedding_pipeline_activity,
            activity_args,
            schedule_to_close_timeout=_EMBEDDING_TIMEOUT,
            retry_policy=_RETRY_POLICY,
        )
        if phase2.errors:
            return IngestDocumentResult(
                source_key=args.source.source_key,
                errors=phase2.errors,
                stored_count=phase2.stored_count,
                processing_log=phase1.processing_log + phase2.processing_log,
            )

        # ── Phase 2b (KG) ────────────────────────────────────────────────
        # Sequential after embedding; gated on enable_kg_phase2b. KG runs
        # in the KGWeave worker fleet — RagWeave dispatches by activity
        # name on KG_TASK_QUEUE, no Python import of KGWeave code.
        kg_status, kg_error_class, kg_entities, kg_triples = await self._run_kg_phase2b(
            args, phase2.processing_log,
        )

        return IngestDocumentResult(
            source_key=args.source.source_key,
            errors=phase2.errors,
            stored_count=phase2.stored_count,
            processing_log=phase1.processing_log + phase2.processing_log,
            kg_status=kg_status,
            kg_error_class=kg_error_class,
            kg_entities_added=kg_entities,
            kg_triples_added=kg_triples,
        )

    async def _run_kg_phase2b(
        self,
        args: IngestDocumentArgs,
        processing_log: list,
    ) -> tuple[str, str, int, int]:
        """Dispatch KG phase 2b on ``KG_TASK_QUEUE`` and record the outcome.

        Returns ``(kg_status, kg_error_class, entities_added, triples_added)``.
        Workflow always returns SUCCESS — KG failures live in the durable
        ledger, drained by ``BackfillKGWorkflow``.
        """
        config = args.config
        if not config.get("enable_kg_phase2b", False):
            return ("skipped", "", 0, 0)

        clean_store_dir = config.get("clean_store_dir", "")
        if not clean_store_dir:
            workflow.logger.warning(
                "kg_phase2b skipped: clean_store_dir not set source_key=%s",
                args.source.source_key,
            )
            return ("skipped", "", 0, 0)

        request = KGIngestRequest(
            source_key=args.source.source_key,
            clean_text="",  # KGWeave reads from CleanDocumentStore by source_key
            meta={
                "source_name": args.source.source_name,
                "source_uri": args.source.source_uri,
            },
            trace_id=workflow.info().workflow_id,
        )

        try:
            result: KGIngestResult = await workflow.execute_activity(
                KG_PHASE2B_ACTIVITY,
                request,
                task_queue=KG_TASK_QUEUE,
                schedule_to_close_timeout=_KG_PHASE2B_TIMEOUT,
                retry_policy=_KG_PHASE2B_RETRY,
                result_type=KGIngestResult,
            )
        except ActivityError as exc:
            error_class = _extract_error_class(exc)
            await workflow.execute_activity(
                record_phase_status_activity,
                RecordPhaseStatusArgs(
                    clean_store_dir=clean_store_dir,
                    source_key=args.source.source_key,
                    phase="phase2b",
                    success=False,
                    error=str(exc)[:500],
                    error_class=error_class,
                ),
                schedule_to_close_timeout=timedelta(seconds=30),
                retry_policy=_LEDGER_RETRY,
            )
            workflow.logger.warning(
                "kg_phase2b failed source_key=%s class=%s — ledger marked",
                args.source.source_key, error_class,
            )
            kg_status = (
                "failed_permanent" if error_class in ("document", "system")
                else "failed_pending_retry"
            )
            return (kg_status, error_class, 0, 0)

        await workflow.execute_activity(
            record_phase_status_activity,
            RecordPhaseStatusArgs(
                clean_store_dir=clean_store_dir,
                source_key=args.source.source_key,
                phase="phase2b",
                success=True,
            ),
            schedule_to_close_timeout=timedelta(seconds=30),
            retry_policy=_LEDGER_RETRY,
        )
        return ("ok", "", result.entities_added, result.triples_added)


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


# ---------------------------------------------------------------------------
# Delete-source workflow (Issue #42, PR-B)
# ---------------------------------------------------------------------------

@workflow.defn
class DeleteSourceWorkflow:
    """Temporal-retryable wrapper around delete_source_activity.

    Provides a durable cleanup primitive callable from operator tools or future
    post-cancel hooks. Idempotent at the activity level, so retries are safe.

    Filters by ``staging_batch_id`` when supplied (post-cancel rollback of one
    in-flight batch) or by ``source_key`` only (manual full purge).

    Wiring into a future cancel-job HTTP endpoint will land separately when
    async ingest submission is added (no synchronous cancel route exists yet).
    """

    @workflow.run
    async def run(self, args: "DeleteSourceArgs") -> "DeleteSourceResult":
        workflow.logger.info(
            "delete_source workflow entry source_key=%s batch=%s",
            args.source_key,
            args.staging_batch_id or "<all>",
        )
        return await workflow.execute_activity(
            delete_source_activity,
            args,
            schedule_to_close_timeout=timedelta(minutes=5),
            retry_policy=_RETRY_POLICY,
        )


# ---------------------------------------------------------------------------
# Phase 2b backfill workflow (Step 11)
# ---------------------------------------------------------------------------


@dataclass
class BackfillKGArgs:
    """Args for ``BackfillKGWorkflow``.

    Args:
        clean_store_dir: Directory backing the ``CleanDocumentStore``.
        source_meta: Optional per-source-key meta to pass to the KG activity.
            Falls back to an empty dict per key when absent.
        max_per_run: Cap on documents to retry in one workflow run so a
            scheduled tick can't fan out unboundedly.
    """

    clean_store_dir: str
    source_meta: dict = field(default_factory=dict)
    max_per_run: int = 100


@dataclass
class BackfillKGResult:
    discovered: int
    succeeded: int
    failed_pending_retry: int
    failed_permanent: int
    skipped: int = 0


@workflow.defn
class BackfillKGWorkflow:
    """Drain ``failed_pending_retry`` Phase 2b entries from the status ledger.

    Designed to be scheduled (Temporal Schedule, cron, etc.). Each run:
      1. Lists pending source keys via ``list_pending_phase2b_activity``.
      2. Caps the worklist at ``max_per_run``.
      3. Dispatches the KG activity by name on ``KG_TASK_QUEUE`` for each.
      4. Records the outcome to the ledger so successes drop out of the queue
         and permanent failures stop being retried.

    Failures during the run do not abort the loop — each source is retried
    independently. Counts are aggregated into ``BackfillKGResult``.
    """

    @workflow.run
    async def run(self, args: BackfillKGArgs) -> BackfillKGResult:
        keys = await workflow.execute_activity(
            list_pending_phase2b_activity,
            ListPendingPhase2bArgs(clean_store_dir=args.clean_store_dir),
            schedule_to_close_timeout=timedelta(seconds=30),
            retry_policy=_LEDGER_RETRY,
        )
        worklist = list(keys)[: args.max_per_run]
        skipped = max(0, len(keys) - len(worklist))
        workflow.logger.info(
            "backfill_kg discovered=%d worklist=%d skipped=%d",
            len(keys), len(worklist), skipped,
        )

        succeeded = 0
        failed_pending = 0
        failed_permanent = 0

        for source_key in worklist:
            request = KGIngestRequest(
                source_key=source_key,
                clean_text="",
                meta=args.source_meta.get(source_key, {}),
                trace_id=workflow.info().workflow_id,
            )
            try:
                await workflow.execute_activity(
                    KG_PHASE2B_ACTIVITY,
                    request,
                    task_queue=KG_TASK_QUEUE,
                    schedule_to_close_timeout=_KG_PHASE2B_TIMEOUT,
                    retry_policy=_KG_PHASE2B_RETRY,
                    result_type=KGIngestResult,
                )
            except ActivityError as exc:
                error_class = _extract_error_class(exc)
                await workflow.execute_activity(
                    record_phase_status_activity,
                    RecordPhaseStatusArgs(
                        clean_store_dir=args.clean_store_dir,
                        source_key=source_key,
                        phase="phase2b",
                        success=False,
                        error=str(exc)[:500],
                        error_class=error_class,
                    ),
                    schedule_to_close_timeout=timedelta(seconds=30),
                    retry_policy=_LEDGER_RETRY,
                )
                if error_class in ("document", "system"):
                    failed_permanent += 1
                else:
                    failed_pending += 1
                continue

            await workflow.execute_activity(
                record_phase_status_activity,
                RecordPhaseStatusArgs(
                    clean_store_dir=args.clean_store_dir,
                    source_key=source_key,
                    phase="phase2b",
                    success=True,
                ),
                schedule_to_close_timeout=timedelta(seconds=30),
                retry_policy=_LEDGER_RETRY,
            )
            succeeded += 1

        return BackfillKGResult(
            discovered=len(keys),
            succeeded=succeeded,
            failed_pending_retry=failed_pending,
            failed_permanent=failed_permanent,
            skipped=skipped,
        )
