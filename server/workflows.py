# @summary
# Temporal workflows for RAG orchestration. RAGQueryWorkflow wraps each user
# query in durable timeout/retry semantics; TurnRetrieveWorkflow wraps the
# turn-loop retrieve_ranked activity (idempotent, no-LLM retrieval primitive,
# caller-supplied timeout_ms; TURN_LOOP_DESIGN.md §3).
# Exports: RAGQueryWorkflow, TurnRetrieveWorkflow, RAG_QUERY_TASK_QUEUE
# Deps: temporalio, server.activities
# @end-summary
"""Temporal workflow definitions for RAG query processing."""

from datetime import timedelta
import math

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from config.settings import RAG_WORKFLOW_DEFAULT_TIMEOUT_MS
    from server.activities import execute_rag_query, retrieve_ranked
    from server.search_attributes import (
        DR_EARLY_STOPPED,
        DR_ENABLED,
        DR_ITERATIONS,
        DR_LLM_CALLS,
        DR_TOPIC_COUNT,
    )

RAG_QUERY_TASK_QUEUE = "rag-query"


def _build_dr_search_attributes(request: dict, result: dict) -> dict:
    """Pure helper: derive DR search-attribute key/value pairs from the
    request + activity result.

    Pulled out so it can be unit-tested without instantiating a workflow.
    Edge cases:
    - DR not requested → only DREnabled=False is set; the other counters
      are not meaningful for baseline runs.
    - DR requested but no metadata.deep_research present (orchestrator
      failed early / sanitizer rejected before DR kicked off) → emit
      zeros so dashboards still see a coherent shape.
    """
    enabled = bool(request.get("deep_research") or False)
    if not enabled:
        return {DR_ENABLED: False}

    metadata = (result or {}).get("metadata") or {}
    dr_meta = metadata.get("deep_research") if isinstance(metadata, dict) else None

    if not isinstance(dr_meta, dict):
        return {
            DR_ENABLED: True,
            DR_ITERATIONS: 0,
            DR_LLM_CALLS: 0,
            DR_EARLY_STOPPED: True,
            DR_TOPIC_COUNT: 0,
        }

    iterations = int(dr_meta.get("iteration_count", 0) or 0)
    llm_calls = int(dr_meta.get("llm_call_count", 0) or 0)
    topic_count = int(dr_meta.get("topic_count", 0) or 0)
    # Early-stopped = orchestrator never ran a multi-topic decomposition.
    early_stopped = not bool(dr_meta.get("decomposed", False))

    return {
        DR_ENABLED: True,
        DR_ITERATIONS: iterations,
        DR_LLM_CALLS: llm_calls,
        DR_EARLY_STOPPED: early_stopped,
        DR_TOPIC_COUNT: topic_count,
    }


@workflow.defn
class RAGQueryWorkflow:
    """Orchestrates a single RAG query through the pipeline.

    The actual inference happens in the activity (worker process where models
    are preloaded). This workflow provides:
    - Durable execution with automatic retries on transient failures
    - Timeout management (query can't hang forever)
    - Visibility in the Temporal UI for debugging/monitoring
    - Workflow ID for deduplication of identical concurrent queries
    - Custom search attributes for Deep Research observability
      (DRIterations, DRLLMCalls, DREarlyStopped, DRTopicCount, DREnabled)
    """

    @workflow.run
    async def run(self, request: dict) -> dict:
        timeout_ms = int(request.get("overall_timeout_ms", RAG_WORKFLOW_DEFAULT_TIMEOUT_MS))
        # Honor client timeout budgets by rounding up milliseconds to seconds.
        timeout_seconds = max(1, math.ceil(timeout_ms / 1000))
        result = await workflow.execute_activity(
            execute_rag_query,
            request,
            start_to_close_timeout=timedelta(seconds=timeout_seconds),
            schedule_to_close_timeout=timedelta(seconds=timeout_seconds),
            retry_policy=RetryPolicy(
                initial_interval=timedelta(seconds=1),
                maximum_interval=timedelta(seconds=10),
                maximum_attempts=3,
                non_retryable_error_types=["ValueError"],
            ),
        )

        # Surface DR observability via Temporal custom search attributes.
        # DISABLED: workflow.upsert_search_attributes() with the deprecated dict
        # API raises a TypeError *inside* the Temporal SDK that leaves a malformed
        # ("empty variant") command in the workflow completion. The try/except
        # below caught the Python exception but could NOT un-queue the bad command,
        # so the completion failed and the (already-computed) query result never
        # returned to the client (every query 500s/times out). Cosmetic
        # observability only — re-enable once migrated to the typed
        # SearchAttribute API (workflow.upsert_search_attributes([SearchAttributePair(...)])).
        #   try:
        #       attrs = _build_dr_search_attributes(request, result)
        #       workflow.upsert_search_attributes(attrs)
        #   except Exception:  # noqa: BLE001
        #       workflow.logger.exception("Failed to upsert DR search attributes")

        return result


@workflow.defn
class TurnRetrieveWorkflow:
    """One turn-loop retrieval round as a durable Temporal execution.

    Thin wrapper over the single ``retrieve_ranked`` activity — the
    API-process turn loop calls this once per RETRIEVE action
    (TURN_LOOP_DESIGN.md §3). The loop itself (controller, HyDE, judge,
    drafting) lives in the API process; only this retrieval primitive is
    durable.

    Idempotency contract: ``retrieve_ranked`` contains no LLM calls and no
    writes — it is a pure embed -> hybrid-search -> rerank -> filter read, so
    re-executing it on a transient failure returns an equivalent result and
    ``maximum_attempts=3`` retries are safe (design §11 risk 2). Validation
    failures raise ``ValueError``, which is non-retryable per the shared
    retry-policy convention.

    Timeout contract: the caller supplies ``request["timeout_ms"]`` — the
    turn-loop injector (``server/turn_loop_runner.build_retrieve_ranked``)
    passes the turn's REMAINING wall-clock budget per call, so the
    per-activity timeout is always strictly below
    ``RAG_TURN_LOOP_WALL_CLOCK_MS``; ``validate_turn_loop_config()`` enforces
    the outer hierarchy ``RAG_TURN_LOOP_WALL_CLOCK_MS`` <
    ``RAG_WORKFLOW_DEFAULT_TIMEOUT_MS``. Absent, it falls back to
    ``RAG_WORKFLOW_DEFAULT_TIMEOUT_MS``.
    """

    @workflow.run
    async def run(self, request: dict) -> dict:
        timeout_ms = int(request.get("timeout_ms", RAG_WORKFLOW_DEFAULT_TIMEOUT_MS))
        # Honor client timeout budgets by rounding up milliseconds to seconds.
        timeout_seconds = max(1, math.ceil(timeout_ms / 1000))
        return await workflow.execute_activity(
            retrieve_ranked,
            request,
            start_to_close_timeout=timedelta(seconds=timeout_seconds),
            schedule_to_close_timeout=timedelta(seconds=timeout_seconds),
            retry_policy=RetryPolicy(
                initial_interval=timedelta(seconds=1),
                maximum_interval=timedelta(seconds=10),
                maximum_attempts=3,
                non_retryable_error_types=["ValueError"],
            ),
        )
