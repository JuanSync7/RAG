# @summary
# Temporal workflow for RAG query orchestration. Each user query becomes a
# workflow execution with durable timeout and retry semantics.
# Exports: RAGQueryWorkflow
# Deps: temporalio, server.activities
# @end-summary
"""Temporal workflow definitions for RAG query processing."""

from datetime import timedelta
import math

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from config.settings import RAG_WORKFLOW_DEFAULT_TIMEOUT_MS
    from server.activities import execute_rag_query

RAG_QUERY_TASK_QUEUE = "rag-query"


@workflow.defn
class RAGQueryWorkflow:
    """Orchestrates a single RAG query through the pipeline.

    The actual inference happens in the activity (worker process where models
    are preloaded). This workflow provides:
    - Durable execution with automatic retries on transient failures
    - Timeout management (query can't hang forever)
    - Visibility in the Temporal UI for debugging/monitoring
    - Workflow ID for deduplication of identical concurrent queries
    """

    @workflow.run
    async def run(self, request: dict) -> dict:
        timeout_ms = int(request.get("overall_timeout_ms", RAG_WORKFLOW_DEFAULT_TIMEOUT_MS))
        # Honor client timeout budgets by rounding up milliseconds to seconds.
        timeout_seconds = max(1, math.ceil(timeout_ms / 1000))
        return await workflow.execute_activity(
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
