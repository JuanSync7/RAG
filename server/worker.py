# @summary
# Temporal worker that preloads RAGChain at startup, then processes query
# activities. Models load once (~22s) and stay in GPU memory for all requests.
# Exports: main
# Deps: temporalio, server.activities, server.workflows, config.settings
# @end-summary
"""Temporal worker — the process where GPU models live.

Start this once. It loads embedding + reranker models into GPU memory,
connects to Temporal, and processes query activities from the task queue.
Every user query runs as inference against already-loaded models.

Usage:
    python -m server.worker
"""

import asyncio
import logging
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from temporalio.client import Client
from temporalio.worker import Worker

from config.settings import RAG_WORKER_CONCURRENCY, TEMPORAL_TARGET_HOST
from server.activities import execute_rag_query, init_rag_chain, shutdown_rag_chain
from server.workflows import RAGQueryWorkflow, RAG_QUERY_TASK_QUEUE

def _configure_console_logging() -> None:
    """Use uvicorn-style console logs for consistent output."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    handler = logging.StreamHandler()
    try:
        from uvicorn.logging import DefaultFormatter

        handler.setFormatter(
            DefaultFormatter(fmt="%(levelprefix)s %(message)s", use_colors=None)
        )
    except Exception:
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    root.addHandler(handler)


_configure_console_logging()
logger = logging.getLogger("rag.server.worker")

MAX_CONCURRENT_ACTIVITIES = RAG_WORKER_CONCURRENCY


async def main() -> None:
    logger.info("=" * 60)
    logger.info("RAG Worker starting")
    logger.info("Temporal host: %s", TEMPORAL_TARGET_HOST)
    logger.info("Task queue: %s", RAG_QUERY_TASK_QUEUE)
    logger.info("Max concurrent activities: %d", MAX_CONCURRENT_ACTIVITIES)
    logger.info("=" * 60)

    # Phase 1: Load models (the expensive one-time cost)
    start = time.time()
    init_rag_chain()
    logger.info("Models loaded in %.1fs — ready to serve queries", time.time() - start)

    # Phase 1b: Pre-warm Ollama so the first query doesn't pay cold-start cost
    from src.retrieval.query.nodes import warm_up_ollama
    warm_up_ollama()

    # Phase 2: Connect to Temporal
    client = await Client.connect(TEMPORAL_TARGET_HOST)
    logger.info("Connected to Temporal at %s", TEMPORAL_TARGET_HOST)

    # Sync activities run in a thread pool (RAGChain.run is synchronous)
    activity_executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_ACTIVITIES)

    worker = Worker(
        client,
        task_queue=RAG_QUERY_TASK_QUEUE,
        workflows=[RAGQueryWorkflow],
        activities=[execute_rag_query],
        activity_executor=activity_executor,
        max_concurrent_activities=MAX_CONCURRENT_ACTIVITIES,
        graceful_shutdown_timeout=timedelta(seconds=10),
    )

    # Graceful shutdown on SIGINT/SIGTERM
    shutdown_event = asyncio.Event()

    def _signal_handler():
        logger.info("Shutdown signal received")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    logger.info("Worker listening on queue '%s' — Ctrl+C to stop", RAG_QUERY_TASK_QUEUE)

    # Run worker until shutdown
    try:
        async with worker:
            await shutdown_event.wait()
    finally:
        shutdown_rag_chain()
        activity_executor.shutdown(wait=False)
        await client.close()
        logger.info("Worker stopped")


if __name__ == "__main__":
    asyncio.run(main())
