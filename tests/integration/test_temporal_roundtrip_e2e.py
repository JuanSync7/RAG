# @summary
# Live-stack Temporal round-trip e2e. Proves the real integration path
# client -> live Temporal server (:7233) -> in-process Worker -> real
# DeleteSourceWorkflow/delete_source_activity -> typed DeleteSourceResult
# decoded back in the workflow. Uses a brand-new unique source_key so the
# delete is an idempotent no-op (weaviate_deleted == 0) and touches no real
# data. This is the GREEN companion to test_ingest_serve_e2e.py: it exercises
# the same in-process-worker machinery on a workflow whose result dataclass
# (DeleteSourceResult.errors: list[str], all f-strings) actually satisfies the
# Temporal payload contract — unlike EmbeddingResult (see
# tests/ingest/temporal/test_payload_contract.py).
# Deps: pytest, temporalio (real), src.ingest.temporal, src.vector_db,
#       src.db, weaviate (real), config.settings
# @end-summary
"""Live Temporal worker round-trip (delete-source) integration test.

Carries ``pytestmark = [slow, integration]`` so the offline gate
(``-m "not slow and not integration"``) deselects it. Runs only when the live
markers are selected and Temporal is reachable; otherwise SKIPS.

Why DeleteSourceWorkflow rather than the full ingest: the ingest workflow is
currently wedged by a real payload-decode bug (dict errors in a list[str]
field — see test_ingest_serve_e2e.py / test_payload_contract.py). The delete
workflow is fast, side-effect-free on a never-seen source_key, and its result
dataclass is contract-clean, so it gives a GENUINELY GREEN proof that the
live Temporal client/server/worker/typed-result round-trip works.

The load-bearing assertion is that a real, typed ``DeleteSourceResult`` comes
back from the live server decoded correctly, with ``errors`` being a true
``list[str]`` — the exact thing ``EmbeddingResult`` fails to honour.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import timedelta
from typing import Any

import pytest


# Dual-marker per project memory (feedback_dual_marker_gating): both slow AND
# integration so ``-m "not slow and not integration"`` reliably excludes it.
pytestmark = [pytest.mark.slow, pytest.mark.integration]


def _real_packages_or_skip() -> None:
    """Evict conftest stubs and bind the real weaviate/langchain packages.

    delete_source_activity touches the real vector_db + db facades, so the
    src.* modules must be rebound to the real packages (mirrors the eviction
    used by the other live integration tests).
    """
    import importlib  # noqa: PLC0415
    import sys  # noqa: PLC0415

    for mod_name in list(sys.modules):
        if mod_name == "weaviate" or mod_name.startswith("weaviate."):
            del sys.modules[mod_name]
    try:
        import weaviate  # noqa: F401, PLC0415
    except Exception as exc:  # pragma: no cover - infra-dependent
        pytest.skip(f"Real weaviate package not importable: {exc}")
    if getattr(sys.modules.get("weaviate"), "__file__", None) is None:  # pragma: no cover
        pytest.skip("Real weaviate could not be loaded over the stub")

    for mod_name in list(sys.modules):
        if any(
            mod_name == p or mod_name.startswith(p + ".")
            for p in ("src.ingest", "src.vector_db", "src.db", "src.core.embeddings")
        ):
            del sys.modules[mod_name]
    importlib.import_module("src.vector_db")


async def _run_delete_workflow(args: Any, queue: str, wf_id: str) -> Any:
    """Start an in-process Worker and execute DeleteSourceWorkflow once.

    Hard-bounded (execution_timeout + asyncio.wait_for) on principle, even
    though a no-op delete returns near-instantly.
    """
    from temporalio.client import Client  # noqa: PLC0415
    from temporalio.service import RPCError  # noqa: PLC0415
    from temporalio.worker import Worker  # noqa: PLC0415
    from temporalio.worker.workflow_sandbox import (  # noqa: PLC0415
        SandboxedWorkflowRunner,
        SandboxRestrictions,
    )

    from src.ingest.temporal.activities import (  # noqa: PLC0415
        delete_source_activity,
        document_processing_activity,
        embedding_pipeline_activity,
        list_pending_phase2b_activity,
        record_phase_status_activity,
    )
    from src.ingest.temporal.workflows import (  # noqa: PLC0415
        BackfillKGWorkflow,
        DeleteSourceWorkflow,
        IngestDirectoryWorkflow,
        IngestDocumentWorkflow,
    )

    try:
        client = await Client.connect("localhost:7233")
    except (RPCError, RuntimeError, OSError) as exc:  # pragma: no cover - infra
        pytest.skip(f"Live Temporal not reachable: {exc}")

    runner = SandboxedWorkflowRunner(
        restrictions=SandboxRestrictions.default.with_passthrough_modules(
            "httpx", "urllib", "src"
        )
    )
    worker = Worker(
        client,
        task_queue=queue,
        workflows=[
            IngestDocumentWorkflow,
            IngestDirectoryWorkflow,
            DeleteSourceWorkflow,
            BackfillKGWorkflow,
        ],
        activities=[
            document_processing_activity,
            embedding_pipeline_activity,
            delete_source_activity,
            record_phase_status_activity,
            list_pending_phase2b_activity,
        ],
        workflow_runner=runner,
    )

    async with worker:
        return await asyncio.wait_for(
            client.execute_workflow(
                DeleteSourceWorkflow.run,
                args,
                id=wf_id,
                task_queue=queue,
                execution_timeout=timedelta(seconds=60),
            ),
            timeout=90,
        )


def test_delete_source_workflow_round_trip() -> None:
    """client -> live Temporal -> in-process worker -> typed result, decoded.

    Load-bearing assertions:
      * a real ``DeleteSourceResult`` is returned (the typed payload decoded
        across the workflow boundary — the thing ``EmbeddingResult`` fails at);
      * ``errors`` is a genuine ``list[str]`` (contract honoured);
      * deleting a never-seen ``source_key`` is an idempotent no-op
        (``weaviate_deleted == 0``), proving the real activity ran and found
        nothing rather than erroring.
    """
    _real_packages_or_skip()

    from src.ingest.temporal.activities import DeleteSourceArgs  # noqa: PLC0415
    from src.ingest.temporal.workflows import DeleteSourceWorkflow  # noqa: F401, PLC0415

    run_id = uuid.uuid4().hex
    queue = f"itest-{run_id}"
    wf_id = f"itest-del-{run_id}"
    # A source_key that has never been ingested -> nothing to delete anywhere.
    source_key = f"itest/{run_id}/never-ingested.md"

    args = DeleteSourceArgs(source_key=source_key)
    result = asyncio.run(_run_delete_workflow(args, queue=queue, wf_id=wf_id))

    # Typed round-trip succeeded (contrast: EmbeddingResult cannot decode).
    assert result is not None
    assert hasattr(result, "weaviate_deleted")
    assert hasattr(result, "minio_deleted")
    assert hasattr(result, "errors")

    # errors honours the list[str] contract — every entry is a real str.
    assert isinstance(result.errors, list)
    assert all(isinstance(e, str) for e in result.errors), (
        f"DeleteSourceResult.errors must be list[str], got {result.errors!r}"
    )

    # No-op delete on a never-seen key: nothing was removed from Weaviate.
    assert isinstance(result.weaviate_deleted, int)
    assert result.weaviate_deleted == 0, (
        f"expected 0 deletions for a fresh source_key, got "
        f"{result.weaviate_deleted}; errors={result.errors}"
    )
    assert isinstance(result.minio_deleted, bool)
