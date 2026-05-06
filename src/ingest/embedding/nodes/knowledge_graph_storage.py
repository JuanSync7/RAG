# @summary
# LangGraph node that stages KG chunk calls for atomic commit (Issue #42).
# Exports: knowledge_graph_storage_node
# Deps: embedding.state
# @end-summary

"""Knowledge-graph storage node — stages add_chunk calls; defers to commit_node."""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger("rag.ingest.embedding.knowledge_graph_storage")

from src.ingest.common import append_processing_log
from src.ingest.embedding.state import EmbeddingPipelineState
from src.ingest.common.observability import node_span


@node_span("knowledge_graph_storage")
def knowledge_graph_storage_node(state: EmbeddingPipelineState) -> dict[str, Any]:
    """Stage chunk texts for KG add_chunk replay at commit_node (Issue #42).

    The actual ``kg_builder.add_chunk`` calls are deferred to ``commit_node``
    so that all three sinks (MinIO, Weaviate, KG) are written in one atomic
    terminal phase.

    Args:
        state: Ingestion pipeline state.

    Returns:
        Partial state update containing ``staged_kg_chunks`` and an updated
        ``processing_log``. When the stage is disabled or the runtime has no
        KG builder, returns an empty ``staged_kg_chunks`` list.
    """
    t0 = time.monotonic()
    config = state["runtime"].config
    kg_builder = state["runtime"].kg_builder
    if (
        not config.enable_knowledge_graph_storage
        or kg_builder is None
        or getattr(config, "enable_kg_phase2b", False)
    ):
        logger.debug("knowledge_graph_storage_node skipped in %.3fs", time.monotonic() - t0)
        return {
            "staged_kg_chunks": [],
            "processing_log": append_processing_log(
                state, "knowledge_graph_storage:skipped"
            ),
        }

    chunks = state.get("chunks", [])
    source_name = state["source_name"]
    staged = [(chunk.text, source_name) for chunk in chunks]

    logger.debug(
        "knowledge_graph_storage_node staged %d chunks in %.3fs",
        len(staged),
        time.monotonic() - t0,
    )
    return {
        "staged_kg_chunks": staged,
        "processing_log": append_processing_log(state, "knowledge_graph_storage:staged"),
    }
