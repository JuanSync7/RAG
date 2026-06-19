# @summary
# Phase 2 orchestrator: compiles and invokes the Embedding Pipeline LangGraph.
# Exports: run_embedding_pipeline
# Deps: src.ingest.embedding.workflow, src.ingest.embedding.state, src.ingest.common.types
# trace_id (FR-3052) and batch_id (FR-3053): accepted and injected into initial state.
# NOTE (Phase 3.2): docling_document parameter retained for backward-compat callers but
#   is no longer injected into EmbeddingPipelineState (field removed). parse_result and
#   parser_instance are populated by structure_detection_node at runtime.
# @end-summary

"""Phase 2 runtime implementation for the embedding pipeline."""

from __future__ import annotations

import warnings
from typing import Any, Optional

from src.ingest.common import Runtime
from src.ingest.embedding.state import EmbeddingPipelineState
from src.ingest.embedding.workflow import build_embedding_graph
from src.platform.observability import get_tracer

_GRAPH = build_embedding_graph()


def run_embedding_pipeline(
    runtime: Runtime,
    source_key: str,
    source_name: str,
    source_uri: str,
    source_id: str,
    connector: str,
    source_version: str,
    clean_text: str,
    clean_hash: str,
    docling_document: Optional[Any] = None,
    parse_result: Optional[Any] = None,
    parser_instance: Optional[Any] = None,
    trace_id: str = "",
    batch_id: str = "",
    staging_batch_id: str = "",
    source_hash: str = "",
) -> EmbeddingPipelineState:
    """Run the Phase 2 Embedding Pipeline for a single clean document.

    Args:
        runtime: Shared runtime dependencies.
        source_key: Stable source identity key.
        source_name: Display name for the source.
        source_uri: Stable URI for the source.
        source_id: OS-level stable identity.
        connector: Connector identifier.
        source_version: Source version string.
        clean_text: Clean Markdown text from CleanDocumentStore.
        clean_hash: SHA-256 of ``clean_text`` for change detection.
        docling_document: Retained for backward compatibility. No longer injected
            into EmbeddingPipelineState (Phase 3.2 removed that field). Passing a
            value here has no effect.
        parse_result: ParseResult produced by structure_detection_node (Phase 1)
            via the ParserRegistry. Forwarded in-memory by ingest_directory so
            chunking_node can use the native / markdown parser-abstraction path.
            None → chunking_node falls back to the legacy markdown chunker.
        parser_instance: The live parser used in Phase 1 (e.g. DoclingParser).
            Transient (never serialised); passed in-process so chunking_node can
            call ``parser_instance.chunk(parse_result)`` for native chunking.
        trace_id: UUID v4 trace ID propagated from Phase 1 (FR-3052). Empty
            string when not provided (backward-compatible default).
        batch_id: Optional batch grouping ID (FR-3053). Empty string when not
            part of a batch run.

    Returns:
        Final ``EmbeddingPipelineState`` after all nodes have run.
    """
    if docling_document is not None:
        warnings.warn(
            "docling_document parameter is deprecated and has no effect "
            "(removed in Phase 3.2). It will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )
    initial_state: EmbeddingPipelineState = {
        "runtime": runtime,
        "source_key": source_key,
        "source_name": source_name,
        "source_uri": source_uri,
        "source_id": source_id,
        "connector": connector,
        "source_version": source_version,
        "raw_text": clean_text,
        "cleaned_text": clean_text,
        "clean_hash": clean_hash,
        # parse_result + parser_instance are produced by structure_detection_node
        # in Phase 1 and forwarded in-memory by ingest_directory. When present,
        # chunking_node uses the native/markdown parser-abstraction path; when
        # None it falls back to the legacy markdown chunker.
        "parse_result": parse_result,
        "parser_instance": parser_instance,
        "chunks": [],
        "metadata_summary": "",
        "metadata_keywords": [],
        "cross_references": [],
        "stored_count": 0,
        "errors": [],
        "processing_log": [],
        "trace_id": trace_id,
        "batch_id": batch_id,
        "staging_batch_id": staging_batch_id,
        "source_hash": source_hash,
        # Staging fields for atomic commit (Issue #42); populated by storage nodes.
        "staged_minio": None,
        "staged_weaviate_records": [],
        "staged_weaviate_delete_old": False,
    }
    tracer = get_tracer()
    with tracer.trace(
        "ingest.embedding",
        {
            "source_key": source_key,
            "source_name": source_name,
            "connector": connector,
            "trace_id": trace_id,
            "batch_id": batch_id,
        },
    ):
        try:
            final_state = _GRAPH.invoke(initial_state)
        except Exception as exc:
            final_state = {**initial_state, "errors": [f"embedding_graph:{exc}"], "stored_count": 0}
    return final_state
