# @summary
# LangGraph TypedDict state contract for the Phase 2 Embedding Pipeline.
# Exports: EmbeddingPipelineState
# Deps: src.ingest.common.types, src.ingest.common.schemas
# Fields: runtime, source_*, raw_text, cleaned_text, clean_hash,
#   document_id, chunks, metadata_*, cross_references, kg_triples,
#   stored_count, errors, processing_log,
#   parse_result (Any, ParseResult from parser abstraction — replaces docling_document),
#   parser_instance (Any, transient parser instance for chunk() call — never serialised),
#   visual_stored_count (int, FR-602), page_images (Optional[list[Any]], FR-602),
#   trace_id (str, FR-3052), batch_id (str, FR-3053),
#   staging_batch_id (str, Issue #42 atomicity), source_hash (str, Issue #42),
#   staged_minio (Optional[dict], Issue #42), staged_weaviate_records (list, Issue #42),
#   staged_weaviate_delete_old (bool, Issue #42), staged_kg_chunks (list, Issue #42)
# NOTE: docling_document field removed (Phase 3.2 / FR-3205 AC 2). It was in-memory
#   only and was never persisted. Replaced by parse_result + parser_instance.
# @end-summary

"""State contract for the Embedding Pipeline (Phase 2, nodes 6–13)."""

from __future__ import annotations

from typing import Any, Optional, TypedDict

from src.ingest.common import ProcessedChunk
from src.ingest.common import Runtime


class EmbeddingPipelineState(TypedDict, total=False):
    """Shared state flowing through the 8-node Embedding Pipeline DAG.

    Initial fields are populated by the orchestrator from CleanDocumentStore
    before the graph is invoked.

    Fields
    ------
    runtime : Runtime
        Shared runtime dependencies.
    source_key : str
        Stable source identity key.
    source_name : str
        Display name.
    source_uri : str
        Stable URI for the source.
    source_id : str
        OS-level stable identity.
    source_version : str
        Source version string.
    connector : str
        Connector identifier.
    raw_text : str
        The clean Markdown text read from CleanDocumentStore (used as raw_text
        for compatibility with chunking/enrichment nodes).
    cleaned_text : str
        Same as raw_text for Phase 2 entry point — the clean text is the input.
    clean_hash : str
        SHA-256 of the clean text (for change detection on Phase 2 re-runs).
    document_id : str
        Stable document UUID set by document_storage_node (node X), links to MinIO.
    chunks : list[ProcessedChunk]
        Chunks produced by chunking_node (node 6).
    metadata_summary : str
        LLM-generated document summary from metadata_generation_node (node 8).
    metadata_keywords : list[str]
        Extracted keywords from metadata_generation_node (node 8).
    cross_references : list[dict[str, str]]
        Pattern-matched cross-references from cross_reference_extraction_node (node 9).
    kg_triples : list[dict[str, Any]]
        Extracted KG triples (subject/predicate/object) from kg_extraction_node (node 10).
    stored_count : int
        Number of chunks successfully stored in Weaviate from embedding_storage_node (node 12).
    errors : list[str]
        Error messages from any node.
    processing_log : list[str]
        Stage completion log entries.
    parse_result : Any | None
        ParseResult from the parser abstraction. None on legacy paths.
    parser_instance : Any | None
        Transient DocumentParser instance from structure_detection_node.
        Never serialised. Only valid during a single document's processing.
    """

    runtime: Runtime
    source_key: str
    source_name: str
    source_uri: str
    source_id: str
    source_version: str
    connector: str
    raw_text: str
    cleaned_text: str
    clean_hash: str
    document_id: str
    chunks: list[ProcessedChunk]
    metadata_summary: str
    metadata_keywords: list[str]
    cross_references: list[dict[str, str]]
    kg_triples: list[dict[str, Any]]
    stored_count: int
    errors: list[str]
    processing_log: list[str]
    parse_result: Optional[Any]
    """ParseResult from the parser abstraction (Phase 3.2 / FR-3201).

    Populated by structure_detection_node when a ParserRegistry is present.
    Read by chunking_node to select the appropriate chunking path.
    None when no parser abstraction ran (legacy regex fallback path).
    """
    parser_instance: Optional[Any]
    """Transient DocumentParser instance retained from structure_detection_node
    for use by chunking_node.chunk() (Phase 3.2 / FR-3206).

    Never serialised, persisted, or checkpointed. Only valid for the duration
    of one document's in-memory processing run.
    """

    # -- Visual embedding extensions (FR-602) --
    visual_stored_count: int  # FR-602: number of visual page objects stored; default 0
    page_images: Optional[list[Any]]  # FR-602: PIL.Image objects; cleared after node (FR-606)

    # -- Data Lifecycle trace / batch (FR-3052, FR-3053) --
    trace_id: str
    """UUID v4 trace ID propagated from Phase 1 (FR-3052). Empty string = absent/legacy."""
    batch_id: str
    """Optional batch grouping ID (FR-3053). Empty string when not part of a batch."""

    # -- Cross-document dedup extensions (FR-3403) --
    dedup_merge_report: list[dict[str, Any]]
    """List of MergeEvent dicts emitted by cross_document_dedup_node."""
    dedup_stats: dict[str, Any]
    """Dedup run statistics: total_input_chunks, exact_matches, fuzzy_matches, novel_chunks, degraded."""

    # -- Staging fields for atomic commit (Issue #42) --
    staging_batch_id: str
    """Propagated from Phase 1; used as rollback marker by commit_node."""
    source_hash: str
    """Propagated from Phase 1; persisted on every chunk record for idempotency."""
    staged_minio: Optional[dict[str, Any]]
    """Payload staged for MinIO write: {document_id, content, metadata, bucket} or None."""
    staged_weaviate_records: list[Any]
    """List of DocumentRecord objects staged for Weaviate flush at commit_node."""
    staged_weaviate_delete_old: bool
    """Whether to delete-by-source_key before flushing (update_mode path)."""
    staged_kg_chunks: list[Any]
    """List of (text, source_name) tuples staged for KG add_chunk replay at commit_node."""
