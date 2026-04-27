# @summary
# LangGraph TypedDict state contract for the Phase 1 Document Processing Pipeline.
# Exports: DocumentProcessingState
# Deps: src.ingest.common.types
# Fields: runtime, source_*, raw_text, structure, multimodal_notes, cleaned_text,
#   refactored_text, errors, should_skip, processing_log, docling_document (Optional[Any]),
#   trace_id (str, FR-3051)
# @end-summary

"""State contract for the Document Processing Pipeline (Phase 1, nodes 1–5)."""

from __future__ import annotations

from typing import Any, Optional, TypedDict

from src.ingest.common import Runtime


class DocumentProcessingState(TypedDict, total=False):
    """Shared state flowing through the 5-node Document Processing DAG.

    Populated progressively as nodes complete.

    Fields
    ------
    runtime : Runtime
        Shared runtime dependencies (config, embedder, weaviate, kg_builder).
    source_path : str
        Absolute path to the source file.
    source_name : str
        Display name (relative path or human-readable label).
    source_uri : str
        Stable URI for the source (e.g. file:///...).
    source_key : str
        Stable source identity key (e.g. local_fs:<dev>:<ino>).
    source_id : str
        OS-level stable identity (dev:inode).
    source_hash : str
        SHA-256 of source file bytes. Renamed from ``content_hash`` in IngestState.
    connector : str
        Connector identifier (e.g. ``local_fs``).
    source_version : str
        Source version string (mtime nanoseconds as string).
    raw_text : str
        Format-converted plain/markdown text from the source file.
    structure : dict
        Structure detection results: ``has_figures`` (bool), ``figures`` (list),
        ``heading_count`` (int), ``docling_enabled`` (bool), ``docling_model`` (str).
        After ``structure_detection_node`` runs, also contains
        ``docling_document_available: bool`` indicating whether a
        ``DoclingDocument`` was successfully obtained.
    multimodal_notes : list[str]
        Vision-generated notes for figures. Empty list if multimodal disabled.
    cleaned_text : str
        Boilerplate-stripped, unicode-normalised Markdown text.
    refactored_text : str | None
        LLM-rewritten text (self-contained paragraphs). None if refactoring disabled.
    errors : list[str]
        Error messages from any node. Non-empty triggers orchestrator failure path.
    should_skip : bool
        True if a strict-mode failure requires skipping downstream nodes.
    processing_log : list[str]
        Stage completion log entries for observability.
    """

    runtime: Runtime
    source_path: str
    source_name: str
    source_uri: str
    source_key: str
    source_id: str
    source_hash: str
    connector: str
    source_version: str
    raw_text: str
    structure: dict[str, Any]
    multimodal_notes: list[str]
    cleaned_text: str
    refactored_text: Optional[str]
    errors: list[str]
    should_skip: bool
    processing_log: list[str]
    trace_id: str
    """UUID v4 trace ID generated at workflow start (FR-3051). Empty string
    when absent for backward compatibility.
    """
    raw_bytes: Optional[bytes]
    """Pre-read file bytes supplied by the caller (e.g. ``ingest_directory``).
    When present, ``document_ingestion_node`` uses these bytes directly and
    skips its own disk read, eliminating the double-read that occurs when the
    caller already read the file for an idempotency hash check.  ``None`` when
    not supplied — the node falls back to reading from ``source_path``.
    """
    docling_document: Optional[Any]
    """Native DoclingDocument object from Docling parse. None if Docling
    parsing was disabled or failed. Propagated to CleanDocumentStore and
    used by Phase 2 HybridChunker path.

    The structure dict will contain docling_document_available: bool
    set by structure_detection_node after it runs.
    """
