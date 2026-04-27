# @summary
# Ingestion common package: shared schemas, utilities, state/config types, node helpers, and inter-phase store.
# Exports: ManifestEntry, SourceIdentity, ProcessedChunk, IngestionConfig, Runtime,
#          PIPELINE_NODE_NAMES, append_processing_log, decode_with_fallbacks, sha256_bytes, sha256_path,
#          load_manifest, save_manifest, CleanDocumentStore, MinioCleanStore
# Deps: src.ingest.common.schemas, src.ingest.common.utils, src.ingest.common.types, src.ingest.common.shared,
#       src.ingest.common.clean_store, src.ingest.common.minio_clean_store
# @end-summary
"""Shared contracts and utilities for the ingestion pipeline.

This package centralizes ingestion schemas, state/config types, and deterministic
helpers that are reused across pipeline nodes. The exports in this module are
intended to provide a stable import surface for other ingestion modules.
"""

from src.ingest.common.schemas import ManifestEntry, ProcessedChunk, SourceIdentity
from src.ingest.common.utils import (
    decode_with_fallbacks,
    load_manifest,
    parse_json_object,
    read_text_with_fallbacks,
    save_manifest,
    sha256_bytes,
    sha256_path,
)
from src.ingest.common.types import (
    IngestionConfig,
    IngestionDesignCheck,
    IngestionRunSummary,
    PIPELINE_NODE_NAMES,
    Runtime,
)
from src.ingest.common.shared import (
    append_processing_log,
    map_chunk_provenance,
    extract_keywords_fallback,
    cross_refs,
    quality_score,
)
from src.ingest.common.edit_log import EditLog
from src.ingest.common.clean_store import CleanDocumentStore
from src.ingest.common.minio_clean_store import MinioCleanStore

__all__ = [
    "ManifestEntry",
    "ProcessedChunk",
    "SourceIdentity",
    "decode_with_fallbacks",
    "sha256_bytes",
    "sha256_path",
    "load_manifest",
    "save_manifest",
    "read_text_with_fallbacks",
    "parse_json_object",
    "IngestionConfig",
    "IngestionDesignCheck",
    "IngestionRunSummary",
    "PIPELINE_NODE_NAMES",
    "Runtime",
    "append_processing_log",
    "map_chunk_provenance",
    "extract_keywords_fallback",
    "cross_refs",
    "quality_score",
    "EditLog",
    "CleanDocumentStore",
    "MinioCleanStore",
]

# --- Auto-generated re-exports (fix_encapsulation.py) ---
from src.ingest.common.types import IngestFileResult
