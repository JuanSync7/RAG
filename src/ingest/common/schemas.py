# @summary
# Shared ingestion-common typed contracts for manifest entries, source identity, and processed chunks.
# Exports: PIPELINE_SCHEMA_VERSION, ManifestEntry, SourceIdentity, ProcessedChunk
# Deps: typing, dataclasses
# @end-summary
"""Shared schema contracts for ingestion.

This module defines lightweight, stable contracts that are used across the
ingestion pipeline to represent:

- The persisted ingestion manifest entries (per source key)
- Source discovery identity payloads
- Processed text chunks ready for embedding/storage
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

# -- Pipeline schema version constant (FR-3100 AC4) --
PIPELINE_SCHEMA_VERSION: str = "1.2.0"
"""Single canonical source of truth for the current pipeline schema version.

Referenced by: manifest writes, Weaviate chunk metadata, MinIO metadata
envelopes, and the migration runner.
"""


class ManifestEntry(TypedDict, total=False):
    """Canonical manifest entry persisted for each source key.

    The manifest is the primary mechanism for supporting idempotent and
    incremental ingestion across runs.

    Keys are optional to allow partial persistence and forward-compatible
    extensions.

    All fields are optional (total=False) per FR-3114 to maintain
    backward compatibility with existing manifests.
    """

    # --- Existing fields (unchanged) ---
    source: str
    source_uri: str
    source_id: str
    source_key: str
    connector: str
    source_version: str
    content_hash: str
    chunk_count: int
    summary: str
    keywords: list[str]
    processing_log: list[str]
    mirror_stem: str
    legacy_name: str

    # --- Data Lifecycle additions (FR-3100, FR-3020, FR-3050, FR-3053, FR-3061) ---
    schema_version: str       # Semantic version, e.g. "1.0.0" (FR-3100)
    trace_id: str             # UUID v4 trace ID for this ingestion run (FR-3050)
    batch_id: str             # Optional batch grouping ID (FR-3053)
    deleted: bool             # True if soft-deleted (FR-3020)
    deleted_at: str           # ISO 8601 timestamp of soft deletion (FR-3020)
    validation: dict          # E2E validation result dict (FR-3061)
    clean_hash: str           # SHA-256 of clean markdown output


class SourceIdentity(TypedDict):
    """Stable identity payload used during source discovery and processing."""

    source_path: str
    source_name: str
    source_uri: str
    source_id: str
    source_key: str
    connector: str
    source_version: str


@dataclass
class ProcessedChunk:
    """A processed document chunk ready for embedding.

    Attributes:
        text: Normalized chunk text suitable for embedding.
        metadata: Arbitrary metadata for retrieval attribution (e.g., source
            path, page number, headings).
    """

    text: str
    metadata: dict = field(default_factory=dict)
