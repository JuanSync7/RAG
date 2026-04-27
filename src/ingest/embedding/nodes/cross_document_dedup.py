# @summary
# Cross-document deduplication node for the Embedding Pipeline (Phase 3.3).
# Detects and eliminates duplicate chunks across documents using
# Tier 1 (SHA-256 exact hash) and optional Tier 2 (MinHash fuzzy) matching.
# Override path: when source_key is in config.dedup_override_sources, all chunks
# are stored independently and an "override_skipped" MergeEvent is emitted per chunk.
# Exports: cross_document_dedup_node
# Deps: src.ingest.common, src.ingest.embedding.common.dedup_utils,
#       src.ingest.embedding.common.types, src.ingest.embedding.state,
#       src.ingest.embedding.support.minhash_engine (Tier 2 only, lazy)
# @end-summary
"""Cross-document deduplication node for the Embedding Pipeline.

Inserted between ``quality_validation`` and ``embedding_storage`` in the DAG
(Phase 3.3 wiring). For each surviving chunk this node:

1. Computes a SHA-256 content hash (Tier 1).
2. Queries Weaviate for an exact hash match.
3. If found: appends the source key to the canonical chunk's provenance array
   and drops the duplicate from the output list.
4. If not found and Tier 2 enabled: computes a MinHash fingerprint, queries
   for a fuzzy match above the configured threshold. Longer chunk wins.
5. If no match: attaches hash (and optionally fingerprint) to metadata and
   passes the chunk through as novel.

The node degrades gracefully: on any unhandled exception it logs and passes
all remaining chunks through unchanged.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from src.ingest.common import append_processing_log
from src.ingest.embedding.common.dedup_utils import (
    compute_content_hash,
    find_chunk_by_content_hash,
    append_source_document,
    remove_source_document_refs,
)
from src.ingest.embedding.common.types import MergeEvent, create_merge_event
from src.ingest.embedding.state import EmbeddingPipelineState

logger = logging.getLogger("rag.ingest.embedding.dedup")


def cross_document_dedup_node(state: EmbeddingPipelineState) -> dict[str, Any]:
    """Detect and eliminate cross-document duplicate chunks.

    For each chunk surviving quality validation:
    1. Compute content hash (Tier 1).
    2. Query Weaviate for exact-match on content_hash.
    3. If match: append source_key to canonical chunk's source_documents,
       record merge event, remove chunk from output list.
    4. If no match and Tier 2 enabled: compute MinHash fingerprint,
       query for fuzzy match above threshold.
    5. If no match at all: attach content_hash (and optionally
       fuzzy_fingerprint) to chunk metadata for downstream storage.

    Args:
        state: Embedding pipeline state after quality_validation.

    Returns:
        Partial state update with deduplicated chunks, merge report,
        dedup stats, and updated processing log.
    """
    t0 = time.monotonic()
    config = state["runtime"].config
    chunks = state["chunks"]

    # --- Bypass path (FR-3402) ---
    if not getattr(config, "enable_cross_document_dedup", True):
        return {
            "chunks": chunks,
            "dedup_merge_report": [],
            "dedup_stats": {},
            "processing_log": append_processing_log(
                state, "cross_document_dedup:skipped"
            ),
        }

    runtime = state["runtime"]
    client = runtime.weaviate_client
    source_key = state["source_key"]

    merge_report: list[dict[str, Any]] = []
    novel_chunks = []
    exact_matches = 0
    fuzzy_matches = 0
    degraded = False

    try:
        # --- Re-ingestion back-reference cleanup (FR-3433) ---
        if getattr(config, "update_mode", False):
            remove_source_document_refs(client, source_key)

        # --- Per-source override check (FR-3450) ---
        override_sources = getattr(config, "dedup_override_sources", [])
        skip_lookup = source_key in override_sources

        for chunk in chunks:
            content_hash = compute_content_hash(chunk.text)
            chunk.metadata["content_hash"] = content_hash

            if skip_lookup:
                # Override: store independently but still compute hash (FR-3450).
                # Emit an override_skipped event so the merge report is queryable.
                chunk.metadata.setdefault("source_documents", [source_key])
                merge_report.append(
                    create_merge_event(
                        canonical_content_hash=content_hash,
                        canonical_chunk_id="",
                        merged_source_key=source_key,
                        merged_section=chunk.metadata.get("heading_path", ""),
                        match_tier="override",
                        similarity_score=0.0,
                        canonical_replaced=False,
                        action="override_skipped",
                    )
                )
                novel_chunks.append(chunk)
                continue

            # --- Tier 1: exact content-hash match (FR-3411) ---
            existing = find_chunk_by_content_hash(client, content_hash)
            if existing is not None:
                append_source_document(client, existing["uuid"], source_key)
                merge_report.append(
                    create_merge_event(
                        canonical_content_hash=content_hash,
                        canonical_chunk_id=existing["uuid"],
                        merged_source_key=source_key,
                        merged_section=chunk.metadata.get("heading_path", ""),
                        match_tier="exact",
                        similarity_score=1.0,
                        canonical_replaced=False,
                        action="merged",
                    )
                )
                exact_matches += 1
                continue

            # --- Tier 2: fuzzy fingerprint match (FR-3420–FR-3424) ---
            if getattr(config, "enable_fuzzy_dedup", False):
                fuzzy_match = _try_fuzzy_dedup(
                    client,
                    chunk,
                    content_hash,
                    source_key,
                    config,
                    merge_report,
                )
                if fuzzy_match:
                    fuzzy_matches += 1
                    continue

            # --- Novel chunk: pass through ---
            chunk.metadata.setdefault("source_documents", [source_key])
            novel_chunks.append(chunk)

    except Exception:
        logger.exception(
            "cross_document_dedup degraded — passing remaining chunks through"
        )
        degraded = True
        # Pass through any chunks not yet processed
        novel_chunks = chunks

    dedup_stats: dict[str, Any] = {
        "total_input_chunks": len(chunks),
        "exact_matches": exact_matches,
        "fuzzy_matches": fuzzy_matches,
        "novel_chunks": len(novel_chunks),
        "degraded": degraded,
    }

    log_tag = "cross_document_dedup:degraded" if degraded else "cross_document_dedup:ok"
    logger.info("cross_document_dedup complete: source=%s", state.get("source_name", ""))
    logger.debug("cross_document_dedup_node completed in %.3fs", time.monotonic() - t0)
    return {
        "chunks": novel_chunks,
        "dedup_merge_report": merge_report,
        "dedup_stats": dedup_stats,
        "processing_log": append_processing_log(state, log_tag),
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _try_fuzzy_dedup(
    client: Any,
    chunk: Any,
    content_hash: str,
    source_key: str,
    config: Any,
    merge_report: list[dict[str, Any]],
) -> bool:
    """Attempt Tier 2 fuzzy fingerprint dedup. Returns True if merged.

    Args:
        client: Weaviate client handle.
        chunk: Candidate chunk (ProcessedChunk).
        content_hash: Pre-computed SHA-256 hash for this chunk.
        source_key: Source document identifier.
        config: Active IngestionConfig instance.
        merge_report: Mutable list to append MergeEvent dicts to.

    Returns:
        True if the chunk was merged into an existing canonical; False otherwise.
    """
    from src.ingest.embedding.support.minhash_engine import (
        compute_fuzzy_fingerprint,
        find_chunk_by_fuzzy_fingerprint,
    )

    threshold: float = getattr(config, "fuzzy_similarity_threshold", 0.95)
    shingle_size: int = getattr(config, "fuzzy_shingle_size", 3)
    num_hashes: int = getattr(config, "fuzzy_num_hashes", 128)

    fingerprint = compute_fuzzy_fingerprint(
        chunk.text, shingle_size=shingle_size, num_hashes=num_hashes
    )
    chunk.metadata["fuzzy_fingerprint"] = fingerprint

    match = find_chunk_by_fuzzy_fingerprint(client, fingerprint, threshold, num_hashes)
    if match is None:
        chunk.metadata.setdefault("source_documents", [source_key])
        return False

    # Canonical selection: longer chunk wins (FR-3424)
    canonical_replaced = False
    if len(chunk.text) > match.get("text_length", 0):
        _replace_canonical(client, match["uuid"], chunk, content_hash, fingerprint)
        canonical_replaced = True

    append_source_document(client, match["uuid"], source_key)
    merge_report.append(
        create_merge_event(
            canonical_content_hash=content_hash,
            canonical_chunk_id=match["uuid"],
            merged_source_key=source_key,
            merged_section=chunk.metadata.get("heading_path", ""),
            match_tier="fuzzy",
            similarity_score=match["similarity"],
            canonical_replaced=canonical_replaced,
            action="replaced" if canonical_replaced else "merged",
        )
    )
    return True


def _replace_canonical(
    client: Any,
    chunk_uuid: str,
    chunk: Any,
    content_hash: str,
    fingerprint: str,
) -> None:
    """Replace a canonical chunk's content when the incoming chunk is longer.

    Args:
        client: Weaviate client handle.
        chunk_uuid: UUID of the existing canonical chunk to overwrite.
        chunk: The incoming (longer) ProcessedChunk.
        content_hash: SHA-256 hex of the incoming chunk's text.
        fingerprint: MinHash hex fingerprint of the incoming chunk.
    """
    from src.vector_db import update_chunk_content  # type: ignore[attr-defined]

    update_chunk_content(
        client,
        chunk_uuid,
        text=chunk.text,
        content_hash=content_hash,
        fuzzy_fingerprint=fingerprint,
    )
