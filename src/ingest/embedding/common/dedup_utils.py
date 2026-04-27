# @summary
# Content hash infrastructure for cross-document deduplication (Tier 1).
# Exports: normalise_chunk_text, compute_content_hash,
#          find_chunk_by_content_hash, append_source_document,
#          remove_source_document_refs, revert_merge, build_fuzzy_fingerprint
# Deps: hashlib, re, logging
# revert_merge added (Phase 3.3 / T7): operational helper to undo a merge event
#   by detaching a source_key from a canonical chunk's source_documents array.
# @end-summary
"""Content hash infrastructure for cross-document deduplication (Tier 1).

Provides SHA-256 content hashing with deterministic text normalisation and
Weaviate helper functions for exact-match dedup lookups and source-document
provenance maintenance.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import TYPE_CHECKING, Any, Optional

from config.settings import VECTOR_COLLECTION_DEFAULT

if TYPE_CHECKING:
    from src.ingest.embedding.common.types import MergeEvent

logger = logging.getLogger("rag.ingest.embedding.dedup_utils")

_WHITESPACE_RE = re.compile(r"\s+")

# Default collection name for chunk lookups. Sourced from config so that
# environment-level overrides (RAG_VECTOR_COLLECTION_DEFAULT) are respected.
_DEFAULT_CHUNK_COLLECTION = VECTOR_COLLECTION_DEFAULT


def normalise_chunk_text(text: str) -> str:
    """Normalise chunk text for hash computation.

    Strips leading/trailing whitespace, collapses interior whitespace
    sequences to a single ASCII space. Case is preserved (FR-3410).

    >>> normalise_chunk_text("  Hello   World\\n")
    'Hello World'
    >>> normalise_chunk_text("Hello World")
    'Hello World'
    """
    return _WHITESPACE_RE.sub(" ", text.strip())


def compute_content_hash(text: str) -> str:
    """Compute SHA-256 content hash of normalised text.

    Returns a lowercase hex string (64 characters).

    Contract: normalise_chunk_text("  Hello   World\\n") and
    normalise_chunk_text("Hello World") produce identical hashes.
    "Hello World" and "hello world" produce different hashes (FR-3410 AC-2).
    """
    normalised = normalise_chunk_text(text)
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def find_chunk_by_content_hash(
    client: Any,
    content_hash: str,
    collection_name: str = _DEFAULT_CHUNK_COLLECTION,
) -> Optional[dict[str, Any]]:
    """Query Weaviate for a chunk with matching content_hash.

    Returns dict with 'uuid', 'source_documents', 'text_length' on match,
    or None if no match or transient error (FR-3411 AC-3).

    Args:
        client: Weaviate client handle.
        content_hash: SHA-256 hex string to match.
        collection_name: Weaviate collection name. Defaults to
            ``_DEFAULT_CHUNK_COLLECTION``.

    Returns:
        Dict with uuid, source_documents, text_length; or None.
    """
    try:
        from weaviate.classes.query import Filter  # lazy import — Weaviate optional

        collection = client.collections.get(collection_name)
        result = collection.query.fetch_objects(
            filters=Filter.by_property("content_hash").equal(content_hash),
            limit=1,
            return_properties=["content_hash", "source_documents", "text"],
        )
        if not result.objects:
            return None
        obj = result.objects[0]
        return {
            "uuid": str(obj.uuid),
            "source_documents": obj.properties.get("source_documents", []),
            "text_length": len(obj.properties.get("text", "")),
        }
    except Exception:
        logger.warning(
            "Weaviate lookup failed for content_hash=%s; treating as novel",
            content_hash[:16],
            exc_info=True,
        )
        return None


def append_source_document(
    client: Any,
    chunk_uuid: str,
    source_key: str,
    collection_name: str = _DEFAULT_CHUNK_COLLECTION,
) -> bool:
    """Append source_key to a canonical chunk's source_documents array.

    Deduplicates: does not append if already present (FR-3431 AC-2).
    Returns True on success, False on failure (FR-3431 AC-4).

    Args:
        client: Weaviate client handle.
        chunk_uuid: UUID of the canonical chunk to update.
        source_key: Source key to append.
        collection_name: Weaviate collection name. Defaults to
            ``_DEFAULT_CHUNK_COLLECTION``.

    Returns:
        True on success, False on error.
    """
    try:
        collection = client.collections.get(collection_name)
        obj = collection.query.fetch_object_by_id(
            chunk_uuid, return_properties=["source_documents"]
        )
        existing = obj.properties.get("source_documents", [])
        if source_key in existing:
            return True  # already present, no-op

        existing.append(source_key)
        collection.data.update(
            uuid=chunk_uuid,
            properties={"source_documents": existing},
        )
        return True
    except Exception:
        logger.error(
            "Failed to append source_document %s to chunk %s",
            source_key,
            chunk_uuid,
            exc_info=True,
        )
        return False


def remove_source_document_refs(
    client: Any,
    source_key: str,
    collection_name: str = _DEFAULT_CHUNK_COLLECTION,
) -> None:
    """Remove source_key from all chunks' source_documents arrays.

    Chunks whose source_documents becomes empty are deleted (FR-3433 AC-3).
    Called during re-ingestion before dedup processing.

    Args:
        client: Weaviate client handle.
        source_key: Source key to remove from all chunk provenance arrays.
        collection_name: Weaviate collection name. Defaults to
            ``_DEFAULT_CHUNK_COLLECTION``.
    """
    try:
        from weaviate.classes.query import Filter  # lazy import — Weaviate optional

        collection = client.collections.get(collection_name)
        results = collection.query.fetch_objects(
            filters=Filter.by_property("source_documents").contains_any([source_key]),
            limit=10_000,
            return_properties=["source_documents"],
        )
        for obj in results.objects:
            sources = obj.properties.get("source_documents", [])
            updated = [s for s in sources if s != source_key]
            if not updated:
                collection.data.delete_by_id(obj.uuid)
            else:
                collection.data.update(
                    uuid=obj.uuid,
                    properties={"source_documents": updated},
                )
    except Exception:
        logger.error(
            "Failed to clean source_document refs for %s",
            source_key,
            exc_info=True,
        )


def revert_merge(
    client: Any,
    event: MergeEvent,
    *,
    collection: Optional[str] = None,
) -> bool:
    """Revert a specific dedup merge by detaching a source_key from the canonical chunk.

    Given a MergeEvent, removes ``event["merged_source_key"]`` from the canonical
    chunk's ``source_documents`` array. If the array becomes empty after removal
    the canonical chunk is deleted from Weaviate (no remaining provenance).

    Idempotent: if ``merged_source_key`` is not present in ``source_documents``
    this is a no-op and returns ``False`` (FR-3451 AC-5).

    This function is intended for operational / CLI use — it is NOT called
    automatically during ingestion.

    Args:
        client: Weaviate client handle.
        event: A ``MergeEvent`` dict produced by ``cross_document_dedup_node``.
            Must contain ``canonical_content_hash`` and ``merged_source_key``.
        collection: Weaviate collection name. Defaults to
            ``_DEFAULT_CHUNK_COLLECTION``.

    Returns:
        ``True`` if a revert was performed (source_key was detached or canonical
        deleted), ``False`` on no-op or error.
    """
    collection_name = collection or _DEFAULT_CHUNK_COLLECTION
    canonical_content_hash: str = event["canonical_content_hash"]
    source_key: str = event["merged_source_key"]

    # Locate the canonical chunk by content hash.
    chunk = find_chunk_by_content_hash(client, canonical_content_hash)
    if chunk is None:
        logger.warning(
            "revert_merge: no chunk found for content_hash=%s",
            canonical_content_hash[:16],
        )
        return False

    sources: list = chunk.get("source_documents", [])
    if source_key not in sources:
        logger.info(
            "revert_merge: source_key=%s not in source_documents for hash=%s — no-op",
            source_key,
            canonical_content_hash[:16],
        )
        return False

    updated = [s for s in sources if s != source_key]

    try:
        coll = client.collections.get(collection_name)
        if not updated:
            # No remaining provenance — delete the canonical chunk.
            coll.data.delete_by_id(chunk["uuid"])
            logger.info(
                "revert_merge: canonical chunk %s deleted (no remaining source_documents)",
                chunk["uuid"],
            )
        else:
            coll.data.update(
                uuid=chunk["uuid"],
                properties={"source_documents": updated},
            )
            logger.info(
                "revert_merge: source_key=%s detached from chunk %s; "
                "remaining sources=%d",
                source_key,
                chunk["uuid"],
                len(updated),
            )
        return True
    except Exception:
        logger.error(
            "revert_merge: failed to update chunk %s for source_key=%s",
            chunk.get("uuid"),
            source_key,
            exc_info=True,
        )
        return False


def build_fuzzy_fingerprint(text: str, config: Any = None) -> str:
    """Compute a MinHash fingerprint and return a hex-encoded string.

    Convenience facade over ``minhash_engine.compute_fuzzy_fingerprint``
    for callers that already have the config object. When ``config`` is None
    the engine defaults (shingle_size=3, num_hashes=128) are used.

    Args:
        text: Raw chunk text (normalised internally).
        config: IngestionConfig instance, or None to use defaults.

    Returns:
        Hex string of the serialised MinHash signature.

    Raises:
        ImportError: If datasketch is not installed.
    """
    from src.ingest.embedding.support.minhash_engine import compute_fuzzy_fingerprint

    shingle_size = getattr(config, "fuzzy_shingle_size", 3) if config else 3
    num_hashes = getattr(config, "fuzzy_num_hashes", 128) if config else 128
    return compute_fuzzy_fingerprint(text, shingle_size=shingle_size, num_hashes=num_hashes)
