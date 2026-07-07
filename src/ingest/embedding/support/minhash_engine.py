# @summary
# MinHash fingerprint engine for Tier 2 fuzzy cross-document deduplication.
# Exports: MinHashEngine, compute_fuzzy_fingerprint, estimate_similarity,
#          find_chunk_by_fuzzy_fingerprint
# Deps: datasketch (runtime optional — ImportError surfaced at call time), numpy,
#       src.ingest.embedding.common.dedup_utils
# @end-summary
"""MinHash fingerprint engine for Tier 2 fuzzy deduplication.

Provides MinHash signature computation, Jaccard similarity estimation, and
Weaviate-backed fuzzy lookup. The ``datasketch`` library must be installed for
Tier 2 to be active — if it is not installed, all public functions raise
``ImportError`` at call time rather than at import time, so the module is
always importable and Tier 1 dedup is unaffected.

Parameters (from IngestionConfig):
    fuzzy_shingle_size  — word-level n-gram size (default 3)
    fuzzy_num_hashes    — MinHash permutation count (default 128)
    fuzzy_similarity_threshold — Jaccard threshold for a match (default 0.95)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from src.ingest.embedding.common.dedup_utils import normalise_chunk_text
from config.settings import (
    RAG_INGEST_FUZZY_SHINGLE_SIZE,
    RAG_INGEST_FUZZY_NUM_HASHES,
    VECTOR_COLLECTION_DEFAULT,
)

logger = logging.getLogger("rag.ingest.embedding.minhash")


def _require_datasketch():
    """Import and return the MinHash class, raising ImportError if unavailable."""
    try:
        from datasketch import MinHash  # type: ignore[import]
        return MinHash
    except ImportError as exc:
        raise ImportError(
            "datasketch is required for Tier 2 fuzzy deduplication. "
            "Install it with: pip install datasketch"
        ) from exc


def _word_shingles(text: str, shingle_size: int = 3) -> list[str]:
    """Generate word-level shingles (contiguous n-grams of words).

    >>> _word_shingles("the quick brown fox", 3)
    ['the quick brown', 'quick brown fox']
    """
    words = text.split()
    if len(words) < shingle_size:
        return [" ".join(words)] if words else []
    return [
        " ".join(words[i : i + shingle_size])
        for i in range(len(words) - shingle_size + 1)
    ]


def compute_fuzzy_fingerprint(
    text: str,
    shingle_size: int = RAG_INGEST_FUZZY_SHINGLE_SIZE,
    num_hashes: int = RAG_INGEST_FUZZY_NUM_HASHES,
) -> str:
    """Compute a MinHash fingerprint for the given text.

    Returns a hex-encoded string of the MinHash signature (FR-3421).

    Args:
        text: Raw chunk text (will be normalised internally).
        shingle_size: Word-level n-gram width. Must be >= 1.
        num_hashes: Number of MinHash permutations. Must be >= 16.

    Returns:
        Hex string of the serialised MinHash signature.

    Raises:
        ImportError: If datasketch is not installed.
    """
    MinHash = _require_datasketch()

    normalised = normalise_chunk_text(text)
    shingles = _word_shingles(normalised, shingle_size)

    mh = MinHash(num_perm=num_hashes)
    for shingle in shingles:
        mh.update(shingle.encode("utf-8"))

    # Serialise hash values to hex string
    return mh.hashvalues.tobytes().hex()


def _deserialise_minhash(hex_str: str, num_hashes: int = RAG_INGEST_FUZZY_NUM_HASHES) -> Any:
    """Reconstruct a MinHash object from a hex-encoded signature.

    Args:
        hex_str: Hex-encoded MinHash signature.
        num_hashes: Must match the num_hashes used during fingerprint creation.

    Returns:
        A datasketch MinHash instance with restored hash values.

    Raises:
        ImportError: If datasketch is not installed.
    """
    import numpy as np  # numpy ships with the project venv

    MinHash = _require_datasketch()

    hash_bytes = bytes.fromhex(hex_str)
    hash_values = np.frombuffer(hash_bytes, dtype=np.uint64)
    mh = MinHash(num_perm=num_hashes)
    mh.hashvalues = hash_values.copy()
    return mh


def estimate_similarity(sig_a: str, sig_b: str, num_hashes: int = RAG_INGEST_FUZZY_NUM_HASHES) -> float:
    """Estimate Jaccard similarity between two MinHash signatures.

    Args:
        sig_a: Hex-encoded MinHash signature.
        sig_b: Hex-encoded MinHash signature.
        num_hashes: Must match the num_hashes used during fingerprint creation.

    Returns:
        Float in [0.0, 1.0].

    Raises:
        ImportError: If datasketch is not installed.
    """
    mh_a = _deserialise_minhash(sig_a, num_hashes)
    mh_b = _deserialise_minhash(sig_b, num_hashes)
    return mh_a.jaccard(mh_b)


def find_chunk_by_fuzzy_fingerprint(
    client: Any,
    fingerprint: str,
    threshold: float,
    num_hashes: int = RAG_INGEST_FUZZY_NUM_HASHES,
    collection_name: str = VECTOR_COLLECTION_DEFAULT,
) -> Optional[dict[str, Any]]:
    """Find the best fuzzy match above threshold in Weaviate.

    For v1, this scans stored fingerprints via Weaviate query rather than
    maintaining a dedicated LSH forest index. Acceptable for corpora under
    100K chunks (see design OQ-1 resolution).

    Returns dict with 'uuid', 'similarity', 'text_length' on match,
    or None if no match found or on error.

    Args:
        client: Weaviate client handle.
        fingerprint: Hex-encoded MinHash signature of the candidate chunk.
        threshold: Minimum Jaccard similarity to consider a match.
        num_hashes: Must match the num_hashes used during fingerprint creation.
        collection_name: Weaviate collection name. Defaults to
            ``VECTOR_COLLECTION_DEFAULT``.

    Returns:
        Best-matching dict with uuid, similarity, text_length; or None.

    Raises:
        ImportError: If datasketch is not installed (propagated from helpers).
    """
    try:
        from weaviate.classes.query import Filter  # lazy import — Weaviate optional

        collection = client.collections.get(collection_name)
        results = collection.query.fetch_objects(
            filters=Filter.by_property("fuzzy_fingerprint").is_not_none(),
            limit=10_000,
            return_properties=["fuzzy_fingerprint", "text", "source_documents"],
        )

        best_match: Optional[dict[str, Any]] = None
        best_similarity = 0.0

        for obj in results.objects:
            stored_fp = obj.properties.get("fuzzy_fingerprint")
            if not stored_fp:
                continue
            try:
                sim = estimate_similarity(fingerprint, stored_fp, num_hashes)
            except Exception:
                logger.debug(
                    "Skipping fingerprint comparison for chunk %s — bad fingerprint data",
                    obj.uuid,
                    exc_info=True,
                )
                continue

            if sim >= threshold and sim > best_similarity:
                best_similarity = sim
                best_match = {
                    "uuid": str(obj.uuid),
                    "similarity": sim,
                    "text_length": len(obj.properties.get("text", "")),
                }

        return best_match

    except Exception:
        logger.warning(
            "Fuzzy fingerprint lookup failed; treating chunk as novel",
            exc_info=True,
        )
        return None


class MinHashEngine:
    """Configurable MinHash engine for Tier 2 fuzzy deduplication.

    Wraps the module-level functions with per-instance configuration so
    nodes can hold one engine object rather than threading config kwargs
    through every call.

    Raises:
        ImportError: On construction if datasketch is not installed.
    """

    def __init__(self, shingle_size: int = RAG_INGEST_FUZZY_SHINGLE_SIZE, num_hashes: int = RAG_INGEST_FUZZY_NUM_HASHES) -> None:
        """Initialise and verify datasketch availability.

        Args:
            shingle_size: Word-level n-gram width. Must be >= 1.
            num_hashes: Number of MinHash permutations. Must be >= 16.

        Raises:
            ValueError: If shingle_size < 1 or num_hashes < 16.
            ImportError: If datasketch is not installed.
        """
        if shingle_size < 1:
            raise ValueError(f"shingle_size must be >= 1, got {shingle_size}")
        if num_hashes < 16:
            raise ValueError(f"num_hashes must be >= 16, got {num_hashes}")
        # Eagerly verify datasketch is present so the error surfaces at engine creation.
        _require_datasketch()
        self.shingle_size = shingle_size
        self.num_hashes = num_hashes

    def fingerprint(self, text: str) -> str:
        """Compute a MinHash fingerprint for the given text.

        Args:
            text: Raw chunk text (normalised internally).

        Returns:
            Hex-encoded MinHash signature string.
        """
        return compute_fuzzy_fingerprint(
            text, shingle_size=self.shingle_size, num_hashes=self.num_hashes
        )

    def jaccard(self, sig_a: str, sig_b: str) -> float:
        """Estimate Jaccard similarity between two hex-encoded signatures.

        Args:
            sig_a: Hex-encoded MinHash signature.
            sig_b: Hex-encoded MinHash signature.

        Returns:
            Estimated Jaccard similarity in [0.0, 1.0].
        """
        return estimate_similarity(sig_a, sig_b, self.num_hashes)
