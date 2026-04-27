# @summary
# Backend-agnostic contracts for the vector store abstraction layer.
# Exports: DocumentRecord, SearchResult, SearchFilter
# Deps: dataclasses, typing
# @end-summary
"""Shared data contracts for the vector_db subsystem.

These types are backend-independent — no Weaviate or other store-specific
imports belong here. All VectorBackend implementations accept and return
these types at their public boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class DocumentRecord:
    """A document to be inserted into the vector store.

    The input-side counterpart to ``SearchResult``. Pipeline code builds
    ``DocumentRecord`` objects from chunks and passes them to
    ``vector_db.add_documents()``.

    Attributes:
        text: The document chunk text.
        embedding: Pre-computed dense vector for this chunk.
        metadata: Document metadata dict (source, heading, chunk_index, etc.).
    """

    text: str
    embedding: list[float]
    metadata: dict[str, Any]


@dataclass
class SearchResult:
    """A single document returned by the vector store search layer.

    Attributes:
        text: The document chunk text.
        score: Relevance score assigned by the backend (higher is better).
        metadata: Document metadata dict (source, heading, chunk_index, etc.).
        object_id: Optional backend-native object identifier.
        collection: Name of the collection this result came from.
    """

    text: str
    score: float
    metadata: dict[str, Any]
    object_id: Optional[str] = None
    collection: Optional[str] = None


@dataclass
class SearchFilter:
    """Generic metadata filter clause for vector store queries.

    Multiple ``SearchFilter`` instances are AND-combined. Each
    ``VectorBackend`` implementation translates these to its native filter
    representation.

    Attributes:
        property: Document metadata property to filter on (e.g. "source").
        operator: Comparison operator. Supported values:
            ``"eq"`` (equal), ``"ne"`` (not equal),
            ``"gt"`` / ``"lt"`` / ``"gte"`` / ``"lte"`` (numeric comparisons),
            ``"like"`` (glob-style pattern match),
            ``"not_in"`` (value is a list; matches when the property is not in it).
        value: Value to compare against. For ``"not_in"`` this must be a list.
    """

    property: str
    operator: str
    value: Any
