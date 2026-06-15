# @summary
# Typed contracts shared across the RAPTOR-lite document-routing slices.
# Pure dataclasses, no I/O — imported by glossary, decomposition, router, and
# the ingest card builder so every slice agrees on the same shapes.
# Exports: DocCard, RoutingResult, DecompositionResult
# Deps: dataclasses (stdlib)
# @end-summary
"""Typed contracts for document routing.

These dataclasses are the stable cross-slice vocabulary for the
RAPTOR-lite document-routing feature:

* :class:`DocCard` — a routing-only document summary card (title + section
  headings + optional LLM summary). **Never sent to the LLM**; used purely to
  build the card index that Stage-1 routing searches over.
* :class:`RoutingResult` — the output of Stage-1 routing: the routed document
  ids, optional per-doc scores, and a ``used`` flag. ``used=False`` signals
  that routing contributed nothing (low confidence / empty index) and the
  caller must fall back to pure flat retrieval.
* :class:`DecompositionResult` — the output of query decomposition: the
  sub-queries, the tier/``method`` that produced them, and a ``decomposed``
  flag. The identity result (``[query]``, ``method="identity"``,
  ``decomposed=False``) is the safe baseline.

All fields use ``field(default_factory=...)`` for mutable defaults so distinct
instances never share state.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DocCard:
    """A routing-only summary card for a single document.

    Attributes:
        document_id: Stable identifier of the source document.
        title: Human-readable document title.
        section_headings: Ordered, de-duplicated list of section headings.
        card_text: The embeddable text for the card index (title + headings
            [+ summary]). This is what Stage-1 routing embeds and searches.
        summary: Optional one-paragraph LLM summary (empty when the baseline
            title+headings card is used).
        num_chunks: Number of leaf chunks the card was built from (diagnostic).
    """

    document_id: str
    title: str
    section_headings: list[str]
    card_text: str
    summary: str = ""
    num_chunks: int = 0


@dataclass
class RoutingResult:
    """The result of Stage-1 document routing.

    Attributes:
        doc_ids: Routed document ids (top-N, never top-1-as-hard-filter).
        scores: Optional per-document card-similarity scores.
        used: ``True`` when routing produced a usable routed set; ``False``
            means routing contributed nothing and the caller should use pure
            flat retrieval.
    """

    doc_ids: list[str]
    scores: dict[str, float] = field(default_factory=dict)
    used: bool = False


@dataclass
class DecompositionResult:
    """The result of query decomposition.

    Attributes:
        subqueries: One or more standalone sub-queries. For the identity
            baseline this is ``[original_query]``.
        method: The tier that produced the result — one of ``"identity"``,
            ``"regex"`` (Tier-1 fallback), or ``"llm"`` (Tier-2 primary).
        decomposed: ``True`` only when the query was actually split into
            multiple sub-queries; ``False`` for the identity baseline.
    """

    subqueries: list[str]
    method: str = "identity"
    decomposed: bool = False
