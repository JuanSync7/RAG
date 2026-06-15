# @summary
# Stage-1 card-index document router (RAPTOR-lite, slice B1). Embeds (caller
# pre-embeds) a query and vector-searches the small document-card collection to
# return up to top-N routed document_ids as a SOFT hint (RoutingResult).
# Soft + never top-1 + degrade-to-flat: returns used=False on low confidence /
# empty / missing collection / ANY error. It filters nothing itself and NEVER
# raises.
# Exports: route_documents
# Deps: config.settings, src.vector_db.search, src.retrieval.routing.schemas
# @end-summary
"""Stage-1 document routing over the card index.

Per ``DOCUMENT_ROUTING_DESIGN.md`` §3, Stage-1 ROUTE answers *"where do I
look?"*: embed the query, search the compact document-**card** collection by
vector similarity, and return up to ``top_n`` routed ``document_id`` values.

This output is a **soft document hint** — *not* answer context and *not* a hard
filter (design §7). The router therefore:

* routes to **top-N, never top-1** (the caller unions routed-doc chunks with the
  unchanged flat path, so a strong chunk from a non-routed doc still survives);
* **falls back to pure flat retrieval** (``used=False``, empty ``doc_ids``) when
  routing confidence is low (top card score below ``min_score``), when the card
  index is empty, when the collection is missing, or on **any** backend error;
* **never raises** — every failure path degrades to the safe empty result, so
  routing can only *help*, never break or silently exclude.

The query embedding is supplied by the caller; this module performs no
embedding and holds no client. The vector search is issued through the
backend-agnostic ``src.vector_db.search`` facade, mirroring the call shape used
by ``RAGChain._do_search``.
"""
from __future__ import annotations

import logging
from typing import Optional

from src.retrieval.routing.schemas import RoutingResult
from src.vector_db import search

logger = logging.getLogger("rag.retrieval.routing.router")

# Vector-first routing: cards are matched by semantic similarity to the query,
# so the hybrid search runs at pure-vector alpha. ``query_text`` (when given)
# only supplies the keyword component; alpha keeps it vector-dominant.
_ROUTING_ALPHA = 1.0


def route_documents(
    query_embedding: list[float],
    *,
    top_n: Optional[int] = None,
    min_score: Optional[float] = None,
    collection: Optional[str] = None,
    query_text: str = "",
) -> RoutingResult:
    """Route a query to up to ``top_n`` candidate documents via the card index.

    Issues a vector-first search over the document-card collection and returns
    the routed ``document_id`` values (rank-ordered, de-duplicated) as a *soft*
    hint. This is the Stage-1 ROUTE step of the design — it produces a soft
    document filter, never answer context, and never a hard exclusion.

    Args:
        query_embedding: Pre-computed dense vector for the query (the caller
            embeds; this function never embeds).
        top_n: Maximum number of routed documents. ``None`` →
            ``settings.RAG_DOCUMENT_ROUTING_TOP_N`` (design §7: never top-1).
        min_score: Card-similarity floor. When the top card hit scores below
            this, routing yields nothing (→ pure flat retrieval). ``None`` →
            ``settings.RAG_DOCUMENT_ROUTING_MIN_SCORE``.
        collection: Card collection name. ``None`` →
            ``settings.RAG_DOCUMENT_CARD_COLLECTION``.
        query_text: Optional keyword/BM25 component for the hybrid search. The
            search stays vector-dominant (``alpha=1.0``) regardless; this is
            forwarded only so the facade has a text query when it needs one.

    Returns:
        A :class:`RoutingResult`. ``used=True`` with rank-ordered ``doc_ids``
        and a ``{document_id: score}`` map when at least one card cleared
        ``min_score``; otherwise the safe empty result
        ``RoutingResult([], {}, used=False)`` signalling "fall back to flat".

    Notes:
        This function **never raises**. The search is wrapped in a broad
        try/except; on any failure (missing collection, backend down, etc.) it
        logs at debug level and returns the unused result.
    """
    from config import settings

    if top_n is None:
        top_n = settings.RAG_DOCUMENT_ROUTING_TOP_N
    if min_score is None:
        min_score = settings.RAG_DOCUMENT_ROUTING_MIN_SCORE
    if collection is None:
        collection = settings.RAG_DOCUMENT_CARD_COLLECTION

    _unused = RoutingResult(doc_ids=[], scores={}, used=False)

    try:
        # Vector-first search over the card collection. Call shape mirrors
        # RAGChain._do_search (query, query_embedding, alpha, limit, collection)
        # but with no client handle — the facade resolves the backend singleton.
        hits = search(
            client=None,
            query=query_text,
            query_embedding=query_embedding,
            alpha=_ROUTING_ALPHA,
            limit=top_n,
            filters=None,
            collection=collection,
        )
    except Exception as exc:  # noqa: BLE001 — degrade to flat on ANY failure.
        logger.debug(
            "route_documents: card search over %r failed (%s) — "
            "falling back to pure flat retrieval.",
            collection, exc,
        )
        return _unused

    if not hits:
        logger.debug(
            "route_documents: no card hits in %r — pure flat retrieval.",
            collection,
        )
        return _unused

    # Low-confidence floor: the *top* card hit must clear ``min_score`` for
    # routing to engage at all (design §7 — soft, fall back when weak).
    top_score = hits[0].score
    if top_score < min_score:
        logger.debug(
            "route_documents: top card score %.4f < min_score %.4f in %r — "
            "low confidence, falling back to pure flat retrieval.",
            top_score, min_score, collection,
        )
        return _unused

    # Collect routed document_ids in rank order, de-duplicating while keeping
    # the first (highest-rank) occurrence and its score.
    doc_ids: list[str] = []
    scores: dict[str, float] = {}
    for hit in hits:
        doc_id = (hit.metadata or {}).get("document_id")
        if not doc_id:
            # A card without a document_id cannot route anything — skip it
            # rather than fail (robustness, not an error).
            continue
        if doc_id in scores:
            continue
        doc_ids.append(doc_id)
        scores[doc_id] = hit.score

    if not doc_ids:
        # Hits existed but none carried a usable document_id.
        return _unused

    # Top-N cap (never top-1 hard — that is enforced by TOP_N>=2 config
    # validation and by the soft union the caller performs).
    routed = doc_ids[:top_n]
    routed_scores = {doc_id: scores[doc_id] for doc_id in routed}

    logger.info(
        "route_documents: routed to %d/%d card hits from %r (top score %.4f).",
        len(routed), len(hits), collection, top_score,
    )
    return RoutingResult(doc_ids=routed, scores=routed_scores, used=True)
