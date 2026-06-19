# @summary
# LangGraph node that emits ONE routing-only document card per document (A3 of
# the RAPTOR-lite document-routing feature). When config.build_document_cards is
# True, it groups state["chunks"] by metadata["document_id"], builds a baseline
# card (title + headings, no LLM) via build_document_card, embeds each card_text
# via runtime.embedder.embed_documents, attaches the vector, and STAGES the
# records on state as staged_card_records (commit_node/A4 performs the write).
# Strict no-op when the flag is off (the default): no card work, no embedder
# call, no staged_card_records key.
# Exports: document_card_emission_node
# Deps: src.ingest.common, src.ingest.embedding.common.card_builder,
#   src.ingest.embedding.state
# @end-summary

"""Document-card emission node (DOCUMENT_ROUTING_DESIGN.md §6.4, §11).

A "card" is a tiny per-document record whose ``card_text`` is embedded into a
small card index for Stage-1 document routing. This node is the ingest-time
producer: it turns each document's chunks into one card, embeds the card text,
and stages the records for the atomic commit phase (A4 / ``commit_node``).

The node is **config-gated** with a conservative default: when
``config.build_document_cards`` is False (the default), it is a strict no-op —
it performs no card construction, never touches the embedder, and returns no
``staged_card_records`` key, so the disabled-path behaviour of the graph is
byte-identical to pre-feature behaviour.

Cards are **routing-only** and are never sent to the LLM (per the design's
safety philosophy). The baseline card is built purely from structural metadata
(title + section headings) with no LLM summary; the ``card_llm_summary`` upgrade
is out of A3's scope, so ``summary`` stays empty here.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from src.ingest.common import append_processing_log
from src.ingest.common.observability import node_span
from src.ingest.embedding.common.card_builder import build_document_card
from src.ingest.embedding.state import EmbeddingPipelineState

logger = logging.getLogger("rag.ingest.embedding.document_card")


def _chunk_document_id(chunk: Any) -> str:
    """Return a chunk's ``document_id`` from its metadata, tolerating shapes.

    Mirrors the accessor tolerance in ``card_builder``: accepts a
    ``ProcessedChunk``-like object (``.metadata`` attribute) or a plain dict
    with a ``"metadata"`` key. Returns ``""`` when absent.
    """
    meta = getattr(chunk, "metadata", None)
    if meta is None and isinstance(chunk, dict):
        meta = chunk.get("metadata")
    if isinstance(meta, dict):
        return str(meta.get("document_id", "") or "")
    return ""


def _group_by_document(chunks: list[Any]) -> dict[str, list[Any]]:
    """Group chunks by ``document_id`` preserving first-seen document order.

    Chunks with no resolvable ``document_id`` are collected under the empty
    string key; the caller decides how to treat that degenerate group (the node
    skips cards whose ``card_text`` is empty so they never reach the index).
    """
    groups: dict[str, list[Any]] = {}
    for chunk in chunks:
        doc_id = _chunk_document_id(chunk)
        groups.setdefault(doc_id, []).append(chunk)
    return groups


@node_span("document_card_emission")
def document_card_emission_node(state: EmbeddingPipelineState) -> dict[str, Any]:
    """Build, embed, and stage one routing card per document (when enabled).

    Resolves ``config`` and ``runtime`` from ``state`` the same way
    ``embedding_storage_node`` does (``state["runtime"]`` →
    ``runtime.config`` / ``runtime.embedder``) and stages records onto the
    returned partial-state dict (``staged_card_records``), mirroring the
    ``staged_weaviate_records`` staging style.

    Behaviour:

    - **Disabled** (``config.build_document_cards`` False — the default):
      strict no-op. Returns only a ``processing_log`` entry; no
      ``staged_card_records`` key, and the embedder is never called.
    - **Enabled**: groups ``state["chunks"]`` by ``metadata["document_id"]``
      (preserving first-seen order), builds a baseline card per document
      (``build_document_card`` with ``with_summary=False``, ``summary=""``),
      batch-embeds all ``card_text`` values in one ``embed_documents`` call,
      attaches each vector as ``card["vector"]``, and stages the list as
      ``staged_card_records``. Documents whose card has no usable
      ``card_text`` are skipped (never embedded, never staged) so a chunk
      lacking structure cannot create a spurious empty card.

    Args:
        state: Embedding pipeline state. Requires ``runtime`` and ``chunks``.

    Returns:
        Partial state update. When enabled: ``{"staged_card_records": [...],
        "processing_log": [...]}``. When disabled or there is nothing to stage:
        just an updated ``processing_log`` (no ``staged_card_records`` key).
    """
    t0 = time.monotonic()
    runtime = state["runtime"]
    config = runtime.config

    if not getattr(config, "build_document_cards", False):
        return {
            "processing_log": append_processing_log(state, "document_card_emission:skip"),
        }

    chunks = list(state.get("chunks") or [])
    if not chunks:
        return {
            "staged_card_records": [],
            "processing_log": append_processing_log(state, "document_card_emission:empty"),
        }

    max_headings = int(getattr(config, "card_max_headings", 60))

    # Build one baseline card per document, preserving first-seen order. Cards
    # with no usable card_text (e.g. a structureless/empty chunk group) are
    # dropped so we never embed/stage a spurious empty card.
    cards: list[dict[str, Any]] = []
    for _doc_id, doc_chunks in _group_by_document(chunks).items():
        card = build_document_card(
            doc_chunks,
            max_headings=max_headings,
            with_summary=False,
            summary="",
        )
        if not card.get("card_text"):
            continue
        cards.append(card)

    if not cards:
        return {
            "staged_card_records": [],
            "processing_log": append_processing_log(
                state, "document_card_emission:no_cards"
            ),
        }

    # Batch-embed all card texts in a single call for efficiency (mirrors the
    # batched embed_documents idiom of embedding_storage), then attach each
    # vector to its card by position.
    card_texts = [card["card_text"] for card in cards]
    vectors = runtime.embedder.embed_documents(card_texts)
    for card, vector in zip(cards, vectors):
        card["vector"] = vector

    logger.info(
        "document_card_emission: staged %d cards from %d chunks",
        len(cards),
        len(chunks),
    )
    logger.debug(
        "document_card_emission_node completed in %.3fs", time.monotonic() - t0
    )
    return {
        "staged_card_records": cards,
        "processing_log": append_processing_log(
            state, f"document_card_emission:staged={len(cards)}"
        ),
    }
