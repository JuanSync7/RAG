# @summary
# Runs each golden's query through src.vector_db.search against a named
# collection, collecting retrieved source paths AND chunk texts in rank
# order for downstream recall@k scoring (P4) and faithfulness judging (P5).
# Exports: QueryRetrievalResult, RetrievalResults, retrieve_for_goldens
# Deps: src.vector_db (search/create_persistent_client/close_client),
#       src.core.embeddings (get_embedding_provider), src.eval.pack.schema.Golden
# @end-summary
"""P4 — retrieval pass over golden queries against an ingested collection.

Module-top imports of ``search``, ``create_persistent_client``,
``close_client``, and ``get_embedding_provider`` are load-bearing: tests
monkeypatch the LOCAL names on this module.

P5 additively extends ``QueryRetrievalResult`` with ``retrieved_chunks`` —
the text of each top-k hit — so the faithfulness judge can score against
the actual chunk contents.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from config.settings import HYBRID_SEARCH_ALPHA
from src.core.embeddings import get_embedding_provider
from src.eval.pack.schema import Golden
from src.vector_db import close_client, create_persistent_client, search


@dataclass(frozen=True)
class QueryRetrievalResult:
    """Retrieval outcome for a single golden query.

    ``retrieved_sources`` is rank-ordered (best-first) and contains the
    ``metadata['source']`` of each top-k chunk. ``retrieved_chunks`` is
    the parallel rank-ordered tuple of chunk *texts* (added in P5 for
    faithfulness judging; defaults to ``()`` so P4 constructions remain
    backward-compatible).
    """

    qid: str
    qtype: str
    query: str
    retrieved_sources: tuple[str, ...]
    k: int
    retrieved_chunks: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievalResults:
    """Aggregated retrieval outcomes across all goldens, keyed by qid."""

    collection_name: str
    k: int
    per_query: Mapping[str, QueryRetrievalResult]


def retrieve_for_goldens(
    collection_name: str,
    goldens: Mapping[str, list[Golden]],
    k: int = 5,
) -> RetrievalResults:
    """Embed each golden query and run hybrid search against the collection.

    Args:
        collection_name: Target Weaviate collection (must already be ingested).
        goldens: Mapping from qtype to list of ``Golden`` objects.
        k: Top-k cutoff per query (also passed as ``limit`` to ``search``).

    Returns:
        ``RetrievalResults`` with one ``QueryRetrievalResult`` per golden,
        keyed by ``qid``.
    """
    client = create_persistent_client()
    try:
        provider = get_embedding_provider()
        per_query: dict[str, QueryRetrievalResult] = {}
        for qtype, golden_list in goldens.items():
            for golden in golden_list:
                query = golden.query
                if not query:
                    # Defensive: skip empty queries (shouldn't happen for valid packs).
                    per_query[golden.qid] = QueryRetrievalResult(
                        qid=golden.qid,
                        qtype=qtype,
                        query=query,
                        retrieved_sources=(),
                        k=k,
                        retrieved_chunks=(),
                    )
                    continue
                query_embedding = provider.embed_documents([query])[0]
                hits = search(
                    client,
                    query,
                    query_embedding,
                    HYBRID_SEARCH_ALPHA,
                    k,
                    collection=collection_name,
                )
                sources = tuple(
                    hit.metadata.get("source", "") for hit in hits
                )
                chunks = tuple(hit.text for hit in hits)
                per_query[golden.qid] = QueryRetrievalResult(
                    qid=golden.qid,
                    qtype=qtype,
                    query=query,
                    retrieved_sources=sources,
                    k=k,
                    retrieved_chunks=chunks,
                )
        return RetrievalResults(
            collection_name=collection_name,
            k=k,
            per_query=per_query,
        )
    finally:
        try:
            close_client(client)
        except Exception:
            pass
