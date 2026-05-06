# @summary
# KGPhase2bRunner — drives KG ingest from CleanDocumentStore using a
# KGIngestClient. The runner is the Phase 2b sibling of the embedding
# pipeline (Phase 2a); both read from the same Phase 1 boundary and write
# to disjoint sinks. No coupling to the embedding LangGraph state.
# Exports: KGPhase2bRunner
# Deps: src.ingest.common.clean_store, src.knowledge_graph.common.protocols
# @end-summary
"""Phase 2b runner — read clean docs, ingest into KG via KGIngestClient.

.. deprecated::
    The canonical Phase 2b path is now Temporal dispatch to the KGWeave
    worker fleet (see ``src/ingest/temporal/workflows.py:_run_kg_phase2b``
    and the BackfillKGWorkflow). This in-process runner remains only for
    CLI/batch tooling that does not have Temporal available; it will be
    removed once the retrieval-side KG migration (Step 14) lands and
    ``src/knowledge_graph/`` can be deleted from RagWeave.
"""

from __future__ import annotations

import logging
from typing import Iterable, List

from src.ingest.common.clean_store import CleanDocumentStore
from kgweave.knowledge_graph.common.protocols import KGIngestClient, KGIngestResult

logger = logging.getLogger("rag.ingest.kg.phase2b")

__all__ = ["KGPhase2bRunner"]


class KGPhase2bRunner:
    """Drive KG ingest from a :class:`CleanDocumentStore` using a Protocol client.

    The runner has no LangGraph state and no shared dependencies with the
    embedding pipeline. It can run inside a Temporal activity, a CLI batch
    job, or a backfill script.

    Args:
        clean_store: Phase 1 → Phase 2 boundary store.
        kg_client: Any implementation of
            :class:`~src.knowledge_graph.common.protocols.KGIngestClient`.
    """

    def __init__(
        self,
        clean_store: CleanDocumentStore,
        kg_client: KGIngestClient,
    ) -> None:
        self._clean_store = clean_store
        self._kg_client = kg_client

    def run_one(self, source_key: str) -> KGIngestResult:
        """Read *source_key* and ingest its clean text into the KG.

        Raises:
            FileNotFoundError: When the clean store has no entry for *source_key*.
        """
        clean_text, meta = self._clean_store.read(source_key)
        result = self._kg_client.extract_and_commit(
            source_key=source_key,
            clean_text=clean_text,
            meta=meta,
        )
        logger.debug(
            "phase2b run_one source_key=%s entities=+%d triples=+%d",
            source_key,
            result.entities_added,
            result.triples_added,
        )
        return result

    def run_many(
        self,
        source_keys: Iterable[str],
        *,
        skip_missing: bool = False,
    ) -> List[KGIngestResult]:
        """Ingest each *source_key* in iteration order.

        Args:
            source_keys: Iterable of keys to ingest.
            skip_missing: When True, missing entries log a warning and are
                skipped. When False (default), the first missing key raises.

        Returns:
            List of ``KGIngestResult`` in the order keys were processed.
        """
        results: List[KGIngestResult] = []
        for key in source_keys:
            try:
                results.append(self.run_one(key))
            except FileNotFoundError:
                if not skip_missing:
                    raise
                logger.warning("phase2b skip_missing source_key=%s", key)
        return results

    def run_all(self) -> List[KGIngestResult]:
        """Ingest every key currently in the clean store."""
        return self.run_many(self._clean_store.list_keys())
