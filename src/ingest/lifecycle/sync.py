# @summary
# SyncEngine: four-store reconciliation that enumerates Weaviate, MinIO, Neo4j,
# and the manifest; builds a StoreInventory; and diffs it into an OrphanReport.
# All operations are read-only (no mutations).
# Exports: SyncEngine
# Deps: src.ingest.common.schemas, src.ingest.lifecycle.schemas,
#       src.ingest.common.minio_clean_store, src.vector_db, src.knowledge_graph
# @end-summary
"""Four-store inventory and orphan-detection engine (FR-3000, FR-3002).

:class:`SyncEngine` is stateless and composable:

1. Call :meth:`SyncEngine.inventory` to enumerate all four stores.
2. Call :meth:`SyncEngine.diff` with the returned :class:`StoreInventory` to
   derive an :class:`OrphanReport`.

No store is mutated during either step.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from src.ingest.common.schemas import ManifestEntry
from src.ingest.lifecycle.schemas import OrphanReport, StoreInventory

logger = logging.getLogger(__name__)


class SyncEngine:
    """Read-only four-store reconciliation engine.

    Args:
        manifest: Current manifest dict keyed by source_key.
        weaviate_client: Weaviate client handle.  Passed to
            ``src.vector_db.aggregate_by_source`` to enumerate keys.
        minio_client: MinIO client handle.  ``None`` if MinIO is not
            configured — the MinIO inventory step is skipped.
        minio_bucket: Target MinIO bucket name.
        neo4j_client: KG backend handle (``GraphStorageBackend`` instance or
            any object with a ``list_source_keys()`` method).  ``None`` if the
            KG is disabled — the Neo4j step is skipped.
        collection: Weaviate collection name.  ``None`` uses the default.
    """

    def __init__(
        self,
        manifest: dict[str, ManifestEntry],
        weaviate_client: Any,
        minio_client: Optional[Any] = None,
        minio_bucket: str = "",
        neo4j_client: Optional[Any] = None,
        collection: Optional[str] = None,
    ) -> None:
        self._manifest = manifest
        self._weaviate_client = weaviate_client
        self._minio_client = minio_client
        self._minio_bucket = minio_bucket
        self._neo4j_client = neo4j_client
        self._collection = collection

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def inventory(self) -> StoreInventory:
        """Enumerate all four stores and return a :class:`StoreInventory`.

        All enumeration errors are caught and recorded in the inventory's
        ``*_error`` fields — they never propagate to the caller.  The engine
        never mutates any store.

        Returns:
            A :class:`StoreInventory` with the key sets from each store and
            error messages for any store that could not be enumerated.
        """
        inv = StoreInventory()

        # -- Manifest -------------------------------------------------
        inv.manifest_keys = {
            k
            for k, v in self._manifest.items()
            if not v.get("deleted", False)
        }

        # -- Weaviate -------------------------------------------------
        try:
            inv.weaviate_keys = self._enumerate_weaviate()
        except Exception as exc:
            inv.weaviate_error = str(exc)
            logger.warning("sync_inventory_weaviate_failed error=%s", exc)

        # -- MinIO ----------------------------------------------------
        if self._minio_client is not None:
            try:
                inv.minio_keys = self._enumerate_minio()
            except Exception as exc:
                inv.minio_error = str(exc)
                logger.warning("sync_inventory_minio_failed error=%s", exc)

        # -- Neo4j / KG -----------------------------------------------
        if self._neo4j_client is not None:
            try:
                inv.neo4j_keys = self._enumerate_neo4j()
            except Exception as exc:
                inv.neo4j_error = str(exc)
                logger.warning("sync_inventory_neo4j_failed error=%s", exc)

        logger.info(
            "sync_inventory manifest=%d weaviate=%d minio=%d neo4j=%d",
            len(inv.manifest_keys),
            len(inv.weaviate_keys),
            len(inv.minio_keys),
            len(inv.neo4j_keys),
        )
        return inv

    def diff(self, inventory: StoreInventory) -> OrphanReport:
        """Diff a :class:`StoreInventory` against the manifest to find orphans.

        An orphan is a key present in a store but absent from the manifest.
        ``manifest_only`` identifies keys in the manifest but in none of the
        stores — these are candidates for hard-delete cleanup.

        Args:
            inventory: Previously built inventory (from :meth:`inventory`).

        Returns:
            :class:`OrphanReport` with per-store orphan lists.
        """
        manifest_keys = inventory.manifest_keys

        report = OrphanReport(
            weaviate_orphans=sorted(inventory.weaviate_keys - manifest_keys),
            minio_orphans=sorted(inventory.minio_keys - manifest_keys),
            neo4j_orphans=sorted(inventory.neo4j_keys - manifest_keys),
        )

        # Keys in manifest but present in none of the active stores
        all_store_keys = (
            inventory.weaviate_keys | inventory.minio_keys | inventory.neo4j_keys
        )
        report.manifest_only = sorted(manifest_keys - all_store_keys)

        logger.info(
            "sync_diff weaviate_orphans=%d minio_orphans=%d "
            "neo4j_orphans=%d manifest_only=%d",
            len(report.weaviate_orphans),
            len(report.minio_orphans),
            len(report.neo4j_orphans),
            len(report.manifest_only),
        )
        return report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _enumerate_weaviate(self) -> set[str]:
        """Return all distinct source_keys present in Weaviate.

        Uses :func:`src.vector_db.aggregate_by_source` which groups chunk
        counts by source_key.  If the function is not available (e.g. the
        vector_db facade does not yet expose it), falls back gracefully to an
        empty set and logs a warning.
        """
        try:
            from src.vector_db import aggregate_by_source

            rows = aggregate_by_source(
                self._weaviate_client,
                collection=self._collection,
            )
            # Each row is expected to have a "source_key" field.
            keys: set[str] = set()
            for row in rows:
                sk = row.get("source_key") or row.get("source") or ""
                if sk:
                    keys.add(sk)
            return keys
        except ImportError:
            logger.warning(
                "sync_enumerate_weaviate: aggregate_by_source not available; "
                "returning empty set"
            )
            return set()

    def _enumerate_minio(self) -> set[str]:
        """Return all source_keys (safe-key form) in the MinIO clean store."""
        from src.ingest.common.minio_clean_store import MinioCleanStore

        store = MinioCleanStore(self._minio_client, self._minio_bucket)
        return set(store.list_keys())

    def _enumerate_neo4j(self) -> set[str]:
        """Return all source_keys present in the KG backend.

        Calls ``neo4j_client.list_source_keys()`` if the method exists.
        Falls back to an empty set so the rest of the inventory is not blocked.
        """
        if hasattr(self._neo4j_client, "list_source_keys"):
            return set(self._neo4j_client.list_source_keys())
        logger.warning(
            "sync_enumerate_neo4j: client has no list_source_keys(); "
            "returning empty set"
        )
        return set()
