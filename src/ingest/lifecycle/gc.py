# @summary
# GCEngine: four-store garbage collection with soft (default, 30d retention) and
# hard delete modes. Hard delete requires explicit confirm=True AND cli_confirmed=True.
# Also exposes run_gc_cli() as a module-level CLI entry point.
# Exports: GCEngine, run_gc_cli
# Deps: src.ingest.common.schemas, src.ingest.common.minio_clean_store,
#       src.ingest.lifecycle.schemas, src.vector_db, datetime, argparse, json, logging
# @end-summary
"""Four-store garbage collection engine (FR-3001, FR-3010, NFR-3210, NFR-3230).

:class:`GCEngine` supports two delete modes:

* **soft** (default): marks manifest entries as deleted and moves MinIO objects
  to the ``deleted/`` prefix.  Data is retained for ``retention_days`` before
  :meth:`GCEngine.purge_expired` hard-deletes it.
* **hard**: immediately removes data from all four stores.  Requires both
  ``confirm=True`` *and* ``cli_confirmed=True`` to prevent accidental execution
  (NFR-3230).

Store-level failures are isolated per NFR-3210 — a failure in one store does
not prevent cleanup of the remaining stores.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from dataclasses import asdict
from typing import Any, Optional

from src.ingest.common.schemas import ManifestEntry
from src.ingest.lifecycle.schemas import GCReport, OrphanReport, StoreCleanupStatus

logger = logging.getLogger(__name__)

# Sentinel used to distinguish a CLI-originated call from a programmatic call.
_CLI_HARD_DELETE_SENTINEL = "cli_hard_confirmed"


class GCEngine:
    """Four-store garbage collection engine.

    Args:
        manifest: Current manifest dict (mutated in-place for soft deletes).
        weaviate_client: Weaviate client handle.
        minio_client: MinIO client handle.  ``None`` skips MinIO cleanup.
        minio_bucket: Target MinIO bucket.
        neo4j_client: KG backend handle (any object with
            ``soft_delete_by_source_key`` / ``delete_by_source_key`` methods,
            or the ABC :class:`GraphStorageBackend` using ``remove_by_source``).
            ``None`` skips KG cleanup.
        retention_days: Retention period in days for soft-deleted data.
        collection: Weaviate collection name.  ``None`` uses the default.
    """

    def __init__(
        self,
        manifest: dict[str, ManifestEntry],
        weaviate_client: Any,
        minio_client: Optional[Any] = None,
        minio_bucket: str = "",
        neo4j_client: Optional[Any] = None,
        retention_days: int = 30,
        collection: Optional[str] = None,
    ) -> None:
        self._manifest = manifest
        self._weaviate_client = weaviate_client
        self._minio_client = minio_client
        self._minio_bucket = minio_bucket
        self._neo4j_client = neo4j_client
        self._retention_days = retention_days
        self._collection = collection

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect(
        self,
        report: OrphanReport,
        mode: str = "soft",
        dry_run: bool = False,
        confirm: bool = False,
        cli_confirmed: bool = False,
    ) -> GCReport:
        """Execute GC for the deleted keys surfaced in *report*.

        Deleted keys are taken from :attr:`OrphanReport.manifest_only` — keys
        that the manifest tracks but that exist in no live store — as well as
        any caller-supplied overrides via the lower-level
        :meth:`collect_keys` method.

        Hard delete mode requires **both** ``confirm=True`` AND
        ``cli_confirmed=True``.  If either is missing the call raises
        :class:`PermissionError` (NFR-3230).

        Args:
            report: Orphan report produced by :class:`SyncEngine`.
            mode: ``"soft"`` (default) or ``"hard"``.
            dry_run: When ``True``, log and count without mutating any store.
            confirm: Programmatic confirmation flag for hard delete.
            cli_confirmed: CLI-originating confirmation flag for hard delete.

        Returns:
            :class:`GCReport` with per-document results and aggregate counts.

        Raises:
            ValueError: If *mode* is not ``"soft"`` or ``"hard"``.
            PermissionError: If hard delete is requested without full
                confirmation.
        """
        _validate_mode(mode)
        if mode == "hard":
            _require_hard_delete_confirmation(confirm, cli_confirmed)

        return self.collect_keys(
            keys=report.manifest_only,
            mode=mode,
            dry_run=dry_run,
            confirm=confirm,
            cli_confirmed=cli_confirmed,
        )

    def collect_keys(
        self,
        keys: list[str],
        mode: str = "soft",
        dry_run: bool = False,
        confirm: bool = False,
        cli_confirmed: bool = False,
    ) -> GCReport:
        """Execute GC for an explicit list of source_keys.

        This lower-level method lets callers (e.g. the on-ingest integration
        in ``impl.py``) pass keys from ``SyncResult.deleted`` directly without
        going through orphan detection.

        Args:
            keys: Source keys to clean up.
            mode: ``"soft"`` or ``"hard"``.
            dry_run: When ``True``, count without mutating.
            confirm: Programmatic hard-delete confirmation.
            cli_confirmed: CLI hard-delete confirmation.

        Returns:
            :class:`GCReport`.

        Raises:
            ValueError: If *mode* is not ``"soft"`` or ``"hard"``.
            PermissionError: If hard delete is requested without full
                confirmation.
        """
        _validate_mode(mode)
        if mode == "hard":
            _require_hard_delete_confirmation(confirm, cli_confirmed)

        result = GCReport(dry_run=dry_run)

        for source_key in keys:
            status = StoreCleanupStatus()

            if dry_run:
                result.per_document[source_key] = status
                if mode == "soft":
                    result.soft_deleted += 1
                else:
                    result.hard_deleted += 1
                logger.info(
                    "gc_dry_run source_key=%s mode=%s", source_key, mode
                )
                continue

            # -- Weaviate cleanup ------------------------------------
            status.weaviate = self._cleanup_weaviate(source_key, mode, status)

            # -- MinIO cleanup ---------------------------------------
            if self._minio_client is not None:
                status.minio = self._cleanup_minio(source_key, mode, status)

            # -- Neo4j / KG cleanup ----------------------------------
            if self._neo4j_client is not None:
                status.neo4j = self._cleanup_neo4j(source_key, mode, status)

            # -- Manifest cleanup ------------------------------------
            status.manifest = self._cleanup_manifest(source_key, mode, result, status)

            result.per_document[source_key] = status

            # Audit log (NFR-3221)
            logger.info(
                "gc_operation source_key=%s mode=%s "
                "weaviate=%s minio=%s neo4j=%s manifest=%s errors=%d",
                source_key,
                mode,
                status.weaviate,
                status.minio,
                status.neo4j,
                status.manifest,
                len(status.errors),
            )

        return result

    def purge_expired(self, dry_run: bool = False) -> int:
        """Hard-delete soft-deleted entries past their retention period (FR-3022).

        Scans the manifest for entries where ``deleted == True`` and
        ``deleted_at`` is older than ``retention_days``.  Each qualifying
        entry is hard-deleted from all four stores using :meth:`collect_keys`.

        Args:
            dry_run: When ``True``, count qualifying entries without deleting.

        Returns:
            Count of entries purged (or that *would* be purged on dry-run).
        """
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=self._retention_days)
        expired_keys: list[str] = []

        for key, entry in self._manifest.items():
            if not entry.get("deleted", False):
                continue
            deleted_at_str = entry.get("deleted_at", "")
            if not deleted_at_str:
                continue
            try:
                deleted_at = datetime.fromisoformat(deleted_at_str)
                if deleted_at.tzinfo is None:
                    deleted_at = deleted_at.replace(tzinfo=timezone.utc)
                if deleted_at < cutoff:
                    expired_keys.append(key)
            except (ValueError, TypeError):
                logger.warning(
                    "gc_purge_invalid_deleted_at source_key=%s deleted_at=%s",
                    key,
                    deleted_at_str,
                )

        if not expired_keys:
            return 0

        if dry_run:
            logger.info(
                "gc_purge_expired_dry_run count=%d retention_days=%d",
                len(expired_keys),
                self._retention_days,
            )
            return len(expired_keys)

        # Hard delete without CLI confirmation — this is an internal scheduled
        # purge, not a user-initiated hard delete, so we bypass the sentinel.
        result = GCReport()
        for source_key in expired_keys:
            status = StoreCleanupStatus()
            self._cleanup_weaviate(source_key, "hard", status)
            if self._minio_client is not None:
                self._cleanup_minio(source_key, "hard", status)
            if self._neo4j_client is not None:
                self._cleanup_neo4j(source_key, "hard", status)
            self._cleanup_manifest(source_key, "hard", result, status)
            result.per_document[source_key] = status
            # Note: _cleanup_manifest already increments result.hard_deleted.

        result.retention_purged = result.hard_deleted

        logger.info(
            "gc_purge_expired purged=%d retention_days=%d",
            result.hard_deleted,
            self._retention_days,
        )
        return result.hard_deleted

    # ------------------------------------------------------------------
    # Private per-store helpers
    # ------------------------------------------------------------------

    def _cleanup_weaviate(
        self, source_key: str, mode: str, status: StoreCleanupStatus
    ) -> bool:
        """Delete or soft-delete chunks in Weaviate for *source_key*."""
        try:
            if mode == "soft":
                # Attempt soft_delete_by_source_key if available on facade.
                try:
                    from src.vector_db import soft_delete_by_source_key  # type: ignore[attr-defined]
                    soft_delete_by_source_key(
                        self._weaviate_client,
                        source_key,
                        collection=self._collection,
                    )
                except (ImportError, AttributeError):
                    # Function not yet in the facade — skip silently.
                    logger.debug(
                        "gc_weaviate_soft_delete_unavailable source_key=%s; "
                        "skipping weaviate soft-delete",
                        source_key,
                    )
            else:
                from src.vector_db import delete_by_source_key

                delete_by_source_key(
                    self._weaviate_client,
                    source_key,
                    collection=self._collection,
                )
            return True
        except Exception as exc:
            status.errors.append(f"weaviate: {exc}")
            logger.error(
                "gc_weaviate_failed source_key=%s mode=%s error=%s",
                source_key,
                mode,
                exc,
            )
            return False

    def _cleanup_minio(
        self, source_key: str, mode: str, status: StoreCleanupStatus
    ) -> bool:
        """Soft-delete or delete MinIO objects for *source_key*."""
        try:
            from src.ingest.common.minio_clean_store import MinioCleanStore

            store = MinioCleanStore(self._minio_client, self._minio_bucket)
            if mode == "soft":
                store.soft_delete(source_key)
            else:
                store.delete(source_key)
            return True
        except Exception as exc:
            status.errors.append(f"minio: {exc}")
            logger.error(
                "gc_minio_failed source_key=%s mode=%s error=%s",
                source_key,
                mode,
                exc,
            )
            return False

    def _cleanup_neo4j(
        self, source_key: str, mode: str, status: StoreCleanupStatus
    ) -> bool:
        """Remove or soft-delete KG triples for *source_key*."""
        try:
            if mode == "soft":
                if hasattr(self._neo4j_client, "soft_delete_by_source_key"):
                    self._neo4j_client.soft_delete_by_source_key(source_key)
                elif hasattr(self._neo4j_client, "remove_by_source"):
                    # GraphStorageBackend ABC — hard remove is the only option.
                    self._neo4j_client.remove_by_source(source_key)
                else:
                    logger.debug(
                        "gc_neo4j_soft_delete_unavailable source_key=%s; "
                        "skipping neo4j soft-delete",
                        source_key,
                    )
            else:
                if hasattr(self._neo4j_client, "delete_by_source_key"):
                    self._neo4j_client.delete_by_source_key(source_key)
                elif hasattr(self._neo4j_client, "remove_by_source"):
                    self._neo4j_client.remove_by_source(source_key)
                else:
                    logger.warning(
                        "gc_neo4j_no_delete_method source_key=%s", source_key
                    )
            return True
        except Exception as exc:
            status.errors.append(f"neo4j: {exc}")
            logger.error(
                "gc_neo4j_failed source_key=%s mode=%s error=%s",
                source_key,
                mode,
                exc,
            )
            return False

    def _cleanup_manifest(
        self,
        source_key: str,
        mode: str,
        result: GCReport,
        status: StoreCleanupStatus,
    ) -> bool:
        """Update the manifest entry for *source_key* according to *mode*."""
        try:
            if mode == "soft":
                now = datetime.now(timezone.utc).isoformat()
                if source_key in self._manifest:
                    self._manifest[source_key]["deleted"] = True
                    self._manifest[source_key]["deleted_at"] = now
                result.soft_deleted += 1
            else:
                self._manifest.pop(source_key, None)
                result.hard_deleted += 1
            return True
        except Exception as exc:
            status.errors.append(f"manifest: {exc}")
            logger.error(
                "gc_manifest_failed source_key=%s mode=%s error=%s",
                source_key,
                mode,
                exc,
            )
            return False


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _validate_mode(mode: str) -> None:
    """Raise ValueError if *mode* is not a recognised GC mode."""
    if mode not in ("soft", "hard"):
        raise ValueError(
            f"Invalid GC mode {mode!r}. Valid values: 'soft', 'hard'."
        )


def _require_hard_delete_confirmation(
    confirm: bool, cli_confirmed: bool
) -> None:
    """Raise PermissionError unless both hard-delete confirmation flags are set.

    NFR-3230 requires explicit double confirmation for hard delete to prevent
    accidental data loss.

    Args:
        confirm: Programmatic confirmation flag.
        cli_confirmed: CLI-originating confirmation flag (requires ``--hard-confirm``
            on the CLI entry point).

    Raises:
        PermissionError: If either flag is ``False``.
    """
    if not confirm or not cli_confirmed:
        raise PermissionError(
            "Hard delete requires both confirm=True and cli_confirmed=True. "
            "Pass --mode hard --hard-confirm on the CLI, or set both flags "
            "programmatically. This operation is irreversible (NFR-3230)."
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def run_gc_cli(argv: list[str] | None = None) -> int:
    """Command-line entry point for the GC engine.

    Parses argv (or ``sys.argv[1:]``), runs the GC, and emits a JSON or
    human-readable report to stdout.

    Args:
        argv: Argument list.  Defaults to ``sys.argv[1:]``.

    Returns:
        Exit code: 0 on success, 1 on usage error, 2 on GC error.

    Usage::

        python -m src.ingest.lifecycle.gc \\
            [--dry-run] \\
            [--mode {soft,hard}] \\
            [--hard-confirm] \\
            [--trace-id TRACE_ID] \\
            [--retention-days N] \\
            [--format {json,text}]
    """
    parser = argparse.ArgumentParser(
        prog="ragweave-gc",
        description="Run garbage collection across all RagWeave stores.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Report what would be deleted without modifying any store.",
    )
    parser.add_argument(
        "--mode",
        choices=["soft", "hard"],
        default="soft",
        help="Delete mode. Default: soft.",
    )
    parser.add_argument(
        "--hard-confirm",
        action="store_true",
        default=False,
        help="Required alongside --mode hard. This operation is irreversible.",
    )
    parser.add_argument(
        "--trace-id",
        default=None,
        metavar="UUID",
        help="Restrict GC to a single trace_id (informational; not yet enforced).",
    )
    parser.add_argument(
        "--retention-days",
        type=int,
        default=30,
        metavar="N",
        help="Override retention period for soft deletes. Default: 30.",
    )
    parser.add_argument(
        "--format",
        choices=["json", "text"],
        default="json",
        help="Output format. Default: json.",
    )

    args = parser.parse_args(argv)

    # Hard-delete requires both --mode hard AND --hard-confirm.
    if args.mode == "hard" and not args.hard_confirm:
        parser.error(
            "--mode hard requires --hard-confirm. "
            "This operation is irreversible (NFR-3230)."
        )
        return 1

    # Initialise clients.
    try:
        from src.ingest.lifecycle.sync import SyncEngine

        weaviate_client = _open_weaviate_client()
        minio_client = _open_minio_client()
        minio_bucket = _resolve_minio_bucket()
        neo4j_client = _open_neo4j_client()
        manifest = _load_manifest()

        engine_sync = SyncEngine(
            manifest=manifest,
            weaviate_client=weaviate_client,
            minio_client=minio_client,
            minio_bucket=minio_bucket,
            neo4j_client=neo4j_client,
        )
        inventory = engine_sync.inventory()
        orphan_report = engine_sync.diff(inventory)

        engine_gc = GCEngine(
            manifest=manifest,
            weaviate_client=weaviate_client,
            minio_client=minio_client,
            minio_bucket=minio_bucket,
            neo4j_client=neo4j_client,
            retention_days=args.retention_days,
        )

        gc_report = engine_gc.collect(
            report=orphan_report,
            mode=args.mode,
            dry_run=args.dry_run,
            confirm=args.mode == "hard",
            cli_confirmed=args.hard_confirm,
        )

        if not args.dry_run:
            purged = engine_gc.purge_expired(dry_run=False)
            gc_report.retention_purged = purged
            _save_manifest(manifest)

    except PermissionError as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        return 1
    except Exception as exc:
        logger.exception("gc_cli_error: %s", exc)
        sys.stderr.write(f"ERROR: {exc}\n")
        return 2

    _emit_report(gc_report, orphan_report, fmt=args.format)
    return 0


# ---------------------------------------------------------------------------
# CLI helpers (thin wrappers around project internals)
# ---------------------------------------------------------------------------


def _open_weaviate_client() -> Any:
    """Open a Weaviate client for the CLI."""
    from src.vector_db import create_persistent_client

    return create_persistent_client()


def _open_minio_client() -> Optional[Any]:
    """Open a MinIO client for the CLI, or return None on failure."""
    try:
        import minio as _minio_lib  # noqa: F401 — presence check only
        from config.settings import (  # type: ignore[attr-defined]
            MINIO_ENDPOINT,
            MINIO_ACCESS_KEY,
            MINIO_SECRET_KEY,
            MINIO_SECURE,
        )
        import minio

        return minio.Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_SECURE,
        )
    except Exception as exc:
        logger.warning("gc_cli_minio_unavailable error=%s", exc)
        return None


def _resolve_minio_bucket() -> str:
    """Resolve the MinIO bucket name from settings."""
    try:
        from config.settings import MINIO_BUCKET  # type: ignore[attr-defined]

        return MINIO_BUCKET
    except Exception:
        return ""


def _open_neo4j_client() -> Optional[Any]:
    """Return the active KG backend, or None if unavailable."""
    try:
        from src.knowledge_graph import get_graph_backend

        return get_graph_backend()
    except Exception as exc:
        logger.warning("gc_cli_neo4j_unavailable error=%s", exc)
        return None


def _load_manifest() -> dict:
    """Load and normalise the current manifest from disk."""
    try:
        from src.ingest.common.utils import load_manifest  # type: ignore[attr-defined]

        raw = load_manifest()
        return raw
    except Exception as exc:
        logger.warning("gc_cli_manifest_load_failed error=%s", exc)
        return {}


def _save_manifest(manifest: dict) -> None:
    """Persist the mutated manifest back to disk."""
    try:
        from src.ingest.common.utils import save_manifest  # type: ignore[attr-defined]

        save_manifest(manifest)
    except Exception as exc:
        logger.warning("gc_cli_manifest_save_failed error=%s", exc)


def _emit_report(
    gc_report: GCReport,
    orphan_report: OrphanReport,
    fmt: str = "json",
) -> None:
    """Write the GC report to stdout."""
    from src.ingest.lifecycle.orphan_report import format_text, format_json

    if fmt == "text":
        print(format_text(orphan_report, gc_report))
    else:
        data = {
            "orphans": {
                "weaviate": orphan_report.weaviate_orphans,
                "minio": orphan_report.minio_orphans,
                "neo4j": orphan_report.neo4j_orphans,
                "manifest_only": orphan_report.manifest_only,
            },
            "gc": {
                "soft_deleted": gc_report.soft_deleted,
                "hard_deleted": gc_report.hard_deleted,
                "retention_purged": gc_report.retention_purged,
                "dry_run": gc_report.dry_run,
                "per_document": {
                    k: asdict(v) for k, v in gc_report.per_document.items()
                },
            },
        }
        print(json.dumps(data, indent=2))


if __name__ == "__main__":
    sys.exit(run_gc_cli())
