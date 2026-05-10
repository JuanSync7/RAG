# @summary
# E2EValidator: per-trace_id and bulk cross-store consistency validation.
# Queries Weaviate, MinIO, Neo4j, and the manifest; reports presence/count
# discrepancies, missing-from-store lists, and split-state trace_ids.
# Exposes run_validation_cli() as a module-level CLI entry point.
# Exports: E2EValidator, run_validation_cli
# Deps: src.ingest.lifecycle.schemas, src.vector_db, src.ingest.common.minio_clean_store,
#       logging, argparse, json, random
# @end-summary
"""End-to-end per-document store consistency validation (FR-3060, FR-3061, FR-3062).

:class:`E2EValidator` queries all four stores (Weaviate, MinIO, Neo4j, manifest)
by ``trace_id`` and ``source_key`` and produces a :class:`ValidationReport` that
records:

* Which stores are missing data for a given trace_id.
* Chunk-count or triple-count discrepancies (informational).
* trace_ids whose data is split across stores (partial ingestion).

Validation is read-only — no store is mutated.  Failures are caught per-entry
so a single store unavailability does not abort the whole run (FR-3060 AC6).

Disabled stores are reported as ``None`` (not ``False``) to distinguish
"not configured" from "configured but broken" (FR-3062).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from src.ingest.lifecycle.schemas import ValidationReport, ValidationFinding

logger = logging.getLogger(__name__)


class E2EValidator:
    """Cross-store consistency validator.

    Args:
        client: Weaviate client handle.
        minio: :class:`MinioCleanStore` instance, or ``None`` if MinIO is not
            configured.
        kg: KG backend client with a ``count_triples_by_trace_id(trace_id)``
            method, or ``None`` if the KG is disabled.
        manifest_path: Path to the manifest JSON file on disk (optional; used
            by CLI helpers).  Programmatic callers can supply a manifest dict
            directly to :meth:`validate_by_trace_id` and :meth:`validate_all`.
        collection: Weaviate collection name.  ``None`` uses the default.
    """

    def __init__(
        self,
        client: Any,
        minio: Optional[Any] = None,
        kg: Optional[Any] = None,
        manifest_path: Optional[str] = None,
        collection: Optional[str] = None,
    ) -> None:
        self._client = client
        self._minio = minio
        self._kg = kg
        self._manifest_path = manifest_path
        self._collection = collection

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_by_trace_id(
        self,
        trace_id: str,
        manifest: Optional[dict] = None,
    ) -> ValidationReport:
        """Validate all four stores for a single trace_id.

        Queries:
        1. Weaviate: count of chunks with this trace_id (>= 1 expected).
        2. MinIO: existence of clean markdown for the source_key bound to
           this trace_id in the manifest.
        3. Neo4j: count of triples with this trace_id (>= 1 if KG enabled).
        4. Manifest: presence of an entry referencing this trace_id.

        Args:
            trace_id: UUID v4 trace ID to validate.
            manifest: Optional manifest dict. When ``None``, loaded from
                ``manifest_path`` if available.

        Returns:
            :class:`ValidationReport` with per-store findings and a
            ``consistent`` flag.
        """
        if manifest is None:
            manifest = self._load_manifest()

        # Find the source_key for this trace_id in the manifest.
        source_key = self._find_source_key_by_trace_id(trace_id, manifest)

        finding = self._check_single(trace_id, source_key, manifest)
        report = ValidationReport(
            validated_at=datetime.now(timezone.utc).isoformat(),
            findings=[finding],
            consistent=finding.consistent,
        )
        logger.info(
            "e2e_validate_by_trace_id trace_id=%s source_key=%s consistent=%s",
            trace_id,
            source_key,
            finding.consistent,
        )
        return report

    def validate_all(
        self,
        *,
        sample_size: Optional[int] = None,
        manifest: Optional[dict] = None,
    ) -> ValidationReport:
        """Validate all (or a random sample of) trace_ids from the manifest.

        Non-deleted manifest entries with a non-empty ``trace_id`` field are
        candidates. If *sample_size* is given, a random sample of at most
        *sample_size* entries is validated.

        Args:
            sample_size: Optional maximum number of entries to validate.
                ``None`` validates all candidates.
            manifest: Optional manifest dict.  Loaded from ``manifest_path``
                when ``None``.

        Returns:
            :class:`ValidationReport` with one :class:`ValidationFinding` per
            validated trace_id.
        """
        if manifest is None:
            manifest = self._load_manifest()

        candidates: list[tuple[str, str]] = []  # [(trace_id, source_key), ...]
        for source_key, entry in manifest.items():
            if entry.get("deleted", False):
                continue
            trace_id = entry.get("trace_id", "")
            if not trace_id:
                continue
            candidates.append((trace_id, source_key))

        if sample_size is not None and sample_size < len(candidates):
            candidates = random.sample(candidates, sample_size)

        findings: list[ValidationFinding] = []
        for trace_id, source_key in candidates:
            finding = self._check_single(trace_id, source_key, manifest)
            findings.append(finding)

        inconsistent = [f for f in findings if not f.consistent]
        overall_consistent = len(inconsistent) == 0

        report = ValidationReport(
            validated_at=datetime.now(timezone.utc).isoformat(),
            findings=findings,
            consistent=overall_consistent,
            total_checked=len(candidates),
            inconsistent_count=len(inconsistent),
        )
        logger.info(
            "e2e_validate_all checked=%d inconsistent=%d",
            len(candidates),
            len(inconsistent),
        )
        return report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_single(
        self,
        trace_id: str,
        source_key: Optional[str],
        manifest: dict,
    ) -> ValidationFinding:
        """Run per-store checks for one (trace_id, source_key) pair."""
        finding = ValidationFinding(
            trace_id=trace_id,
            source_key=source_key or "",
            checked_at=datetime.now(timezone.utc).isoformat(),
        )

        # -- Manifest check --
        if source_key and source_key in manifest:
            finding.manifest_ok = True
        else:
            finding.manifest_ok = False
            finding.missing_stores.append("manifest")

        # -- Weaviate check --
        finding.weaviate_chunk_count = self._count_weaviate(trace_id)
        if finding.weaviate_chunk_count is None:
            # Error occurred — flag as missing.
            finding.missing_stores.append("weaviate")
            finding.weaviate_ok = False
        elif finding.weaviate_chunk_count > 0:
            finding.weaviate_ok = True
        else:
            finding.weaviate_ok = False
            finding.missing_stores.append("weaviate")

        # -- MinIO check --
        if source_key and self._minio is not None:
            minio_result = self._check_minio(source_key)
            finding.minio_ok = minio_result
            if not minio_result:
                finding.missing_stores.append("minio")
        elif self._minio is None:
            finding.minio_ok = None  # Not configured.
        else:
            # No source_key resolved — cannot check MinIO.
            finding.minio_ok = None

        # -- Neo4j / KG check --
        if self._kg is not None:
            finding.kg_triple_count = self._count_kg(trace_id)
            if finding.kg_triple_count is None:
                finding.neo4j_ok = False
                finding.missing_stores.append("neo4j")
            elif finding.kg_triple_count > 0:
                finding.neo4j_ok = True
            else:
                finding.neo4j_ok = False
                finding.missing_stores.append("neo4j")
        else:
            finding.neo4j_ok = None  # KG disabled.

        # -- Overall consistency --
        # A store counts toward consistency only when explicitly True/False
        # (None = disabled, excluded from check per FR-3062).
        enabled = [
            v
            for v in (finding.weaviate_ok, finding.minio_ok, finding.neo4j_ok)
            if v is not None
        ]
        # Always require manifest and at least Weaviate.
        finding.consistent = (
            finding.manifest_ok is True
            and finding.weaviate_ok is True
            and all(enabled)
        )

        return finding

    def _count_weaviate(self, trace_id: str) -> Optional[int]:
        """Return chunk count for *trace_id* in Weaviate, or ``None`` on error."""
        try:
            from src.vector_db import count_by_trace_id  # type: ignore[attr-defined]

            return count_by_trace_id(
                self._client,
                trace_id,
                collection=self._collection,
            )
        except (ImportError, AttributeError):
            # Function not yet in facade — treat as 0 chunks.
            logger.debug(
                "e2e_validator_weaviate_count_unavailable trace_id=%s", trace_id
            )
            return 0
        except Exception as exc:
            logger.warning(
                "e2e_validator_weaviate_count_failed trace_id=%s error=%s",
                trace_id,
                exc,
            )
            return None

    def _check_minio(self, source_key: str) -> bool:
        """Return True if the clean document exists in MinIO for *source_key*."""
        try:
            return self._minio.exists(source_key)
        except Exception as exc:
            logger.warning(
                "e2e_validator_minio_check_failed source_key=%s error=%s",
                source_key,
                exc,
            )
            return False

    def _count_kg(self, trace_id: str) -> Optional[int]:
        """Return triple count for *trace_id* in KG, or ``None`` on error."""
        try:
            if hasattr(self._kg, "count_triples_by_trace_id"):
                return self._kg.count_triples_by_trace_id(trace_id)
            logger.debug(
                "e2e_validator_kg_count_unavailable trace_id=%s", trace_id
            )
            return 0
        except Exception as exc:
            logger.warning(
                "e2e_validator_kg_count_failed trace_id=%s error=%s",
                trace_id,
                exc,
            )
            return None

    def _find_source_key_by_trace_id(
        self, trace_id: str, manifest: dict
    ) -> Optional[str]:
        """Scan manifest for an entry whose trace_id matches."""
        for source_key, entry in manifest.items():
            if entry.get("trace_id") == trace_id:
                return source_key
        return None

    def _load_manifest(self) -> dict:
        """Load manifest from ``manifest_path``, or return empty dict."""
        if not self._manifest_path:
            return {}
        try:
            import json as _json

            return _json.loads(
                open(self._manifest_path, encoding="utf-8").read()
            )
        except Exception as exc:
            logger.warning(
                "e2e_validator_manifest_load_failed path=%s error=%s",
                self._manifest_path,
                exc,
            )
            return {}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def run_validation_cli(argv: list[str] | None = None) -> int:
    """Command-line entry point for the E2E validator.

    Args:
        argv: Argument list.  Defaults to ``sys.argv[1:]``.

    Returns:
        Exit code: 0 on success (consistent), 1 on inconsistency found,
        2 on runtime error.

    Usage::

        python -m src.ingest.lifecycle.validation \\
            [--trace-id UUID] \\
            [--all] \\
            [--sample N] \\
            [--format {json,text}]
    """
    parser = argparse.ArgumentParser(
        prog="ragweave-validate",
        description="Validate cross-store consistency for ingested documents.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--trace-id",
        metavar="UUID",
        help="Validate a single trace_id.",
    )
    group.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Validate all trace_ids in the manifest.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Randomly sample N entries when used with --all.",
    )
    parser.add_argument(
        "--format",
        choices=["json", "text"],
        default="json",
        help="Output format. Default: json.",
    )

    args = parser.parse_args(argv)

    try:
        weaviate_client = _open_weaviate_client()
        minio = _build_minio_store()
        kg = _open_kg_client()
        manifest = _load_manifest_cli()

        validator = E2EValidator(
            client=weaviate_client,
            minio=minio,
            kg=kg,
        )

        if args.trace_id:
            report = validator.validate_by_trace_id(
                args.trace_id, manifest=manifest
            )
        else:
            report = validator.validate_all(
                sample_size=args.sample, manifest=manifest
            )

        _emit_validation_report(report, fmt=args.format)
        return 0 if report.consistent else 1

    except Exception as exc:
        logger.exception("validation_cli_error: %s", exc)
        sys.stderr.write(f"ERROR: {exc}\n")
        return 2


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def _open_weaviate_client() -> Any:
    try:
        from src.vector_db import create_persistent_client  # type: ignore[attr-defined]

        return create_persistent_client()
    except Exception as exc:
        logger.warning("validation_cli_weaviate_unavailable error=%s", exc)
        return None


def _build_minio_store() -> Optional[Any]:
    try:
        import minio as _minio_lib  # noqa: F401
        from config.settings import (  # type: ignore[attr-defined]
            MINIO_ENDPOINT,
            MINIO_ACCESS_KEY,
            MINIO_SECRET_KEY,
            MINIO_SECURE,
            MINIO_BUCKET,
        )
        from src.ingest.common.minio_clean_store import MinioCleanStore  # type: ignore[attr-defined]
        import minio

        client = minio.Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_SECURE,
        )
        return MinioCleanStore(client, MINIO_BUCKET)
    except Exception as exc:
        logger.warning("validation_cli_minio_unavailable error=%s", exc)
        return None


def _open_kg_client() -> Optional[Any]:
    try:
        from src.knowledge_graph import get_graph_backend  # type: ignore[attr-defined]

        return get_graph_backend()
    except Exception as exc:
        logger.warning("validation_cli_kg_unavailable error=%s", exc)
        return None


def _load_manifest_cli() -> dict:
    try:
        from src.ingest.common.utils import load_manifest  # type: ignore[attr-defined]

        return load_manifest()
    except Exception as exc:
        logger.warning("validation_cli_manifest_load_failed error=%s", exc)
        return {}


def _emit_validation_report(report: ValidationReport, fmt: str = "json") -> None:
    """Write the validation report to stdout."""
    if fmt == "text":
        ts = report.validated_at
        print(f"=== E2E Validation Report  [{ts}] ===")
        print(f"  Overall consistent : {report.consistent}")
        print(f"  Total checked      : {report.total_checked}")
        print(f"  Inconsistent count : {report.inconsistent_count}")
        print()
        for finding in report.findings:
            marker = "OK" if finding.consistent else "INCONSISTENT"
            print(
                f"  [{marker}] {finding.trace_id} "
                f"(source_key={finding.source_key})"
            )
            print(
                f"    manifest={finding.manifest_ok} "
                f"weaviate={finding.weaviate_ok} (chunks={finding.weaviate_chunk_count}) "
                f"minio={finding.minio_ok} "
                f"neo4j={finding.neo4j_ok} (triples={finding.kg_triple_count})"
            )
            if finding.missing_stores:
                print(f"    Missing from: {', '.join(finding.missing_stores)}")
    else:
        data = {
            "validated_at": report.validated_at,
            "consistent": report.consistent,
            "total_checked": report.total_checked,
            "inconsistent_count": report.inconsistent_count,
            "findings": [
                {
                    "trace_id": f.trace_id,
                    "source_key": f.source_key,
                    "checked_at": f.checked_at,
                    "consistent": f.consistent,
                    "manifest_ok": f.manifest_ok,
                    "weaviate_ok": f.weaviate_ok,
                    "weaviate_chunk_count": f.weaviate_chunk_count,
                    "minio_ok": f.minio_ok,
                    "neo4j_ok": f.neo4j_ok,
                    "kg_triple_count": f.kg_triple_count,
                    "missing_stores": f.missing_stores,
                }
                for f in report.findings
            ],
        }
        print(json.dumps(data, indent=2))


if __name__ == "__main__":
    sys.exit(run_validation_cli())
