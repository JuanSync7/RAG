# @summary
# Atomic persistent store for clean Markdown documents and optional DoclingDocument
# JSON between pipeline phases.
# Exports: CleanDocumentStore
# Deps: orjson, hashlib, pathlib, docling_core (lazy)
# @end-summary

"""CleanDocumentStore — atomic read/write of clean Markdown between pipeline phases."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import orjson
from config.settings import RAG_INGEST_CLEAN_STORE_MAX_ATTEMPTS

logger = logging.getLogger(__name__)


_VALID_PHASES = ("phase2a", "phase2b")
_VALID_ERROR_CLASSES = ("transient", "document", "system")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty_phase() -> dict[str, Any]:
    return {
        "status": "pending",
        "attempts": 0,
        "last_error": None,
        "error_class": None,
        "last_attempt_ts": None,
        "ok_ts": None,
    }


def _empty_status() -> dict[str, dict[str, Any]]:
    return {phase: _empty_phase() for phase in _VALID_PHASES}


class CleanDocumentStore:
    """Persistent store for clean Markdown output from Phase 1.

    Stores each document as up to three files:
      - ``{store_dir}/{source_key}.md``           — clean Markdown text
      - ``{store_dir}/{source_key}.meta.json``     — source identity metadata
      - ``{store_dir}/{source_key}.docling.json``  — serialized DoclingDocument (optional)

    All writes are atomic: content is written to a ``.tmp`` file and
    then renamed into place, preventing partial reads on failure.

    The ``.docling.json`` file uses an envelope format with ``_schema_version``
    for future migration safety.

    Args:
        store_dir: Directory in which to store documents. Created on first write.
    """

    def __init__(self, store_dir: Path) -> None:
        self._dir = Path(store_dir)

    def _safe_key(self, source_key: str) -> str:
        """Sanitize source key for use in filesystem paths."""
        return source_key.replace("/", "_").replace(":", "_").replace("..", "__")

    def _md_path(self, source_key: str) -> Path:
        return self._dir / f"{self._safe_key(source_key)}.md"

    def _meta_path(self, source_key: str) -> Path:
        return self._dir / f"{self._safe_key(source_key)}.meta.json"

    def _status_path(self, source_key: str) -> Path:
        """Return the path for the per-source status ledger sidecar."""
        return self._dir / f"{self._safe_key(source_key)}.status.json"

    def _docling_path(self, source_key: str) -> Path:
        """Return the path for the serialized DoclingDocument JSON file.

        Args:
            source_key: Stable source identity key.

        Returns:
            Path of the form ``{store_dir}/{safe_key}.docling.json``.
        """
        return self._dir / f"{self._safe_key(source_key)}.docling.json"

    def write_docling(self, source_key: str, docling_document: Any) -> None:
        """Atomically serialize and persist a DoclingDocument.

        Writes to a ``.tmp`` file first, then renames into place.
        Wraps the document in the envelope::

            {"_schema_version": "docling-native-v1", "document": {...}}

        Args:
            source_key: Stable source identity key.
            docling_document: Native DoclingDocument (docling_core Pydantic model).
                Must support ``.model_dump_json()`` serialization.

        Raises:
            OSError: If the atomic write fails (tmp write or rename).
            ValueError: If the document cannot be serialized.
        """
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._docling_path(source_key)
        tmp_path = path.with_suffix(".tmp")
        try:
            doc_dict = json.loads(docling_document.model_dump_json())
            envelope = {"_schema_version": "docling-native-v1", "document": doc_dict}
            tmp_path.write_bytes(orjson.dumps(envelope))
            os.replace(tmp_path, path)
        except (OSError, ValueError):
            tmp_path.unlink(missing_ok=True)
            raise
        except Exception as exc:
            tmp_path.unlink(missing_ok=True)
            raise ValueError(
                f"CleanDocumentStore: failed to serialize DoclingDocument for {source_key!r}: {exc}"
            ) from exc

    def read_docling(self, source_key: str) -> Any | None:
        """Deserialize and return a DoclingDocument for the given source key.

        Checks ``_schema_version == "docling-native-v1"`` before deserializing.
        Imports ``DoclingDocument`` lazily to avoid a hard ``docling-core`` import at
        module load time.

        Returns:
            The deserialized DoclingDocument, or ``None`` if:

            - The ``.docling.json`` file does not exist.
            - The file contains invalid JSON.
            - ``_schema_version`` does not match ``"docling-native-v1"``.
            - Deserialization fails for any reason.

        Logs a warning on any failure. Never raises.
        """
        path = self._docling_path(source_key)
        try:
            raw = path.read_bytes()
            data = orjson.loads(raw)
            version = data.get("_schema_version")
            if version != "docling-native-v1":
                logger.warning(
                    "CleanDocumentStore: unsupported _schema_version %r for %r — skipping",
                    version,
                    source_key,
                )
                return None
            from docling_core.types.doc import DoclingDocument  # noqa: PLC0415
            return DoclingDocument.model_validate(data["document"])
        except FileNotFoundError:
            return None
        except Exception as exc:
            logger.warning(
                "CleanDocumentStore: failed to read/deserialize docling document for %r: %s",
                source_key,
                exc,
            )
            return None

    def write(
        self,
        source_key: str,
        text: str,
        meta: dict[str, Any],
        docling_document: Any | None = None,
    ) -> None:
        """Atomically write clean text, metadata, and optional DoclingDocument.

        Existing behavior for text and meta is unchanged (atomic tmp → rename).
        When ``docling_document`` is not ``None``, calls :meth:`write_docling` after
        the md + meta write. A ``write_docling`` failure is logged but does NOT
        roll back the already-committed md/meta files.

        Args:
            source_key: Stable source identity key.
            text: Clean markdown text.
            meta: Metadata dict to serialize as JSON.
            docling_document: Optional native DoclingDocument.
        """
        self._dir.mkdir(parents=True, exist_ok=True)
        md_path = self._md_path(source_key)
        meta_path = self._meta_path(source_key)

        tmp_md = md_path.with_suffix(".md.tmp")
        tmp_meta = meta_path.with_suffix(".meta.json.tmp")

        try:
            tmp_md.write_text(text, encoding="utf-8")
            tmp_meta.write_bytes(orjson.dumps(meta))
            tmp_md.replace(md_path)
            tmp_meta.replace(meta_path)
        except (OSError, ValueError):
            tmp_md.unlink(missing_ok=True)
            tmp_meta.unlink(missing_ok=True)
            raise

        if docling_document is not None:
            try:
                self.write_docling(source_key, docling_document)
            except Exception as exc:
                logger.error(
                    "CleanDocumentStore: write_docling failed for %r (md/meta preserved): %s",
                    source_key,
                    exc,
                )

    def read(self, source_key: str) -> tuple[str, dict[str, Any]]:
        """Read clean text and metadata for a source key."""
        md_path = self._md_path(source_key)
        meta_path = self._meta_path(source_key)
        if not md_path.exists():
            raise FileNotFoundError(f"CleanDocumentStore: no entry for {source_key!r}")
        text = md_path.read_text(encoding="utf-8")
        meta = orjson.loads(meta_path.read_bytes()) if meta_path.exists() else {}
        return text, meta

    def exists(self, source_key: str) -> bool:
        """Return True if a clean document entry exists for this key."""
        return self._md_path(source_key).exists()

    def clean_hash(self, source_key: str) -> str:
        """Return the SHA-256 hash of the stored clean text."""
        md_path = self._md_path(source_key)
        if not md_path.exists():
            raise FileNotFoundError(f"CleanDocumentStore: no entry for {source_key!r}")
        return hashlib.sha256(md_path.read_bytes()).hexdigest()

    def delete(self, source_key: str) -> None:
        """Remove the clean document entry for this key (all sidecars).

        Removes ``{safe_key}.md``, ``{safe_key}.meta.json``,
        ``{safe_key}.docling.json``, and ``{safe_key}.status.json``.
        Missing files are silently ignored.
        """
        self._md_path(source_key).unlink(missing_ok=True)
        self._meta_path(source_key).unlink(missing_ok=True)
        self._docling_path(source_key).unlink(missing_ok=True)
        self._status_path(source_key).unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Status ledger (Step 7) — durable per-phase ingest status
    # ------------------------------------------------------------------

    def get_status(self, source_key: str) -> dict[str, dict[str, Any]]:
        """Return the per-phase status ledger for *source_key*.

        Phases without a recorded attempt return defaults
        (``status="pending"``, ``attempts=0``).

        Raises:
            FileNotFoundError: If no clean entry exists for this key.
        """
        if not self.exists(source_key):
            raise FileNotFoundError(
                f"CleanDocumentStore: no entry for {source_key!r}"
            )
        path = self._status_path(source_key)
        if not path.exists():
            return _empty_status()
        try:
            data = orjson.loads(path.read_bytes())
        except Exception as exc:
            logger.warning(
                "CleanDocumentStore: status sidecar unreadable for %r: %s",
                source_key, exc,
            )
            return _empty_status()
        merged = _empty_status()
        for phase in _VALID_PHASES:
            if isinstance(data.get(phase), dict):
                merged[phase].update(data[phase])
        return merged

    def _write_status(
        self, source_key: str, status: dict[str, dict[str, Any]]
    ) -> None:
        """Atomically persist the status ledger for *source_key*."""
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._status_path(source_key)
        tmp = path.with_suffix(".json.tmp")
        try:
            tmp.write_bytes(orjson.dumps(status))
            os.replace(tmp, path)
        except OSError:
            tmp.unlink(missing_ok=True)
            raise

    def record_attempt(
        self,
        source_key: str,
        *,
        phase: str,
        success: bool,
        error: str | None = None,
        error_class: str | None = None,
        max_attempts: int = RAG_INGEST_CLEAN_STORE_MAX_ATTEMPTS,
    ) -> dict[str, Any]:
        """Increment the attempt counter for *phase* and record outcome.

        Status transitions:
          * ``success=True``       → ``"ok"`` (clears last_error).
          * transient + attempts < max_attempts → ``"failed_pending_retry"``.
          * transient + attempts >= max_attempts → ``"failed_permanent"``.
          * ``error_class="document"`` or ``"system"`` → ``"failed_permanent"``.

        Args:
            source_key: Stable source identity key.
            phase: One of ``"phase2a"`` or ``"phase2b"``.
            success: Whether the attempt succeeded.
            error: Stringified error message (failure path only).
            error_class: ``"transient"``, ``"document"``, or ``"system"``.
            max_attempts: Cap for transient retries before marking permanent.

        Returns:
            The updated phase entry (dict).

        Raises:
            FileNotFoundError: If no clean entry exists for this key.
            ValueError: For unknown ``phase`` or ``error_class``.
        """
        if phase not in _VALID_PHASES:
            raise ValueError(
                f"unknown phase {phase!r}; expected one of {_VALID_PHASES}"
            )
        if not success and error_class is not None and error_class not in _VALID_ERROR_CLASSES:
            raise ValueError(
                f"unknown error_class {error_class!r}; "
                f"expected one of {_VALID_ERROR_CLASSES}"
            )

        status = self.get_status(source_key)
        entry = status[phase]
        entry["attempts"] = int(entry.get("attempts", 0)) + 1
        entry["last_attempt_ts"] = _now_iso()

        if success:
            entry["status"] = "ok"
            entry["last_error"] = None
            entry["error_class"] = None
            entry["ok_ts"] = entry["last_attempt_ts"]
        else:
            entry["last_error"] = error
            entry["error_class"] = error_class
            if error_class in ("document", "system"):
                entry["status"] = "failed_permanent"
            elif entry["attempts"] >= max_attempts:
                entry["status"] = "failed_permanent"
            else:
                entry["status"] = "failed_pending_retry"

        status[phase] = entry
        self._write_status(source_key, status)
        return entry

    def list_pending(
        self, *, phase: str, status: str = "failed_pending_retry"
    ) -> list[str]:
        """Return source keys whose *phase* matches *status*.

        Used by the backfill workflow to discover documents that need a
        bounded retry of phase 2b. Backfill drains
        ``status="failed_pending_retry"``; permanent failures stay out so
        they can be triaged by an operator.
        """
        if phase not in _VALID_PHASES:
            raise ValueError(
                f"unknown phase {phase!r}; expected one of {_VALID_PHASES}"
            )
        if not self._dir.exists():
            return []
        keys: list[str] = []
        for md_path in self._dir.glob("*.md"):
            source_key = md_path.stem
            try:
                s = self.get_status(source_key)
            except FileNotFoundError:
                continue
            if s[phase]["status"] == status:
                keys.append(source_key)
        return sorted(keys)

    def list_keys(self) -> list[str]:
        """Return all source keys currently stored."""
        if not self._dir.exists():
            return []
        return [p.stem for p in self._dir.glob("*.md") if p.suffix == ".md"]
