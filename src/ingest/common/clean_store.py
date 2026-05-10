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
import orjson
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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
        """Remove the clean document entry for this key (all three files).

        Removes ``{safe_key}.md``, ``{safe_key}.meta.json``, and
        ``{safe_key}.docling.json``. Missing files are silently ignored.
        """
        self._md_path(source_key).unlink(missing_ok=True)
        self._meta_path(source_key).unlink(missing_ok=True)
        self._docling_path(source_key).unlink(missing_ok=True)

    def list_keys(self) -> list[str]:
        """Return all source keys currently stored."""
        if not self._dir.exists():
            return []
        return [p.stem for p in self._dir.glob("*.md") if p.suffix == ".md"]
