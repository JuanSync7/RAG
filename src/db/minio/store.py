# @summary
# MinIO low-level helpers: bucket management, document put/get/delete/exists, presigned URLs, listing,
# and page image storage/deletion for the visual embedding pipeline.
# Exports: create_client, ensure_bucket, put_document, get_document,
#          delete_document, document_exists, get_document_url, build_document_id, list_documents,
#          store_page_images, delete_page_images, get_page_image_url
# Deps: minio, minio.error, json, io, uuid, datetime, config.settings, src.platform.observability, PIL
# @end-summary
"""Low-level MinIO document store operations."""

from __future__ import annotations

import io
import json
import logging
import uuid
from datetime import timedelta
from typing import Optional

from minio import Minio
from minio.error import S3Error

from config.settings import (
    MINIO_ENDPOINT,
    MINIO_ACCESS_KEY,
    MINIO_SECRET_KEY,
    MINIO_BUCKET,
    MINIO_SECURE,
)
from src.platform.observability import get_tracer

logger = logging.getLogger("rag.db.minio.store")
tracer = get_tracer()

_METADATA_SUFFIX = ".meta.json"
_CONTENT_SUFFIX = ".md"


def build_document_id(source_key: str) -> str:
    """Deterministic UUID for a document, stable across ingestion runs."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"doc:{source_key}"))


def create_client() -> Minio:
    """Create a MinIO client. The client is stateless and safe to share."""
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE,
    )


def ensure_bucket(client: Minio, bucket: str = MINIO_BUCKET) -> None:
    """Create the bucket if it does not exist (idempotent)."""
    span = tracer.start_span("document_store.ensure_bucket", {"bucket": bucket})
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
    span.end(status="ok")


def put_document(
    client: Minio,
    document_id: str,
    content: str,
    metadata: Optional[dict] = None,
    bucket: str = MINIO_BUCKET,
) -> None:
    """Store or overwrite a document and its metadata sidecar.

    Two objects are written:
      - ``<document_id>.md``        — the markdown content
      - ``<document_id>.meta.json`` — the metadata dict as JSON
    """
    span = tracer.start_span(
        "document_store.put_document",
        {"document_id": document_id, "bucket": bucket, "content_bytes": len(content)},
    )
    content_bytes = content.encode("utf-8")
    client.put_object(
        bucket,
        f"{document_id}{_CONTENT_SUFFIX}",
        io.BytesIO(content_bytes),
        length=len(content_bytes),
        content_type="text/markdown; charset=utf-8",
    )
    meta_payload = json.dumps(metadata or {}, ensure_ascii=False).encode("utf-8")
    client.put_object(
        bucket,
        f"{document_id}{_METADATA_SUFFIX}",
        io.BytesIO(meta_payload),
        length=len(meta_payload),
        content_type="application/json; charset=utf-8",
    )
    span.end(status="ok")


def get_document(
    client: Minio,
    document_id: str,
    bucket: str = MINIO_BUCKET,
) -> Optional[dict]:
    """Fetch a document's content and metadata.

    Returns:
        Dict with ``document_id``, ``content``, ``metadata`` keys, or ``None``
        if not found.
    """
    span = tracer.start_span(
        "document_store.get_document",
        {"document_id": document_id, "bucket": bucket},
    )
    try:
        resp = client.get_object(bucket, f"{document_id}{_CONTENT_SUFFIX}")
        try:
            content = resp.read().decode("utf-8")
        finally:
            resp.close()
            resp.release_conn()
    except S3Error as exc:
        if exc.code in ("NoSuchKey", "NoSuchBucket"):
            span.end(status="ok")
            return None
        raise

    metadata: dict = {}
    try:
        resp_meta = client.get_object(bucket, f"{document_id}{_METADATA_SUFFIX}")
        try:
            metadata = json.loads(resp_meta.read().decode("utf-8"))
        finally:
            resp_meta.close()
            resp_meta.release_conn()
    except S3Error as exc:
        if exc.code not in ("NoSuchKey", "NoSuchBucket"):
            raise
        # metadata sidecar missing — not fatal

    span.end(status="ok")
    return {"document_id": document_id, "content": content, "metadata": metadata}


def delete_document(
    client: Minio,
    document_id: str,
    bucket: str = MINIO_BUCKET,
) -> bool:
    """Remove a document and its metadata sidecar.

    Returns:
        True if the content object existed, False otherwise.
    """
    span = tracer.start_span(
        "document_store.delete_document",
        {"document_id": document_id, "bucket": bucket},
    )
    existed = document_exists(client, document_id, bucket)
    for suffix in (_CONTENT_SUFFIX, _METADATA_SUFFIX):
        try:
            client.remove_object(bucket, f"{document_id}{suffix}")
        except S3Error as exc:
            if exc.code not in ("NoSuchKey", "NoSuchBucket"):
                raise
    span.end(status="ok")
    return existed


def document_exists(
    client: Minio,
    document_id: str,
    bucket: str = MINIO_BUCKET,
) -> bool:
    """Return True if the content object exists in the bucket."""
    try:
        client.stat_object(bucket, f"{document_id}{_CONTENT_SUFFIX}")
        return True
    except S3Error as exc:
        if exc.code not in ("NoSuchKey", "NoSuchBucket"):
            raise
        return False


def get_document_url(
    client: Minio,
    document_id: str,
    bucket: str = MINIO_BUCKET,
    expires_in_seconds: int = 3600,
) -> str:
    """Return a presigned URL for direct download of the document content."""
    return client.presigned_get_object(
        bucket,
        f"{document_id}{_CONTENT_SUFFIX}",
        expires=timedelta(seconds=expires_in_seconds),
    )


def list_documents(
    client: Minio,
    bucket: str = MINIO_BUCKET,
    prefix: str = "",
    limit: int = 1000,
    offset: int = 0,
) -> list[dict]:
    """List content objects in a bucket, excluding metadata sidecars.

    Returns dicts with: document_id, source_key, size_bytes, last_modified (ISO-8601).
    Sidecars (.meta.json) are excluded from results.
    Applies Python-side pagination (offset/limit) after collecting all matches.
    Logs WARNING and falls back to object name stem when sidecar is missing.

    Raises:
        S3Error: re-raised if bucket is unreachable (not NoSuchKey/NoSuchBucket).
    """
    span = tracer.start_span(
        "document_store.list_documents",
        {"bucket": bucket, "prefix": prefix, "limit": limit, "offset": offset},
    )
    results: list[dict] = []
    for obj in client.list_objects(bucket, prefix=prefix, recursive=True):
        name: str = obj.object_name
        if not name.endswith(_CONTENT_SUFFIX):
            continue
        stem = name[: -len(_CONTENT_SUFFIX)]
        source_key = stem
        try:
            resp = client.get_object(bucket, f"{stem}{_METADATA_SUFFIX}")
            try:
                meta = json.loads(resp.read().decode("utf-8"))
            finally:
                resp.close()
                resp.release_conn()
            source_key = meta.get("source_key", stem)
        except S3Error as exc:
            if exc.code not in ("NoSuchKey", "NoSuchBucket"):
                raise
            logger.warning(
                "list_documents: sidecar missing for %r; using stem as source_key", name
            )
        results.append({
            "document_id": build_document_id(source_key),
            "source_key": source_key,
            "size_bytes": obj.size,
            "last_modified": obj.last_modified.isoformat() if obj.last_modified else None,
        })
    span.end(status="ok")
    return results[offset: offset + limit]


def get_page_image_url(
    client: Minio,
    minio_key: str,
    bucket: str = "",
    expires_in_seconds: int = 0,
) -> str:
    """Generate a presigned GET URL for a page image object.

    Unlike ``get_document_url()``, this function does NOT append any suffix
    to the key. The ``minio_key`` is used as-is (e.g.,
    ``pages/{document_id}/{page_number:04d}.jpg``).

    If the object does not exist in MinIO, a presigned URL is still
    generated (MinIO does not verify existence at signing time).

    Args:
        client: MinIO client handle.
        minio_key: Full MinIO object key for the page image (FR-401).
        bucket: Target bucket. Defaults to ``MINIO_BUCKET`` when empty
            string or not provided (FR-403).
        expires_in_seconds: URL expiry duration in seconds. Defaults to
            ``RAG_VISUAL_RETRIEVAL_URL_EXPIRY_SECONDS`` when 0 or not
            provided (FR-403).

    Returns:
        Presigned GET URL string for the page image.
    """
    # FR-403: sentinel default resolution
    if not bucket:
        bucket = MINIO_BUCKET
    if expires_in_seconds == 0:
        from config.settings import RAG_VISUAL_RETRIEVAL_URL_EXPIRY_SECONDS
        expires_in_seconds = RAG_VISUAL_RETRIEVAL_URL_EXPIRY_SECONDS

    span = tracer.start_span(
        "document_store.get_page_image_url",
        {"minio_key": minio_key, "bucket": bucket},
    )
    # FR-401: raw key — no suffix appended
    url = client.presigned_get_object(
        bucket,
        minio_key,
        expires=timedelta(seconds=expires_in_seconds),
    )
    span.end(status="ok")
    return url


def store_figure_image(
    client: Minio,
    source_key: str,
    figure_label: str,
    image_bytes: bytes,
    mime_type: str,
    bucket: str = MINIO_BUCKET,
) -> str | None:
    """Store a single figure image (markdown image ref) in MinIO.

    Companion to ``store_page_images`` — that one stores PDF page screenshots
    for visual retrieval; this one stores standalone figure images referenced
    in markdown so citation UIs can render them via presigned URL.

    Key pattern: ``figures/{safe_source_key}/{safe_figure_label}.{ext}``

    Args:
        client: MinIO client handle.
        source_key: Stable source identity (sanitised internally).
        figure_label: Human label like ``"Figure 1"``.
        image_bytes: Raw image bytes.
        mime_type: MIME type, used to derive the file extension.
        bucket: Target bucket (defaults to ``MINIO_BUCKET``).

    Returns:
        The MinIO object key on success, or ``None`` if the upload failed
        (failure is logged; never raises).
    """
    safe_source = source_key.replace("/", "_").replace(":", "_").replace("..", "__")
    safe_label = figure_label.replace("/", "_").replace(" ", "_").lower()
    ext = mime_type.split("/", 1)[-1] if "/" in mime_type else "bin"
    if ext == "jpeg":
        ext = "jpg"
    key = f"figures/{safe_source}/{safe_label}.{ext}"
    try:
        client.put_object(
            bucket,
            key,
            io.BytesIO(image_bytes),
            length=len(image_bytes),
            content_type=mime_type or "application/octet-stream",
        )
        return key
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "store_figure_image: failed to upload %s for %r: %s",
            figure_label,
            source_key,
            exc,
        )
        return None


def store_page_images(
    client: Minio,
    document_id: str,
    pages: list[tuple[int, object]],
    quality: int = 85,
    bucket: str = MINIO_BUCKET,
) -> list[str]:
    """Store page images as JPEG in MinIO.

    Key pattern: pages/{document_id}/{page_number:04d}.jpg (FR-401)
    JPEG compression at specified quality (FR-402).
    Returns list of successfully stored MinIO keys (FR-403).

    Per-page errors are isolated: one failed upload does not block the rest (FR-401).

    Args:
        client: MinIO client handle.
        document_id: Unique document identifier.
        pages: List of (1-indexed page_number, PIL.Image) tuples.
        quality: JPEG compression quality 1-100. FR-402
        bucket: Target bucket. Uses MINIO_BUCKET default if not provided.

    Returns:
        List of MinIO object keys that were successfully stored. FR-403
    """
    stored_keys: list[str] = []
    for page_number, image in pages:
        key = f"pages/{document_id}/{page_number:04d}.jpg"
        try:
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=quality)  # type: ignore[union-attr]
            # Determine buffer length then reset to start for upload
            buffer.seek(0, 2)
            length = buffer.tell()
            buffer.seek(0)
            client.put_object(
                bucket,
                key,
                buffer,
                length=length,
                content_type="image/jpeg",
            )
            stored_keys.append(key)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "store_page_images: failed to upload page %d for document %r: %s",
                page_number,
                document_id,
                exc,
            )
    return stored_keys


def delete_page_images(
    client: Minio,
    document_id: str,
    bucket: str = MINIO_BUCKET,
) -> int:
    """Delete all page images for a document from MinIO.

    Deletes all objects matching prefix pages/{document_id}/ (FR-404).
    Used for pre-storage cleanup in update mode (FR-405).

    Args:
        client: MinIO client handle.
        document_id: Unique document identifier.
        bucket: Target bucket. Uses MINIO_BUCKET default if not provided.

    Returns:
        Number of objects deleted.
    """
    prefix = f"pages/{document_id}/"
    deleted = 0
    try:
        for obj in client.list_objects(bucket, prefix=prefix, recursive=True):
            try:
                client.remove_object(bucket, obj.object_name)
                deleted += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "delete_page_images: failed to delete %r for document %r: %s",
                    obj.object_name,
                    document_id,
                    exc,
                )
                return deleted
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "delete_page_images: failed to list objects for document %r: %s",
            document_id,
            exc,
        )
    return deleted
