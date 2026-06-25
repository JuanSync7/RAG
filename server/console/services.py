# @summary
# Shared console service helpers for UI serving, static asset resolution, log snapshots, source previews, and rendering.
# Exports: CONSOLE_HTML_PATH, USER_CONSOLE_HTML_PATH, CONSOLE_STATIC_DIR, USER_CONSOLE_STATIC_DIR, resolve_console_html_path, resolve_user_console_html_path, resolve_console_static_asset, resolve_user_console_static_asset, is_ollama_reachable, tail_log_lines, resolve_console_source_path, is_remote_view_uri, build_source_preview_payload, render_source_document_html, read_clean_document_from_minio
# Deps: config.settings, server.schemas, fastapi
# @end-summary
"""Console service helpers."""

from __future__ import annotations

import logging
import os
from collections import deque
from html import escape
from pathlib import Path
from urllib.parse import unquote, urlparse

from fastapi import HTTPException

from config.settings import (
    DOCUMENTS_DIR,
    OLLAMA_BASE_URL,
    PROJECT_ROOT,
    RAG_CONSOLE_PREVIEW_CONTEXT_CAP,
    RAG_CONSOLE_PREVIEW_CONTEXT_MIN,
    RAG_CONSOLE_PREVIEW_MAX_CHARS_CAP,
    RAG_CONSOLE_PREVIEW_MIN_CHARS,
)
from server.schemas import ConsoleLogsResponse

logger = logging.getLogger(__name__)

_CONSOLE_DIR = Path(__file__).resolve().parent

# --- Admin Console (existing tabbed debug/ops interface) ---
_CONSOLE_HTML_CANDIDATES = (
    _CONSOLE_DIR / "static" / "console.html",
    _CONSOLE_DIR.parent / "console.html",  # Backward-compat for older local checkouts.
)
CONSOLE_STATIC_DIR = _CONSOLE_DIR / "static"

# --- User Console (modern chat interface) ---
USER_CONSOLE_STATIC_DIR = _CONSOLE_DIR / "static" / "user"
_USER_CONSOLE_HTML_PATH = USER_CONSOLE_STATIC_DIR / "index.html"


def resolve_console_html_path() -> Path:
    """Resolve admin console HTML path with fallback for legacy locations."""
    for candidate in _CONSOLE_HTML_CANDIDATES:
        if candidate.exists():
            return candidate
    return _CONSOLE_HTML_CANDIDATES[0]


CONSOLE_HTML_PATH = resolve_console_html_path()


def resolve_user_console_html_path() -> Path:
    """Resolve User Console HTML path."""
    return _USER_CONSOLE_HTML_PATH


USER_CONSOLE_HTML_PATH = resolve_user_console_html_path()


def resolve_console_static_asset(asset_path: str) -> Path:
    """Resolve and validate static console asset path (admin console)."""
    candidate = (CONSOLE_STATIC_DIR / asset_path).resolve()
    static_root = CONSOLE_STATIC_DIR.resolve()
    if not str(candidate).startswith(str(static_root)):
        raise HTTPException(status_code=404, detail="Console asset not found")
    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Console asset not found")
    return candidate


def resolve_user_console_static_asset(asset_path: str) -> Path:
    """Resolve and validate static asset path for the User Console."""
    candidate = (USER_CONSOLE_STATIC_DIR / asset_path).resolve()
    static_root = USER_CONSOLE_STATIC_DIR.resolve()
    if not str(candidate).startswith(str(static_root)):
        raise HTTPException(status_code=404, detail="User console asset not found")
    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="User console asset not found")
    return candidate


def is_ollama_reachable() -> bool:
    """Best-effort Ollama reachability probe from API process."""
    from urllib.request import Request, urlopen

    req = Request(f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags", method="GET")
    try:
        with urlopen(req, timeout=3):
            return True
    except Exception:
        logger.debug("Ollama reachability probe failed", exc_info=True)
        return False


def tail_log_lines(lines: int = 120) -> ConsoleLogsResponse:
    """Return a tail snapshot across common local log files."""
    logs_dir = Path(__file__).resolve().parent.parent.parent / "logs"
    log_files = [
        logs_dir / "query_processing.log",
        logs_dir / "ingest.log",
        logs_dir / "server.log",
    ]
    out_lines: deque[str] = deque(maxlen=max(10, lines))
    found_files: list[str] = []
    for candidate in log_files:
        if not candidate.exists():
            continue
        found_files.append(str(candidate))
        try:
            text = candidate.read_text(encoding="utf-8", errors="replace")
            for line in text.splitlines()[-lines:]:
                out_lines.append(f"[{candidate.name}] {line}")
        except Exception as exc:
            out_lines.append(f"[{candidate.name}] <read_error> {exc}")
    if not found_files:
        out_lines.append("No log files found in ./logs")
    return ConsoleLogsResponse(files=found_files, lines=list(out_lines))


_BINARY_SOURCE_EXTS = {".docx", ".pptx", ".xlsx", ".pdf", ".doc", ".ppt", ".xls"}


def _allowed_source_roots() -> list[Path]:
    """Roots under which the console may read source documents.

    Defaults to ``DOCUMENTS_DIR``. Extra colon-separated roots may be added
    via the ``RAG_CONSOLE_SOURCE_ROOTS`` env var — required for the
    dev-against-prod-stack workflow where chunks reference the worker's
    clone path (e.g., ``/home/.../RagWeave/docs``) rather than the frontend
    project's ``documents/`` directory.
    """
    roots: list[Path] = [DOCUMENTS_DIR.resolve()]
    extra = os.environ.get("RAG_CONSOLE_SOURCE_ROOTS", "")
    for entry in extra.split(":"):
        entry = entry.strip()
        if not entry:
            continue
        try:
            roots.append(Path(entry).expanduser().resolve())
        except Exception:
            logger.warning("invalid_console_source_root entry=%r", entry)
    return roots


def is_remote_view_uri(source_uri: str | None) -> str | None:
    """Return the URI when it points to a remote origin the console may redirect to.

    Gated by ``RAG_CONSOLE_REMOTE_VIEW_ENABLED`` (default off) so flipping on
    open-redirect behaviour is opt-in. ``RAG_CONSOLE_REMOTE_VIEW_HOST_ALLOWLIST``
    (comma-separated host suffixes) optionally narrows further. Returns the
    original URI when allowed, ``None`` otherwise.
    """
    if not source_uri:
        return None
    enabled = os.environ.get("RAG_CONSOLE_REMOTE_VIEW_ENABLED", "false").lower() in {
        "1", "true", "yes", "on",
    }
    if not enabled:
        return None
    parsed = urlparse(source_uri)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    allowlist = os.environ.get("RAG_CONSOLE_REMOTE_VIEW_HOST_ALLOWLIST", "").strip()
    if allowlist:
        suffixes = [s.strip().lower() for s in allowlist.split(",") if s.strip()]
        host = parsed.netloc.lower()
        if not any(host == suf or host.endswith("." + suf) for suf in suffixes):
            return None
    return source_uri


def resolve_console_source_path(source: str | None, source_uri: str | None) -> Path:
    """Resolve console source reference to a local file under an allowed root."""
    if source_uri:
        parsed = urlparse(source_uri)
        if parsed.scheme and parsed.scheme != "file":
            raise HTTPException(status_code=404, detail="Source URI is not a local file")
        candidate = Path(unquote(parsed.path)) if parsed.scheme == "file" else Path(source_uri)
    elif source:
        candidate = DOCUMENTS_DIR / Path(source).name
    else:
        raise HTTPException(status_code=400, detail="source or source_uri is required")

    try:
        resolved = candidate.resolve()
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Invalid source path: {candidate}"
        ) from exc

    roots = _allowed_source_roots()
    if not any(str(resolved).startswith(str(root)) for root in roots):
        roots_str = ", ".join(str(r) for r in roots)
        raise HTTPException(
            status_code=400,
            detail=(
                f"Source path {resolved} is outside allowed roots ({roots_str}). "
                "Set RAG_CONSOLE_SOURCE_ROOTS to allow additional directories."
            ),
        )
    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=404, detail=f"Source document not found: {resolved}")
    return resolved


def build_source_preview_payload(
    *,
    target: Path,
    source_uri: str | None,
    text: str,
    start: int | None,
    end: int | None,
    context_chars: int,
    max_chars: int,
) -> dict:
    """Build payload for `/console/source-document` preview endpoint."""
    total_chars = len(text)
    # Deliberate preview-size guardrails clamping user-supplied params (200..20000 chars).
    effective_max_chars = max(
        RAG_CONSOLE_PREVIEW_MIN_CHARS, min(max_chars, RAG_CONSOLE_PREVIEW_MAX_CHARS_CAP)
    )
    preview_start = 0
    preview_end = min(total_chars, effective_max_chars)
    highlight_start = None
    highlight_end = None

    if start is not None and end is not None and end > start:
        safe_start = max(0, min(start, total_chars))
        safe_end = max(safe_start, min(end, total_chars))
        # Deliberate preview-size guardrails clamping user-supplied params (100..5000 chars).
        context = max(
            RAG_CONSOLE_PREVIEW_CONTEXT_MIN, min(context_chars, RAG_CONSOLE_PREVIEW_CONTEXT_CAP)
        )
        preview_start = max(0, safe_start - context)
        preview_end = min(total_chars, safe_end + context)
        if preview_end - preview_start > effective_max_chars:
            preview_end = min(total_chars, preview_start + effective_max_chars)
            if preview_end < safe_end:
                preview_end = safe_end
                preview_start = max(0, preview_end - effective_max_chars)
        highlight_start = safe_start
        highlight_end = safe_end

    clipped = text[preview_start:preview_end]
    return {
        "source": target.name,
        "path": str(target),
        "source_uri": source_uri or target.as_uri(),
        "preview": clipped,
        "truncated": preview_start > 0 or preview_end < total_chars,
        "total_chars": total_chars,
        "preview_start": preview_start,
        "preview_end": preview_end,
        "highlight_start": highlight_start,
        "highlight_end": highlight_end,
    }


def read_clean_document_from_minio(source_key: str) -> tuple[str, dict]:
    """Read clean markdown + metadata from MinIO by source_key.

    Two MinIO layouts can hold a document's clean markdown, and they are
    populated by different paths:

    1. **Document store** (``<document_id>.md`` where
       ``document_id = build_document_id(source_key)``) — written by the
       embedding pipeline's ``commit_node`` on every normal ingest when
       ``store_documents`` is enabled (the default). This is the layout the
       CLI/Temporal ingest actually fills, for *all* formats (the clean
       markdown rendering of pdf/docx/pptx/xlsx, not just .md sources).
    2. **MinioCleanStore** (``clean/{safe_key}.md``) — only populated by the
       lifecycle tooling (migration/sync). Normal ingest never writes it.

    We therefore try the document store first (so document viewing works on a
    standard ingest with no backfill), and fall back to ``MinioCleanStore`` for
    environments where lifecycle migration populated it instead.

    Returns:
        (markdown_text, metadata_dict)

    Raises:
        HTTPException(404) if MinIO is unreachable or the document is missing
        from both layouts.
    """
    try:
        from src.db import build_document_id, get_document
        from src.db.minio import create_client
        from src.ingest.common.minio_clean_store import MinioCleanStore
        from config.settings import MINIO_BUCKET
    except Exception as exc:
        logger.warning("minio_clean_store_import_failed error=%s", exc)
        raise HTTPException(status_code=404, detail="Clean document store unavailable") from exc

    try:
        client = create_client()

        # Primary: the document store written by normal ingest (commit_node).
        # ``get_document`` returns a ``StoredDocument`` dataclass (attribute
        # access — NOT a dict), or ``None`` when the id is absent.
        document_id = build_document_id(source_key)
        doc = get_document(client, document_id, MINIO_BUCKET)
        if doc is not None and getattr(doc, "content", None):
            return doc.content, dict(getattr(doc, "metadata", None) or {})

        # Fallback: lifecycle-populated MinioCleanStore (clean/ prefix).
        store = MinioCleanStore(client, MINIO_BUCKET)
        if store.exists(source_key):
            text, meta = store.read(source_key)
            return text, meta

        raise HTTPException(
            status_code=404,
            detail=(
                f"Clean document not found in MinIO "
                f"(document_id={document_id}, source_key={source_key!r})"
            ),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("minio_clean_store_read_failed source_key=%s error=%s", source_key, exc)
        raise HTTPException(status_code=404, detail="Failed to read clean document") from exc


def _render_markdown_to_html(md_text: str) -> str:
    """Render markdown to sanitized HTML using markdown-it-py.

    Falls back to escaped <pre> on import failure.
    """
    try:
        from markdown_it import MarkdownIt
    except Exception:
        return f"<pre>{escape(md_text)}</pre>"
    md = MarkdownIt("commonmark", {"html": False, "linkify": True, "breaks": True}).enable("table")
    return md.render(md_text)


def render_source_document_html(
    *,
    target: Path,
    text: str,
    start: int | None,
    end: int | None,
    chunk: int | None,
    is_markdown: bool = False,
    title: str | None = None,
    chunk_text: str | None = None,
) -> str:
    """Render HTML document with optional highlighted source range.

    When ``is_markdown`` is True, the document is rendered as parsed markdown
    (e.g., the clean markdown from MinIO). The cited [start..end] range is
    shown as a quoted excerpt panel above the rendered body, which avoids
    breaking markdown structure with raw <mark> insertion.
    """
    total_chars = len(text)
    safe_start = 0
    safe_end = 0
    has_range = start is not None and end is not None and end > start
    if has_range:
        safe_start = max(0, min(start, total_chars))
        safe_end = max(safe_start, min(end, total_chars))

    excerpt_html = ""
    # Prefer the explicit chunk_text from retrieval — guarantees parity with the
    # citation card. Fall back to slicing the document by offsets.
    chosen = (chunk_text or "").strip()
    if not chosen and has_range:
        chosen = text[safe_start:safe_end].strip()
    if chosen:
        rendered = _render_markdown_to_html(chosen) if is_markdown else f"<pre>{escape(chosen)}</pre>"
        excerpt_html = f"<div class='excerpt-label'>Cited chunk</div><blockquote class='excerpt'>{rendered}</blockquote>"

    if is_markdown:
        body = _render_markdown_to_html(text)
        body_class = "doc"
    elif has_range:
        before = escape(text[:safe_start])
        highlighted = escape(text[safe_start:safe_end]) or "&nbsp;"
        after = escape(text[safe_end:])
        body = f"{before}<mark>{highlighted}</mark>{after}"
        body_class = "mono"
    else:
        body = escape(text)
        body_class = "mono"

    display_name = title or target.name
    chunk_label = f"Chunk {chunk}" if chunk is not None and chunk > 0 else "Document View"
    range_label = f"chars {safe_start}..{safe_end}" if has_range else "no highlight range provided"
    source_label = "MinIO clean store" if is_markdown else str(target)
    return (
        "<!doctype html><html><head><meta charset='utf-8' />"
        f"<title>{escape(display_name)} - {escape(chunk_label)}</title>"
        "<style>"
        "body{font-family:ui-sans-serif,system-ui;background:#0f1115;color:#e7ecf3;margin:0;padding:16px;max-width:920px;margin-left:auto;margin-right:auto;}"
        ".meta{margin-bottom:14px;padding:10px 12px;border:1px solid #2a3040;border-radius:8px;background:#171a21;font-size:13px;color:#aebbcc;}"
        ".meta strong{color:#e7ecf3;font-size:14px;}"
        ".excerpt-label{font-size:11px;font-weight:600;letter-spacing:0.08em;color:#4da3ff;text-transform:uppercase;margin-bottom:4px;}"
        ".excerpt{margin:0 0 16px 0;padding:10px 14px;border-left:3px solid #ffe08a;background:#1a1f2b;border-radius:0 6px 6px 0;color:#e7ecf3;font-size:13px;line-height:1.5;}"
        ".excerpt p{margin:0.5em 0;}.excerpt p:first-child{margin-top:0;}.excerpt p:last-child{margin-bottom:0;}"
        ".excerpt pre{background:#0a0d14;border:1px solid #2a3040;border-radius:6px;padding:8px;overflow-x:auto;margin:0;white-space:pre-wrap;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;color:#aebbcc;}"
        ".excerpt table{border-collapse:collapse;margin:6px 0;font-size:12px;}.excerpt th,.excerpt td{border:1px solid #2a3040;padding:4px 8px;}.excerpt th{background:#0f1115;}"
        ".mono{white-space:pre-wrap;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;line-height:1.45;border:1px solid #2a3040;border-radius:8px;padding:12px;background:#111522;}"
        ".doc{line-height:1.65;font-size:15px;}"
        ".doc h1,.doc h2,.doc h3,.doc h4{color:#e7ecf3;margin-top:1.4em;}"
        ".doc h1{font-size:1.6em;border-bottom:1px solid #2a3040;padding-bottom:6px;}"
        ".doc h2{font-size:1.3em;border-bottom:1px solid #2a3040;padding-bottom:4px;}"
        ".doc h3{font-size:1.1em;}"
        ".doc p{margin:0.7em 0;}"
        ".doc code{background:#1a1f2b;border:1px solid #2a3040;border-radius:4px;padding:1px 5px;font-size:0.9em;color:#ffd28a;}"
        ".doc pre{background:#0a0d14;border:1px solid #2a3040;border-radius:8px;padding:12px;overflow-x:auto;}"
        ".doc pre code{background:none;border:none;padding:0;color:#e7ecf3;}"
        ".doc blockquote{border-left:3px solid #4da3ff;padding-left:12px;color:#aebbcc;margin:0.7em 0;}"
        ".doc table{border-collapse:collapse;margin:0.7em 0;}"
        ".doc th,.doc td{border:1px solid #2a3040;padding:6px 10px;}"
        ".doc th{background:#171a21;}"
        ".doc ul,.doc ol{padding-left:24px;}"
        ".doc img{max-width:100%;}"
        "mark{background:#ffe08a;color:#111;padding:0 2px;border-radius:2px;}"
        "a{color:#4da3ff;}"
        "</style></head><body>"
        f"<div class='meta'><strong>{escape(display_name)}</strong><br/>"
        f"<span>{escape(source_label)}</span><br/>"
        f"<span>{escape(chunk_label)} | {escape(range_label)}</span><br/>"
        f"<span>total chars: {total_chars}</span></div>"
        f"{excerpt_html}"
        f"<div class='{body_class}'>{body}</div>"
        "</body></html>"
    )

