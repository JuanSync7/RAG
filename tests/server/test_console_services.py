"""Unit tests for server.console.services console helpers.

Covers path-traversal guards, open-redirect gate, source-path resolution,
preview windowing arithmetic, log tailing, HTML rendering, MinIO clean-store
reads, and Ollama reachability. Security-relevant guards are exercised first.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

import server.console.services as services


# --------------------------------------------------------------------------- #
# Static asset resolution (path-traversal guard) — SECURITY
# --------------------------------------------------------------------------- #


def test_resolve_console_static_asset_happy(monkeypatch, tmp_path):
    monkeypatch.setattr(services, "CONSOLE_STATIC_DIR", tmp_path)
    asset = tmp_path / "app.js"
    asset.write_text("console.log(1)")
    resolved = services.resolve_console_static_asset("app.js")
    assert resolved == asset.resolve()


def test_resolve_user_console_static_asset_happy(monkeypatch, tmp_path):
    monkeypatch.setattr(services, "USER_CONSOLE_STATIC_DIR", tmp_path)
    asset = tmp_path / "index.html"
    asset.write_text("<html></html>")
    resolved = services.resolve_user_console_static_asset("index.html")
    assert resolved == asset.resolve()


def test_resolve_console_static_asset_traversal_escape_404(monkeypatch, tmp_path):
    static_root = tmp_path / "static"
    static_root.mkdir()
    # Create a secret OUTSIDE the static root that "../secret" would resolve to.
    (tmp_path / "secret").write_text("top secret")
    monkeypatch.setattr(services, "CONSOLE_STATIC_DIR", static_root)
    with pytest.raises(HTTPException) as exc:
        services.resolve_console_static_asset("../secret")
    assert exc.value.status_code == 404


def test_resolve_user_console_static_asset_traversal_escape_404(monkeypatch, tmp_path):
    static_root = tmp_path / "static"
    static_root.mkdir()
    (tmp_path / "secret").write_text("top secret")
    monkeypatch.setattr(services, "USER_CONSOLE_STATIC_DIR", static_root)
    with pytest.raises(HTTPException) as exc:
        services.resolve_user_console_static_asset("../secret")
    assert exc.value.status_code == 404


def test_resolve_console_static_asset_absolute_escape_404(monkeypatch, tmp_path):
    static_root = tmp_path / "static"
    static_root.mkdir()
    monkeypatch.setattr(services, "CONSOLE_STATIC_DIR", static_root)
    # An absolute path escaping the root must be rejected by the prefix guard.
    with pytest.raises(HTTPException) as exc:
        services.resolve_console_static_asset("../../etc/passwd")
    assert exc.value.status_code == 404


def test_resolve_console_static_asset_missing_404(monkeypatch, tmp_path):
    monkeypatch.setattr(services, "CONSOLE_STATIC_DIR", tmp_path)
    with pytest.raises(HTTPException) as exc:
        services.resolve_console_static_asset("nope.js")
    assert exc.value.status_code == 404


def test_resolve_console_static_asset_directory_not_file_404(monkeypatch, tmp_path):
    monkeypatch.setattr(services, "CONSOLE_STATIC_DIR", tmp_path)
    (tmp_path / "subdir").mkdir()
    with pytest.raises(HTTPException) as exc:
        services.resolve_console_static_asset("subdir")
    assert exc.value.status_code == 404


# --------------------------------------------------------------------------- #
# is_remote_view_uri — open-redirect gate — SECURITY
# --------------------------------------------------------------------------- #


def test_is_remote_view_uri_none_input(monkeypatch):
    monkeypatch.delenv("RAG_CONSOLE_REMOTE_VIEW_ENABLED", raising=False)
    assert services.is_remote_view_uri(None) is None
    assert services.is_remote_view_uri("") is None


def test_is_remote_view_uri_disabled_by_default(monkeypatch):
    monkeypatch.delenv("RAG_CONSOLE_REMOTE_VIEW_ENABLED", raising=False)
    # Even a perfectly valid https URL is rejected when the flag is unset.
    assert services.is_remote_view_uri("https://example.com/doc.pdf") is None


def test_is_remote_view_uri_enabled_valid_https(monkeypatch):
    monkeypatch.setenv("RAG_CONSOLE_REMOTE_VIEW_ENABLED", "true")
    monkeypatch.delenv("RAG_CONSOLE_REMOTE_VIEW_HOST_ALLOWLIST", raising=False)
    uri = "https://example.com/doc.pdf"
    assert services.is_remote_view_uri(uri) == uri


def test_is_remote_view_uri_enabled_non_http_scheme(monkeypatch):
    monkeypatch.setenv("RAG_CONSOLE_REMOTE_VIEW_ENABLED", "1")
    monkeypatch.delenv("RAG_CONSOLE_REMOTE_VIEW_HOST_ALLOWLIST", raising=False)
    assert services.is_remote_view_uri("ftp://example.com/doc.pdf") is None


def test_is_remote_view_uri_enabled_no_netloc(monkeypatch):
    monkeypatch.setenv("RAG_CONSOLE_REMOTE_VIEW_ENABLED", "yes")
    monkeypatch.delenv("RAG_CONSOLE_REMOTE_VIEW_HOST_ALLOWLIST", raising=False)
    # Scheme present but no netloc (e.g., a bare path) -> rejected.
    assert services.is_remote_view_uri("http:///nohost/path") is None


def test_is_remote_view_uri_allowlist_exact_host_match(monkeypatch):
    monkeypatch.setenv("RAG_CONSOLE_REMOTE_VIEW_ENABLED", "on")
    monkeypatch.setenv("RAG_CONSOLE_REMOTE_VIEW_HOST_ALLOWLIST", "example.com")
    uri = "https://example.com/doc.pdf"
    assert services.is_remote_view_uri(uri) == uri


def test_is_remote_view_uri_allowlist_dot_suffix_match(monkeypatch):
    monkeypatch.setenv("RAG_CONSOLE_REMOTE_VIEW_ENABLED", "true")
    monkeypatch.setenv("RAG_CONSOLE_REMOTE_VIEW_HOST_ALLOWLIST", "example.com")
    uri = "https://docs.example.com/doc.pdf"
    assert services.is_remote_view_uri(uri) == uri


def test_is_remote_view_uri_allowlist_no_match(monkeypatch):
    monkeypatch.setenv("RAG_CONSOLE_REMOTE_VIEW_ENABLED", "true")
    monkeypatch.setenv("RAG_CONSOLE_REMOTE_VIEW_HOST_ALLOWLIST", "example.com")
    # evil-example.com must NOT match the "example.com" suffix.
    assert services.is_remote_view_uri("https://evilexample.com/doc.pdf") is None
    assert services.is_remote_view_uri("https://example.com.evil.net/x") is None


# --------------------------------------------------------------------------- #
# resolve_console_source_path — SECURITY
# --------------------------------------------------------------------------- #


def test_resolve_console_source_path_neither_arg_400(monkeypatch, tmp_path):
    monkeypatch.setattr(services, "DOCUMENTS_DIR", tmp_path)
    with pytest.raises(HTTPException) as exc:
        services.resolve_console_source_path(None, None)
    assert exc.value.status_code == 400


def test_resolve_console_source_path_non_file_scheme_404(monkeypatch, tmp_path):
    monkeypatch.setattr(services, "DOCUMENTS_DIR", tmp_path)
    with pytest.raises(HTTPException) as exc:
        services.resolve_console_source_path(None, "https://example.com/doc.pdf")
    assert exc.value.status_code == 404


def test_resolve_console_source_path_file_uri_under_root(monkeypatch, tmp_path):
    monkeypatch.setattr(services, "DOCUMENTS_DIR", tmp_path)
    monkeypatch.delenv("RAG_CONSOLE_SOURCE_ROOTS", raising=False)
    doc = tmp_path / "doc.txt"
    doc.write_text("hello")
    resolved = services.resolve_console_source_path(None, doc.resolve().as_uri())
    assert resolved == doc.resolve()


def test_resolve_console_source_path_outside_roots_400(monkeypatch, tmp_path):
    roots = tmp_path / "allowed"
    roots.mkdir()
    monkeypatch.setattr(services, "DOCUMENTS_DIR", roots)
    monkeypatch.delenv("RAG_CONSOLE_SOURCE_ROOTS", raising=False)
    outside = tmp_path / "outside.txt"
    outside.write_text("nope")
    with pytest.raises(HTTPException) as exc:
        services.resolve_console_source_path(None, outside.resolve().as_uri())
    assert exc.value.status_code == 400


def test_resolve_console_source_path_under_root_missing_404(monkeypatch, tmp_path):
    monkeypatch.setattr(services, "DOCUMENTS_DIR", tmp_path)
    monkeypatch.delenv("RAG_CONSOLE_SOURCE_ROOTS", raising=False)
    missing = tmp_path / "ghost.txt"
    with pytest.raises(HTTPException) as exc:
        services.resolve_console_source_path(None, missing.resolve().as_uri())
    assert exc.value.status_code == 404


def test_resolve_console_source_path_source_name_under_documents(monkeypatch, tmp_path):
    monkeypatch.setattr(services, "DOCUMENTS_DIR", tmp_path)
    monkeypatch.delenv("RAG_CONSOLE_SOURCE_ROOTS", raising=False)
    doc = tmp_path / "report.txt"
    doc.write_text("body")
    # `source` is treated as a bare name resolved under DOCUMENTS_DIR.
    resolved = services.resolve_console_source_path("/anything/report.txt", None)
    assert resolved == doc.resolve()


# --------------------------------------------------------------------------- #
# _allowed_source_roots
# --------------------------------------------------------------------------- #


def test_allowed_source_roots_default(monkeypatch, tmp_path):
    monkeypatch.setattr(services, "DOCUMENTS_DIR", tmp_path)
    monkeypatch.delenv("RAG_CONSOLE_SOURCE_ROOTS", raising=False)
    roots = services._allowed_source_roots()
    assert roots == [tmp_path.resolve()]


def test_allowed_source_roots_extra_colon_split(monkeypatch, tmp_path):
    monkeypatch.setattr(services, "DOCUMENTS_DIR", tmp_path)
    extra1 = tmp_path / "e1"
    extra2 = tmp_path / "e2"
    extra1.mkdir()
    extra2.mkdir()
    monkeypatch.setenv("RAG_CONSOLE_SOURCE_ROOTS", f"{extra1}:{extra2}")
    roots = services._allowed_source_roots()
    assert roots == [tmp_path.resolve(), extra1.resolve(), extra2.resolve()]


def test_allowed_source_roots_skips_empty_entries(monkeypatch, tmp_path):
    monkeypatch.setattr(services, "DOCUMENTS_DIR", tmp_path)
    extra = tmp_path / "e1"
    extra.mkdir()
    # Leading/trailing colons and whitespace-only entries are skipped.
    monkeypatch.setenv("RAG_CONSOLE_SOURCE_ROOTS", f":  :{extra}:")
    roots = services._allowed_source_roots()
    assert roots == [tmp_path.resolve(), extra.resolve()]


# --------------------------------------------------------------------------- #
# build_source_preview_payload — pure windowing arithmetic
# --------------------------------------------------------------------------- #


def test_build_preview_no_range_clamp_high(tmp_path):
    target = tmp_path / "doc.txt"
    # Text LONGER than the 20000 ceiling so the upper clamp is load-bearing.
    text = "x" * 25000
    payload = services.build_source_preview_payload(
        target=target,
        source_uri=None,
        text=text,
        start=None,
        end=None,
        context_chars=300,
        max_chars=999999,  # clamps down to 20000 ceiling
    )
    # effective_max_chars clamps to 20000; preview_end == 20000 (not 25000).
    assert payload["preview_start"] == 0
    assert payload["preview_end"] == 20000
    assert payload["preview"] == text[:20000]
    assert payload["truncated"] is True
    assert payload["total_chars"] == 25000
    assert payload["highlight_start"] is None
    assert payload["highlight_end"] is None
    assert payload["source"] == "doc.txt"
    assert payload["path"] == str(target)
    assert payload["source_uri"] == target.as_uri()
    assert set(payload) == {
        "source", "path", "source_uri", "preview", "truncated",
        "total_chars", "preview_start", "preview_end",
        "highlight_start", "highlight_end",
    }


def test_build_preview_no_range_clamp_low(tmp_path):
    target = tmp_path / "doc.txt"
    text = "y" * 1000
    payload = services.build_source_preview_payload(
        target=target,
        source_uri=None,
        text=text,
        start=None,
        end=None,
        context_chars=300,
        max_chars=5,  # clamps UP to 200
    )
    # effective_max_chars clamps to floor 200 -> preview_end == 200, truncated.
    assert payload["preview_end"] == 200
    assert payload["preview_start"] == 0
    assert payload["preview"] == text[:200]
    assert payload["truncated"] is True
    assert payload["total_chars"] == 1000


def test_build_preview_with_range_context_expansion(tmp_path):
    target = tmp_path / "doc.txt"
    text = "a" * 10000
    payload = services.build_source_preview_payload(
        target=target,
        source_uri="custom://uri",
        text=text,
        start=4000,
        end=4100,
        context_chars=500,  # within [100, 5000]
        max_chars=20000,
    )
    # safe_start=4000, safe_end=4100, context=500
    # preview_start = 4000 - 500 = 3500; preview_end = 4100 + 500 = 4600
    assert payload["preview_start"] == 3500
    assert payload["preview_end"] == 4600
    assert payload["highlight_start"] == 4000
    assert payload["highlight_end"] == 4100
    assert payload["preview"] == text[3500:4600]
    assert payload["truncated"] is True
    assert payload["source_uri"] == "custom://uri"


def test_build_preview_with_range_window_reclip(tmp_path):
    target = tmp_path / "doc.txt"
    text = "z" * 30000
    # context large enough that window exceeds effective_max_chars and re-clips.
    payload = services.build_source_preview_payload(
        target=target,
        source_uri=None,
        text=text,
        start=10000,
        end=10100,
        context_chars=5000,
        max_chars=300,  # effective_max_chars = 300
    )
    # safe_start=10000, safe_end=10100, context=5000
    # initial preview_start = 5000, preview_end = 15100, width 10100 > 300
    # reclip: preview_end = 5000 + 300 = 5300; 5300 < safe_end(10100) so:
    #   preview_end = 10100; preview_start = max(0, 10100 - 300) = 9800
    assert payload["preview_start"] == 9800
    assert payload["preview_end"] == 10100
    assert payload["highlight_start"] == 10000
    assert payload["highlight_end"] == 10100
    assert payload["preview"] == text[9800:10100]
    assert payload["truncated"] is True


def test_build_preview_range_clamped_to_bounds(tmp_path):
    target = tmp_path / "doc.txt"
    text = "b" * 100
    payload = services.build_source_preview_payload(
        target=target,
        source_uri=None,
        text=text,
        start=-50,      # clamps to 0
        end=99999,      # clamps to total_chars=100
        context_chars=10,
        max_chars=20000,
    )
    assert payload["highlight_start"] == 0
    assert payload["highlight_end"] == 100
    assert payload["preview_start"] == 0
    assert payload["preview_end"] == 100


# --------------------------------------------------------------------------- #
# tail_log_lines
# --------------------------------------------------------------------------- #


def _patch_logs_dir(monkeypatch, tmp_path):
    """Redirect tail_log_lines' computed logs dir into tmp_path/logs."""
    fake_services = tmp_path / "logs_root" / "server" / "console" / "services.py"
    fake_services.parent.mkdir(parents=True)
    fake_services.write_text("x")
    logs_dir = tmp_path / "logs_root" / "logs"
    logs_dir.mkdir()
    monkeypatch.setattr(services, "Path", lambda *a, **k: fake_services)
    return logs_dir


def test_tail_log_lines_prefixes_and_lists_files(monkeypatch, tmp_path):
    logs_dir = _patch_logs_dir(monkeypatch, tmp_path)
    (logs_dir / "query_processing.log").write_text("qline1\nqline2\n")
    (logs_dir / "server.log").write_text("sline1\n")
    resp = services.tail_log_lines(lines=120)
    assert "[query_processing.log] qline1" in resp.lines
    assert "[query_processing.log] qline2" in resp.lines
    assert "[server.log] sline1" in resp.lines
    assert any(p.endswith("query_processing.log") for p in resp.files)
    assert any(p.endswith("server.log") for p in resp.files)
    # ingest.log was not created -> not listed.
    assert not any(p.endswith("ingest.log") for p in resp.files)


def test_tail_log_lines_no_files_message(monkeypatch, tmp_path):
    _patch_logs_dir(monkeypatch, tmp_path)
    resp = services.tail_log_lines(lines=120)
    assert resp.files == []
    assert resp.lines == ["No log files found in ./logs"]


def test_tail_log_lines_deque_maxlen_truncates(monkeypatch, tmp_path):
    logs_dir = _patch_logs_dir(monkeypatch, tmp_path)
    # Each file is first sliced to its last `lines` lines, then appended to a
    # deque(maxlen=max(10, lines)). With lines=8 and three files each
    # contributing 8 lines (24 total appended), the deque caps output at 10.
    for name in ("query_processing.log", "ingest.log", "server.log"):
        body = "\n".join(f"{name[0]}{i}" for i in range(20)) + "\n"
        (logs_dir / name).write_text(body)
    resp = services.tail_log_lines(lines=8)
    # 3 files * 8 sliced lines = 24 appended, but maxlen=max(10,8)=10.
    assert len(resp.lines) == 10
    # deque keeps the LAST 10 appended -> tail of the last file (server.log).
    assert resp.lines[-1] == "[server.log] s19"


# --------------------------------------------------------------------------- #
# render_source_document_html
# --------------------------------------------------------------------------- #


def test_render_markdown_mode(tmp_path):
    html = services.render_source_document_html(
        target=tmp_path / "doc.md",
        text="# Heading\n\nbody text",
        start=None,
        end=None,
        chunk=None,
        is_markdown=True,
    )
    assert "<div class='doc'>" in html
    assert "MinIO clean store" in html  # source_label for markdown mode
    assert "Document View" in html


def test_render_mono_mode_escapes(tmp_path):
    html = services.render_source_document_html(
        target=tmp_path / "doc.txt",
        text="<script>alert(1)</script>",
        start=None,
        end=None,
        chunk=None,
        is_markdown=False,
    )
    assert "<div class='mono'>" in html
    # Raw script tag must be escaped, not injected.
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_render_highlight_mode_wraps_mark(tmp_path):
    text = "before MATCH after"
    html = services.render_source_document_html(
        target=tmp_path / "doc.txt",
        text=text,
        start=7,
        end=12,  # "MATCH"
        chunk=None,
        is_markdown=False,
    )
    assert "<mark>MATCH</mark>" in html
    assert "<div class='mono'>" in html


def test_render_highlight_escapes_injection(tmp_path):
    text = "x<script>y</script>z"
    html = services.render_source_document_html(
        target=tmp_path / "doc.txt",
        text=text,
        start=1,
        end=10,
        chunk=None,
        is_markdown=False,
    )
    assert "<script>y</script>" not in html
    assert "&lt;script&gt;" in html


def test_render_chunk_text_precedence_over_offset(tmp_path):
    text = "AAAAAAAAAAAAAAAAAAAA"
    html = services.render_source_document_html(
        target=tmp_path / "doc.txt",
        text=text,
        start=0,
        end=5,  # offset slice would be "AAAAA"
        chunk=None,
        is_markdown=False,
        chunk_text="EXPLICIT_CHUNK",
    )
    # Excerpt panel must quote the explicit chunk_text, not the offset slice.
    assert "Cited chunk" in html
    assert "EXPLICIT_CHUNK" in html


def test_render_offset_slice_excerpt_when_no_chunk_text(tmp_path):
    text = "0123456789DISTINCT9876543210"
    html = services.render_source_document_html(
        target=tmp_path / "doc.txt",
        text=text,
        start=10,
        end=18,  # "DISTINCT"
        chunk=None,
        is_markdown=False,
        chunk_text=None,
    )
    assert "Cited chunk" in html
    assert "DISTINCT" in html


def test_render_title_overrides_target_name(tmp_path):
    html = services.render_source_document_html(
        target=tmp_path / "real_name.txt",
        text="body",
        start=None,
        end=None,
        chunk=None,
        title="Custom Title",
    )
    assert "Custom Title" in html


def test_render_chunk_label_numbered(tmp_path):
    html = services.render_source_document_html(
        target=tmp_path / "doc.txt",
        text="body",
        start=None,
        end=None,
        chunk=7,
    )
    assert "Chunk 7" in html
    assert "Document View" not in html


# --------------------------------------------------------------------------- #
# read_clean_document_from_minio
# --------------------------------------------------------------------------- #


class _FakeStore:
    def __init__(self, *, exists=True, text="md", meta=None, read_raises=False):
        self._exists = exists
        self._text = text
        self._meta = meta or {"k": "v"}
        self._read_raises = read_raises

    def exists(self, key):
        return self._exists

    def read(self, key):
        if self._read_raises:
            raise RuntimeError("boom")
        return self._text, self._meta


def _install_minio(monkeypatch, store):
    monkeypatch.setattr("src.db.minio.create_client", lambda *a, **k: object())
    monkeypatch.setattr(
        "src.ingest.common.minio_clean_store.MinioCleanStore",
        lambda client, bucket: store,
    )


def test_read_clean_document_happy(monkeypatch):
    store = _FakeStore(exists=True, text="# clean", meta={"source": "x"})
    _install_minio(monkeypatch, store)
    text, meta = services.read_clean_document_from_minio("key1")
    assert text == "# clean"
    assert meta == {"source": "x"}


def test_read_clean_document_not_exists_404(monkeypatch):
    store = _FakeStore(exists=False)
    _install_minio(monkeypatch, store)
    with pytest.raises(HTTPException) as exc:
        services.read_clean_document_from_minio("key1")
    assert exc.value.status_code == 404


def test_read_clean_document_read_error_404(monkeypatch):
    store = _FakeStore(exists=True, read_raises=True)
    _install_minio(monkeypatch, store)
    with pytest.raises(HTTPException) as exc:
        services.read_clean_document_from_minio("key1")
    assert exc.value.status_code == 404


def test_read_clean_document_create_client_error_404(monkeypatch):
    # create_client raising is caught by the SECOND try -> 404 "Failed to read".
    def _boom(*a, **k):
        raise RuntimeError("no minio")

    monkeypatch.setattr("src.db.minio.create_client", _boom)
    monkeypatch.setattr(
        "src.ingest.common.minio_clean_store.MinioCleanStore",
        lambda client, bucket: _FakeStore(),
    )
    with pytest.raises(HTTPException) as exc:
        services.read_clean_document_from_minio("key1")
    assert exc.value.status_code == 404


# --------------------------------------------------------------------------- #
# is_ollama_reachable
# --------------------------------------------------------------------------- #


class _UrlopenCtx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_is_ollama_reachable_success(monkeypatch):
    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _UrlopenCtx())
    assert services.is_ollama_reachable() is True


def test_is_ollama_reachable_failure(monkeypatch):
    import urllib.request

    def _boom(*a, **k):
        raise OSError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    assert services.is_ollama_reachable() is False
