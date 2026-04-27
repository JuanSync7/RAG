"""Tests for src/ingest/common/minio_clean_store.py — mock-based."""

from __future__ import annotations

import io
from unittest.mock import MagicMock, call, patch

import pytest

from src.ingest.common.minio_clean_store import MinioCleanStore, _safe_key


# ---------------------------------------------------------------------------
# Pure logic: _safe_key()
# ---------------------------------------------------------------------------


def test_mock_safe_key_replaces_unsafe_chars():
    assert _safe_key("path/to:file*name") == "path_to_file_name"


def test_mock_safe_key_replaces_double_dot():
    # "/" replaced first by _UNSAFE_CHARS, then ".." replaced → "__"
    result = _safe_key("../../../etc/passwd")
    assert ".." not in result
    assert "etc" in result


def test_mock_safe_key_replaces_question_mark():
    assert _safe_key("file?query") == "file_query"


def test_mock_safe_key_clean_key_unchanged():
    assert _safe_key("docs_clean_report") == "docs_clean_report"


def test_mock_safe_key_replaces_dotdot():
    result = _safe_key("a..b")
    assert ".." not in result


# ---------------------------------------------------------------------------
# Static methods: _object_key_md() and _object_key_meta()
# ---------------------------------------------------------------------------


def test_mock_object_key_md_format():
    key = MinioCleanStore._object_key_md("my/doc.txt")
    assert key.startswith("clean/")
    assert key.endswith(".md")


def test_mock_object_key_meta_format():
    key = MinioCleanStore._object_key_meta("my/doc.txt")
    assert key.startswith("clean/")
    assert key.endswith(".meta.json")


def test_mock_object_key_md_sanitizes_source():
    key = MinioCleanStore._object_key_md("path/to/file")
    assert "/" not in key[len("clean/"):]  # slash replaced in the safe_key portion


def test_mock_object_key_meta_sanitizes_source():
    key = MinioCleanStore._object_key_meta("path/to/file")
    assert "/" not in key[len("clean/"):]


# ---------------------------------------------------------------------------
# write() — mock-based
# ---------------------------------------------------------------------------


def _make_store() -> tuple[MinioCleanStore, MagicMock]:
    client = MagicMock()
    store = MinioCleanStore(client=client, bucket="test-bucket")
    return store, client


def test_mock_write_markdown_written_first_meta_second():
    store, client = _make_store()
    meta = {"source_hash": "abc", "clean_hash": "def", "schema_version": 1, "trace_id": "t1"}
    store.write("doc/report", "# Hello\n\nContent", meta)

    assert client.put_object.call_count == 2
    first_call_args = client.put_object.call_args_list[0]
    second_call_args = client.put_object.call_args_list[1]

    # First call: .md
    assert first_call_args[0][1].endswith(".md")
    assert first_call_args[1]["content_type"] == "text/markdown"
    # Second call: .meta.json
    assert second_call_args[0][1].endswith(".meta.json")
    assert second_call_args[1]["content_type"] == "application/json"


def test_mock_write_adds_created_at_if_missing():
    store, client = _make_store()
    meta = {"source_hash": "x"}
    store.write("k", "text", meta)
    assert "created_at" in meta


def test_mock_write_preserves_existing_created_at():
    store, client = _make_store()
    meta = {"created_at": "2024-01-01T00:00:00+00:00", "source_hash": "x"}
    store.write("k", "text", meta)
    assert meta["created_at"] == "2024-01-01T00:00:00+00:00"


# ---------------------------------------------------------------------------
# read() — mock-based
# ---------------------------------------------------------------------------


def test_mock_read_returns_text_and_meta():
    store, client = _make_store()

    md_resp = MagicMock()
    md_resp.read.return_value = b"# Hello"
    md_resp.close = MagicMock()
    md_resp.release_conn = MagicMock()

    import orjson

    meta_dict = {"source_hash": "abc", "trace_id": "t1"}
    meta_resp = MagicMock()
    meta_resp.read.return_value = orjson.dumps(meta_dict)
    meta_resp.close = MagicMock()
    meta_resp.release_conn = MagicMock()

    client.get_object.side_effect = [md_resp, meta_resp]

    text, meta = store.read("doc/report")

    assert text == "# Hello"
    assert meta["source_hash"] == "abc"
    md_resp.close.assert_called_once()
    md_resp.release_conn.assert_called_once()
    meta_resp.close.assert_called_once()
    meta_resp.release_conn.assert_called_once()


# ---------------------------------------------------------------------------
# exists() — mock-based
# ---------------------------------------------------------------------------


def test_mock_exists_returns_true_when_stat_succeeds():
    store, client = _make_store()
    client.stat_object.return_value = MagicMock()
    assert store.exists("doc/report") is True


def test_mock_exists_returns_false_when_stat_raises():
    store, client = _make_store()
    client.stat_object.side_effect = Exception("Not found")
    assert store.exists("doc/report") is False


# ---------------------------------------------------------------------------
# delete() — mock-based
# ---------------------------------------------------------------------------


def test_mock_delete_calls_remove_object_twice():
    store, client = _make_store()
    client.remove_object.return_value = None
    store.delete("doc/report")
    assert client.remove_object.call_count == 2


def test_mock_delete_exception_does_not_raise(caplog):
    import logging
    store, client = _make_store()
    client.remove_object.side_effect = Exception("Storage error")
    # Should not raise
    store.delete("doc/report")
    # Errors should be logged as warnings
    assert client.remove_object.call_count == 2


# ---------------------------------------------------------------------------
# soft_delete() — mock-based
# ---------------------------------------------------------------------------


def test_mock_soft_delete_exception_does_not_raise():
    """soft_delete swallows per-object exceptions per NFR-3210."""
    store, client = _make_store()
    client.copy_object.side_effect = Exception("Copy error")
    # Should not raise even when copy fails
    store.soft_delete("doc/report")


def test_mock_soft_delete_copies_then_removes(monkeypatch):
    """Inject a stub CopySource so we can verify copy_object + remove_object are called."""
    import minio.commonconfig as mc_mod
    # Inject a no-op CopySource into the stub module so the inline import succeeds
    FakeCopySource = type("CopySource", (), {"__init__": lambda self, *a, **kw: None})
    monkeypatch.setattr(mc_mod, "CopySource", FakeCopySource, raising=False)

    store, client = _make_store()
    client.copy_object.return_value = None
    client.remove_object.return_value = None

    store.soft_delete("mykey")

    assert client.copy_object.call_count == 2
    assert client.remove_object.call_count == 2
    dst_keys = [c[0][1] for c in client.copy_object.call_args_list]
    for dst_key in dst_keys:
        assert dst_key.startswith("deleted/")


# ---------------------------------------------------------------------------
# list_keys() — mock-based
# ---------------------------------------------------------------------------


def test_mock_list_keys_returns_safe_keys():
    store, client = _make_store()

    obj1 = MagicMock()
    obj1.object_name = "clean/doc_report.meta.json"
    obj2 = MagicMock()
    obj2.object_name = "clean/another_file.meta.json"
    obj3 = MagicMock()
    obj3.object_name = "clean/ignore_this.md"  # should be ignored (not .meta.json)

    client.list_objects.return_value = [obj1, obj2, obj3]

    keys = store.list_keys()
    assert "doc_report" in keys
    assert "another_file" in keys
    assert len(keys) == 2


def test_mock_list_keys_empty_bucket():
    store, client = _make_store()
    client.list_objects.return_value = []
    assert store.list_keys() == []


def test_mock_list_keys_uses_clean_prefix():
    store, client = _make_store()
    client.list_objects.return_value = []
    store.list_keys()
    call_kwargs = client.list_objects.call_args
    # Accept either positional or keyword argument for prefix
    positional_args = call_kwargs[0] if call_kwargs[0] else ()
    keyword_args = call_kwargs[1] if call_kwargs[1] else {}
    prefix_used = (
        (len(positional_args) > 1 and positional_args[1] == "clean/")
        or keyword_args.get("prefix") == "clean/"
    )
    assert prefix_used, f"Expected 'clean/' prefix, got args={positional_args} kwargs={keyword_args}"
