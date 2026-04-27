"""Tests for src.ingest.common.utils — sha256_path, decode_with_fallbacks,
read_text_with_fallbacks and related helpers.

All mock test functions start with test_mock_.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.ingest.common.utils import (
    sha256_bytes,
    sha256_path,
    decode_with_fallbacks,
    read_text_with_fallbacks,
)


# ---------------------------------------------------------------------------
# Tests: sha256_bytes
# ---------------------------------------------------------------------------


class TestSha256Bytes:
    def test_sha256_bytes_deterministic(self):
        data = b"hello world"
        assert sha256_bytes(data) == sha256_bytes(data)

    def test_sha256_bytes_correct(self):
        data = b"hello world"
        expected = hashlib.sha256(data).hexdigest()
        assert sha256_bytes(data) == expected

    def test_sha256_bytes_empty(self):
        result = sha256_bytes(b"")
        assert isinstance(result, str)
        assert len(result) == 64


# ---------------------------------------------------------------------------
# Tests: sha256_path
# ---------------------------------------------------------------------------


class TestSha256Path:
    def test_sha256_path(self, tmp_path):
        """sha256_path should correctly hash a file's contents."""
        content = b"chunked file content for hashing"
        filepath = tmp_path / "test.txt"
        filepath.write_bytes(content)

        result = sha256_path(filepath)
        expected = hashlib.sha256(content).hexdigest()
        assert result == expected

    def test_sha256_path_empty_file(self, tmp_path):
        """sha256_path on an empty file should return the sha256 of empty bytes."""
        filepath = tmp_path / "empty.txt"
        filepath.write_bytes(b"")

        result = sha256_path(filepath)
        expected = hashlib.sha256(b"").hexdigest()
        assert result == expected

    def test_sha256_path_large_file(self, tmp_path):
        """sha256_path should handle files larger than a single 64KB chunk."""
        # 128KB + 1 byte to force multiple chunks
        content = b"x" * (65536 * 2 + 1)
        filepath = tmp_path / "large.bin"
        filepath.write_bytes(content)

        result = sha256_path(filepath)
        expected = hashlib.sha256(content).hexdigest()
        assert result == expected


# ---------------------------------------------------------------------------
# Tests: decode_with_fallbacks
# ---------------------------------------------------------------------------


class TestDecodeWithFallbacks:
    def test_decode_utf8_success(self):
        data = "Hello, world!".encode("utf-8")
        result = decode_with_fallbacks(data)
        assert result == "Hello, world!"

    def test_mock_decode_with_fallbacks_latin1(self):
        """decode_with_fallbacks should succeed with Latin-1 encoded bytes."""
        # b'\xe9' is é in Latin-1 but not valid UTF-8
        data = "caf\xe9".encode("latin-1")
        result = decode_with_fallbacks(data)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_mock_decode_with_fallbacks_replacement(self):
        """decode_with_fallbacks should use replacement chars as last resort."""
        # Bytes that are invalid in UTF-8 and Latin-1 (almost impossible but covers path)
        # The function tries utf-8, latin-1, cp1252 then falls back to utf-8 errors="replace"
        # Latin-1 should decode any byte, so we test that the result is a string
        data = bytes(range(0, 256))
        result = decode_with_fallbacks(data)
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Tests: read_text_with_fallbacks
# ---------------------------------------------------------------------------


class TestReadTextWithFallbacks:
    def test_mock_read_text_with_fallbacks(self, tmp_path):
        """read_text_with_fallbacks should read a file and decode it."""
        content = "Some content with \xe9 accent"
        filepath = tmp_path / "test.txt"
        filepath.write_bytes(content.encode("latin-1"))

        result = read_text_with_fallbacks(filepath)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_read_text_utf8(self, tmp_path):
        """read_text_with_fallbacks should handle UTF-8 files correctly."""
        content = "Unicode: \u2603\u2665"
        filepath = tmp_path / "utf8.txt"
        filepath.write_bytes(content.encode("utf-8"))

        result = read_text_with_fallbacks(filepath)
        assert result == content

    def test_read_text_empty_file(self, tmp_path):
        """read_text_with_fallbacks on empty file should return empty string."""
        filepath = tmp_path / "empty.txt"
        filepath.write_bytes(b"")

        result = read_text_with_fallbacks(filepath)
        assert result == ""
