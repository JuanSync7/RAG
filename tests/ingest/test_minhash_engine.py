"""Tests for src.ingest.embedding.support.minhash_engine.

All tests are pure-logic or use mock for the Weaviate client path.
datasketch IS installed in this environment.
"""

from __future__ import annotations

import sys
import types
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _import_module():
    from src.ingest.embedding.support import minhash_engine
    return minhash_engine


@contextmanager
def _weaviate_filter_stub():
    """Context manager that injects a stub weaviate.classes.query module
    and restores the original sys.modules state on exit."""
    filter_stub = MagicMock()
    filter_stub.by_property.return_value.is_not_none.return_value = MagicMock()

    weaviate_query = types.ModuleType("weaviate.classes.query")
    weaviate_query.Filter = filter_stub

    # Snapshot current state so we can restore it
    saved = {
        "weaviate": sys.modules.get("weaviate"),
        "weaviate.classes": sys.modules.get("weaviate.classes"),
        "weaviate.classes.query": sys.modules.get("weaviate.classes.query"),
    }

    # Inject stubs only for keys that aren't already real modules
    if "weaviate" not in sys.modules:
        weaviate_mod = types.ModuleType("weaviate")
        weaviate_classes = types.ModuleType("weaviate.classes")
        weaviate_mod.classes = weaviate_classes
        sys.modules["weaviate"] = weaviate_mod
        sys.modules["weaviate.classes"] = weaviate_classes
    # Always override the query submodule so our Filter stub is used
    sys.modules["weaviate.classes.query"] = weaviate_query

    try:
        yield filter_stub
    finally:
        # Restore original state
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


def _make_mock_client(objects):
    """Build a minimal mock Weaviate client (no sys.modules side-effects)."""
    client = MagicMock()
    collection = MagicMock()
    client.collections.get.return_value = collection
    results = MagicMock()
    results.objects = objects
    collection.query.fetch_objects.return_value = results
    return client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWordShingles:
    """_word_shingles() — pure word-level n-gram generator."""

    def test_standard_four_words_size_three(self):
        mod = _import_module()
        result = mod._word_shingles("the quick brown fox", 3)
        assert result == ["the quick brown", "quick brown fox"]

    def test_exact_shingle_size_single_shingle(self):
        mod = _import_module()
        result = mod._word_shingles("hello world foo", 3)
        assert result == ["hello world foo"]

    def test_fewer_words_than_shingle_size_returns_all_as_one(self):
        mod = _import_module()
        result = mod._word_shingles("hello world", 3)
        assert result == ["hello world"]

    def test_single_word_shingle_size_one(self):
        mod = _import_module()
        result = mod._word_shingles("hello", 1)
        assert result == ["hello"]

    def test_single_word_shingle_size_three(self):
        mod = _import_module()
        result = mod._word_shingles("hello", 3)
        assert result == ["hello"]

    def test_empty_string_returns_empty_list(self):
        mod = _import_module()
        assert mod._word_shingles("", 3) == []

    def test_whitespace_only_returns_empty_list(self):
        mod = _import_module()
        assert mod._word_shingles("   ", 3) == []

    def test_default_shingle_size_is_three(self):
        mod = _import_module()
        words = "a b c d e"
        result = mod._word_shingles(words)
        assert result == ["a b c", "b c d", "c d e"]

    def test_shingle_size_two(self):
        mod = _import_module()
        result = mod._word_shingles("one two three", 2)
        assert result == ["one two", "two three"]


class TestRequireDatesketch:
    """_require_datasketch() — returns MinHash class when installed."""

    def test_returns_minhash_class(self):
        mod = _import_module()
        MinHash = mod._require_datasketch()
        from datasketch import MinHash as RealMinHash
        assert MinHash is RealMinHash

    def test_raises_import_error_when_unavailable(self, monkeypatch):
        import builtins
        mod = _import_module()
        real_import = builtins.__import__

        # Save and remove cached datasketch before patching __import__
        saved_datasketch = sys.modules.pop("datasketch", None)

        def _blocking_import(name, *args, **kwargs):
            if name == "datasketch":
                raise ImportError("datasketch not available (mocked)")
            return real_import(name, *args, **kwargs)

        # Use monkeypatch so cleanup is automatic even on test failure
        monkeypatch.setattr(builtins, "__import__", _blocking_import)
        try:
            with pytest.raises(ImportError, match="datasketch"):
                mod._require_datasketch()
        finally:
            # Always restore datasketch to sys.modules
            if saved_datasketch is not None:
                sys.modules["datasketch"] = saved_datasketch


class TestComputeFuzzyFingerprint:
    """compute_fuzzy_fingerprint() — hex string determinism."""

    def test_returns_hex_string(self):
        mod = _import_module()
        fp = mod.compute_fuzzy_fingerprint("Hello world, this is a test sentence.")
        assert isinstance(fp, str)
        # Must be a valid hex string
        int(fp, 16)

    def test_deterministic_for_same_input(self):
        mod = _import_module()
        text = "The quick brown fox jumps over the lazy dog."
        fp1 = mod.compute_fuzzy_fingerprint(text)
        fp2 = mod.compute_fuzzy_fingerprint(text)
        assert fp1 == fp2

    def test_different_texts_different_fingerprints(self):
        mod = _import_module()
        fp1 = mod.compute_fuzzy_fingerprint("apple banana cherry date elderberry")
        fp2 = mod.compute_fuzzy_fingerprint("zebra yacht xray walrus vulture")
        assert fp1 != fp2

    def test_length_reflects_num_hashes(self):
        mod = _import_module()
        # 128 hashes * 8 bytes/uint64 = 1024 bytes → 2048 hex chars
        fp = mod.compute_fuzzy_fingerprint("sample text for length test", num_hashes=128)
        assert len(fp) == 2048

    def test_custom_num_hashes(self):
        mod = _import_module()
        fp = mod.compute_fuzzy_fingerprint("sample", num_hashes=64)
        # 64 * 8 = 512 bytes → 1024 hex chars
        assert len(fp) == 1024


class TestDeserialiseMinHash:
    """_deserialise_minhash() — roundtrip fidelity."""

    def test_roundtrip_hashvalues_match(self):
        import numpy as np
        mod = _import_module()
        text = "roundtrip test with several distinct words here now"
        fp = mod.compute_fuzzy_fingerprint(text, num_hashes=128)
        mh = mod._deserialise_minhash(fp, num_hashes=128)
        mh2 = mod._deserialise_minhash(fp, num_hashes=128)
        assert np.array_equal(mh.hashvalues, mh2.hashvalues)

    def test_deserialised_object_has_correct_num_perm(self):
        mod = _import_module()
        fp = mod.compute_fuzzy_fingerprint("test content", num_hashes=64)
        mh = mod._deserialise_minhash(fp, num_hashes=64)
        assert len(mh.hashvalues) == 64


class TestEstimateSimilarity:
    """estimate_similarity() — Jaccard approximation."""

    def test_identical_texts_near_one(self):
        mod = _import_module()
        text = "the quick brown fox jumps over the lazy dog again and again"
        fp1 = mod.compute_fuzzy_fingerprint(text)
        fp2 = mod.compute_fuzzy_fingerprint(text)
        sim = mod.estimate_similarity(fp1, fp2)
        assert sim >= 0.99

    def test_very_different_texts_low_similarity(self):
        mod = _import_module()
        fp1 = mod.compute_fuzzy_fingerprint(
            "apple banana cherry date elderberry fig grape honeydew iris jasmine"
        )
        fp2 = mod.compute_fuzzy_fingerprint(
            "quantum physics relativity entropy thermodynamics momentum electron photon"
        )
        sim = mod.estimate_similarity(fp1, fp2)
        assert sim < 0.3

    def test_return_type_is_float(self):
        mod = _import_module()
        fp1 = mod.compute_fuzzy_fingerprint("some text here to test")
        fp2 = mod.compute_fuzzy_fingerprint("other text there to verify")
        sim = mod.estimate_similarity(fp1, fp2)
        assert isinstance(sim, float)

    def test_similarity_in_zero_one_range(self):
        mod = _import_module()
        fp1 = mod.compute_fuzzy_fingerprint("partially overlapping text words here")
        fp2 = mod.compute_fuzzy_fingerprint("partially overlapping text content there")
        sim = mod.estimate_similarity(fp1, fp2)
        assert 0.0 <= sim <= 1.0


class TestMinHashEngineClass:
    """MinHashEngine — configuration validation and method delegates."""

    def test_default_construction_succeeds(self):
        mod = _import_module()
        engine = mod.MinHashEngine()
        assert engine.shingle_size == 3
        assert engine.num_hashes == 128

    def test_custom_params(self):
        mod = _import_module()
        engine = mod.MinHashEngine(shingle_size=2, num_hashes=64)
        assert engine.shingle_size == 2
        assert engine.num_hashes == 64

    def test_shingle_size_zero_raises_value_error(self):
        mod = _import_module()
        with pytest.raises(ValueError, match="shingle_size"):
            mod.MinHashEngine(shingle_size=0)

    def test_shingle_size_negative_raises_value_error(self):
        mod = _import_module()
        with pytest.raises(ValueError, match="shingle_size"):
            mod.MinHashEngine(shingle_size=-1)

    def test_num_hashes_below_16_raises_value_error(self):
        mod = _import_module()
        with pytest.raises(ValueError, match="num_hashes"):
            mod.MinHashEngine(num_hashes=15)

    def test_num_hashes_exactly_16_succeeds(self):
        mod = _import_module()
        engine = mod.MinHashEngine(num_hashes=16)
        assert engine.num_hashes == 16

    def test_fingerprint_returns_hex_string(self):
        mod = _import_module()
        engine = mod.MinHashEngine()
        fp = engine.fingerprint("the quick brown fox jumps over the lazy dog")
        assert isinstance(fp, str)
        int(fp, 16)  # valid hex

    def test_fingerprint_deterministic(self):
        mod = _import_module()
        engine = mod.MinHashEngine()
        text = "deterministic fingerprint test input here"
        assert engine.fingerprint(text) == engine.fingerprint(text)

    def test_jaccard_identical_texts_near_one(self):
        mod = _import_module()
        engine = mod.MinHashEngine()
        text = "the quick brown fox jumps over the lazy dog across the field"
        fp1 = engine.fingerprint(text)
        fp2 = engine.fingerprint(text)
        assert engine.jaccard(fp1, fp2) >= 0.99

    def test_jaccard_returns_float_in_range(self):
        mod = _import_module()
        engine = mod.MinHashEngine()
        fp1 = engine.fingerprint("text one with some words")
        fp2 = engine.fingerprint("text two with other words")
        result = engine.jaccard(fp1, fp2)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0


class TestFindChunkByFuzzyFingerprint:
    """find_chunk_by_fuzzy_fingerprint() — mock-based Weaviate path tests.

    The weaviate.classes.query stub is injected/restored per-test via the
    _weaviate_filter_stub() context manager to avoid polluting sys.modules
    for other test modules.
    """

    def test_mock_match_found_above_threshold(self):
        mod = _import_module()
        text = "the quick brown fox jumps over the lazy dog in the park"
        fp = mod.compute_fuzzy_fingerprint(text)

        obj = MagicMock()
        obj.uuid = "aaaa-bbbb-cccc"
        obj.properties = {
            "fuzzy_fingerprint": fp,
            "text": text,
            "source_documents": [],
        }

        with _weaviate_filter_stub():
            client = _make_mock_client([obj])
            result = mod.find_chunk_by_fuzzy_fingerprint(client, fp, threshold=0.9)

        assert result is not None
        assert result["uuid"] == "aaaa-bbbb-cccc"
        assert result["similarity"] >= 0.9
        assert "text_length" in result

    def test_mock_no_match_below_threshold(self):
        mod = _import_module()
        fp_query = mod.compute_fuzzy_fingerprint(
            "apple banana cherry date elderberry fig grape"
        )
        fp_stored = mod.compute_fuzzy_fingerprint(
            "quantum physics thermodynamics electron photon wave"
        )

        obj = MagicMock()
        obj.uuid = "dddd-eeee-ffff"
        obj.properties = {
            "fuzzy_fingerprint": fp_stored,
            "text": "quantum physics text",
            "source_documents": [],
        }

        with _weaviate_filter_stub():
            client = _make_mock_client([obj])
            result = mod.find_chunk_by_fuzzy_fingerprint(client, fp_query, threshold=0.95)

        assert result is None

    def test_mock_empty_collection_returns_none(self):
        mod = _import_module()
        fp = mod.compute_fuzzy_fingerprint("some text to search for")
        with _weaviate_filter_stub():
            client = _make_mock_client([])
            result = mod.find_chunk_by_fuzzy_fingerprint(client, fp, threshold=0.8)
        assert result is None

    def test_mock_exception_path_returns_none(self):
        mod = _import_module()
        fp = mod.compute_fuzzy_fingerprint("exception test input text")

        client = MagicMock()
        client.collections.get.side_effect = RuntimeError("Weaviate unavailable")

        # Exception path does NOT call the lazy weaviate import — no stub needed
        result = mod.find_chunk_by_fuzzy_fingerprint(client, fp, threshold=0.8)
        assert result is None

    def test_mock_object_with_missing_fingerprint_skipped(self):
        mod = _import_module()
        fp = mod.compute_fuzzy_fingerprint("valid fingerprint search query text here")

        obj_bad = MagicMock()
        obj_bad.uuid = "no-fp-uuid"
        obj_bad.properties = {"fuzzy_fingerprint": None, "text": "some text"}

        with _weaviate_filter_stub():
            client = _make_mock_client([obj_bad])
            result = mod.find_chunk_by_fuzzy_fingerprint(client, fp, threshold=0.5)
        assert result is None

    def test_mock_best_match_selected_among_multiple(self):
        mod = _import_module()
        base_text = "the quick brown fox jumps over the lazy dog near the river"
        fp_query = mod.compute_fuzzy_fingerprint(base_text)

        # High-similarity object
        obj_high = MagicMock()
        obj_high.uuid = "high-sim-uuid"
        obj_high.properties = {
            "fuzzy_fingerprint": fp_query,  # same fingerprint → sim ~1.0
            "text": base_text,
        }

        # Low-similarity object
        fp_low = mod.compute_fuzzy_fingerprint(
            "completely different content about quantum mechanics and entropy"
        )
        obj_low = MagicMock()
        obj_low.uuid = "low-sim-uuid"
        obj_low.properties = {
            "fuzzy_fingerprint": fp_low,
            "text": "quantum text",
        }

        with _weaviate_filter_stub():
            client = _make_mock_client([obj_low, obj_high])
            result = mod.find_chunk_by_fuzzy_fingerprint(client, fp_query, threshold=0.5)

        assert result is not None
        assert result["uuid"] == "high-sim-uuid"
