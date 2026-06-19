"""Tests for tokenizer wiring in chunk_markdown_via_docling.

Verifies:
  1. Sentence-per-line markdown (OpenTitan-style) is consolidated by HybridChunker
     when a tokenizer is supplied (regression: bare HybridChunker invocation
     without a tokenizer produces one chunk per source line).
  2. The hybrid_chunker_tokenizer_model config knob is respected when set.
  3. When the knob is None, the embedding-model config (EMBEDDING_MODEL_PATH)
     is used as the tokenizer source.
"""
from __future__ import annotations

# Eagerly import PIL submodules so root conftest doesn't replace them with a
# stub before docling-core's page module imports them.
try:
    import PIL  # noqa: F401
    import PIL.Image  # noqa: F401
    import PIL.ImageColor  # noqa: F401
    import PIL.ImageDraw  # noqa: F401
    import PIL.ImageFont  # noqa: F401
except ImportError:
    pass

from dataclasses import dataclass

import pytest


AES_MARKDOWN = """##### AES

AES is the primary symmetric encryption and decryption mechanism used in OpenTitan protocols.
AES runs with key sizes of 128b, 192b, or 256b.
The module can select encryption or decryption of data that arrives in 16 byte quantities to be encrypted or decrypted using different block cipher modes of operation.
It supports ECB mode, CBC mode, CFB mode, OFB mode and CTR mode.

The GCM mode is not implemented in hardware, but can be constructed using AES in counter mode.
The integrity tag calculation can be implemented in Ibex and accelerated via bitmanip instructions.
"""


@dataclass
class _FakeIngestionConfig:
    """Minimal stand-in for IngestionConfig (avoids import-time settings cost)."""
    enable_vision_processing: bool = False
    hybrid_chunker_max_tokens: int = 512
    hybrid_chunker_tokenizer_model: str | None = None


@pytest.fixture(autouse=True)
def _clear_tokenizer_cache():
    """Ensure each test starts with a clean tokenizer cache."""
    from src.ingest.support import docling as docling_mod
    cache = getattr(docling_mod, "_TOKENIZER_CACHE", None)
    if cache is not None:
        cache.clear()
    yield
    if cache is not None:
        cache.clear()


def test_aes_section_consolidates_to_few_chunks():
    """Sentence-per-line AES section should produce <=2 chunks, not 6+.

    This is the primary regression test for the bug where HybridChunker was
    invoked without a tokenizer, defeating merge_peers.
    """
    pytest.importorskip("transformers")
    pytest.importorskip("docling")
    pytest.importorskip("docling_core")
    # The root conftest stubs PIL when it isn't already in sys.modules. Real
    # docling's page module imports PIL.ImageColor, which the stub doesn't
    # provide. Skip this integration test in that environment — the unit
    # tests below cover the resolution logic without invoking real docling.
    try:
        from PIL import ImageColor as _IC  # noqa: F401
    except ImportError:
        pytest.skip("real PIL.ImageColor not available (PIL is stubbed)")

    # Other ingest tests context-manage MagicMock submodules into
    # ``sys.modules`` (e.g. ``docling.datamodel.base_models``) and restore by
    # popping them in ``finally``. Once popped, Python's import system
    # remembers their absence in some cases — re-importing the real submodule
    # then yields a partial module without DocumentStream / InputFormat.
    # Force a clean re-import by purging any docling-related sys.modules
    # entries before this integration test runs.
    import sys as _sys
    for _mod in list(_sys.modules):
        if (
            _mod == "docling"
            or _mod.startswith("docling.")
            or _mod == "docling_core"
            or _mod.startswith("docling_core.")
        ):
            _sys.modules.pop(_mod, None)

    from src.ingest.support.docling import chunk_markdown_via_docling

    chunks = chunk_markdown_via_docling(
        AES_MARKDOWN,
        max_tokens=512,
        config=_FakeIngestionConfig(
            hybrid_chunker_tokenizer_model="BAAI/bge-m3",
        ),
    )

    assert len(chunks) >= 1, "expected at least one chunk"
    assert len(chunks) <= 2, (
        f"AES section should consolidate to <=2 chunks, got {len(chunks)}: "
        f"{[c.text[:60] for c in chunks]}"
    )

    combined = " ".join(c.text for c in chunks)
    for phrase in [
        "primary symmetric encryption",
        "128b, 192b, or 256b",
        "ECB mode",
        "GCM mode",
        "Ibex",
    ]:
        assert phrase in combined, f"missing phrase: {phrase!r}"

    # Heading metadata: each chunk should carry "AES" in its heading path.
    for c in chunks:
        path = (c.section_path or "") + " " + (c.heading or "")
        assert "AES" in path, f"AES not in heading metadata for chunk: {c!r}"


def test_config_tokenizer_model_is_used(monkeypatch):
    """hybrid_chunker_tokenizer_model wins over embedding model fallback."""
    from src.ingest.support import docling as docling_mod

    seen: dict = {}

    class _DummyTokenizer:
        def __init__(self, model_id, max_tokens):
            seen["model_id"] = model_id
            seen["max_tokens"] = max_tokens

    def _fake_build(model_id, max_tokens):
        return _DummyTokenizer(model_id, max_tokens)

    monkeypatch.setattr(docling_mod, "_build_hf_tokenizer", _fake_build)

    config = _FakeIngestionConfig(
        hybrid_chunker_tokenizer_model="BAAI/bge-small-en-v1.5",
    )
    resolved = docling_mod._resolve_tokenizer_model_id(config)
    assert resolved == "BAAI/bge-small-en-v1.5"

    tok = docling_mod._get_or_build_tokenizer(resolved, max_tokens=256)
    assert seen["model_id"] == "BAAI/bge-small-en-v1.5"
    assert seen["max_tokens"] == 256
    # Cache hit on second call.
    tok2 = docling_mod._get_or_build_tokenizer(resolved, max_tokens=256)
    assert tok is tok2


def test_falls_back_to_embedding_model(monkeypatch):
    """When hybrid_chunker_tokenizer_model is None, EMBEDDING_MODEL_PATH is used."""
    from src.ingest.support import docling as docling_mod

    monkeypatch.setattr(
        docling_mod,
        "EMBEDDING_MODEL_PATH",
        "BAAI/bge-small-en-v1.5",
        raising=False,
    )

    config = _FakeIngestionConfig(hybrid_chunker_tokenizer_model=None)
    resolved = docling_mod._resolve_tokenizer_model_id(config)
    assert resolved == "BAAI/bge-small-en-v1.5"


def test_final_fallback_to_bge_m3(monkeypatch):
    """No config + no embedding model -> 'BAAI/bge-m3' fallback."""
    from src.ingest.support import docling as docling_mod

    monkeypatch.setattr(docling_mod, "EMBEDDING_MODEL_PATH", "", raising=False)

    config = _FakeIngestionConfig(hybrid_chunker_tokenizer_model=None)
    resolved = docling_mod._resolve_tokenizer_model_id(config)
    assert resolved == "BAAI/bge-m3"


def test_missing_local_embedding_path_falls_back(monkeypatch):
    """A local EMBEDDING_MODEL_PATH that does not exist must NOT be passed to
    AutoTokenizer (it would raise HFValidationError, breaking native chunking on
    TEI/remote-embedding deployments). Falls back to the downloadable default."""
    from src.ingest.support import docling as docling_mod

    monkeypatch.setattr(
        docling_mod, "EMBEDDING_MODEL_PATH", "/root/models/baai/bge-m3", raising=False
    )
    config = _FakeIngestionConfig(hybrid_chunker_tokenizer_model=None)
    resolved = docling_mod._resolve_tokenizer_model_id(config)
    assert resolved == "BAAI/bge-m3"


def test_existing_local_embedding_path_is_used(monkeypatch, tmp_path):
    """A local EMBEDDING_MODEL_PATH that DOES exist is used as-is (prod with a
    materialised local model)."""
    from src.ingest.support import docling as docling_mod

    monkeypatch.setattr(
        docling_mod, "EMBEDDING_MODEL_PATH", str(tmp_path), raising=False
    )
    config = _FakeIngestionConfig(hybrid_chunker_tokenizer_model=None)
    resolved = docling_mod._resolve_tokenizer_model_id(config)
    assert resolved == str(tmp_path)
