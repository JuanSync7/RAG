# @summary
# Tests for the ingest-time chunk-ROLE TAG step that replaces the legacy regex
# DROP in the chunking node (Slice B). Asserts the core invariants: NOTHING is
# dropped (every chunk survives), the classifier's role lands in
# metadata["chunk_role"], a classifier failure FAILS OPEN (all chunks tagged the
# default "content" role), the legacy DROP only runs under the explicit
# back-compat escape hatch, and the tag step is a no-op when disabled.
# @end-summary

"""Tests for ``_tag_chunk_roles`` and its wiring into ``chunking_node`` (Slice B).

These tests use a FAKE classifier (no live model / no provider) injected into the
tag helper, so they assert the node's *contract* — survive-everything, tag-by-
role, fail-open — independent of the LLM. The shared classifier itself is tested
in ``tests/ingest/test_role_classify.py``.
"""

from types import SimpleNamespace

import pytest

from src.ingest.embedding.nodes import chunking as chunking_mod
from src.ingest.embedding.nodes.chunking import _tag_chunk_roles


def _c(text: str, **meta) -> SimpleNamespace:
    """A ProcessedChunk-like object: ``.text`` + a mutable ``.metadata`` dict."""
    return SimpleNamespace(text=text, metadata=dict(meta))


# ---------------------------------------------------------------------------
# _tag_chunk_roles — drop nothing, tag by role
# ---------------------------------------------------------------------------

def test_tags_every_chunk_and_drops_nothing():
    chunks = [
        _c("The reset value of REG37 is 0x25 and it is sticky across warm reset."),
        _c("A1.3 AXI Architecture .................... A1-22"),
        _c("Copyright 2024 Example Corp. All rights reserved."),
    ]

    def fake_classify(provider, batch):
        # One role per input chunk, same order (the shared classifier contract).
        return ["content", "navigation", "boilerplate"]

    out = _tag_chunk_roles(
        chunks, source_name="doc", provider=object(), classify_fn=fake_classify
    )

    # Nothing dropped: same objects, same order, same count.
    assert out is chunks
    assert len(out) == 3
    assert [c.metadata["chunk_role"] for c in out] == [
        "content",
        "navigation",
        "boilerplate",
    ]


def test_content_chunks_get_content_role():
    chunks = [_c("Substantive body paragraph describing the AWVALID handshake.")]

    def fake_classify(provider, batch):
        return ["content"]

    out = _tag_chunk_roles(
        chunks, source_name="doc", provider=object(), classify_fn=fake_classify
    )
    assert out[0].metadata["chunk_role"] == "content"


def test_chunks_with_dict_metadata_are_tagged():
    """Chunks may carry a real dict in ``.metadata`` (ProcessedChunk shape)."""
    chunks = [_c("nav text", chunk_type="text"), _c("body text", chunk_type="text")]

    def fake_classify(provider, batch):
        return ["navigation", "content"]

    out = _tag_chunk_roles(
        chunks, source_name="doc", provider=object(), classify_fn=fake_classify
    )
    # Pre-existing metadata is preserved; chunk_role is added alongside.
    assert out[0].metadata["chunk_type"] == "text"
    assert out[0].metadata["chunk_role"] == "navigation"
    assert out[1].metadata["chunk_role"] == "content"


# ---------------------------------------------------------------------------
# FAIL-OPEN: classifier failure must tag ALL chunks "content", never drop
# ---------------------------------------------------------------------------

def test_classifier_exception_fails_open_to_content():
    chunks = [
        _c("A1.3 AXI Architecture .................... A1-22"),
        _c("Copyright notice front matter."),
        _c("Real body content about burst types."),
    ]

    def exploding_classify(provider, batch):
        raise RuntimeError("router down")

    out = _tag_chunk_roles(
        chunks, source_name="doc", provider=object(), classify_fn=exploding_classify
    )

    # Drop nothing; every chunk fails open to the default role "content".
    assert out is chunks
    assert len(out) == 3
    assert all(c.metadata["chunk_role"] == "content" for c in out)


def test_length_mismatch_fails_open_to_content():
    """If the classifier returns the wrong number of roles, do not mis-align —
    fail open: every chunk gets the default role, nothing is dropped."""
    chunks = [_c("a"), _c("b"), _c("c")]

    def short_classify(provider, batch):
        return ["navigation"]  # too few

    out = _tag_chunk_roles(
        chunks, source_name="doc", provider=object(), classify_fn=short_classify
    )
    assert len(out) == 3
    assert all(c.metadata["chunk_role"] == "content" for c in out)


def test_empty_chunk_list_is_noop():
    out = _tag_chunk_roles(
        [], source_name="doc", provider=object(), classify_fn=lambda p, b: []
    )
    assert out == []


# ---------------------------------------------------------------------------
# chunking_node wiring: TAG when enabled, legacy DROP only via escape hatch
# ---------------------------------------------------------------------------

def _run_node_with(config, chunks, monkeypatch, fake_roles=None, classify_raises=False):
    """Drive ``chunking_node`` through the legacy-markdown path with a stubbed
    chunk producer, capturing the chunks that exit the node.

    Returns the node's returned ``chunks`` list (post tag/drop).
    """
    captured = {}

    # Stub the chunk producer so the node deterministically yields ``chunks``.
    monkeypatch.setattr(
        chunking_mod, "_chunk_with_markdown_legacy",
        lambda state, cfg, base: list(chunks),
    )
    # Stub metadata extraction so we do not depend on real text heuristics.
    monkeypatch.setattr(chunking_mod, "extract_metadata", lambda raw, name: object())
    monkeypatch.setattr(chunking_mod, "metadata_to_dict", lambda md: {})

    # Stub the classifier the tag step uses.
    def fake_classify_from_config(provider, batch):
        if classify_raises:
            raise RuntimeError("boom")
        return list(fake_roles) if fake_roles is not None else ["content"] * len(batch)

    monkeypatch.setattr(
        chunking_mod, "classify_roles_sync", fake_classify_from_config
    )
    # Stub provider acquisition so no live router is needed.
    monkeypatch.setattr(chunking_mod, "get_llm_provider", lambda: object())

    runtime = SimpleNamespace(config=config, embedder=None)
    state = {
        "runtime": runtime,
        "raw_text": "raw",
        "source_name": "doc",
        "source_uri": "uri",
        "source_key": "key",
        "source_id": "id",
        "connector": "conn",
        "source_version": "v1",
        "parse_result": None,
        "parser_instance": None,
        "errors": [],
        "processing_log": [],
    }
    result = chunking_mod.chunking_node(state)
    captured["result"] = result
    return result


def _legacy_config(**over):
    """Minimal config object for the legacy path with role-tag knobs."""
    base = dict(
        chunker="legacy",
        nav_classify=True,
        drop_navigational=False,
        nav_max_chars=320,
        nav_role_default="content",
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_node_tags_roles_when_nav_classify_enabled(monkeypatch):
    chunks = [_c("body one"), _c("toc line ... 12"), _c("legal notice")]
    cfg = _legacy_config(nav_classify=True)
    result = _run_node_with(
        cfg, chunks, monkeypatch,
        fake_roles=["content", "navigation", "boilerplate"],
    )
    out = result["chunks"]
    # Nothing dropped; all three roles applied.
    assert len(out) == 3
    assert [c.metadata["chunk_role"] for c in out] == [
        "content", "navigation", "boilerplate",
    ]


def test_node_fails_open_when_classifier_raises(monkeypatch):
    chunks = [_c("a"), _c("b"), _c("c")]
    cfg = _legacy_config(nav_classify=True)
    result = _run_node_with(cfg, chunks, monkeypatch, classify_raises=True)
    out = result["chunks"]
    assert len(out) == 3  # nothing dropped
    assert all(c.metadata["chunk_role"] == "content" for c in out)


def test_node_does_not_drop_when_classify_enabled(monkeypatch):
    """Even nav-looking chunks survive: the classifier TAGS, it never DROPS.
    With nav_classify on and drop_navigational off, a ToC-looking chunk that the
    classifier calls 'navigation' is still present in the output."""
    chunks = [_c("A1.3 AXI Architecture .................... A1-22")]
    cfg = _legacy_config(nav_classify=True, drop_navigational=False)
    result = _run_node_with(cfg, chunks, monkeypatch, fake_roles=["navigation"])
    out = result["chunks"]
    assert len(out) == 1
    assert out[0].metadata["chunk_role"] == "navigation"


def test_node_no_tag_when_classify_disabled_and_no_legacy_drop(monkeypatch):
    """nav_classify off + drop_navigational off -> no role key, nothing dropped."""
    chunks = [_c("body one"), _c("A1.3 AXI Architecture .......... A1-22")]
    cfg = _legacy_config(nav_classify=False, drop_navigational=False)
    result = _run_node_with(cfg, chunks, monkeypatch)
    out = result["chunks"]
    assert len(out) == 2  # nothing dropped
    assert all("chunk_role" not in c.metadata for c in out)


def test_node_legacy_drop_escape_hatch(monkeypatch):
    """Back-compat: nav_classify off AND drop_navigational on -> the OLD regex
    DROP runs and removes the ToC chunk (and tags nothing)."""
    chunks = [
        _c("The reset value of REG37 is 0x25 and it is sticky across warm reset."),
        _c("A1.3 AXI Architecture .................... A1-22"),
    ]
    cfg = _legacy_config(nav_classify=False, drop_navigational=True)
    result = _run_node_with(cfg, chunks, monkeypatch)
    out = result["chunks"]
    # The legacy regex drops the ToC chunk.
    assert len(out) == 1
    assert "REG37" in out[0].text
    assert all("chunk_role" not in c.metadata for c in out)


def test_node_classify_takes_precedence_over_legacy_drop(monkeypatch):
    """If BOTH flags are on, the classifier TAG wins (drop nothing) — the escape
    hatch only fires when nav_classify is off."""
    chunks = [
        _c("body content here"),
        _c("A1.3 AXI Architecture .................... A1-22"),
    ]
    cfg = _legacy_config(nav_classify=True, drop_navigational=True)
    result = _run_node_with(
        cfg, chunks, monkeypatch, fake_roles=["content", "navigation"]
    )
    out = result["chunks"]
    assert len(out) == 2  # nothing dropped despite drop_navigational=True
    assert [c.metadata["chunk_role"] for c in out] == ["content", "navigation"]
