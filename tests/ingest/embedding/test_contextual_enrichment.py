# @summary
# Tests for src/ingest/embedding/nodes/contextual_enrichment.py.
# Covers: disabled/skip, enabled sets embed_text = context + body while leaving
# the stored text (chunk.text / enriched_content) untouched, count-mismatch and
# LLM-error fail-open (embed_text stays unset), and empty-doc/empty-chunks guard.
# @end-summary
"""Tests for the contextual_enrichment_node pipeline stage."""

from unittest.mock import MagicMock, patch

from src.ingest.common.schemas import ProcessedChunk
from src.ingest.common.types import IngestionConfig, Runtime

_PROVIDER = "src.ingest.embedding.nodes.contextual_enrichment.get_llm_provider"


def _make_chunk(text: str) -> ProcessedChunk:
    c = ProcessedChunk(text=text, metadata={})
    c.metadata["enriched_content"] = text  # as chunk_enrichment_node sets it
    return c


def _make_state(chunks, *, enabled=True, batch_size=8, cleaned="The document body about topic Z."):
    config = IngestionConfig(
        enable_contextual_chunking=enabled,
        contextual_batch_size=batch_size,
        contextual_doc_max_chars=8000,
        contextual_model_alias="controller",
        llm_temperature=0.0,
        llm_timeout_seconds=10,
    )
    runtime = Runtime(config=config, embedder=MagicMock(), weaviate_client=MagicMock())
    return {
        "chunks": chunks,
        "cleaned_text": cleaned,
        "source_name": "doc.md",
        "errors": [],
        "processing_log": [],
        "runtime": runtime,
    }


def _fake_provider(contents):
    """Provider whose json_completion returns successive .content strings."""
    prov = MagicMock()
    resps = [MagicMock(content=c) for c in contents]
    prov.json_completion.side_effect = resps
    return prov


def _run(state):
    from src.ingest.embedding.nodes.contextual_enrichment import contextual_enrichment_node
    return contextual_enrichment_node(state)


def test_disabled_is_noop():
    chunks = [_make_chunk("body one")]
    state = _make_state(chunks, enabled=False)
    with patch(_PROVIDER) as gp:
        _run(state)
        gp.assert_not_called()
    assert "embed_text" not in chunks[0].metadata  # no contextualization


def test_enabled_sets_embed_text_prefix_and_leaves_stored_text():
    chunks = [_make_chunk("body one"), _make_chunk("body two")]
    state = _make_state(chunks)
    prov = _fake_provider(
        ['{"contexts": ["From section A about one.", "From section B about two."]}']
    )
    with patch(_PROVIDER, return_value=prov):
        _run(state)
    # EMBED text = context + "\n\n" + body
    assert chunks[0].metadata["embed_text"] == "From section A about one.\n\nbody one"
    assert chunks[1].metadata["embed_text"] == "From section B about two.\n\nbody two"
    # STORED text (chunk.text / enriched_content) is UNCHANGED
    assert chunks[0].text == "body one"
    assert chunks[0].metadata["enriched_content"] == "body one"
    # Routes to the configured INSTRUCT alias (NOT the default reasoning model
    # that returns empty content) — guards the empty-context regression.
    assert prov.json_completion.call_args.kwargs["model_alias"] == "controller"


def test_count_mismatch_fails_open():
    chunks = [_make_chunk("body one"), _make_chunk("body two")]
    state = _make_state(chunks)
    with patch(_PROVIDER, return_value=_fake_provider(['{"contexts": ["only one context"]}'])):
        _run(state)
    assert "embed_text" not in chunks[0].metadata
    assert "embed_text" not in chunks[1].metadata


def test_llm_error_fails_open():
    chunks = [_make_chunk("body one")]
    state = _make_state(chunks)
    prov = MagicMock()
    prov.json_completion.side_effect = RuntimeError("llm boom")
    with patch(_PROVIDER, return_value=prov):
        _run(state)
    assert "embed_text" not in chunks[0].metadata


def test_empty_context_string_skipped():
    chunks = [_make_chunk("body one"), _make_chunk("body two")]
    state = _make_state(chunks)
    with patch(_PROVIDER, return_value=_fake_provider(['{"contexts": ["", "ctx for two"]}'])):
        _run(state)
    assert "embed_text" not in chunks[0].metadata  # empty ctx skipped
    assert chunks[1].metadata["embed_text"] == "ctx for two\n\nbody two"


def test_batching_makes_one_call_per_batch():
    chunks = [_make_chunk(f"body {i}") for i in range(5)]
    state = _make_state(chunks, batch_size=2)  # 5 chunks -> batches of 2,2,1 -> 3 calls
    prov = _fake_provider([
        '{"contexts": ["c0", "c1"]}',
        '{"contexts": ["c2", "c3"]}',
        '{"contexts": ["c4"]}',
    ])
    with patch(_PROVIDER, return_value=prov):
        _run(state)
    assert prov.json_completion.call_count == 3
    assert chunks[4].metadata["embed_text"] == "c4\n\nbody 4"


def test_empty_doc_skips():
    chunks = [_make_chunk("body one")]
    state = _make_state(chunks, cleaned="   ")
    with patch(_PROVIDER) as gp:
        _run(state)
        gp.assert_not_called()
    assert "embed_text" not in chunks[0].metadata
