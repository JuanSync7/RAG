# @summary
# Tests for Phase 4 batch embedding optimisation (FR-1210–FR-1214).
# Covers: _form_batches order/partial/empty, config validation,
#   _embed_batches retry isolation, observability logs, and
#   embedding_storage_node integration with batching.
# Exports: (test functions)
# Deps: pytest, src.ingest.embedding.nodes.embedding_storage,
#   src.ingest.common.types.IngestionConfig, src.ingest.impl.verify_core_design
# @end-summary

"""Tests for batch embedding helpers and embedding_storage_node batch behaviour."""

from __future__ import annotations

import importlib
import logging
from unittest.mock import MagicMock, call, patch

import pytest

from src.ingest.common.schemas import ProcessedChunk
from src.ingest.common.types import IngestionConfig, Runtime
from src.ingest.embedding.nodes.embedding_storage import (
    _embed_batches,
    _form_batches,
    _log_batch_metrics,
    _log_batch_summary,
    embedding_storage_node,
)
from src.ingest.impl import verify_core_design

# Patch targets
_ADD_DOCS = "src.ingest.embedding.nodes.embedding_storage.add_documents"
_DELETE = "src.ingest.embedding.nodes.embedding_storage.delete_by_source_key"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(text: str = "chunk text") -> ProcessedChunk:
    return ProcessedChunk(text=text, metadata={})


def _make_state(
    chunks=None,
    batch_size: int = 64,
    update_mode: bool = False,
    source_key: str = "doc-001",
    source_name: str = "doc.md",
    embed_return=None,
):
    config = IngestionConfig(
        update_mode=update_mode,
        embedding_batch_size=batch_size,
    )
    mock_weaviate = MagicMock()
    mock_embedder = MagicMock()
    if embed_return is not None:
        mock_embedder.embed_documents.return_value = embed_return
    else:
        mock_embedder.embed_documents.side_effect = lambda texts: [[0.1] * 3] * len(texts)
    runtime = Runtime(
        config=config,
        embedder=mock_embedder,
        weaviate_client=mock_weaviate,
    )
    return {
        "chunks": chunks or [],
        "source_key": source_key,
        "source_name": source_name,
        "stored_count": 0,
        "errors": [],
        "processing_log": [],
        "runtime": runtime,
    }


def _safe_cfg(**overrides) -> IngestionConfig:
    """Build an IngestionConfig with safe defaults plus overrides."""
    defaults = dict(
        chunk_size=512,
        chunk_overlap=64,
        enable_docling_parser=False,
        vlm_mode="disabled",
        enable_multimodal_processing=False,
        enable_vision_processing=False,
        enable_visual_embedding=False,
        parser_strategy="auto",
        chunker="native",
    )
    defaults.update(overrides)
    return IngestionConfig(**defaults)


# ---------------------------------------------------------------------------
# FR-1210 / FR-1212 — Batch Formation
# ---------------------------------------------------------------------------


class TestFormBatches:
    def test_standard_split_preserves_order(self):
        result = _form_batches([1, 2, 3, 4, 5], 2)
        assert result == [[1, 2], [3, 4], [5]]

    def test_empty_input_returns_empty(self):
        assert _form_batches([], 64) == []

    def test_exact_fit_single_batch(self):
        assert _form_batches([1, 2, 3, 4], 4) == [[1, 2, 3, 4]]

    def test_single_item_smaller_than_batch(self):
        assert _form_batches([1], 64) == [[1]]

    def test_batch_size_one(self):
        result = _form_batches([10, 20, 30], 1)
        assert result == [[10], [20], [30]]

    def test_batch_size_larger_than_input(self):
        result = _form_batches([1, 2, 3], 100)
        assert result == [[1, 2, 3]]

    def test_order_preserved_strings(self):
        items = ["a", "b", "c", "d", "e"]
        result = _form_batches(items, 3)
        assert result == [["a", "b", "c"], ["d", "e"]]
        # Verify no items dropped
        flat = [x for batch in result for x in batch]
        assert flat == items


# ---------------------------------------------------------------------------
# FR-1211 — Config validation
# ---------------------------------------------------------------------------


class TestEmbeddingBatchSizeValidation:
    def test_zero_batch_size_fails(self):
        cfg = _safe_cfg(embedding_batch_size=0)
        result = verify_core_design(cfg)
        assert not result.ok
        assert any("embedding_batch_size" in e for e in result.errors)

    def test_over_max_batch_size_fails(self):
        cfg = _safe_cfg(embedding_batch_size=2049)
        result = verify_core_design(cfg)
        assert not result.ok
        assert any("embedding_batch_size" in e for e in result.errors)

    def test_negative_batch_size_fails(self):
        cfg = _safe_cfg(embedding_batch_size=-1)
        result = verify_core_design(cfg)
        assert not result.ok
        assert any("embedding_batch_size" in e for e in result.errors)

    def test_default_batch_size_passes(self):
        cfg = _safe_cfg(embedding_batch_size=64)
        result = verify_core_design(cfg)
        assert result.ok, f"Expected ok but got errors: {result.errors}"

    def test_min_boundary_passes(self):
        cfg = _safe_cfg(embedding_batch_size=1)
        result = verify_core_design(cfg)
        assert result.ok

    def test_max_boundary_passes(self):
        cfg = _safe_cfg(embedding_batch_size=2048)
        result = verify_core_design(cfg)
        assert result.ok

    def test_env_var_flows_to_settings(self, monkeypatch):
        """RAGWEAVE_EMBEDDING_BATCH_SIZE env var flows to settings module."""
        monkeypatch.setenv("RAGWEAVE_EMBEDDING_BATCH_SIZE", "32")
        import config.settings as settings_mod
        importlib.reload(settings_mod)
        assert settings_mod.RAG_INGESTION_EMBEDDING_BATCH_SIZE == 32
        # Restore to avoid polluting other tests
        monkeypatch.delenv("RAGWEAVE_EMBEDDING_BATCH_SIZE", raising=False)
        importlib.reload(settings_mod)


# ---------------------------------------------------------------------------
# FR-1213 — Batch retry isolation
# ---------------------------------------------------------------------------


class TestEmbedBatches:
    def test_all_succeed_returns_all_vectors(self):
        embedder = MagicMock()
        embedder.embed_documents.side_effect = lambda texts: [[0.1] * 3] * len(texts)
        batches = [["a", "b"], ["c", "d", "e"]]
        vectors, errors, success_mask = _embed_batches(embedder, batches)
        assert len(vectors) == 5
        assert errors == []
        assert success_mask == [True, True]

    def test_first_batch_fails_permanently_second_succeeds(self):
        """First batch fails 3x; second batch succeeds; only second vectors returned."""
        fail_count = [0]

        def embed_side_effect(texts):
            # First call group (batch 0) always fails; after 3 attempts batch 1 is tried.
            # We track total calls to route behavior.
            fail_count[0] += 1
            if fail_count[0] <= 3:
                raise RuntimeError("embed error")
            return [[0.5] * 3] * len(texts)

        embedder = MagicMock()
        embedder.embed_documents.side_effect = embed_side_effect
        batches = [["a", "b"], ["c"]]

        with patch("src.ingest.embedding.nodes.embedding_storage.time.sleep"):
            vectors, errors, success_mask = _embed_batches(embedder, batches)

        assert success_mask == [False, True]
        assert len(vectors) == 1  # only second batch (1 item)
        assert len(errors) == 1
        assert errors[0]["type"] == "batch_embedding_failure"
        assert errors[0]["batch_index"] == 1
        assert errors[0]["chunk_range"] == "0-1"  # chunks 0 and 1

    def test_single_batch_fails_permanently_error_dict(self):
        embedder = MagicMock()
        embedder.embed_documents.side_effect = RuntimeError("always fails")
        batches = [["x", "y", "z"]]

        with patch("src.ingest.embedding.nodes.embedding_storage.time.sleep"):
            vectors, errors, success_mask = _embed_batches(embedder, batches)

        assert vectors == []
        assert success_mask == [False]
        assert len(errors) == 1
        err = errors[0]
        assert err["type"] == "batch_embedding_failure"
        assert err["batch_index"] == 1
        assert err["chunk_range"] == "0-2"
        assert "always fails" in err["error"]

    def test_successful_batches_not_re_embedded(self):
        """Verify successful batches are embedded exactly once each (no re-embedding)."""
        call_count = [0]

        def embed_side_effect(texts):
            call_count[0] += 1
            # Batch 2 (second call group) fails first two attempts, then succeeds.
            # Batch 1 always succeeds on first try.
            # So: call 1 = batch 1 success, calls 2-3 = batch 2 fail, call 4 = batch 2 success
            if call_count[0] == 2 or call_count[0] == 3:
                raise RuntimeError("transient")
            return [[0.1] * 3] * len(texts)

        embedder = MagicMock()
        embedder.embed_documents.side_effect = embed_side_effect
        batches = [["a", "b"], ["c", "d"]]

        with patch("src.ingest.embedding.nodes.embedding_storage.time.sleep"):
            vectors, errors, success_mask = _embed_batches(embedder, batches)

        # Batch 1: 1 call; Batch 2: 3 calls (2 fail + 1 succeed) = 4 total
        assert embedder.embed_documents.call_count == 4
        assert success_mask == [True, True]
        assert errors == []
        assert len(vectors) == 4

    def test_retry_delay_called_between_attempts(self):
        """time.sleep called between retry attempts with linearly increasing delay."""
        embedder = MagicMock()
        embedder.embed_documents.side_effect = [
            RuntimeError("fail 1"),
            RuntimeError("fail 2"),
            [[0.1] * 3],  # succeeds on 3rd attempt
        ]
        batches = [["x"]]

        with patch("src.ingest.embedding.nodes.embedding_storage.time.sleep") as mock_sleep:
            _embed_batches(embedder, batches)

        assert mock_sleep.call_count == 2
        # Delays: _BATCH_RETRY_DELAY * 1 = 0.3, then _BATCH_RETRY_DELAY * 2 = 0.6
        assert mock_sleep.call_args_list[0] == call(0.3)
        assert mock_sleep.call_args_list[1] == call(0.6)

    def test_empty_batches_list(self):
        embedder = MagicMock()
        vectors, errors, success_mask = _embed_batches(embedder, [])
        assert vectors == []
        assert errors == []
        assert success_mask == []
        embedder.embed_documents.assert_not_called()


# ---------------------------------------------------------------------------
# FR-1214 — Observability logs
# ---------------------------------------------------------------------------


class TestBatchObservability:
    def test_log_batch_metrics_emits_structured_record(self, caplog):
        with caplog.at_level(logging.INFO, logger="rag.ingest.embedding.storage"):
            _log_batch_metrics(
                batch_idx=1,
                total_batches=3,
                chunk_count=10,
                latency_ms=42.5,
            )
        records = [r for r in caplog.records if r.message == "embedding_batch_complete"]
        assert len(records) == 1
        rec = records[0]
        assert rec.__dict__["batch_index"] == 1
        assert rec.__dict__["total_batches"] == 3
        assert rec.__dict__["chunk_count"] == 10
        assert rec.__dict__["latency_ms"] == 42.5

    def test_log_batch_summary_emits_structured_record(self, caplog):
        with caplog.at_level(logging.INFO, logger="rag.ingest.embedding.storage"):
            _log_batch_summary(total_chunks=20, total_batches=2, total_ms=500.0)
        records = [r for r in caplog.records if r.message == "embedding_batch_summary"]
        assert len(records) == 1
        rec = records[0]
        assert rec.__dict__["total_chunks"] == 20
        assert rec.__dict__["total_batches"] == 2
        assert rec.__dict__["total_ms"] == 500.0
        assert rec.__dict__["throughput_chunks_per_sec"] == 40.0

    def test_embed_batches_emits_complete_and_summary_logs(self, caplog):
        embedder = MagicMock()
        embedder.embed_documents.side_effect = lambda texts: [[0.1] * 3] * len(texts)
        batches = [["a", "b"], ["c"]]

        with caplog.at_level(logging.INFO, logger="rag.ingest.embedding.storage"):
            _embed_batches(embedder, batches)

        complete_records = [r for r in caplog.records if r.message == "embedding_batch_complete"]
        summary_records = [r for r in caplog.records if r.message == "embedding_batch_summary"]
        assert len(complete_records) == 2  # one per batch
        assert len(summary_records) == 1

    def test_summary_throughput_zero_when_no_elapsed(self, caplog):
        with caplog.at_level(logging.INFO, logger="rag.ingest.embedding.storage"):
            _log_batch_summary(total_chunks=5, total_batches=1, total_ms=0.0)
        records = [r for r in caplog.records if r.message == "embedding_batch_summary"]
        assert records[0].__dict__["throughput_chunks_per_sec"] == 0.0


# ---------------------------------------------------------------------------
# Integration — embedding_storage_node with batching
# ---------------------------------------------------------------------------


class TestEmbeddingStorageNodeBatching:
    def test_5_chunks_batch_size_2_three_batches_all_staged(self):
        """5 chunks + batch_size=2 → 3 batches → all 5 records staged (Issue #42).

        Since the staging refactor, add_documents is NOT called in this node.
        The node stages records in staged_weaviate_records for commit_node.
        stored_count stays 0 here; commit_node sets the real count.
        """
        chunks = [_make_chunk(f"chunk {i}") for i in range(5)]
        state = _make_state(chunks=chunks, batch_size=2)
        with patch(_ADD_DOCS) as mock_add, patch(_DELETE):
            result = embedding_storage_node(state)
        mock_add.assert_not_called()
        assert result["stored_count"] == 0
        # Verify 5 records staged
        staged = result.get("staged_weaviate_records", [])
        assert len(staged) == 5

    def test_batch_2_fails_chunks_excluded_from_staged_records(self):
        """Batch 2 (chunks 2–3) fails permanently → 3 records staged; batch_embedding_failure in errors."""
        chunks = [_make_chunk(f"chunk {i}") for i in range(5)]
        state = _make_state(chunks=chunks, batch_size=2)

        call_n = [0]

        def embed_side_effect(texts):
            call_n[0] += 1
            # Calls 1 = batch 1 success, calls 2-4 = batch 2 fail 3x, call 5 = batch 3 success
            if call_n[0] in (2, 3, 4):
                raise RuntimeError("batch 2 broken")
            return [[0.9] * 3] * len(texts)

        state["runtime"].embedder.embed_documents.side_effect = embed_side_effect

        with patch(_ADD_DOCS) as mock_add, patch(_DELETE), \
             patch("src.ingest.embedding.nodes.embedding_storage.time.sleep"):
            result = embedding_storage_node(state)

        # Since Issue #42, add_documents is NOT called here.
        mock_add.assert_not_called()
        # batch 1 (chunks 0,1) succeeds, batch 2 (chunks 2,3) fails, batch 3 (chunk 4) succeeds
        staged = result.get("staged_weaviate_records", [])
        assert len(staged) == 3  # 2 + 0 + 1

        # errors enter state as STRINGS (list[str] contract; must decode across
        # the Temporal boundary as EmbeddingResult.errors) — the batch index and
        # chunk range are preserved in the rendered string.
        batch_errors = [
            e for e in result["errors"]
            if isinstance(e, str) and "batch_embedding_failure" in e
        ]
        assert len(batch_errors) == 1
        err = batch_errors[0]
        assert "batch=2" in err
        assert "chunk_range=2-3" in err

    def test_all_batches_succeed_no_errors(self):
        chunks = [_make_chunk(f"text {i}") for i in range(4)]
        state = _make_state(chunks=chunks, batch_size=2)
        with patch(_ADD_DOCS) as mock_add, patch(_DELETE):
            result = embedding_storage_node(state)
        mock_add.assert_not_called()  # Issue #42: staging only
        assert result["stored_count"] == 0  # real count set by commit_node
        assert result.get("errors", []) == []
        assert len(result.get("staged_weaviate_records", [])) == 4

    def test_lifecycle_meta_attached_to_staged_records(self):
        """trace_id, schema_version, and batch_id appear in every staged record's metadata."""
        chunks = [_make_chunk("hello"), _make_chunk("world")]
        state = _make_state(chunks=chunks, batch_size=10)
        state["trace_id"] = "trace-abc"
        state["batch_id"] = "batch-xyz"

        with patch(_ADD_DOCS) as mock_add, patch(_DELETE):
            result = embedding_storage_node(state)

        mock_add.assert_not_called()  # Issue #42: staging only
        staged = result.get("staged_weaviate_records", [])
        assert len(staged) == 2
        for rec in staged:
            assert rec.metadata["trace_id"] == "trace-abc"
            assert rec.metadata["batch_id"] == "batch-xyz"
            assert "schema_version" in rec.metadata

    def test_should_skip_returns_zero_stored(self):
        chunks = [_make_chunk()]
        state = _make_state(chunks=chunks)
        state["should_skip"] = True
        with patch(_ADD_DOCS) as mock_add:
            result = embedding_storage_node(state)
        assert result["stored_count"] == 0
        mock_add.assert_not_called()

    def test_embed_exception_caught_and_recorded(self):
        """An unrecoverable embed error propagates to state errors, not raised."""
        chunks = [_make_chunk()]
        state = _make_state(chunks=chunks)
        state["runtime"].embedder.embed_documents.side_effect = RuntimeError("crash")

        with patch("src.ingest.embedding.nodes.embedding_storage.time.sleep"):
            result = embedding_storage_node(state)

        # All retries exhausted → error recorded as batch_embedding_failure dict
        assert result.get("stored_count", 0) == 0 or "errors" in result
        errors = result.get("errors", [])
        assert len(errors) >= 1
