# @summary
# TDD tests for A4 (RAPTOR-lite document routing): commit_node commits the
# staged document cards into the card collection as part of the SAME atomic
# commit as the chunks, with rollback consistency, and a STRICT no-op when there
# are no staged cards (cards disabled — the default).
# Covers: enabled path writes cards (ensure + add) alongside chunks; disabled
#   path never touches the card store; rollback removes the just-written cards
#   for the affected document_id(s); chunk-commit/return shape unchanged.
# @end-summary
"""TDD tests for committing document cards atomically (A4).

Mirrors the commit/rollback mocking style of ``test_atomicity_staging.py``:
patch the direct-write helpers at the ``commit_node`` import site and assert on
the calls. The card helpers (``ensure_card_collection`` / ``add_document_cards``
/ ``delete_document_cards``) are patched the same way so no live Weaviate is
needed.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Shared helpers (mirror tests/ingest/test_atomicity_staging.py)
# ---------------------------------------------------------------------------


def _make_runtime(
    *,
    store_documents: bool = True,
    build_document_cards: bool = False,
    card_collection: str = "TestCards",
    db_client=None,
    weaviate_client=None,
    target_bucket: str = "test-bucket",
    target_collection: str = "TestCol",
):
    """Build a minimal Runtime with controlled config (cards configurable)."""
    from src.ingest.common.types import IngestionConfig, Runtime

    embedder = MagicMock()
    embedder.embed_documents = MagicMock(return_value=[[0.1, 0.2]])

    config = IngestionConfig(
        store_documents=store_documents,
        target_bucket=target_bucket,
        target_collection=target_collection,
        build_document_cards=build_document_cards,
        card_collection=card_collection,
    )
    return Runtime(
        config=config,
        embedder=embedder,
        weaviate_client=weaviate_client or MagicMock(),
        db_client=db_client or MagicMock(),
    )


def _make_base_state(runtime, **overrides) -> dict:
    """Minimal state dict that satisfies the commit node."""
    from src.ingest.common.schemas import ProcessedChunk

    chunk = ProcessedChunk(
        text="hello world",
        metadata={"source": "test.md", "source_key": "k1", "chunk_index": 0},
    )
    state = {
        "runtime": runtime,
        "source_key": "k1",
        "source_name": "test.md",
        "source_uri": "file:///tmp/test.md",
        "source_id": "test:1",
        "source_version": "1",
        "connector": "local_fs",
        "raw_text": "# Hello",
        "cleaned_text": "# Hello",
        "clean_hash": "abc",
        "chunks": [chunk],
        "errors": [],
        "processing_log": [],
        "staging_batch_id": "batch-test-001",
        "source_hash": "deadbeef",
        "trace_id": "",
        "batch_id": "",
        "document_id": "",
        "staged_minio": None,
        "staged_weaviate_records": [],
        "staged_weaviate_delete_old": False,
    }
    state.update(overrides)
    return state


def _make_card(document_id: str, source: str = "test.md") -> dict:
    """A staged card record as document_card_emission_node would produce."""
    return {
        "document_id": document_id,
        "title": "Doc Title",
        "source": source,
        "section_headings": ["Intro", "Body"],
        "card_text": "Doc Title\n## Intro\n## Body",
        "summary": "",
        "num_chunks": 3,
        "vector": [0.3, 0.4],
    }


def _staged_chunk_state(*, build_document_cards: bool, cards=None, **overrides):
    """Build a fully-staged state (MinIO + Weaviate chunk + optional cards)."""
    from src.vector_db import DocumentRecord

    runtime = _make_runtime(
        store_documents=True, build_document_cards=build_document_cards
    )
    doc_id = overrides.pop("document_id", "doc-uuid-1234")
    record = DocumentRecord(
        text="chunk text",
        embedding=[0.1, 0.2],
        metadata={"source": "test.md", "source_key": "k1", "chunk_index": 0},
    )
    extra = {}
    if cards is not None:
        extra["staged_card_records"] = cards
    state = _make_base_state(
        runtime,
        document_id=doc_id,
        staged_minio={
            "document_id": doc_id,
            "content": "# Hello",
            "metadata": {"source_key": "k1"},
            "bucket": "test-bucket",
        },
        staged_weaviate_records=[record],
        staged_weaviate_delete_old=False,
        **extra,
        **overrides,
    )
    return state, doc_id


# ---------------------------------------------------------------------------
# Enabled path: cards committed alongside chunks
# ---------------------------------------------------------------------------


class TestCommitWritesCardsWhenStaged:
    def test_commit_writes_cards_alongside_chunks(self):
        from src.ingest.embedding.nodes.commit_node import commit_node

        card = _make_card("doc-uuid-1234")
        state, _ = _staged_chunk_state(build_document_cards=True, cards=[card])

        with (
            patch("src.ingest.embedding.nodes.commit_node.put_document") as mock_put,
            patch(
                "src.ingest.embedding.nodes.commit_node.add_documents", return_value=1
            ) as mock_add,
            patch(
                "src.ingest.embedding.nodes.commit_node.ensure_card_collection"
            ) as mock_ensure,
            patch(
                "src.ingest.embedding.nodes.commit_node.add_document_cards",
                return_value=1,
            ) as mock_add_cards,
        ):
            result = commit_node(state)

        # Chunk commit still happens normally.
        mock_put.assert_called_once()
        mock_add.assert_called_once()

        # Card collection ensured + cards written with the configured collection.
        mock_ensure.assert_called_once()
        ensure_args = mock_ensure.call_args
        assert "TestCards" in str(ensure_args)
        assert ensure_args.args[0] is state["runtime"].weaviate_client

        mock_add_cards.assert_called_once()
        add_cards_args, add_cards_kwargs = mock_add_cards.call_args
        # client first positional, records second positional
        assert add_cards_args[0] is state["runtime"].weaviate_client
        assert add_cards_args[1] == [card]
        assert add_cards_kwargs.get("collection") == "TestCards"

        # Chunk return shape unchanged.
        assert result["stored_count"] == 1
        assert not result.get("errors")
        assert any("commit:ok" in e for e in result.get("processing_log", []))

    def test_ensure_called_before_add_cards(self):
        from src.ingest.embedding.nodes.commit_node import commit_node

        order: list[str] = []
        card = _make_card("doc-uuid-1234")
        state, _ = _staged_chunk_state(build_document_cards=True, cards=[card])

        with (
            patch("src.ingest.embedding.nodes.commit_node.put_document"),
            patch(
                "src.ingest.embedding.nodes.commit_node.add_documents", return_value=1
            ),
            patch(
                "src.ingest.embedding.nodes.commit_node.ensure_card_collection",
                side_effect=lambda *a, **k: order.append("ensure"),
            ),
            patch(
                "src.ingest.embedding.nodes.commit_node.add_document_cards",
                side_effect=lambda *a, **k: order.append("add") or 1,
            ),
        ):
            commit_node(state)

        assert order == ["ensure", "add"]


# ---------------------------------------------------------------------------
# Disabled path: strict no-op (the default, cards disabled)
# ---------------------------------------------------------------------------


class TestCommitNoCardsWhenAbsent:
    def test_no_card_calls_when_staged_card_records_absent(self):
        """Default path (no staged_card_records key) → never touch the card store."""
        from src.ingest.embedding.nodes.commit_node import commit_node

        state, _ = _staged_chunk_state(build_document_cards=False, cards=None)
        assert "staged_card_records" not in state

        with (
            patch("src.ingest.embedding.nodes.commit_node.put_document") as mock_put,
            patch(
                "src.ingest.embedding.nodes.commit_node.add_documents", return_value=1
            ) as mock_add,
            patch(
                "src.ingest.embedding.nodes.commit_node.ensure_card_collection"
            ) as mock_ensure,
            patch(
                "src.ingest.embedding.nodes.commit_node.add_document_cards"
            ) as mock_add_cards,
        ):
            result = commit_node(state)

        # Chunk commit identical to today.
        mock_put.assert_called_once()
        mock_add.assert_called_once()
        # Strict no-op for cards.
        mock_ensure.assert_not_called()
        mock_add_cards.assert_not_called()

        assert result["stored_count"] == 1
        assert not result.get("errors")

    def test_no_card_calls_when_staged_card_records_empty(self):
        """Empty staged_card_records list → still a strict no-op for cards."""
        from src.ingest.embedding.nodes.commit_node import commit_node

        state, _ = _staged_chunk_state(build_document_cards=True, cards=[])

        with (
            patch("src.ingest.embedding.nodes.commit_node.put_document"),
            patch(
                "src.ingest.embedding.nodes.commit_node.add_documents", return_value=1
            ),
            patch(
                "src.ingest.embedding.nodes.commit_node.ensure_card_collection"
            ) as mock_ensure,
            patch(
                "src.ingest.embedding.nodes.commit_node.add_document_cards"
            ) as mock_add_cards,
        ):
            result = commit_node(state)

        mock_ensure.assert_not_called()
        mock_add_cards.assert_not_called()
        assert result["stored_count"] == 1


# ---------------------------------------------------------------------------
# Rollback consistency: card write failure rolls back chunks + cards
# ---------------------------------------------------------------------------


class TestCardWriteFailureRollsBack:
    def test_card_write_failure_triggers_full_rollback(self):
        """If add_document_cards raises, the SAME rollback fires (chunks + cards)."""
        from src.ingest.embedding.nodes.commit_node import commit_node

        card = _make_card("doc-uuid-card-fail")
        state, doc_id = _staged_chunk_state(
            build_document_cards=True,
            cards=[card],
            document_id="doc-uuid-card-fail",
            staging_batch_id="batch-card-fail",
        )

        with (
            patch("src.ingest.embedding.nodes.commit_node.put_document") as mock_put,
            patch(
                "src.ingest.embedding.nodes.commit_node.add_documents", return_value=1
            ) as mock_add,
            patch(
                "src.ingest.embedding.nodes.commit_node.ensure_card_collection"
            ),
            patch(
                "src.ingest.embedding.nodes.commit_node.add_document_cards",
                side_effect=RuntimeError("card store down"),
            ),
            patch(
                "src.ingest.embedding.nodes.commit_node.delete_documents_by_staging_batch"
            ) as mock_del_wv,
            patch(
                "src.ingest.embedding.nodes.commit_node.delete_document"
            ) as mock_del_minio,
            patch(
                "src.ingest.embedding.nodes.commit_node.delete_document_cards"
            ) as mock_del_cards,
        ):
            result = commit_node(state)

        # Chunk write succeeded before the card write failed.
        mock_put.assert_called_once()
        mock_add.assert_called_once()

        # Existing chunk-rollback assertions still hold.
        mock_del_wv.assert_called_once()
        assert "batch-card-fail" in str(mock_del_wv.call_args)
        mock_del_minio.assert_called_once()
        assert doc_id in str(mock_del_minio.call_args)

        # New: card cleanup fires for the affected document_id.
        mock_del_cards.assert_called_once()
        del_cards_args = mock_del_cards.call_args
        assert doc_id in str(del_cards_args)
        assert "TestCards" in str(del_cards_args)

        assert result.get("stored_count") == 0
        errors = result.get("errors", [])
        assert any(e.startswith("commit:") for e in errors)
        assert any("commit:rolled_back" in e for e in result.get("processing_log", []))

    def test_chunk_write_failure_cleans_cards_only_if_written(self):
        """If chunks fail BEFORE cards are written, card cleanup must NOT fire."""
        from src.ingest.embedding.nodes.commit_node import commit_node

        card = _make_card("doc-uuid-wv-fail")
        state, doc_id = _staged_chunk_state(
            build_document_cards=True,
            cards=[card],
            document_id="doc-uuid-wv-fail",
            staging_batch_id="batch-wv-fail",
        )

        with (
            patch("src.ingest.embedding.nodes.commit_node.put_document") as mock_put,
            patch(
                "src.ingest.embedding.nodes.commit_node.add_documents",
                side_effect=RuntimeError("weaviate down"),
            ),
            patch(
                "src.ingest.embedding.nodes.commit_node.ensure_card_collection"
            ) as mock_ensure,
            patch(
                "src.ingest.embedding.nodes.commit_node.add_document_cards"
            ) as mock_add_cards,
            patch(
                "src.ingest.embedding.nodes.commit_node.delete_documents_by_staging_batch"
            ) as mock_del_wv,
            patch(
                "src.ingest.embedding.nodes.commit_node.delete_document"
            ) as mock_del_minio,
            patch(
                "src.ingest.embedding.nodes.commit_node.delete_document_cards"
            ) as mock_del_cards,
        ):
            result = commit_node(state)

        mock_put.assert_called_once()
        # Cards were never reached because the chunk write failed first.
        mock_ensure.assert_not_called()
        mock_add_cards.assert_not_called()
        # Chunk rollback still fires.
        mock_del_wv.assert_called_once()
        mock_del_minio.assert_called_once()
        # No card cleanup because no cards were written.
        mock_del_cards.assert_not_called()

        assert result.get("stored_count") == 0

    def test_card_rollback_best_effort_does_not_mask_original_error(self):
        """A failure in the card cleanup itself must not raise / mask the commit error."""
        from src.ingest.embedding.nodes.commit_node import commit_node

        card = _make_card("doc-uuid-card-fail2")
        state, _ = _staged_chunk_state(
            build_document_cards=True,
            cards=[card],
            document_id="doc-uuid-card-fail2",
            staging_batch_id="batch-card-fail2",
        )

        with (
            patch("src.ingest.embedding.nodes.commit_node.put_document"),
            patch(
                "src.ingest.embedding.nodes.commit_node.add_documents", return_value=1
            ),
            patch("src.ingest.embedding.nodes.commit_node.ensure_card_collection"),
            patch(
                "src.ingest.embedding.nodes.commit_node.add_document_cards",
                side_effect=RuntimeError("card store down"),
            ),
            patch(
                "src.ingest.embedding.nodes.commit_node.delete_documents_by_staging_batch"
            ) as mock_del_wv,
            patch("src.ingest.embedding.nodes.commit_node.delete_document"),
            patch(
                "src.ingest.embedding.nodes.commit_node.delete_document_cards",
                side_effect=RuntimeError("card cleanup also down"),
            ) as mock_del_cards,
        ):
            result = commit_node(state)

        # Card cleanup was attempted even though it raised.
        mock_del_cards.assert_called_once()
        # Chunk rollback still fired.
        mock_del_wv.assert_called_once()
        # Original commit error surfaces; node did not raise.
        errors = result.get("errors", [])
        assert any("card store down" in e for e in errors)
        assert result.get("stored_count") == 0
