# @summary
# Tests for the card-backfill tool (RAPTOR-lite document-routing §9 rollout
# step 1): build RAGDocumentCards from EXISTING RAGDocuments chunks WITHOUT
# re-ingesting the source files.
# Covers: src/ingest/embedding/common/card_backfill.py
#   (backfill_cards_from_corpus) — document enumeration, paginated per-document
#   chunk fetch, baseline card build (reusing build_document_card), batch embed,
#   ensure_card_collection once + add_document_cards write, dry-run, empty-card
#   skip, per-document error isolation, and deterministic idempotency — plus the
#   scripts/backfill_document_cards.py argparse surface.
# Exports: (pytest test functions)
# Deps: pytest, unittest.mock, src.ingest.embedding.common.card_backfill
# @end-summary
"""Tests for ``backfill_cards_from_corpus`` and its runnable wrapper.

A FAKE Weaviate client (mirroring the v4 ``collections.get(name).query
.fetch_objects`` / ``.aggregate.over_all`` read surface) and a FAKE embedder
(deterministic vectors) keep these tests fully offline — no live Weaviate, no
TEI. The mocking style mirrors ``tests/vector_db/test_card_store.py`` and
``tests/ingest/test_document_card_node.py``.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.ingest.embedding.common.card_backfill import backfill_cards_from_corpus


# ---------------------------------------------------------------------------
# Fakes — a minimal Weaviate v4 read surface + a deterministic embedder.
# ---------------------------------------------------------------------------

def _obj(uuid: str, **props):
    """A fake Weaviate object: ``.uuid`` + ``.properties`` (a dict)."""
    return SimpleNamespace(uuid=uuid, properties=dict(props))


class _FakeQuery:
    """Records ``fetch_objects`` calls and returns canned, offset-paginated rows.

    ``rows_by_doc`` maps a document_id to the ordered list of fake objects that
    a filter on that document_id should return. ``fetch_objects`` honours the
    weaviate v4 offset contract: ``limit`` caps the page and ``offset`` skips
    prior pages, so a per-document fetch terminates when a short page comes back.
    (Weaviate forbids the cursor ``after`` together with a ``where`` filter, so
    the read helper paginates by offset.)
    """

    def __init__(self, rows_by_doc: dict[str, list], page_size_seen: list[int]):
        self._rows_by_doc = rows_by_doc
        self._page_size_seen = page_size_seen
        self.calls: list[dict] = []

    def fetch_objects(self, *, filters=None, limit=None, offset=0, return_properties=None):
        # The fake filter carries the document_id it was built for.
        doc_id = getattr(filters, "_doc_id", None)
        rows = list(self._rows_by_doc.get(doc_id, []))
        self.calls.append({"doc_id": doc_id, "limit": limit, "offset": offset})
        if limit is not None:
            self._page_size_seen.append(limit)
        start = offset or 0
        page = rows[start: start + limit] if limit is not None else rows[start:]
        return SimpleNamespace(objects=page)


class _FakeAggregate:
    """Returns a group-by-over_all response listing distinct document_ids."""

    def __init__(self, doc_ids: list[str]):
        self._doc_ids = doc_ids
        self.calls: list[dict] = []

    def over_all(self, *, group_by=None, total_count=False, filters=None):
        prop = getattr(group_by, "prop", None) if group_by is not None else None
        self.calls.append({"prop": prop})
        groups = [
            SimpleNamespace(grouped_by=SimpleNamespace(value=d), total_count=1)
            for d in self._doc_ids
        ]
        return SimpleNamespace(groups=groups)


class _FakeCollection:
    def __init__(self, rows_by_doc, doc_ids, page_size_seen):
        self.query = _FakeQuery(rows_by_doc, page_size_seen)
        self.aggregate = _FakeAggregate(doc_ids)


class _FakeClient:
    """A fake Weaviate client exposing only the read surface the tool uses."""

    def __init__(self, rows_by_doc, doc_ids):
        self.page_size_seen: list[int] = []
        self._col = _FakeCollection(rows_by_doc, doc_ids, self.page_size_seen)
        self.collections = MagicMock()
        self.collections.get.return_value = self._col
        self.collections.exists.return_value = False


class _FakeEmbedder:
    """Deterministic embedder: assigns ``[i, i+0.5]`` to the i-th text."""

    def __init__(self):
        self.batches: list[list[str]] = []

    def embed_documents(self, texts):
        self.batches.append(list(texts))
        return [[float(i), float(i) + 0.5] for i, _ in enumerate(texts)]


# ---------------------------------------------------------------------------
# Canned corpus: two documents, each with title/source/heading_path/node_kind.
# ---------------------------------------------------------------------------

def _make_rows():
    rows_by_doc = {
        "doc-A": [
            _obj("a0", text="body", document_id="doc-A", title="AXI4 Spec",
                 source="specs/axi4.pdf", heading_path=["Overview"],
                 heading="Overview", node_kind="chunk"),
            _obj("a1", text="body", document_id="doc-A", title="AXI4 Spec",
                 source="specs/axi4.pdf", heading_path=["Channels"],
                 heading="Channels", node_kind="chunk"),
        ],
        "doc-B": [
            _obj("b0", text="body", document_id="doc-B", title="CHI Spec",
                 source="specs/chi.pdf", heading_path=["Intro"],
                 heading="Intro", node_kind="chunk"),
        ],
    }
    return rows_by_doc, ["doc-A", "doc-B"]


def _patched_filter():
    """Patch the Filter used by the read helpers so ``.equal`` stamps the doc id.

    The core function fetches a document's chunks via
    ``Filter.by_property("document_id").equal(doc_id)``. Under the fake client we
    intercept that to carry the requested document_id so ``fetch_objects`` can
    return the right canned rows without a live Weaviate filter engine.
    """
    fake_filter = MagicMock()

    def _by_property(name):
        prop = MagicMock()

        def _equal(value):
            flt = SimpleNamespace(_doc_id=value if name == "document_id" else None)
            return flt

        prop.equal.side_effect = _equal
        return prop

    fake_filter.by_property.side_effect = _by_property
    return fake_filter


# ---------------------------------------------------------------------------
# Core behaviour
# ---------------------------------------------------------------------------

class TestBackfillCore:
    def test_two_documents_build_two_cards_written(self):
        rows_by_doc, doc_ids = _make_rows()
        client = _FakeClient(rows_by_doc, doc_ids)
        embedder = _FakeEmbedder()

        with patch("src.vector_db.weaviate.store.Filter", _patched_filter()), \
             patch("src.ingest.embedding.common.card_backfill.ensure_card_collection") as ensure, \
             patch("src.ingest.embedding.common.card_backfill.add_document_cards", return_value=2) as add:
            stats = backfill_cards_from_corpus(
                client, embedder,
                document_ids=["doc-A", "doc-B"],
            )

        assert stats["documents_seen"] == 2
        assert stats["cards_built"] == 2
        assert stats["cards_written"] == 2
        assert stats["skipped_empty"] == 0
        assert stats["errors"] == []
        # ensure_card_collection called exactly once.
        ensure.assert_called_once()
        # add_document_cards called with 2 records, each carrying a vector.
        add.assert_called_once()
        written = add.call_args[0][1] if len(add.call_args[0]) > 1 else add.call_args.kwargs["cards"]
        assert len(written) == 2
        for card in written:
            assert "vector" in card and isinstance(card["vector"], list) and card["vector"]
            assert card["card_text"]

    def test_card_collection_passed_through(self):
        rows_by_doc, doc_ids = _make_rows()
        client = _FakeClient(rows_by_doc, doc_ids)
        embedder = _FakeEmbedder()

        with patch("src.vector_db.weaviate.store.Filter", _patched_filter()), \
             patch("src.ingest.embedding.common.card_backfill.ensure_card_collection") as ensure, \
             patch("src.ingest.embedding.common.card_backfill.add_document_cards", return_value=2) as add:
            backfill_cards_from_corpus(
                client, embedder,
                document_ids=["doc-A", "doc-B"],
                card_collection="MyCards",
            )

        # collection name threaded to both store helpers.
        assert ensure.call_args[0][1] == "MyCards" or ensure.call_args.kwargs.get("collection") == "MyCards"
        add_kwargs = add.call_args.kwargs
        assert add_kwargs.get("collection") == "MyCards"

    def test_embedder_called_with_card_texts(self):
        rows_by_doc, doc_ids = _make_rows()
        client = _FakeClient(rows_by_doc, doc_ids)
        embedder = _FakeEmbedder()

        with patch("src.vector_db.weaviate.store.Filter", _patched_filter()), \
             patch("src.ingest.embedding.common.card_backfill.ensure_card_collection"), \
             patch("src.ingest.embedding.common.card_backfill.add_document_cards", return_value=2):
            backfill_cards_from_corpus(client, embedder, document_ids=["doc-A", "doc-B"])

        # All card texts embedded; headings present in the embedded text.
        joined = "\n".join(t for batch in embedder.batches for t in batch)
        assert "Overview" in joined and "Channels" in joined and "Intro" in joined

    def test_document_ids_filter_restricts(self):
        rows_by_doc, doc_ids = _make_rows()
        client = _FakeClient(rows_by_doc, doc_ids)
        embedder = _FakeEmbedder()

        with patch("src.vector_db.weaviate.store.Filter", _patched_filter()), \
             patch("src.ingest.embedding.common.card_backfill.ensure_card_collection"), \
             patch("src.ingest.embedding.common.card_backfill.add_document_cards", return_value=1) as add:
            stats = backfill_cards_from_corpus(client, embedder, document_ids=["doc-A"])

        # Aggregate (full enumeration) must NOT be used when ids are given.
        assert client._col.aggregate.calls == []
        assert stats["documents_seen"] == 1
        assert stats["cards_built"] == 1
        written = add.call_args[0][1] if len(add.call_args[0]) > 1 else add.call_args.kwargs["cards"]
        assert [c["document_id"] for c in written] == ["doc-A"]

    def test_enumerates_documents_when_ids_omitted(self):
        rows_by_doc, doc_ids = _make_rows()
        client = _FakeClient(rows_by_doc, doc_ids)
        embedder = _FakeEmbedder()

        with patch("src.vector_db.weaviate.store.Filter", _patched_filter()), \
             patch("src.ingest.embedding.common.card_backfill.ensure_card_collection"), \
             patch("src.ingest.embedding.common.card_backfill.add_document_cards", return_value=2):
            stats = backfill_cards_from_corpus(client, embedder)

        # Distinct document_ids discovered via group-by aggregation.
        assert client._col.aggregate.calls
        assert client._col.aggregate.calls[0]["prop"] == "document_id"
        assert stats["documents_seen"] == 2
        assert stats["cards_built"] == 2

    def test_empty_card_text_is_skipped(self):
        # doc-C has only a structureless chunk → empty card_text → skipped.
        rows_by_doc = {
            "doc-C": [_obj("c0", text="", document_id="", node_kind="chunk")],
        }
        client = _FakeClient(rows_by_doc, ["doc-C"])
        embedder = _FakeEmbedder()

        with patch("src.vector_db.weaviate.store.Filter", _patched_filter()), \
             patch("src.ingest.embedding.common.card_backfill.ensure_card_collection"), \
             patch("src.ingest.embedding.common.card_backfill.add_document_cards", return_value=0) as add:
            stats = backfill_cards_from_corpus(client, embedder, document_ids=["doc-C"])

        assert stats["skipped_empty"] == 1
        assert stats["cards_built"] == 0
        assert stats["cards_written"] == 0
        # Nothing to write → add_document_cards not called with cards.
        if add.called:
            written = add.call_args[0][1] if len(add.call_args[0]) > 1 else add.call_args.kwargs.get("cards", [])
            assert written == []

    def test_dry_run_does_not_write(self):
        rows_by_doc, doc_ids = _make_rows()
        client = _FakeClient(rows_by_doc, doc_ids)
        embedder = _FakeEmbedder()

        with patch("src.vector_db.weaviate.store.Filter", _patched_filter()), \
             patch("src.ingest.embedding.common.card_backfill.ensure_card_collection") as ensure, \
             patch("src.ingest.embedding.common.card_backfill.add_document_cards") as add:
            stats = backfill_cards_from_corpus(
                client, embedder, document_ids=["doc-A", "doc-B"], dry_run=True,
            )

        add.assert_not_called()
        assert stats["cards_built"] == 2
        assert stats["cards_written"] == 0

    def test_per_document_error_isolated(self):
        rows_by_doc, doc_ids = _make_rows()
        client = _FakeClient(rows_by_doc, doc_ids)
        embedder = _FakeEmbedder()

        # Make the chunk fetch raise for doc-A only; doc-B must still produce a card.
        original_fetch = client._col.query.fetch_objects

        def _boom(*, filters=None, **kw):
            if getattr(filters, "_doc_id", None) == "doc-A":
                raise RuntimeError("simulated weaviate read failure")
            return original_fetch(filters=filters, **kw)

        client._col.query.fetch_objects = _boom

        with patch("src.vector_db.weaviate.store.Filter", _patched_filter()), \
             patch("src.ingest.embedding.common.card_backfill.ensure_card_collection"), \
             patch("src.ingest.embedding.common.card_backfill.add_document_cards", return_value=1) as add:
            stats = backfill_cards_from_corpus(client, embedder, document_ids=["doc-A", "doc-B"])

        assert stats["documents_seen"] == 2
        assert stats["cards_built"] == 1
        assert stats["cards_written"] == 1
        assert len(stats["errors"]) == 1
        assert "doc-A" in stats["errors"][0]
        written = add.call_args[0][1] if len(add.call_args[0]) > 1 else add.call_args.kwargs["cards"]
        assert [c["document_id"] for c in written] == ["doc-B"]

    def test_progress_callable_invoked(self):
        rows_by_doc, doc_ids = _make_rows()
        client = _FakeClient(rows_by_doc, doc_ids)
        embedder = _FakeEmbedder()
        seen: list[str] = []

        with patch("src.vector_db.weaviate.store.Filter", _patched_filter()), \
             patch("src.ingest.embedding.common.card_backfill.ensure_card_collection"), \
             patch("src.ingest.embedding.common.card_backfill.add_document_cards", return_value=2):
            backfill_cards_from_corpus(
                client, embedder, document_ids=["doc-A", "doc-B"],
                progress=seen.append,
            )

        assert seen  # progress was reported at least once.

    def test_idempotent_rebuild_is_deterministic(self):
        # Re-running yields byte-identical cards (same card_text / section_headings),
        # so the deterministic uuid5(document_id) upsert refreshes in place.
        rows_by_doc, doc_ids = _make_rows()
        embedder = _FakeEmbedder()

        def _run():
            client = _FakeClient(rows_by_doc, doc_ids)
            with patch("src.vector_db.weaviate.store.Filter", _patched_filter()), \
                 patch("src.ingest.embedding.common.card_backfill.ensure_card_collection"), \
                 patch("src.ingest.embedding.common.card_backfill.add_document_cards", return_value=2) as add:
                backfill_cards_from_corpus(client, embedder, document_ids=["doc-A", "doc-B"])
                return add.call_args[0][1] if len(add.call_args[0]) > 1 else add.call_args.kwargs["cards"]

        first = _run()
        second = _run()
        strip = lambda cards: [
            {k: v for k, v in c.items() if k != "vector"} for c in cards
        ]
        assert strip(first) == strip(second)

    def test_pagination_uses_fetch_page_size(self):
        # A document with more chunks than the page size must paginate (cursor),
        # not silently truncate.
        rows_by_doc = {
            "doc-A": [
                _obj(f"a{i}", text="body", document_id="doc-A", title="AXI4 Spec",
                     source="specs/axi4.pdf", heading_path=[f"H{i}"],
                     heading=f"H{i}", node_kind="chunk")
                for i in range(5)
            ],
        }
        client = _FakeClient(rows_by_doc, ["doc-A"])
        embedder = _FakeEmbedder()

        with patch("src.vector_db.weaviate.store.Filter", _patched_filter()), \
             patch("src.ingest.embedding.common.card_backfill.ensure_card_collection"), \
             patch("src.ingest.embedding.common.card_backfill.add_document_cards", return_value=1) as add:
            backfill_cards_from_corpus(
                client, embedder, document_ids=["doc-A"], fetch_page_size=2,
            )

        # All 5 headings must reach the card despite page size 2.
        written = add.call_args[0][1] if len(add.call_args[0]) > 1 else add.call_args.kwargs["cards"]
        assert len(written[0]["section_headings"]) == 5
        # Offset pagination issued > 1 fetch_objects call for the document.
        assert len(client._col.query.calls) >= 3
        assert all(c == 2 for c in client.page_size_seen)


# ---------------------------------------------------------------------------
# Script-level argparse surface (no live services).
# ---------------------------------------------------------------------------

class TestScriptArgparse:
    def _load_script_module(self):
        import importlib.util
        from pathlib import Path

        root = Path(__file__).resolve().parents[2]
        path = root / "scripts" / "backfill_document_cards.py"
        spec = importlib.util.spec_from_file_location("backfill_document_cards", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_parses_core_flags(self):
        mod = self._load_script_module()
        args = mod._parse_args([
            "--card-collection", "MyCards",
            "--source-collection", "RAGDocuments",
            "--document-ids", "d1,d2,d3",
            "--max-headings", "10",
            "--embed-batch-size", "8",
            "--dry-run",
        ])
        assert args.card_collection == "MyCards"
        assert args.source_collection == "RAGDocuments"
        assert args.document_ids == "d1,d2,d3"
        assert args.max_headings == 10
        assert args.embed_batch_size == 8
        assert args.dry_run is True

    def test_document_ids_parsed_to_list(self):
        mod = self._load_script_module()
        assert mod._split_ids("d1,d2 , d3,") == ["d1", "d2", "d3"]
        assert mod._split_ids(None) is None
        assert mod._split_ids("") is None
