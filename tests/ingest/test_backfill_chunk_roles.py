# @summary
# Tests for the chunk-ROLE backfill tool (nav-classify rollout, Slice D): tag
# chunk_role on EXISTING corpus chunks WITHOUT re-ingesting, using the SAME shared
# LLM classifier (classify_roles_from_config) that ingest tagging uses.
# Covers: src/ingest/embedding/common/role_backfill.py
#   (backfill_roles_from_corpus) — document enumeration, paginated per-document
#   chunk fetch (text-bearing), page-wise classification, in-place update-by-uuid
#   write, dry-run (writes nothing), per-document error isolation, fail-open to the
#   default role, and idempotency — plus the scripts/backfill_chunk_roles.py
#   argparse surface.
# Exports: (pytest test functions)
# Deps: pytest, unittest.mock, src.ingest.embedding.common.role_backfill
# @end-summary
"""Tests for ``backfill_roles_from_corpus`` and its runnable wrapper.

A FAKE Weaviate client (mirroring the v4 ``collections.get(name).query
.fetch_objects`` / ``.aggregate.over_all`` / ``.data.update`` surface) and a FAKE
classifier (no live router) keep these tests fully offline — no live Weaviate, no
LLM. The mocking style mirrors ``tests/ingest/test_card_backfill.py``.

DO NOT run against any live store: every external dependency is faked here.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.ingest.embedding.common.role_backfill import backfill_roles_from_corpus


# ---------------------------------------------------------------------------
# Fakes — a minimal Weaviate v4 read+update surface + a deterministic classifier.
# ---------------------------------------------------------------------------

def _obj(uuid: str, **props):
    """A fake Weaviate object: ``.uuid`` + ``.properties`` (a dict)."""
    return SimpleNamespace(uuid=uuid, properties=dict(props))


class _FakeQuery:
    """Records ``fetch_objects`` calls and returns canned, offset-paginated rows."""

    def __init__(self, rows_by_doc: dict[str, list], calls: list[dict]):
        self._rows_by_doc = rows_by_doc
        self.calls = calls

    def fetch_objects(self, *, filters=None, limit=None, offset=0, return_properties=None):
        doc_id = getattr(filters, "_doc_id", None)
        rows = list(self._rows_by_doc.get(doc_id, []))
        self.calls.append({
            "doc_id": doc_id,
            "limit": limit,
            "offset": offset,
            "return_properties": list(return_properties or []),
        })
        start = offset or 0
        page = rows[start: start + limit] if limit is not None else rows[start:]
        return SimpleNamespace(objects=page)


class _FakeAggregate:
    """Returns a group-by-over_all response listing distinct document_ids."""

    def __init__(self, doc_ids: list[str]):
        self._doc_ids = doc_ids

    def over_all(self, *, group_by=None, total_count=False, filters=None):
        groups = [
            SimpleNamespace(grouped_by=SimpleNamespace(value=d), total_count=1)
            for d in self._doc_ids
        ]
        return SimpleNamespace(groups=groups)


class _FakeData:
    """Records ``update`` calls — the in-place update-by-uuid write surface."""

    def __init__(self, updates: list[dict], raise_on=None):
        self.updates = updates
        self._raise_on = raise_on or set()

    def update(self, *, uuid=None, properties=None):
        if uuid in self._raise_on:
            raise RuntimeError(f"boom updating {uuid}")
        self.updates.append({"uuid": uuid, "properties": dict(properties or {})})


class _FakeCollection:
    def __init__(self, rows_by_doc, doc_ids, calls, updates, raise_on):
        self.query = _FakeQuery(rows_by_doc, calls)
        self.aggregate = _FakeAggregate(doc_ids)
        self.data = _FakeData(updates, raise_on)


class _FakeClient:
    """A fake Weaviate client exposing the read + update surface the tool uses."""

    def __init__(self, rows_by_doc, doc_ids, raise_on=None):
        self.fetch_calls: list[dict] = []
        self.updates: list[dict] = []
        self._col = _FakeCollection(
            rows_by_doc, doc_ids, self.fetch_calls, self.updates, raise_on
        )
        self.collections = MagicMock()
        self.collections.get.return_value = self._col


def _patched_filter():
    """Patch the Filter used by the read helper so ``.equal`` stamps the doc id."""
    fake_filter = MagicMock()

    def _by_property(name):
        prop = MagicMock()

        def _equal(value):
            return SimpleNamespace(_doc_id=value if name == "document_id" else None)

        prop.equal.side_effect = _equal
        return prop

    fake_filter.by_property.side_effect = _by_property
    return fake_filter


# ---------------------------------------------------------------------------
# Canned corpus + fake classifier.
# ---------------------------------------------------------------------------

def _make_rows():
    rows_by_doc = {
        "doc-A": [
            _obj("a0", text="Table of contents ......... 5", document_id="doc-A"),
            _obj("a1", text="The AXI4 protocol has five channels.", document_id="doc-A"),
        ],
        "doc-B": [
            _obj("b0", text="Copyright 2024. All rights reserved.", document_id="doc-B"),
        ],
    }
    return rows_by_doc, ["doc-A", "doc-B"]


def _classifier(mapping=None, *, raise_=False, wrong_length=False):
    """Build a fake classify_fn ``(provider, chunks) -> list[str]``.

    ``mapping`` maps a chunk's text to a role; anything unmapped -> "content".
    ``raise_`` makes the classifier raise (exercises fail-open).
    ``wrong_length`` returns a mis-sized list (exercises the length guard).
    """
    calls: list[list] = []

    def _fn(provider, chunks):
        calls.append(list(chunks))
        if raise_:
            raise RuntimeError("classifier exploded")
        if wrong_length:
            return ["content"]  # deliberately too short
        out = []
        for c in chunks:
            text = c.get("text") if isinstance(c, dict) else getattr(c, "text", "")
            out.append((mapping or {}).get(text, "content"))
        return out

    _fn.calls = calls  # type: ignore[attr-defined]
    return _fn


# ---------------------------------------------------------------------------
# Core behaviour
# ---------------------------------------------------------------------------

class TestBackfillCore:
    def test_real_run_updates_once_per_chunk_with_valid_role(self):
        rows_by_doc, doc_ids = _make_rows()
        client = _FakeClient(rows_by_doc, doc_ids)
        classify = _classifier({
            "Table of contents ......... 5": "navigation",
            "Copyright 2024. All rights reserved.": "boilerplate",
            "The AXI4 protocol has five channels.": "content",
        })

        with patch("src.vector_db.weaviate.store.Filter", _patched_filter()):
            stats = backfill_roles_from_corpus(
                client,
                document_ids=["doc-A", "doc-B"],
                classify_fn=classify,
            )

        assert stats["documents_seen"] == 2
        assert stats["chunks_seen"] == 3
        assert stats["chunks_updated"] == 3
        assert stats["errors"] == []
        # One update per chunk, each setting a valid chunk_role.
        assert len(client.updates) == 3
        by_uuid = {u["uuid"]: u["properties"]["chunk_role"] for u in client.updates}
        assert by_uuid == {
            "a0": "navigation",
            "a1": "content",
            "b0": "boilerplate",
        }
        for u in client.updates:
            assert set(u["properties"].keys()) == {"chunk_role"}
            assert u["properties"]["chunk_role"] in (
                "content", "navigation", "boilerplate"
            )

    def test_text_is_fetched_for_classification(self):
        """The backfill must request the chunk TEXT (card-backfill omits it)."""
        rows_by_doc, doc_ids = _make_rows()
        client = _FakeClient(rows_by_doc, doc_ids)
        classify = _classifier()

        with patch("src.vector_db.weaviate.store.Filter", _patched_filter()):
            backfill_roles_from_corpus(
                client, document_ids=["doc-A"], classify_fn=classify,
            )

        # The fetch asked for text + uuid-bearing read (text present in props).
        assert client.fetch_calls, "expected at least one fetch_objects call"
        for call in client.fetch_calls:
            assert "text" in call["return_properties"]
        # The classifier saw the actual chunk text, not empty strings.
        seen_texts = [
            c["text"] for batch in classify.calls for c in batch  # type: ignore[index]
        ]
        assert "The AXI4 protocol has five channels." in seen_texts


class TestDryRun:
    def test_dry_run_writes_nothing(self):
        rows_by_doc, doc_ids = _make_rows()
        client = _FakeClient(rows_by_doc, doc_ids)
        classify = _classifier({
            "Table of contents ......... 5": "navigation",
        })

        with patch("src.vector_db.weaviate.store.Filter", _patched_filter()):
            stats = backfill_roles_from_corpus(
                client,
                document_ids=["doc-A", "doc-B"],
                classify_fn=classify,
                dry_run=True,
            )

        # Classified + counted, but NOTHING written.
        assert client.updates == []
        assert stats["chunks_updated"] == 0
        assert stats["chunks_seen"] == 3
        # Dry-run still reports what it WOULD have done, per-role.
        assert stats["role_counts"]["navigation"] == 1
        assert stats["role_counts"]["content"] == 2


class TestFailOpen:
    def test_classifier_exception_fails_open_to_default_role(self):
        """A classifier that raises -> every chunk tagged default_role, not dropped."""
        rows_by_doc, doc_ids = _make_rows()
        client = _FakeClient(rows_by_doc, doc_ids)
        classify = _classifier(raise_=True)

        with patch("src.vector_db.weaviate.store.Filter", _patched_filter()):
            stats = backfill_roles_from_corpus(
                client,
                document_ids=["doc-A", "doc-B"],
                classify_fn=classify,
                default_role="content",
            )

        # Every chunk still written, all default_role — never demoted on error.
        assert stats["chunks_updated"] == 3
        assert all(u["properties"]["chunk_role"] == "content" for u in client.updates)
        # A classifier blow-up is NOT a per-document error (it fails open silently).
        assert stats["errors"] == []

    def test_length_mismatch_fails_open_to_default_role(self):
        rows_by_doc, doc_ids = _make_rows()
        client = _FakeClient(rows_by_doc, doc_ids)
        classify = _classifier(wrong_length=True)

        with patch("src.vector_db.weaviate.store.Filter", _patched_filter()):
            stats = backfill_roles_from_corpus(
                client,
                document_ids=["doc-A"],  # 2 chunks, classifier returns 1
                classify_fn=classify,
                default_role="content",
            )

        assert stats["chunks_updated"] == 2
        assert all(u["properties"]["chunk_role"] == "content" for u in client.updates)

    def test_update_error_is_isolated_not_fatal(self):
        """A failed update on one chunk records an error but doesn't stop the rest."""
        rows_by_doc, doc_ids = _make_rows()
        client = _FakeClient(rows_by_doc, doc_ids, raise_on={"a0"})
        classify = _classifier()

        with patch("src.vector_db.weaviate.store.Filter", _patched_filter()):
            stats = backfill_roles_from_corpus(
                client, document_ids=["doc-A", "doc-B"], classify_fn=classify,
            )

        # a0 failed; a1 and b0 still updated.
        assert {u["uuid"] for u in client.updates} == {"a1", "b0"}
        assert stats["chunks_updated"] == 2
        assert len(stats["errors"]) == 1
        assert "a0" in stats["errors"][0]

    def test_enumeration_failure_is_fatal_but_no_raise(self):
        """If document enumeration fails, return stats with an error, never raise."""
        client = _FakeClient({}, [])

        def _boom(*a, **k):
            raise RuntimeError("aggregate down")

        with patch(
            "src.ingest.embedding.common.role_backfill.iter_document_ids",
            side_effect=_boom,
        ):
            stats = backfill_roles_from_corpus(client, classify_fn=_classifier())

        assert stats["documents_seen"] == 0
        assert stats["chunks_updated"] == 0
        assert len(stats["errors"]) == 1


class TestEnumerationAndLimit:
    def test_enumerates_all_documents_when_no_ids(self):
        rows_by_doc, doc_ids = _make_rows()
        client = _FakeClient(rows_by_doc, doc_ids)
        classify = _classifier()

        with patch("src.vector_db.weaviate.store.Filter", _patched_filter()):
            stats = backfill_roles_from_corpus(client, classify_fn=classify)

        assert stats["documents_seen"] == 2
        assert stats["chunks_updated"] == 3

    def test_limit_caps_documents_processed(self):
        rows_by_doc, doc_ids = _make_rows()
        client = _FakeClient(rows_by_doc, doc_ids)
        classify = _classifier()

        with patch("src.vector_db.weaviate.store.Filter", _patched_filter()):
            stats = backfill_roles_from_corpus(
                client, classify_fn=classify, limit=1,
            )

        assert stats["documents_seen"] == 1
        # Only doc-A's chunks touched.
        assert {u["uuid"] for u in client.updates} == {"a0", "a1"}

    def test_idempotent_rerun_same_result(self):
        """Re-running yields the same in-place updates (deterministic, no dupes)."""
        rows_by_doc, doc_ids = _make_rows()
        classify = _classifier({"Table of contents ......... 5": "navigation"})

        results = []
        for _ in range(2):
            client = _FakeClient(rows_by_doc, doc_ids)
            with patch("src.vector_db.weaviate.store.Filter", _patched_filter()):
                backfill_roles_from_corpus(
                    client, document_ids=["doc-A", "doc-B"], classify_fn=classify,
                )
            results.append(
                {u["uuid"]: u["properties"]["chunk_role"] for u in client.updates}
            )
        assert results[0] == results[1]


# ---------------------------------------------------------------------------
# Runnable wrapper argparse surface.
# ---------------------------------------------------------------------------

class TestScriptArgparse:
    def test_arg_defaults_and_overrides(self):
        from scripts.backfill_chunk_roles import _parse_args

        ns = _parse_args([])
        assert ns.dry_run is False
        assert ns.limit is None

        ns2 = _parse_args([
            "--collection", "MyChunks",
            "--batch-size", "12",
            "--dry-run",
            "--limit", "3",
        ])
        assert ns2.collection == "MyChunks"
        assert ns2.batch_size == 12
        assert ns2.dry_run is True
        assert ns2.limit == 3
