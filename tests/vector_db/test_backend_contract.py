"""Contract + delegation tests for the vector store backend layer.

This suite isolates the *backend abstraction* (``src.vector_db.backend.VectorBackend``
and ``src.vector_db.weaviate.backend.WeaviateBackend``) from the underlying
Weaviate store functions, which live in ``src.vector_db.weaviate.store`` /
``visual_store`` and are exercised elsewhere. Those store functions are imported
into ``backend.py`` under ``_wv_*`` aliases; every delegation test monkeypatches
those aliases on the backend module with ``MagicMock`` spies. This proves the
backend's own logic — default collection resolution (``_col``), argument
forwarding, ``SearchResult`` mapping, and the load-bearing
``SearchFilter`` → ``weaviate.classes.query.Filter`` translation — without
touching real Weaviate, network, or store internals.

The ``client`` handle is an opaque sentinel (a bare ``object()``); the backend
never inspects it, it only threads it through to the store layer.

For the load-bearing filter translation, the backend does a function-local
``from weaviate.classes.query import Filter as WeaviateFilter``. The test env's
``weaviate.classes.query`` module is shared and (depending on collection order)
may be a conftest-installed stub, so these tests inject a *recording* fake
``Filter`` into that module via the ``fake_weaviate_filter`` fixture. The fake
records every ``by_property(...).<op>(value)`` call and every ``&`` combination
as a small immutable node tree (``_Leaf`` / ``_And``), letting the tests assert
the exact property/operator/value mapping and the AND-combination structure the
backend builds — independent of whichever real/stub weaviate is loaded.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

import src.vector_db.weaviate.backend as wb
from src.vector_db.backend import VectorBackend
from src.vector_db.common import DocumentRecord, SearchFilter, SearchResult
from config.settings import WEAVIATE_COLLECTION_NAME

VISUAL_DEFAULT = wb.WeaviateBackend._VISUAL_COLLECTION_DEFAULT


# --------------------------------------------------------------------------- #
# Recording fake Filter for deterministic translation teeth
# --------------------------------------------------------------------------- #
class _Leaf:
    """A recorded ``by_property(prop).<op>(value)`` leaf clause."""

    def __init__(self, prop, op, value):
        self.prop = prop
        self.op = op
        self.value = value

    def __and__(self, other):
        return _And([self, other])

    def __or__(self, other):
        return _Or([self, other])

    def as_tuple(self):
        return (self.prop, self.op, self.value)


class _And:
    """A recorded AND-combination of clauses (left-folded by the backend)."""

    def __init__(self, clauses):
        self.clauses = list(clauses)

    def __and__(self, other):
        return _And([*self.clauses, other])

    def __or__(self, other):
        return _Or([self, other])

    def leaves(self):
        return [c.as_tuple() for c in self.clauses]


class _Or:
    """A recorded OR-combination — distinguishes an AND→OR translation bug."""

    def __init__(self, clauses):
        self.clauses = list(clauses)

    def __or__(self, other):
        return _Or([*self.clauses, other])

    def __and__(self, other):
        return _And([self, other])

    def leaves(self):
        return [c.as_tuple() for c in self.clauses]


class _RecordingFilter:
    """Stand-in for ``weaviate.classes.query.Filter`` that records calls."""

    @staticmethod
    def by_property(name):
        class _Prop:
            def equal(self, v):
                return _Leaf(name, "eq", v)

            def not_equal(self, v):
                return _Leaf(name, "ne", v)

            def like(self, v):
                return _Leaf(name, "like", v)

            def greater_than(self, v):
                return _Leaf(name, "gt", v)

            def less_than(self, v):
                return _Leaf(name, "lt", v)

            def greater_or_equal(self, v):
                return _Leaf(name, "gte", v)

            def less_or_equal(self, v):
                return _Leaf(name, "lte", v)

        return _Prop()


@pytest.fixture
def fake_weaviate_filter(monkeypatch):
    """Install the recording Filter into ``weaviate.classes.query``.

    The backend imports ``Filter`` lazily inside ``_single_filter``, so patching
    the module attribute is sufficient and survives whichever weaviate module
    (real or conftest stub) is already loaded.
    """
    mod = sys.modules["weaviate.classes.query"]
    monkeypatch.setattr(mod, "Filter", _RecordingFilter, raising=False)
    return _RecordingFilter


# --------------------------------------------------------------------------- #
# ABC contract
# --------------------------------------------------------------------------- #
class TestAbcContract:
    def test_vector_backend_cannot_be_instantiated(self):
        """VectorBackend is abstract — direct instantiation raises TypeError."""
        with pytest.raises(TypeError):
            VectorBackend()

    def test_weaviate_backend_instantiates(self):
        """WeaviateBackend is concrete and constructs without arguments."""
        assert isinstance(wb.WeaviateBackend(), VectorBackend)

    def test_weaviate_backend_overrides_all_abstractmethods(self):
        """WeaviateBackend implements every abstractmethod (no abstracts remain)."""
        assert wb.WeaviateBackend.__abstractmethods__ == frozenset()

    def test_vector_backend_declares_expected_abstractmethods(self):
        """The ABC pins the full set of required backend operations."""
        assert VectorBackend.__abstractmethods__ == frozenset({
            "create_persistent_client",
            "get_ephemeral_client",
            "ensure_collection",
            "collection_exists",
            "add_documents",
            "update_chunk_content",
            "search",
            "delete_collection",
            "delete_by_source",
            "delete_by_source_key",
            "aggregate_by_source",
            "get_collection_stats",
            "list_collections",
            "ensure_visual_collection",
            "add_visual_documents",
            "delete_visual_by_source_key",
            "search_visual",
        })


# --------------------------------------------------------------------------- #
# _col resolution (load-bearing)
# --------------------------------------------------------------------------- #
class TestColResolution:
    def test_none_resolves_to_default(self):
        """_col(None) falls back to the configured WEAVIATE_COLLECTION_NAME."""
        assert wb.WeaviateBackend()._col(None) == WEAVIATE_COLLECTION_NAME

    def test_explicit_collection_preserved(self):
        """_col('custom') returns the caller-supplied collection unchanged."""
        assert wb.WeaviateBackend()._col("custom") == "custom"

    def test_threaded_none_resolves_to_default(self, monkeypatch):
        """ensure_collection(None) forwards collection=WEAVIATE_COLLECTION_NAME."""
        spy = MagicMock()
        monkeypatch.setattr(wb, "_wv_ensure_collection", spy)
        client = object()

        wb.WeaviateBackend().ensure_collection(client, collection=None)

        spy.assert_called_once_with(client, collection=WEAVIATE_COLLECTION_NAME)

    def test_threaded_explicit_collection_forwarded(self, monkeypatch):
        """ensure_collection('custom') forwards collection='custom' verbatim."""
        spy = MagicMock()
        monkeypatch.setattr(wb, "_wv_ensure_collection", spy)
        client = object()

        wb.WeaviateBackend().ensure_collection(client, collection="custom")

        spy.assert_called_once_with(client, collection="custom")


# --------------------------------------------------------------------------- #
# Simple collection-scoped delegations
# --------------------------------------------------------------------------- #
class TestCollectionScopedDelegation:
    def test_collection_exists_resolves_and_returns(self, monkeypatch):
        """collection_exists threads resolved collection + returns store bool."""
        spy = MagicMock(return_value=True)
        monkeypatch.setattr(wb, "_wv_collection_exists", spy)
        client = object()

        result = wb.WeaviateBackend().collection_exists(client, collection=None)

        assert result is True
        spy.assert_called_once_with(client, collection=WEAVIATE_COLLECTION_NAME)

    def test_delete_collection_resolves_explicit(self, monkeypatch):
        """delete_collection threads an explicit collection unchanged."""
        spy = MagicMock(return_value=None)
        monkeypatch.setattr(wb, "_wv_delete_collection", spy)
        client = object()

        wb.WeaviateBackend().delete_collection(client, collection="custom")

        spy.assert_called_once_with(client, collection="custom")

    def test_delete_by_source_forwards_source_and_resolved(self, monkeypatch):
        """delete_by_source forwards source positionally + resolved collection."""
        spy = MagicMock(return_value=4)
        monkeypatch.setattr(wb, "_wv_delete_by_source", spy)
        client = object()

        result = wb.WeaviateBackend().delete_by_source(
            client, "src/path.pdf", collection=None
        )

        assert result == 4
        spy.assert_called_once_with(
            client, "src/path.pdf", collection=WEAVIATE_COLLECTION_NAME
        )

    def test_delete_by_source_key_forwards_legacy_and_resolved(self, monkeypatch):
        """delete_by_source_key forwards source_key + legacy_source + resolved."""
        spy = MagicMock(return_value=2)
        monkeypatch.setattr(wb, "_wv_delete_by_source_key", spy)
        client = object()

        result = wb.WeaviateBackend().delete_by_source_key(
            client, "key-1", "legacy/path", collection="custom"
        )

        assert result == 2
        spy.assert_called_once_with(
            client, "key-1", "legacy/path", collection="custom"
        )

    def test_aggregate_by_source_forwards_filters_and_resolved(self, monkeypatch):
        """aggregate_by_source forwards both filters + resolved collection."""
        sentinel = [{"source_key": "k", "count": 3}]
        spy = MagicMock(return_value=sentinel)
        monkeypatch.setattr(wb, "_wv_aggregate_by_source", spy)
        client = object()

        result = wb.WeaviateBackend().aggregate_by_source(
            client,
            collection=None,
            source_filter="src",
            connector_filter="conn",
        )

        assert result is sentinel
        spy.assert_called_once_with(
            client,
            collection=WEAVIATE_COLLECTION_NAME,
            source_filter="src",
            connector_filter="conn",
        )

    def test_get_collection_stats_resolves_and_returns(self, monkeypatch):
        """get_collection_stats threads resolved collection + returns store dict."""
        sentinel = {"count": 9}
        spy = MagicMock(return_value=sentinel)
        monkeypatch.setattr(wb, "_wv_get_collection_stats", spy)
        client = object()

        result = wb.WeaviateBackend().get_collection_stats(client, collection=None)

        assert result is sentinel
        spy.assert_called_once_with(client, collection=WEAVIATE_COLLECTION_NAME)

    def test_list_collections_delegates_without_collection(self, monkeypatch):
        """list_collections delegates with only the client (no collection arg)."""
        sentinel = [{"name": "RAGDocuments", "count": 1}]
        spy = MagicMock(return_value=sentinel)
        monkeypatch.setattr(wb, "_wv_list_collections", spy)
        client = object()

        result = wb.WeaviateBackend().list_collections(client)

        assert result is sentinel
        spy.assert_called_once_with(client)


# --------------------------------------------------------------------------- #
# add_documents — unpacks DocumentRecord lists
# --------------------------------------------------------------------------- #
class TestAddDocuments:
    def test_unpacks_records_and_resolves_collection(self, monkeypatch):
        """add_documents splits records into texts/embeddings/metadatas lists."""
        spy = MagicMock(return_value=2)
        monkeypatch.setattr(wb, "_wv_add_documents", spy)
        client = object()
        docs = [
            DocumentRecord(text="t0", embedding=[0.1, 0.2], metadata={"i": 0}),
            DocumentRecord(text="t1", embedding=[0.3, 0.4], metadata={"i": 1}),
        ]

        result = wb.WeaviateBackend().add_documents(client, docs, collection=None)

        assert result == 2
        spy.assert_called_once_with(
            client,
            texts=["t0", "t1"],
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            metadatas=[{"i": 0}, {"i": 1}],
            collection=WEAVIATE_COLLECTION_NAME,
        )

    def test_explicit_collection_forwarded(self, monkeypatch):
        """add_documents threads an explicit collection through unchanged."""
        spy = MagicMock(return_value=1)
        monkeypatch.setattr(wb, "_wv_add_documents", spy)
        client = object()
        docs = [DocumentRecord(text="only", embedding=[1.0], metadata={"k": "v"})]

        wb.WeaviateBackend().add_documents(client, docs, collection="custom")

        spy.assert_called_once_with(
            client,
            texts=["only"],
            embeddings=[[1.0]],
            metadatas=[{"k": "v"}],
            collection="custom",
        )


# --------------------------------------------------------------------------- #
# update_chunk_content
# --------------------------------------------------------------------------- #
class TestUpdateChunkContent:
    def test_forwards_all_kwargs_and_resolved(self, monkeypatch):
        """update_chunk_content forwards uuid + text/hash/fingerprint + resolved."""
        spy = MagicMock(return_value=True)
        monkeypatch.setattr(wb, "_wv_update_chunk_content", spy)
        client = object()

        result = wb.WeaviateBackend().update_chunk_content(
            client,
            "uuid-1",
            text="new text",
            content_hash="h123",
            fuzzy_fingerprint="fp456",
            collection=None,
        )

        assert result is True
        spy.assert_called_once_with(
            client,
            "uuid-1",
            text="new text",
            content_hash="h123",
            fuzzy_fingerprint="fp456",
            collection=WEAVIATE_COLLECTION_NAME,
        )

    def test_default_fingerprint_is_none(self, monkeypatch):
        """An unspecified fuzzy_fingerprint defaults to None on the store call."""
        spy = MagicMock(return_value=False)
        monkeypatch.setattr(wb, "_wv_update_chunk_content", spy)
        client = object()

        result = wb.WeaviateBackend().update_chunk_content(
            client, "uuid-2", text="t", content_hash="h", collection="custom"
        )

        assert result is False
        spy.assert_called_once_with(
            client,
            "uuid-2",
            text="t",
            content_hash="h",
            fuzzy_fingerprint=None,
            collection="custom",
        )


# --------------------------------------------------------------------------- #
# search — delegation, param forwarding, result mapping
# --------------------------------------------------------------------------- #
class TestSearchDelegation:
    def test_forwards_query_params_and_resolved_collection(self, monkeypatch):
        """search forwards query/embedding/alpha/limit + resolved collection."""
        spy = MagicMock(return_value=[])
        monkeypatch.setattr(wb, "_wv_hybrid_search", spy)
        client = object()

        wb.WeaviateBackend().search(
            client,
            "the query",
            [0.5, 0.6],
            alpha=0.25,
            limit=11,
            filters=None,
            collection=None,
        )

        spy.assert_called_once_with(
            client,
            "the query",
            [0.5, 0.6],
            0.25,
            11,
            None,
            collection=WEAVIATE_COLLECTION_NAME,
        )

    def test_no_filters_passes_none(self, monkeypatch):
        """With filters=None the translated filter argument is None."""
        spy = MagicMock(return_value=[])
        monkeypatch.setattr(wb, "_wv_hybrid_search", spy)

        wb.WeaviateBackend().search(
            object(), "q", [0.1], alpha=0.5, limit=5, filters=None, collection="c"
        )

        assert spy.call_args.args[5] is None

    def test_empty_filter_list_passes_none(self, monkeypatch):
        """An empty filter list also yields a None translated filter."""
        spy = MagicMock(return_value=[])
        monkeypatch.setattr(wb, "_wv_hybrid_search", spy)

        wb.WeaviateBackend().search(
            object(), "q", [0.1], alpha=0.5, limit=5, filters=[], collection="c"
        )

        assert spy.call_args.args[5] is None

    def test_maps_raw_results_to_search_results(self, monkeypatch):
        """Raw store dicts are mapped onto SearchResult with the resolved collection."""
        raw = [
            {"text": "a", "score": 0.9, "metadata": {"i": 0}},
            {"text": "b", "score": 0.4, "metadata": {"i": 1}},
        ]
        spy = MagicMock(return_value=raw)
        monkeypatch.setattr(wb, "_wv_hybrid_search", spy)

        results = wb.WeaviateBackend().search(
            object(), "q", [0.1], alpha=0.5, limit=5, filters=None, collection="myc"
        )

        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)
        assert results[0].text == "a"
        assert results[0].score == 0.9
        assert results[0].metadata == {"i": 0}
        assert results[0].collection == "myc"
        assert results[1].text == "b"
        assert results[1].collection == "myc"


# --------------------------------------------------------------------------- #
# Filter translation (load-bearing teeth) — uses a recording fake Filter
# --------------------------------------------------------------------------- #
class TestFilterTranslation:
    def test_none_and_empty_translate_to_none(self):
        """No clauses → None (no filter applied at the store layer)."""
        be = wb.WeaviateBackend()
        assert be._translate_filters(None) is None
        assert be._translate_filters([]) is None

    def test_single_eq_maps_property_operator_value(self, fake_weaviate_filter):
        """A single eq clause maps property/eq/value field-for-field."""
        be = wb.WeaviateBackend()
        clause = be._translate_filters(
            [SearchFilter(property="source", operator="eq", value="x.pdf")]
        )
        assert clause.as_tuple() == ("source", "eq", "x.pdf")

    def test_each_operator_maps_to_its_filter_method(self, fake_weaviate_filter):
        """ne/like/gt/lt/gte/lte each route to the matching Filter method."""
        be = wb.WeaviateBackend()
        for op in ("ne", "like", "gt", "lt", "gte", "lte"):
            clause = be._translate_filters(
                [SearchFilter(property="p", operator=op, value=3)]
            )
            assert clause.as_tuple() == ("p", op, 3), op

    def test_operator_is_case_insensitive(self, fake_weaviate_filter):
        """Operator matching lowercases the input ('EQ' behaves as 'eq')."""
        be = wb.WeaviateBackend()
        clause = be._translate_filters(
            [SearchFilter(property="p", operator="EQ", value=1)]
        )
        assert clause.as_tuple() == ("p", "eq", 1)

    def test_two_clauses_combine_with_and(self, fake_weaviate_filter):
        """Two SearchFilters AND-combine into a single AND of both leaves in order."""
        be = wb.WeaviateBackend()
        combined = be._translate_filters([
            SearchFilter(property="source", operator="eq", value="x.pdf"),
            SearchFilter(property="page", operator="gt", value=2),
        ])
        assert isinstance(combined, _And)
        assert combined.leaves() == [
            ("source", "eq", "x.pdf"),
            ("page", "gt", 2),
        ]

    def test_not_in_builds_and_chain_of_not_equals(self, fake_weaviate_filter):
        """not_in expands to an AND-chain of not_equal clauses, one per value."""
        be = wb.WeaviateBackend()
        combined = be._translate_filters(
            [SearchFilter(property="tag", operator="not_in", value=["a", "b", "c"])]
        )
        assert isinstance(combined, _And)
        assert combined.leaves() == [
            ("tag", "ne", "a"),
            ("tag", "ne", "b"),
            ("tag", "ne", "c"),
        ]

    def test_not_in_single_value_is_a_leaf_not_equal(self, fake_weaviate_filter):
        """A one-element not_in is a single not_equal leaf, not a combination."""
        be = wb.WeaviateBackend()
        clause = be._translate_filters(
            [SearchFilter(property="tag", operator="not_in", value=["only"])]
        )
        assert isinstance(clause, _Leaf)
        assert clause.as_tuple() == ("tag", "ne", "only")

    def test_empty_not_in_drops_clause_to_none(self, fake_weaviate_filter):
        """A not_in with an empty value list contributes no clause → None."""
        be = wb.WeaviateBackend()
        assert (
            be._translate_filters(
                [SearchFilter(property="tag", operator="not_in", value=[])]
            )
            is None
        )

    def test_unsupported_operator_raises_value_error(self, fake_weaviate_filter):
        """An unknown operator raises ValueError listing valid operators."""
        be = wb.WeaviateBackend()
        with pytest.raises(ValueError, match="Unsupported filter operator"):
            be._translate_filters(
                [SearchFilter(property="p", operator="contains", value="x")]
            )

    def test_translated_filter_threaded_into_hybrid_search(
        self, monkeypatch, fake_weaviate_filter
    ):
        """The translated Filter object is the one passed to _wv_hybrid_search."""
        spy = MagicMock(return_value=[])
        monkeypatch.setattr(wb, "_wv_hybrid_search", spy)

        wb.WeaviateBackend().search(
            object(),
            "q",
            [0.1],
            alpha=0.5,
            limit=5,
            filters=[SearchFilter(property="source", operator="eq", value="x.pdf")],
            collection="c",
        )

        passed = spy.call_args.args[5]
        assert isinstance(passed, _Leaf)
        assert passed.as_tuple() == ("source", "eq", "x.pdf")


# --------------------------------------------------------------------------- #
# Visual collection operations
# --------------------------------------------------------------------------- #
class TestVisualDelegation:
    def test_ensure_visual_collection_default(self, monkeypatch):
        """ensure_visual_collection(None) uses the visual default (positional)."""
        spy = MagicMock()
        monkeypatch.setattr(wb, "_wv_ensure_visual_collection", spy)
        client = object()

        wb.WeaviateBackend().ensure_visual_collection(client, collection=None)

        spy.assert_called_once_with(client, VISUAL_DEFAULT)

    def test_ensure_visual_collection_explicit(self, monkeypatch):
        """An explicit visual collection name is forwarded unchanged."""
        spy = MagicMock()
        monkeypatch.setattr(wb, "_wv_ensure_visual_collection", spy)
        client = object()

        wb.WeaviateBackend().ensure_visual_collection(client, collection="VC")

        spy.assert_called_once_with(client, "VC")

    def test_add_visual_documents_forwards_docs_and_default(self, monkeypatch):
        """add_visual_documents forwards the docs list + visual default."""
        spy = MagicMock(return_value=3)
        monkeypatch.setattr(wb, "_wv_add_visual_documents", spy)
        client = object()
        docs = [{"page": 1}, {"page": 2}, {"page": 3}]

        result = wb.WeaviateBackend().add_visual_documents(
            client, docs, collection=None
        )

        assert result == 3
        spy.assert_called_once_with(client, docs, VISUAL_DEFAULT)

    def test_delete_visual_by_source_key_forwards_key_and_default(self, monkeypatch):
        """delete_visual_by_source_key forwards source_key + visual default."""
        spy = MagicMock(return_value=5)
        monkeypatch.setattr(wb, "_wv_delete_visual_by_source_key", spy)
        client = object()

        result = wb.WeaviateBackend().delete_visual_by_source_key(
            client, "key-9", collection=None
        )

        assert result == 5
        spy.assert_called_once_with(client, "key-9", VISUAL_DEFAULT)

    def test_search_visual_forwards_all_params_and_default(self, monkeypatch):
        """search_visual forwards vector/limit/threshold/tenant + visual default."""
        sentinel = [{"page": 1, "score": 0.8}]
        spy = MagicMock(return_value=sentinel)
        monkeypatch.setattr(wb, "_wv_visual_search", spy)
        client = object()

        result = wb.WeaviateBackend().search_visual(
            client,
            [0.1, 0.2],
            limit=7,
            score_threshold=0.6,
            tenant_id="t-1",
            collection=None,
        )

        assert result is sentinel
        spy.assert_called_once_with(
            client, [0.1, 0.2], 7, 0.6, "t-1", VISUAL_DEFAULT
        )

    def test_search_visual_explicit_collection_and_no_tenant(self, monkeypatch):
        """search_visual threads explicit collection + a None tenant_id default."""
        spy = MagicMock(return_value=[])
        monkeypatch.setattr(wb, "_wv_visual_search", spy)
        client = object()

        wb.WeaviateBackend().search_visual(
            client, [0.3], limit=2, score_threshold=0.1, collection="VC"
        )

        spy.assert_called_once_with(client, [0.3], 2, 0.1, None, "VC")


# --------------------------------------------------------------------------- #
# Client lifecycle
# --------------------------------------------------------------------------- #
class TestClientLifecycle:
    def test_create_persistent_client_delegates(self, monkeypatch):
        """create_persistent_client returns whatever the store factory returns."""
        spy = MagicMock(return_value="CLIENT")
        monkeypatch.setattr(wb, "_wv_create_persistent", spy)

        assert wb.WeaviateBackend().create_persistent_client() == "CLIENT"
        spy.assert_called_once_with()

    def test_get_ephemeral_client_yields_factory_result(self, monkeypatch):
        """The context manager yields the store factory's client sentinel."""
        sentinel = object()
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=sentinel)
        cm.__exit__ = MagicMock(return_value=False)
        spy = MagicMock(return_value=cm)
        monkeypatch.setattr(wb, "_wv_get_ephemeral", spy)

        with wb.WeaviateBackend().get_ephemeral_client() as client:
            assert client is sentinel
        spy.assert_called_once_with()
        cm.__exit__.assert_called_once()

    def test_close_client_calls_client_close(self):
        """close_client invokes client.close() for a non-None client."""
        client = MagicMock()
        wb.WeaviateBackend().close_client(client)
        client.close.assert_called_once_with()

    def test_close_client_none_is_noop(self):
        """close_client(None) returns None without raising."""
        assert wb.WeaviateBackend().close_client(None) is None

    def test_close_client_swallows_close_errors(self):
        """A client.close() exception is swallowed (logged, not raised)."""
        client = MagicMock()
        client.close.side_effect = RuntimeError("boom")
        # Must not raise.
        assert wb.WeaviateBackend().close_client(client) is None
