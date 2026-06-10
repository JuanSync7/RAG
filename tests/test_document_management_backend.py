# @summary
# Unit tests for MinIO list_documents and Weaviate aggregate functions.
# Covers: list_documents, aggregate_by_source, get_collection_stats, list_collections
# Deps: pytest, unittest.mock, minio, weaviate, src.db.minio.store, src.vector_db.weaviate.store
# @end-summary
"""Backend unit tests for Document & Collection Management (MinIO + Weaviate).

Tests mock all external clients (MinIO, Weaviate) and verify return shapes,
pagination, sidecar fallback, filter composition, and error propagation.
"""

from __future__ import annotations

import json
import sys
import types
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from minio.error import S3Error


def _s3error(code: str) -> S3Error:
    """Build an S3Error whose ``.code`` is exactly ``code``.

    The REAL ``minio.error.S3Error`` signature is
    ``(response, code, message, resource, request_id, host_id, ...)`` — ``code``
    is the SECOND positional arg, not the first. Building it positionally as
    ``S3Error("NoSuchKey", ...)`` therefore sets ``response="NoSuchKey"`` and
    leaves ``.code`` as the next arg, so production's
    ``exc.code in ("NoSuchKey", "NoSuchBucket")`` check (src/db/minio/store.py)
    would NOT match and the swallow/propagate branch under test would not be the
    one actually exercised. (It only "worked" offline because the conftest stub
    happens to take ``code`` first.) Passing ``code=`` by keyword pins ``.code``
    correctly against BOTH the stub and the real package. Mirrors the helper in
    tests/db/test_minio_store.py.
    """
    return S3Error(
        response=None,
        code=code,
        message="test-message",
        resource="test-resource",
        request_id="test-request-id",
        host_id="test-host-id",
    )


# ---------------------------------------------------------------------------
# Ensure weaviate.classes.aggregate stub exists (conftest does not register it)
# ---------------------------------------------------------------------------

def _ensure_weaviate_aggregate_stub() -> None:
    """Register a weaviate.classes.aggregate module stub if missing."""
    if "weaviate.classes.aggregate" not in sys.modules:
        agg_mod = types.ModuleType("weaviate.classes.aggregate")

        class GroupByAggregate:
            def __init__(self, prop: str):
                self.prop = prop

        agg_mod.GroupByAggregate = GroupByAggregate
        sys.modules["weaviate.classes.aggregate"] = agg_mod
        # Also attach to the parent namespace if it exists
        import weaviate
        if not hasattr(weaviate, "classes"):
            weaviate.classes = types.ModuleType("weaviate.classes")
            sys.modules["weaviate.classes"] = weaviate.classes
        weaviate.classes.aggregate = agg_mod


_ensure_weaviate_aggregate_stub()


# ---------------------------------------------------------------------------
# MinIO helpers
# ---------------------------------------------------------------------------


def _make_minio_object(name: str, size: int = 100, last_modified: datetime | None = None) -> MagicMock:
    """Return a minimal stub matching minio Object attributes."""
    obj = MagicMock()
    obj.object_name = name
    obj.size = size
    obj.last_modified = last_modified or datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    return obj


def _make_minio_mock(
    content_objects: list[dict],
    *,
    sidecars: dict[str, dict] | None = None,
    missing_sidecars: set[str] | None = None,
    list_error: Exception | None = None,
) -> MagicMock:
    """Build a MagicMock that behaves like a minio.Minio client.

    Args:
        content_objects: list of dicts with ``name``, ``size``, and optional ``last_modified``.
        sidecars: mapping from stem -> sidecar JSON dict.
        missing_sidecars: stems that should raise S3Error when sidecar is fetched.
        list_error: if set, list_objects will raise this exception.
    """
    client = MagicMock()
    sidecars = sidecars or {}
    missing_sidecars = missing_sidecars or set()

    # Build Object stubs — includes both .md and .meta.json
    objs = [_make_minio_object(
        d["name"], d.get("size", 100), d.get("last_modified")
    ) for d in content_objects]

    if list_error:
        client.list_objects.side_effect = list_error
    else:
        client.list_objects.return_value = iter(objs)

    def _get_object(_bucket, key):
        if key.endswith(".meta.json"):
            stem = key[: -len(".meta.json")]
            if stem in missing_sidecars:
                raise _s3error("NoSuchKey")
            if stem in sidecars:
                resp = MagicMock()
                resp.read.return_value = json.dumps(sidecars[stem]).encode("utf-8")
                return resp
            raise _s3error("NoSuchKey")
        raise _s3error("NoSuchKey")

    client.get_object.side_effect = _get_object
    return client


# ---------------------------------------------------------------------------
# Weaviate helpers
# ---------------------------------------------------------------------------


def _make_group(source_key: str, source: str, connector: str, count: int) -> MagicMock:
    """Return a mock aggregate group matching Weaviate group_by shape."""
    group = MagicMock()
    group.grouped_by.value = source_key
    group.total_count = count
    group.properties = {
        "source": {"top_occurrences": [{"value": source}]},
        "connector": {"top_occurrences": [{"value": connector}]},
    }
    return group


def _make_agg_response(groups: list[MagicMock], total_count: int | None = None) -> MagicMock:
    """Return a mock aggregate response."""
    resp = MagicMock()
    resp.groups = groups
    resp.total_count = total_count
    return resp


def _make_weaviate_client(
    *,
    collection_exists: bool = True,
    agg_groups: list[MagicMock] | None = None,
    total_count: int | None = None,
    collections_list: list[str] | None = None,
    agg_error: Exception | None = None,
    get_error: Exception | None = None,
    list_error: Exception | None = None,
    exists_error: Exception | None = None,
) -> MagicMock:
    """Build a MagicMock that behaves like a weaviate.WeaviateClient."""
    client = MagicMock()
    agg_groups = agg_groups or []
    collections_list = collections_list or []

    # collections.exists
    if exists_error:
        client.collections.exists.side_effect = exists_error
    else:
        client.collections.exists.return_value = collection_exists

    # collections.list_all
    if list_error:
        client.collections.list_all.side_effect = list_error
    else:
        client.collections.list_all.return_value = collections_list

    # collections.get -> collection stub
    col = MagicMock()
    if agg_error:
        col.aggregate.over_all.side_effect = agg_error
    else:
        col.aggregate.over_all.return_value = _make_agg_response(agg_groups, total_count)

    if get_error:
        client.collections.get.side_effect = get_error
    else:
        client.collections.get.return_value = col

    return client


# ===========================================================================
# MinIO list_documents tests
# ===========================================================================


class TestListDocuments:
    """Tests for src.db.minio.store.list_documents (MT-M-01 through MT-M-13)."""

    def _call(self, client, **kwargs):
        from src.db.minio.store import list_documents
        return list_documents(client, **kwargs)

    # -- Happy Path --------------------------------------------------------

    def test_happy_path_returns_documents(self):
        """MT-M-01: Bucket with 3 docs returns list of 3 dicts."""
        objs = [
            {"name": "docs/a.md"},
            {"name": "docs/b.md"},
            {"name": "docs/c.md"},
        ]
        sidecars = {
            "docs/a": {"source_key": "wiki/a"},
            "docs/b": {"source_key": "wiki/b"},
            "docs/c": {"source_key": "wiki/c"},
        }
        client = _make_minio_mock(objs, sidecars=sidecars)
        results = self._call(client)
        assert len(results) == 3
        for doc in results:
            assert "document_id" in doc
            assert "source_key" in doc
            assert "size_bytes" in doc
            assert "last_modified" in doc

    def test_document_id_is_uuid5_of_source_key(self):
        """MT-M-02: document_id == build_document_id(source_key)."""
        from src.db.minio.store import build_document_id
        objs = [{"name": "docs/foo.md"}]
        sidecars = {"docs/foo": {"source_key": "docs/foo"}}
        client = _make_minio_mock(objs, sidecars=sidecars)
        results = self._call(client)
        assert results[0]["document_id"] == build_document_id("docs/foo")

    def test_meta_json_excluded_from_results(self):
        """MT-M-03: .meta.json sidecars are NOT returned as documents."""
        objs = [
            {"name": "docs/a.md"},
            {"name": "docs/a.meta.json"},
            {"name": "docs/b.md"},
            {"name": "docs/b.meta.json"},
        ]
        sidecars = {
            "docs/a": {"source_key": "a"},
            "docs/b": {"source_key": "b"},
        }
        client = _make_minio_mock(objs, sidecars=sidecars)
        results = self._call(client)
        assert len(results) == 2
        for doc in results:
            assert not doc["source_key"].endswith(".meta.json")

    def test_prefix_forwarded_to_list_objects(self):
        """MT-M-04: prefix param is forwarded to minio list_objects."""
        client = _make_minio_mock([])
        self._call(client, prefix="docs/")
        call_kwargs = client.list_objects.call_args
        assert call_kwargs.kwargs.get("prefix") == "docs/" or call_kwargs[1].get("prefix") == "docs/"

    # -- Empty / Boundary --------------------------------------------------

    def test_empty_bucket(self):
        """MT-M-05: Empty bucket returns []."""
        client = _make_minio_mock([])
        results = self._call(client)
        assert results == []

    def test_offset_beyond_result_count(self):
        """MT-M-06: offset beyond result count returns []."""
        objs = [{"name": "a.md"}, {"name": "b.md"}]
        client = _make_minio_mock(objs, missing_sidecars={"a", "b"})
        results = self._call(client, offset=5)
        assert results == []

    def test_limit_one(self):
        """MT-M-07: limit=1 on 5 docs returns exactly 1."""
        objs = [{"name": f"d{i}.md"} for i in range(5)]
        stems = {f"d{i}" for i in range(5)}
        client = _make_minio_mock(objs, missing_sidecars=stems)
        results = self._call(client, limit=1, offset=0)
        assert len(results) == 1

    def test_offset_and_limit(self):
        """MT-M-08: offset=2, limit=2 on 5 docs returns 2 docs."""
        objs = [{"name": f"d{i}.md"} for i in range(5)]
        stems = {f"d{i}" for i in range(5)}
        client = _make_minio_mock(objs, missing_sidecars=stems)
        results = self._call(client, offset=2, limit=2)
        assert len(results) == 2

    def test_pagination_clamped_at_end(self):
        """MT-M-09: offset near end returns remaining docs, not overflow."""
        objs = [{"name": f"d{i}.md"} for i in range(100)]
        stems = {f"d{i}" for i in range(100)}
        client = _make_minio_mock(objs, missing_sidecars=stems)
        results = self._call(client, offset=90, limit=20)
        assert len(results) == 10

    # -- Missing Sidecar ---------------------------------------------------

    def test_missing_sidecar_falls_back_to_stem(self):
        """MT-M-10: Missing sidecar -> source_key is the stem; no exception."""
        objs = [{"name": "stem.md"}]
        client = _make_minio_mock(objs, missing_sidecars={"stem"})
        results = self._call(client)
        assert len(results) == 1
        assert results[0]["source_key"] == "stem"

    def test_partial_sidecars(self):
        """MT-M-11: 1 of 3 sidecars missing; still returns 3 docs."""
        objs = [{"name": "a.md"}, {"name": "b.md"}, {"name": "c.md"}]
        sidecars = {"a": {"source_key": "wiki/a"}, "c": {"source_key": "wiki/c"}}
        client = _make_minio_mock(objs, sidecars=sidecars, missing_sidecars={"b"})
        results = self._call(client)
        assert len(results) == 3
        assert results[1]["source_key"] == "b"

    # -- Error Scenarios ---------------------------------------------------

    def test_list_objects_s3error_propagates(self):
        """MT-M-12: S3Error from list_objects propagates to caller."""
        client = _make_minio_mock(
            [], list_error=_s3error("ServiceUnavailable")
        )
        with pytest.raises(S3Error):
            self._call(client)

    def test_sidecar_s3error_swallowed(self):
        """MT-M-13: S3Error on sidecar fetch is swallowed; doc returned with stem."""
        objs = [{"name": "x.md"}]
        client = _make_minio_mock(objs, missing_sidecars={"x"})
        results = self._call(client)
        assert len(results) == 1
        assert results[0]["source_key"] == "x"


# ===========================================================================
# Weaviate aggregate_by_source tests
# ===========================================================================


class TestAggregateBySource:
    """Tests for src.vector_db.weaviate.store.aggregate_by_source (MT-W-01 through MT-W-08)."""

    def _call(self, client, **kwargs):
        from src.vector_db.weaviate.store import aggregate_by_source
        return aggregate_by_source(client, **kwargs)

    def test_groups_correctly(self):
        """MT-W-01: Two groups return two dicts with correct chunk counts."""
        groups = [
            _make_group("a", "wiki/a", "confluence", 5),
            _make_group("b", "wiki/b", "local_fs", 3),
        ]
        client = _make_weaviate_client(agg_groups=groups)
        results = self._call(client)
        assert len(results) == 2
        assert results[0]["source_key"] == "a"
        assert results[0]["chunk_count"] == 5
        assert results[1]["source_key"] == "b"
        assert results[1]["chunk_count"] == 3

    @patch("src.vector_db.weaviate.store.Filter")
    def test_source_filter_forwarded(self, mock_filter_cls):
        """MT-W-02: source_filter creates a like filter on 'source'."""
        mock_filter_instance = MagicMock()
        mock_filter_cls.by_property.return_value = mock_filter_instance
        mock_filter_instance.like.return_value = mock_filter_instance

        client = _make_weaviate_client(agg_groups=[])
        self._call(client, source_filter="wiki")
        mock_filter_cls.by_property.assert_any_call("source")

    @patch("src.vector_db.weaviate.store.Filter")
    def test_connector_filter_forwarded(self, mock_filter_cls):
        """MT-W-03: connector_filter creates an equal filter on 'connector'."""
        mock_filter_instance = MagicMock()
        mock_filter_cls.by_property.return_value = mock_filter_instance
        mock_filter_instance.equal.return_value = mock_filter_instance

        client = _make_weaviate_client(agg_groups=[])
        self._call(client, connector_filter="confluence")
        mock_filter_cls.by_property.assert_any_call("connector")

    @patch("src.vector_db.weaviate.store.Filter")
    def test_both_filters_compose_with_all_of(self, mock_filter_cls):
        """MT-W-04: Both filters compose with Filter.all_of([...])."""
        mock_f = MagicMock()
        mock_filter_cls.by_property.return_value = mock_f
        mock_f.like.return_value = "source_f"
        mock_f.equal.return_value = "connector_f"
        mock_filter_cls.all_of.return_value = "combined"

        client = _make_weaviate_client(agg_groups=[])
        self._call(client, source_filter="wiki", connector_filter="confluence")
        mock_filter_cls.all_of.assert_called_once()

    def test_no_filters_passes_none(self):
        """MT-W-05: No filters -> filters=None passed to over_all."""
        client = _make_weaviate_client(agg_groups=[])
        self._call(client)
        col = client.collections.get.return_value
        call_kwargs = col.aggregate.over_all.call_args
        # The filters keyword arg should be None
        assert call_kwargs.kwargs.get("filters") is None or call_kwargs[1].get("filters") is None

    def test_source_connector_from_top_occurrences(self):
        """MT-W-06: source/connector extracted from top_occurrences."""
        groups = [_make_group("key", "my_source", "my_conn", 10)]
        client = _make_weaviate_client(agg_groups=groups)
        results = self._call(client)
        assert results[0]["source"] == "my_source"
        assert results[0]["connector"] == "my_conn"

    def test_collection_not_found_keyerror(self):
        """MT-W-07: KeyError propagates when collection not found."""
        client = _make_weaviate_client(get_error=KeyError("not found"))
        with pytest.raises(KeyError):
            self._call(client)

    def test_weaviate_query_error_propagates(self):
        """MT-W-08: WeaviateQueryError propagates from over_all."""
        client = _make_weaviate_client(agg_error=RuntimeError("query failed"))
        with pytest.raises(RuntimeError):
            self._call(client)


# ===========================================================================
# Weaviate get_collection_stats tests
# ===========================================================================


class TestGetCollectionStats:
    """Tests for src.vector_db.weaviate.store.get_collection_stats (MT-W-09 through MT-W-13)."""

    def _call(self, client, **kwargs):
        from src.vector_db.weaviate.store import get_collection_stats
        return get_collection_stats(client, **kwargs)

    def test_collection_exists_full_stats(self):
        """MT-W-09: Collection exists -> full stats returned."""
        # We need fine-grained control over multiple over_all calls.
        client = MagicMock()
        client.collections.exists.return_value = True

        col = MagicMock()
        client.collections.get.return_value = col

        # First call: total count
        total_resp = MagicMock()
        total_resp.total_count = 50

        # Second call: by source_key (groups)
        src_groups = [MagicMock(), MagicMock(), MagicMock()]
        by_source_resp = MagicMock()
        by_source_resp.groups = src_groups

        # Third call: by connector
        conn_group_a = MagicMock()
        conn_group_a.grouped_by.value = "confluence"
        conn_group_a.total_count = 30
        conn_group_b = MagicMock()
        conn_group_b.grouped_by.value = "local_fs"
        conn_group_b.total_count = 20
        by_connector_resp = MagicMock()
        by_connector_resp.groups = [conn_group_a, conn_group_b]

        col.aggregate.over_all.side_effect = [total_resp, by_source_resp, by_connector_resp]

        result = self._call(client)
        assert result is not None
        assert result["chunk_count"] == 50
        assert result["document_count"] == 3
        assert result["connector_breakdown"] == {"confluence": 30, "local_fs": 20}

    def test_collection_missing_returns_none(self):
        """MT-W-10: Collection missing -> returns None."""
        client = _make_weaviate_client(collection_exists=False)
        result = self._call(client)
        assert result is None

    def test_three_queries_issued(self):
        """MT-W-11: Three over_all calls issued when collection exists."""
        client = MagicMock()
        client.collections.exists.return_value = True
        col = MagicMock()
        client.collections.get.return_value = col

        total_resp = MagicMock()
        total_resp.total_count = 0
        by_source_resp = MagicMock()
        by_source_resp.groups = []
        by_connector_resp = MagicMock()
        by_connector_resp.groups = []
        col.aggregate.over_all.side_effect = [total_resp, by_source_resp, by_connector_resp]

        self._call(client)
        assert col.aggregate.over_all.call_count == 3

    def test_empty_collection(self):
        """MT-W-12: Empty collection returns zeroed stats."""
        client = MagicMock()
        client.collections.exists.return_value = True
        col = MagicMock()
        client.collections.get.return_value = col

        total_resp = MagicMock()
        total_resp.total_count = 0
        by_source_resp = MagicMock()
        by_source_resp.groups = []
        by_connector_resp = MagicMock()
        by_connector_resp.groups = []
        col.aggregate.over_all.side_effect = [total_resp, by_source_resp, by_connector_resp]

        result = self._call(client)
        assert result["chunk_count"] == 0
        assert result["document_count"] == 0
        assert result["connector_breakdown"] == {}

    def test_query_error_propagates(self):
        """MT-W-13: WeaviateQueryError propagates from over_all."""
        client = MagicMock()
        client.collections.exists.return_value = True
        col = MagicMock()
        client.collections.get.return_value = col
        col.aggregate.over_all.side_effect = RuntimeError("query boom")

        with pytest.raises(RuntimeError):
            self._call(client)


# ===========================================================================
# Weaviate list_collections tests
# ===========================================================================


class TestListCollections:
    """Tests for src.vector_db.weaviate.store.list_collections (MT-W-14 through MT-W-17)."""

    def _call(self, client):
        from src.vector_db.weaviate.store import list_collections
        return list_collections(client)

    def test_returns_all_collections(self):
        """MT-W-14: Returns all collections with chunk counts."""
        client = MagicMock()
        client.collections.list_all.return_value = ["A", "B"]
        col = MagicMock()
        agg_resp = MagicMock()
        agg_resp.total_count = 42
        col.aggregate.over_all.return_value = agg_resp
        client.collections.get.return_value = col

        results = self._call(client)
        assert len(results) == 2
        assert results[0]["collection_name"] == "A"
        assert results[0]["chunk_count"] == 42

    def test_n_plus_1_round_trips(self):
        """MT-W-15: N+1 round trips — over_all called N times for N collections."""
        client = MagicMock()
        client.collections.list_all.return_value = ["X", "Y", "Z"]
        col = MagicMock()
        agg_resp = MagicMock()
        agg_resp.total_count = 10
        col.aggregate.over_all.return_value = agg_resp
        client.collections.get.return_value = col

        self._call(client)
        assert col.aggregate.over_all.call_count == 3

    def test_empty_client(self):
        """MT-W-16: Empty client returns []."""
        client = _make_weaviate_client(collections_list=[])
        client.collections.list_all.return_value = []
        results = self._call(client)
        assert results == []

    def test_connection_error_propagates(self):
        """MT-W-17: WeaviateConnectionError propagates from list_all."""
        client = _make_weaviate_client(list_error=ConnectionError("refused"))
        with pytest.raises(ConnectionError):
            self._call(client)
