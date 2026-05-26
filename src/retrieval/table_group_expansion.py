# @summary
# Query-time helper that expands table-aware retrieval hits by attaching the
# sibling chunks (summary <-> rows) from the same ``table_group_id``. Opt-in,
# deduped, budget-bounded; preserves the original ranking by inserting each
# expansion immediately adjacent to its source hit.
# Exports: expand_table_group_hits
# Deps: weaviate (Filter), src.vector_db.weaviate.store (filter pattern)
# @end-summary
"""Table-aware retrieval expansion.

The chunker writes two flavours of table chunk: a ``table_summary`` (one per
table, holds the caption + full markdown reconstruction) and many
``table_row`` chunks (one per body row). Both share a stable
``table_group_id``. By default, retrieval only returns whatever the query
ranked highest — which is often a single row in isolation, missing the
column headers and caption that make it interpretable.

``expand_table_group_hits`` is the query-time consumer that fixes this:
given a list of retrieval hits, it fetches the matching sibling(s) for any
table hit and inserts them adjacent to the source hit.

Design contract:

1. **Opt-in** — caller decides via the keyword args; nothing fires unless
   the caller explicitly asks for it (callers wire this into their post-rank
   stage).
2. **Deduped** — never attach a chunk that's already in the result list.
   Dedup keys on ``chunk_id`` (preferred) and falls back to ``uuid``.
3. **Budget-bounded** — ``max_groups_to_expand`` caps how many distinct
   ``table_group_id``s trigger a Weaviate read; ``max_rows_per_group`` caps
   the per-group row fan-out on summary hits.
4. **Ranking-preserving** — expansions are inserted immediately adjacent to
   their source hit (summary before row; rows after summary). Hits not part
   of an expansion keep their original positions.

The helper expects each hit to be a ``dict`` of the shape returned by
``src.vector_db.weaviate.store.hybrid_search``: ``{"text", "score", "uuid",
"metadata": {...}}``. ``chunk_type`` / ``table_group_id`` / ``chunk_id`` are
read from ``metadata``. Inserted (expanded) hits are tagged with
``metadata["expanded_from"] = <table_group_id>`` so consumers (the document
formatter, the LLM-prompt builder) can render them as context-attached
rather than as primary retrieval results.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


__all__ = ["expand_table_group_hits"]


# ---------------------------------------------------------------------------
# Filter builder (factored out so tests can stub the Weaviate Filter API)
# ---------------------------------------------------------------------------


def _filter_for_group(table_group_id: str) -> Any:
    """Return a Weaviate ``Filter`` matching ``table_group_id``.

    Factored out so tests can patch it with a stub that side-loads the gid
    onto the returned object (lets the mock client route the right fixture).
    Production callers get the real ``Filter.by_property(...).equal(...)``.
    """
    # Imported lazily so the module is import-safe even where the weaviate
    # client lib is not installed (e.g. lightweight unit-test runs).
    from weaviate.classes.query import Filter  # type: ignore

    return Filter.by_property("table_group_id").equal(table_group_id)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


_TABLE_SUMMARY = "table_summary"
_TABLE_ROW = "table_row"

# Properties we read off Weaviate objects when building expansion hits.
_RETURN_PROPS = [
    "text",
    "chunk_type",
    "chunk_id",
    "table_group_id",
    "table_row_index",
    "table_id",
    "table_caption",
    "table_markdown",
    "source",
    "source_key",
    "source_uri",
    "section_path",
    "heading",
]


def _hit_key(hit: dict) -> str:
    """Stable dedup key for a hit dict (prefers metadata.chunk_id, then uuid)."""
    meta = hit.get("metadata") or {}
    return str(meta.get("chunk_id") or hit.get("uuid") or "")


def _obj_to_hit(obj: Any, *, expanded_from: str) -> dict:
    """Convert a Weaviate object to the standard retrieval-hit dict shape."""
    props = dict(getattr(obj, "properties", {}) or {})
    uuid = str(getattr(obj, "uuid", "") or "")
    chunk_id = str(props.get("chunk_id") or uuid)
    meta = {
        "chunk_id": chunk_id,
        "chunk_type": props.get("chunk_type", ""),
        "table_group_id": props.get("table_group_id", ""),
        "table_row_index": props.get("table_row_index", -1),
        "table_id": props.get("table_id", ""),
        "table_caption": props.get("table_caption", ""),
        "table_markdown": props.get("table_markdown", ""),
        "source": props.get("source", ""),
        "source_key": props.get("source_key", ""),
        "source_uri": props.get("source_uri", ""),
        "section_path": props.get("section_path", ""),
        "heading": props.get("heading", ""),
        "expanded_from": expanded_from,
    }
    return {
        "text": props.get("text", ""),
        "score": 0.0,
        "uuid": uuid,
        "metadata": meta,
    }


def _fetch_group(
    *,
    client: Any,
    collection: str,
    table_group_id: str,
    limit: int,
) -> list[Any]:
    """Fetch all chunks for one ``table_group_id`` (bounded by ``limit``).

    Returns the raw Weaviate objects or ``[]`` on any failure (we never
    raise from the expansion path — the caller already has results to
    return).
    """
    try:
        col = client.collections.get(collection)
        flt = _filter_for_group(table_group_id)
        response = col.query.fetch_objects(
            filters=flt,
            limit=limit,
            return_properties=_RETURN_PROPS,
        )
        return list(getattr(response, "objects", []) or [])
    except Exception:  # pragma: no cover - defensive
        logger.debug(
            "table_group_expansion fetch failed for group=%s",
            table_group_id, exc_info=True,
        )
        return []


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------


def expand_table_group_hits(
    hits: list[dict],
    *,
    client: Any,
    collection: str,
    max_rows_per_group: int = 0,
    fetch_summary_for_row_hits: bool = True,
    max_groups_to_expand: int = 8,
) -> list[dict]:
    """Expand table-aware retrieval hits with their adjacent siblings.

    Args:
        hits: Ordered list of retrieval-hit dicts. Each must have a
            ``metadata`` sub-dict; ``chunk_type`` / ``table_group_id`` /
            ``chunk_id`` are read from there.
        client: Live Weaviate client (``weaviate.WeaviateClient``). Only its
            ``collections.get(...).query.fetch_objects(...)`` surface is
            used; pass a mock for unit tests.
        collection: Collection name to query (must match the one the hits
            came from).
        max_rows_per_group: Maximum row chunks to fetch for any
            ``table_summary`` hit. ``0`` disables summary->row expansion.
        fetch_summary_for_row_hits: When ``True`` (default), each
            ``table_row`` hit triggers a fetch of its ``table_summary``
            sibling. Set ``False`` to disable that side of the expansion.
        max_groups_to_expand: Hard cap on the number of distinct
            ``table_group_id``s that may trigger a Weaviate read in this
            call. Groups beyond the cap pass through unexpanded (still
            present, just not enriched).

    Returns:
        A new list. Original hits keep their relative order; each expansion
        is inserted immediately adjacent to its source hit (summary inserted
        before its row; rows inserted after their summary). Inserted hits
        carry ``metadata["expanded_from"] = <table_group_id>``.

    Notes:
        - If both a row and its summary are already in ``hits``, neither
          side fetches.
        - The Weaviate read is at most one ``fetch_objects`` per expanded
          group; budget caps prevent runaway fan-out on long hit lists.
        - On any Weaviate error the affected source hit passes through
          unchanged (we never raise — retrieval results already exist).
    """
    if not hits:
        return list(hits)

    # 1. Build dedup set from the current results so we never re-attach a
    #    chunk that the caller already has.
    existing_keys: set[str] = {_hit_key(h) for h in hits}
    existing_keys.discard("")  # empty keys must not collapse multiple hits

    # 2. Plan: for each hit decide whether/how it expands, with a budget on
    #    distinct groups touched.
    groups_touched: set[str] = set()
    expansions: dict[int, list[dict]] = {}  # source_index -> inserted hits
    insert_before: set[int] = set()  # source_index whose expansion goes BEFORE

    # First pass: detect groups present in both row and summary form among
    # the existing hits. When the summary is already present we don't need
    # to fetch it for any row hit in the same group; when a row is already
    # present for a group we still allow summary->rows expansion (the
    # presence of the row doesn't cover the *other* rows).
    summary_present_groups: set[str] = set()
    for h in hits:
        meta = h.get("metadata") or {}
        if meta.get("chunk_type") == _TABLE_SUMMARY:
            gid = str(meta.get("table_group_id") or "")
            if gid:
                summary_present_groups.add(gid)

    for idx, hit in enumerate(hits):
        meta = hit.get("metadata") or {}
        ct = meta.get("chunk_type") or ""
        gid = str(meta.get("table_group_id") or "")
        if not gid:
            continue

        if ct == _TABLE_ROW and fetch_summary_for_row_hits:
            # Already have the summary? skip - dedup.
            if gid in summary_present_groups:
                continue
            if (gid not in groups_touched
                    and len(groups_touched) >= max_groups_to_expand):
                continue
            objs = _fetch_group(
                client=client, collection=collection,
                table_group_id=gid,
                # Need at most the summary; cap small.
                limit=max(2, max_rows_per_group + 2),
            )
            groups_touched.add(gid)
            summary = next(
                (o for o in objs
                 if (getattr(o, "properties", {}) or {}).get("chunk_type")
                 == _TABLE_SUMMARY),
                None,
            )
            if summary is None:
                continue
            new_hit = _obj_to_hit(summary, expanded_from=gid)
            key = _hit_key(new_hit)
            if key and key in existing_keys:
                continue
            existing_keys.add(key)
            summary_present_groups.add(gid)
            expansions.setdefault(idx, []).append(new_hit)
            insert_before.add(idx)

        elif ct == _TABLE_SUMMARY and max_rows_per_group > 0:
            if (gid not in groups_touched
                    and len(groups_touched) >= max_groups_to_expand):
                continue
            # Fetch enough to get the cap + the summary itself.
            objs = _fetch_group(
                client=client, collection=collection,
                table_group_id=gid,
                limit=max_rows_per_group + 1,
            )
            groups_touched.add(gid)
            rows = [
                o for o in objs
                if (getattr(o, "properties", {}) or {}).get("chunk_type")
                == _TABLE_ROW
            ]
            rows.sort(
                key=lambda o: int(
                    (getattr(o, "properties", {}) or {}).get(
                        "table_row_index", 0) or 0
                )
            )
            inserted = 0
            for o in rows:
                if inserted >= max_rows_per_group:
                    break
                new_hit = _obj_to_hit(o, expanded_from=gid)
                key = _hit_key(new_hit)
                if key and key in existing_keys:
                    continue
                existing_keys.add(key)
                expansions.setdefault(idx, []).append(new_hit)
                inserted += 1

    if not expansions:
        return list(hits)

    # 3. Stitch the output: walk original list, splicing expansions adjacent
    #    to their source hit.
    out: list[dict] = []
    for idx, hit in enumerate(hits):
        extras = expansions.get(idx, [])
        if extras and idx in insert_before:
            out.extend(extras)
            out.append(hit)
        elif extras:
            out.append(hit)
            out.extend(extras)
        else:
            out.append(hit)
    return out
