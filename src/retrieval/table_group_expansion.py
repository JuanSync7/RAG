# @summary
# Query-time helper that expands a table-aware retrieval hit by attaching the
# sibling row-block chunks from the same ``table_group_id``. Opt-in, deduped,
# budget-bounded; preserves the original ranking by inserting each fetched
# sibling immediately after its source hit (ordered by block index).
# Exports: expand_table_group_hits
# Deps: weaviate (Filter), src.vector_db.weaviate.store (filter pattern)
# @end-summary
"""Table-aware retrieval expansion.

The chunker splits each table into one or more ``table_row`` chunks ("row
blocks"). Every block restates the caption + column headers and packs as many
whole body rows as fit the token budget, so a single retrieved block is
already self-interpretable (it is never header-blind). All blocks of one table
share a stable ``table_group_id`` and are ordered by ``table_row_block_index``.

What a lone hit still misses is the *rest of the table*: a query may rank one
block of a large multi-block table highest, leaving the other blocks (and their
rows) out of context. ``expand_table_group_hits`` is the query-time consumer
that fixes this: given a list of retrieval hits, for any ``table_row`` hit it
fetches the sibling blocks of the same ``table_group_id`` and inserts them
immediately after the source hit, in block order.

Design contract:

1. **Opt-in** — nothing fires unless the caller passes ``max_rows_per_group >
   0`` (the cap on sibling blocks attached per group). At ``0`` the helper is a
   pure no-op and issues no Weaviate reads.
2. **Deduped** — never attach a chunk that is already in the result list.
   Dedup keys on ``chunk_id`` (preferred) and falls back to ``uuid``.
3. **Budget-bounded** — ``max_groups_to_expand`` caps how many distinct
   ``table_group_id``s trigger a Weaviate read; ``max_rows_per_group`` caps the
   per-group sibling-block fan-out.
4. **Ranking-preserving** — fetched siblings are inserted immediately after
   their source hit, ordered by ``table_row_block_index``. Hits not part of an
   expansion keep their original positions.

The helper expects each hit to be a ``dict`` of the shape returned by
``src.vector_db.weaviate.store.hybrid_search``: ``{"text", "score", "uuid",
"metadata": {...}}``. ``chunk_type`` / ``table_group_id`` / ``chunk_id`` are
read from ``metadata``. Inserted (expanded) hits are tagged with
``metadata["expanded_from"] = <table_group_id>`` so consumers (the document
formatter, the LLM-prompt builder) can render them as context-attached rather
than as primary retrieval results.
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


_TABLE_ROW = "table_row"

# Properties we read off Weaviate objects when building expansion hits.
_RETURN_PROPS = [
    "text",
    "chunk_type",
    "chunk_id",
    "table_group_id",
    "table_row_index",
    "table_row_block_index",
    "table_block_total",
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


def _block_order(obj_or_props: Any) -> int:
    """Sort key for a sibling block: prefer block index, fall back to row index."""
    props = (
        obj_or_props
        if isinstance(obj_or_props, dict)
        else (getattr(obj_or_props, "properties", {}) or {})
    )
    for key in ("table_row_block_index", "table_row_index"):
        val = props.get(key)
        if val is not None:
            try:
                return int(val)
            except (TypeError, ValueError):
                continue
    return 0


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
        "table_row_block_index": props.get("table_row_block_index", -1),
        "table_block_total": props.get("table_block_total", -1),
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
    max_groups_to_expand: int = 8,
) -> list[dict]:
    """Expand each ``table_row`` hit with its sibling row blocks.

    Args:
        hits: Ordered list of retrieval-hit dicts. Each must have a
            ``metadata`` sub-dict; ``chunk_type`` / ``table_group_id`` /
            ``chunk_id`` are read from there.
        client: Live Weaviate client (``weaviate.WeaviateClient``). Only its
            ``collections.get(...).query.fetch_objects(...)`` surface is
            used; pass a mock for unit tests.
        collection: Collection name to query (must match the one the hits
            came from).
        max_rows_per_group: Maximum sibling ``table_row`` block-chunks to
            attach per ``table_group_id``. ``0`` (default) disables expansion
            entirely — the helper becomes a pure no-op and issues no reads.
        max_groups_to_expand: Hard cap on the number of distinct
            ``table_group_id``s that may trigger a Weaviate read in this
            call. Groups beyond the cap pass through unexpanded (still
            present, just not enriched).

    Returns:
        A new list. Original hits keep their relative order; each fetched
        sibling block is inserted immediately after its source hit, ordered
        by ``table_row_block_index``. Inserted hits carry
        ``metadata["expanded_from"] = <table_group_id>``.

    Notes:
        - Only the FIRST hit of a given group expands; siblings already
          present in ``hits`` are deduped and never re-attached.
        - The Weaviate read is at most one ``fetch_objects`` per expanded
          group; budget caps prevent runaway fan-out on long hit lists.
        - On any Weaviate error the affected source hit passes through
          unchanged (we never raise — retrieval results already exist).
    """
    if not hits or max_rows_per_group <= 0:
        return list(hits)

    # 1. Build dedup set from the current results so we never re-attach a
    #    chunk that the caller already has.
    existing_keys: set[str] = {_hit_key(h) for h in hits}
    existing_keys.discard("")  # empty keys must not collapse multiple hits

    # 2. Plan: for the first row hit of each group, fetch and attach sibling
    #    blocks, with a budget on distinct groups touched.
    groups_touched: set[str] = set()
    expansions: dict[int, list[dict]] = {}  # source_index -> inserted hits

    for idx, hit in enumerate(hits):
        meta = hit.get("metadata") or {}
        if meta.get("chunk_type") != _TABLE_ROW:
            continue
        gid = str(meta.get("table_group_id") or "")
        if not gid or gid in groups_touched:
            continue
        if len(groups_touched) >= max_groups_to_expand:
            continue
        # Fetch enough to cover the cap plus the source block itself.
        objs = _fetch_group(
            client=client, collection=collection,
            table_group_id=gid,
            limit=max_rows_per_group + 1,
        )
        groups_touched.add(gid)
        siblings = [
            o for o in objs
            if (getattr(o, "properties", {}) or {}).get("chunk_type") == _TABLE_ROW
        ]
        siblings.sort(key=_block_order)
        inserted = 0
        for o in siblings:
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

    # 3. Stitch the output: walk original list, splicing each group's sibling
    #    blocks immediately after their source hit.
    out: list[dict] = []
    for idx, hit in enumerate(hits):
        out.append(hit)
        out.extend(expansions.get(idx, []))
    return out
