# @summary
# Query-time helper that expands retrieval hits by following the
# cross-reference edges stamped at ingest (``xref_targets`` metadata).
# Scoped MVP: resolves ``section`` / ``section_symbol`` refs only against
# chunks whose ``section_path`` contains the cited section; table/figure/
# appendix resolution is TODO (needs a caption registry — out of scope).
# One-hop only — does NOT transitively expand the targets themselves.
# Exports: expand_xref_hits
# Deps: weaviate (Filter), config.settings
# @end-summary
"""Cross-reference (xref) retrieval expansion — MVP.

The ingestion pipeline stamps each chunk with
``metadata["xref_targets"]`` — a JSON-encoded list of
``{"type": <ref_type>, "value": <ref_value>}`` produced by
``src.ingest.common.shared.cross_refs``. This module is the query-time
consumer: given a list of retrieval hits, it decodes the targets and, for
the *scoped* set of ref types it knows how to resolve (currently
``section`` and ``section_symbol``), fetches the target chunk(s) and
inserts them adjacent to the source hit.

Design contract:

1. **Opt-in** — caller passes ``enabled=True`` (config flag
   ``RAG_XREF_EXPANSION_ENABLED`` defaults to off).
2. **One hop only** — the inserted chunks are NOT themselves re-expanded.
   Transitive walks would amplify cost and increase the risk of pulling
   irrelevant siblings.
3. **Budget-bounded** — ``max_per_hit`` caps expansions per source hit,
   ``max_total`` caps expansions across the whole call.
4. **Deduped** — never attach a chunk already in the result list.
5. **Ranking-preserving** — expansions are inserted immediately after
   their source hit.
6. **Provenance** — each inserted hit carries
   ``metadata["expanded_from"] = f"xref:{ref_type}:{value}"`` so traces
   stay debuggable. Same convention as table-group expansion's
   ``expanded_from``.

**Out of scope (TODO):** ``table`` / ``figure`` / ``appendix`` ref
resolution. These require a caption-to-chunk registry (e.g. an index
keyed on ``table_caption`` / figure caption text) that the current schema
does not yet maintain. We log a debug line and pass through.
"""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


__all__ = ["expand_xref_hits"]


# Ref types the MVP resolver currently knows how to follow.
_SUPPORTED_REF_TYPES = frozenset({"section", "section_symbol"})

# Properties we read off Weaviate objects when building expansion hits.
_RETURN_PROPS = [
    "text",
    "chunk_type",
    "chunk_id",
    "table_group_id",
    "section_path",
    "heading",
    "source",
    "source_key",
    "source_uri",
    "document_id",
    "page_no",
    "page_label",
]


# ---------------------------------------------------------------------------
# Filter builder (factored out so tests can stub the Weaviate Filter API)
# ---------------------------------------------------------------------------


def _filter_for_section(section_value: str) -> Any:
    """Return a Weaviate ``Filter`` matching chunks whose ``section_path``
    contains ``section_value``.

    The match is a substring-style ``like`` because ``section_path`` is a
    breadcrumb (``"Doc > 3.1 Subsection"``) while the ref value is just
    ``"3.1"`` — a contains-style filter is the simplest correct join.
    """
    # Lazy import so the module stays importable in light unit-test envs
    # where the weaviate client lib may not be installed.
    from weaviate.classes.query import Filter  # type: ignore

    return Filter.by_property("section_path").like(f"*{section_value}*")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _hit_key(hit: dict) -> str:
    """Stable dedup key (prefers metadata.chunk_id, then uuid)."""
    meta = hit.get("metadata") or {}
    return str(meta.get("chunk_id") or hit.get("uuid") or "")


def _decode_targets(raw: Any) -> list[dict]:
    """Decode the JSON ``xref_targets`` payload; return ``[]`` on bad input."""
    if not raw:
        return []
    if isinstance(raw, list):
        return raw  # already decoded (some hit dicts may pre-parse)
    try:
        decoded = json.loads(raw)
        if isinstance(decoded, list):
            return decoded
        return []
    except (TypeError, ValueError):
        return []


def _normalise_section_value(ref_type: str, value: str) -> str:
    """Strip the leading ``§`` / "Section" so we can match against
    ``section_path`` which holds bare numbers like ``"3.1"`` in the
    breadcrumb."""
    v = value.strip()
    if ref_type == "section_symbol":
        v = v.lstrip("§").strip()
    elif ref_type == "section":
        # "Section 4" -> "4"; "Section 3.1.2" -> "3.1.2"
        parts = v.split()
        if len(parts) >= 2:
            v = parts[-1]
    return v


def _obj_to_hit(obj: Any, *, expanded_from: str) -> dict:
    """Convert a Weaviate object to the standard retrieval-hit dict shape."""
    props = dict(getattr(obj, "properties", {}) or {})
    uuid = str(getattr(obj, "uuid", "") or "")
    chunk_id = str(props.get("chunk_id") or uuid)
    meta = {
        "chunk_id": chunk_id,
        "chunk_type": props.get("chunk_type", ""),
        "section_path": props.get("section_path", ""),
        "heading": props.get("heading", ""),
        "source": props.get("source", ""),
        "source_key": props.get("source_key", ""),
        "source_uri": props.get("source_uri", ""),
        "document_id": props.get("document_id", ""),
        "page_no": props.get("page_no", 0),
        "page_label": props.get("page_label", ""),
        "expanded_from": expanded_from,
    }
    return {
        "text": props.get("text", ""),
        "score": 0.0,
        "uuid": uuid,
        "metadata": meta,
    }


def _fetch_section_chunks(
    *,
    client: Any,
    collection: str,
    section_value: str,
    limit: int,
) -> list[Any]:
    """Fetch chunks whose ``section_path`` contains ``section_value``.

    Returns ``[]`` on any failure — we never raise from the expansion path.
    """
    try:
        col = client.collections.get(collection)
        flt = _filter_for_section(section_value)
        response = col.query.fetch_objects(
            filters=flt,
            limit=limit,
            return_properties=_RETURN_PROPS,
        )
        return list(getattr(response, "objects", []) or [])
    except Exception:  # pragma: no cover - defensive
        logger.debug(
            "xref_expansion fetch failed for section=%s", section_value,
            exc_info=True,
        )
        return []


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------


def expand_xref_hits(
    hits: list[dict],
    *,
    client: Any,
    collection: str,
    enabled: bool = False,
    max_per_hit: int = 2,
    max_total: int = 6,
) -> list[dict]:
    """Expand retrieval hits by following chunk-stamped xref edges.

    Args:
        hits: Ordered list of retrieval-hit dicts. Each is expected to have
            ``metadata["xref_targets"]`` as a JSON-encoded list (the format
            written by ingest); absent / malformed payloads are tolerated.
        client: Live Weaviate client. Only its
            ``collections.get(...).query.fetch_objects(...)`` surface is
            used; pass a mock for unit tests.
        collection: Collection name to query.
        enabled: When ``False`` (default), returns ``hits`` unchanged.
            Wired to ``config.settings.RAG_XREF_EXPANSION_ENABLED``.
        max_per_hit: Hard cap on expansions inserted per source hit.
        max_total: Hard cap on total expansions across the whole call.

    Returns:
        A new list with original hits in their relative order; each
        expansion is inserted immediately after its source hit. Inserted
        hits carry ``metadata["expanded_from"] = "xref:{type}:{value}"``.

    Scope:
        Only ``section`` and ``section_symbol`` ref types are resolved
        (against ``section_path`` substring match). Other ref types
        (``table`` / ``figure`` / ``appendix``) are intentionally
        pass-through in this MVP — see module docstring.

        One-hop only — inserted chunks are NOT themselves expanded.
    """
    if not enabled or not hits:
        return list(hits)

    existing_keys: set[str] = {_hit_key(h) for h in hits}
    existing_keys.discard("")

    expansions: dict[int, list[dict]] = {}
    total_inserted = 0

    for idx, hit in enumerate(hits):
        if total_inserted >= max_total:
            break
        meta = hit.get("metadata") or {}
        targets = _decode_targets(meta.get("xref_targets"))
        if not targets:
            continue

        per_hit_inserted = 0
        for ref in targets:
            if total_inserted >= max_total or per_hit_inserted >= max_per_hit:
                break
            ref_type = str(ref.get("type") or "")
            ref_value = str(ref.get("value") or "")
            if not ref_type or not ref_value:
                continue
            if ref_type not in _SUPPORTED_REF_TYPES:
                # TODO: implement table/figure/appendix resolution. Needs a
                # caption registry (e.g. a property index on table_caption
                # / figure caption text). Out of scope for the MVP.
                logger.debug(
                    "xref_expansion: ref_type=%s not yet supported (value=%r)",
                    ref_type, ref_value,
                )
                continue

            section_value = _normalise_section_value(ref_type, ref_value)
            if not section_value:
                continue

            objs = _fetch_section_chunks(
                client=client, collection=collection,
                section_value=section_value,
                # Fetch a small window — the per-hit cap clamps anyway.
                limit=max(2, max_per_hit + 1),
            )
            if not objs:
                continue

            stamped = f"xref:{ref_type}:{ref_value}"
            for obj in objs:
                if total_inserted >= max_total or per_hit_inserted >= max_per_hit:
                    break
                new_hit = _obj_to_hit(obj, expanded_from=stamped)
                key = _hit_key(new_hit)
                if key and key in existing_keys:
                    continue
                existing_keys.add(key)
                expansions.setdefault(idx, []).append(new_hit)
                per_hit_inserted += 1
                total_inserted += 1

    if not expansions:
        return list(hits)

    out: list[dict] = []
    for idx, hit in enumerate(hits):
        out.append(hit)
        extras = expansions.get(idx)
        if extras:
            out.extend(extras)
    return out
