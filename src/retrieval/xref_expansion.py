# @summary
# Query-time helper that expands retrieval hits by following the
# cross-reference edges stamped at ingest (``xref_targets`` metadata).
# Resolves ``section`` / ``section_symbol`` refs against chunks whose
# ``section_path`` contains the cited section, and ``table`` refs against
# chunks whose ``caption_label`` matches (document-scoped). ``figure`` /
# ``appendix`` resolution is still TODO (no caption registry yet).
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

**Out of scope (TODO):** ``figure`` / ``appendix`` ref resolution.
These need a caption-to-chunk registry that the current schema does not
yet maintain. ``table`` is supported via the ``caption_label`` property
(stamped at ingest), scoped to the source hit's ``document_id``.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


__all__ = ["expand_xref_hits"]


# Ref types the resolver currently knows how to follow.
# - section / section_symbol  → section_path contains-match (boundary post-filter)
# - table                     → caption_label exact-match, scoped by document_id
# TODO: figure / appendix still need a caption registry.
_SUPPORTED_REF_TYPES = frozenset({"section", "section_symbol", "table", "figure"})


# Anchor-equivalent to ``_FIGURE_CAPTION_LABEL_RE`` in
# ``src.ingest.support.docling`` — duplicated here so the resolver does not
# depend on the ingest module. Both sides MUST agree on the canonical form
# ``"Figure N"`` so the round-trip stays symmetric (FIG-2 contract).
_FIGURE_LABEL_RE = re.compile(
    r"^\s*(?:Figure|Fig\.?)\s+(\d+(?:[.-]\d+)?)",
    re.IGNORECASE,
)


def _normalize_figure_label(value: str) -> str:
    """Normalise a figure ref value to the canonical ``"Figure N"`` form.

    Returns ``""`` when no figure-prefix matches — the caller treats empty
    as "skip this ref". This is the only place the resolver applies its
    own normalisation, so the ingest side's stamping of ``"Figure N"`` is
    a fast-path; ``"Fig. N"`` / ``"Fig N"`` also round-trip.
    """
    if not value:
        return ""
    m = _FIGURE_LABEL_RE.match(value)
    if not m:
        return ""
    return f"Figure {m.group(1)}"

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
    "caption_label",
    "page_no",
    "page_label",
]


# ---------------------------------------------------------------------------
# Filter builder (factored out so tests can stub the Weaviate Filter API)
# ---------------------------------------------------------------------------


def _filter_for_section(section_value: str) -> Any:
    """Return a Weaviate ``Filter`` matching chunks whose ``section_path``
    contains ``section_value``.

    NOTE: This is a **coarse prefilter** only. Weaviate's ``Filter.like``
    is substring/glob-based and over-matches: a ref value ``"3.1"`` will
    pull chunks whose ``section_path`` contains ``"3.10"``, ``"13.1"``,
    etc. ``_section_value_matches_path`` applies the boundary-aware
    post-filter in Python after objects come back.
    """
    # Lazy import so the module stays importable in light unit-test envs
    # where the weaviate client lib may not be installed.
    from weaviate.classes.query import Filter  # type: ignore

    return Filter.by_property("section_path").like(f"*{section_value}*")


def _filter_for_figure(label: str, document_id: str) -> Any:
    """Return a Weaviate ``Filter`` matching figure chunks whose
    ``caption_label`` equals ``label`` AND ``document_id`` matches AND
    ``chunk_type`` is ``"figure"``.

    Mirrors :func:`_filter_for_table` — the document_id scoping is
    load-bearing (figure captions like ``"Figure 4-1"`` recur across
    datasheets), and the ``chunk_type`` clause prevents accidental
    cross-pollination if a non-figure chunk ever carried a figure-shaped
    ``caption_label``.
    """
    from weaviate.classes.query import Filter  # type: ignore

    return (
        Filter.by_property("caption_label").equal(label)
        & Filter.by_property("document_id").equal(document_id)
        & Filter.by_property("chunk_type").equal("figure")
    )


def _filter_for_table(label: str, document_id: str) -> Any:
    """Return a Weaviate ``Filter`` matching chunks whose ``caption_label``
    equals ``label`` AND whose ``document_id`` equals ``document_id``.

    Tables share captions across documents (every datasheet has its own
    "Table 5-2"), so document_id scoping is **load-bearing**, not advisory.
    Callers MUST pass a non-empty document_id; this is enforced at the
    dispatch site, not here.
    """
    from weaviate.classes.query import Filter  # type: ignore

    return (
        Filter.by_property("caption_label").equal(label)
        & Filter.by_property("document_id").equal(document_id)
    )


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


def _section_value_matches_path(section_value: str, section_path: str) -> bool:
    """Boundary-aware match of a section ref value against a section_path.

    The Weaviate ``Filter.like`` prefilter is substring-based and
    over-matches (``"3.1"`` glob-matches ``"3.10"`` / ``"13.1"`` /
    ``"3.11"``). This helper enforces numeric token boundaries on both
    sides of the candidate match:

    - Left lookbehind ``(?<![\\d.])`` rejects ``"13.1"`` or ``"3.13.1"``
      as a match for ``"3.1"`` (the char before the ``"3"`` is a digit or
      a ``.``).
    - Right lookahead ``(?![\\d])`` rejects ``"3.10"`` as a match for
      ``"3.1"`` but **allows** ``"3.1.4"`` (``.`` is a permitted suffix —
      ``"3.1"`` is a valid prefix of ``"3.1.4"``).

    Empty inputs are treated as no-match.
    """
    if not section_value or not section_path:
        return False
    pattern = r"(?<![\d.])" + re.escape(section_value) + r"(?![\d])"
    return re.search(pattern, section_path) is not None


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
        "caption_label": props.get("caption_label", ""),
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

    The Weaviate ``Filter.like`` filter is a **coarse prefilter** — it
    over-matches at the substring level (``"3.1"`` will pull ``"3.10"``,
    ``"13.1"``). We apply ``_section_value_matches_path`` as a
    boundary-aware Python post-filter on the returned objects so the
    effective match respects numeric token boundaries.

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
        raw = list(getattr(response, "objects", []) or [])
        # Boundary-aware post-filter — Weaviate's like(...) over-matches.
        filtered: list[Any] = []
        for obj in raw:
            props = getattr(obj, "properties", {}) or {}
            sp = str(props.get("section_path") or "")
            if _section_value_matches_path(section_value, sp):
                filtered.append(obj)
        return filtered
    except Exception:  # pragma: no cover - defensive
        logger.debug(
            "xref_expansion fetch failed for section=%s", section_value,
            exc_info=True,
        )
        return []


def _fetch_figure_chunks(
    *,
    client: Any,
    collection: str,
    label: str,
    document_id: str,
    limit: int,
) -> list[Any]:
    """Fetch figure chunks whose ``caption_label`` matches ``label`` and
    whose ``document_id`` matches the source-hit's document.

    Returns ``[]`` on any failure — we never raise from the expansion path.
    """
    try:
        col = client.collections.get(collection)
        flt = _filter_for_figure(label, document_id)
        response = col.query.fetch_objects(
            filters=flt,
            limit=limit,
            return_properties=_RETURN_PROPS,
        )
        return list(getattr(response, "objects", []) or [])
    except Exception:  # pragma: no cover - defensive
        logger.debug(
            "xref_expansion fetch failed for figure=%s doc=%s",
            label, document_id, exc_info=True,
        )
        return []


def _fetch_table_chunks(
    *,
    client: Any,
    collection: str,
    label: str,
    document_id: str,
    limit: int,
) -> list[Any]:
    """Fetch chunks whose ``caption_label`` exactly matches ``label`` AND
    ``document_id`` matches the source-hit's document.

    Returns ``[]`` on any failure — we never raise from the expansion path.
    """
    try:
        col = client.collections.get(collection)
        flt = _filter_for_table(label, document_id)
        response = col.query.fetch_objects(
            filters=flt,
            limit=limit,
            return_properties=_RETURN_PROPS,
        )
        return list(getattr(response, "objects", []) or [])
    except Exception:  # pragma: no cover - defensive
        logger.debug(
            "xref_expansion fetch failed for table=%s doc=%s",
            label, document_id, exc_info=True,
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
        ``section`` / ``section_symbol`` resolve against ``section_path``
        substring + boundary post-filter. ``table`` and ``figure`` resolve
        against ``caption_label`` (exact match) scoped to the source hit's
        ``document_id`` AND ``chunk_type``. ``appendix`` remains pass-through.

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
                # TODO: implement figure/appendix resolution. Needs a
                # caption registry (e.g. a property index on figure caption
                # text). Table resolution lives below.
                logger.debug(
                    "xref_expansion: ref_type=%s not yet supported (value=%r)",
                    ref_type, ref_value,
                )
                continue

            if ref_type == "table":
                # Tables are document-scoped: a bare "Table 5-2" is
                # ambiguous across datasheets. Without the source hit's
                # document_id we cannot safely resolve.
                src_doc_id = str(meta.get("document_id") or "")
                if not src_doc_id:
                    logger.debug(
                        "xref_expansion: skipping table ref %r — source hit "
                        "has no document_id to scope against",
                        ref_value,
                    )
                    continue
                objs = _fetch_table_chunks(
                    client=client, collection=collection,
                    label=ref_value, document_id=src_doc_id,
                    limit=max(2, max_per_hit + 1),
                )
            elif ref_type == "figure":
                # Figures share the table contract: caption_label is unique
                # within a document but recurs across documents, so the
                # document_id scoping is load-bearing. Normalise the ref
                # value first so "Fig. 4-1" still resolves the chunk that
                # was stamped with caption_label="Figure 4-1".
                src_doc_id = str(meta.get("document_id") or "")
                if not src_doc_id:
                    logger.debug(
                        "xref_expansion: skipping figure ref %r — source hit "
                        "has no document_id to scope against",
                        ref_value,
                    )
                    continue
                norm_label = _normalize_figure_label(ref_value)
                if not norm_label:
                    # False positive (e.g. "Refigure 4-1") — anchored regex
                    # rejected it. Do not query.
                    continue
                objs = _fetch_figure_chunks(
                    client=client, collection=collection,
                    label=norm_label, document_id=src_doc_id,
                    limit=max(2, max_per_hit + 1),
                )
            else:
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
