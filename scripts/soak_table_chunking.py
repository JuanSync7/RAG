# @summary
# Offline soak harness for the adaptive table-chunking pipeline.
# Runs DoclingParser end-to-end on a real datasheet PDF twice (for
# cross-reparse determinism), aggregates table / chunk / heading metrics,
# and emits a markdown quality report. Also runs FIG-3 figure-artifact
# metrics (caption coverage, image-uri sanitisation, chunk-id
# idempotency, xref emission/resolvability in two flag modes).
# Exports: main, run_soak, run_figure_soak
# Deps: src.ingest.support.docling, src.ingest.common.types
# @end-summary

"""Soak the adaptive table-chunking pipeline against a real datasheet PDF.

Usage:
    uv run python scripts/soak_table_chunking.py <pdf_path> [<report_path>]

The script is re-runnable. It does NOT modify ingestion state or write to any
vector store -- this is pure parse + chunk in-process. It writes a markdown
report at <report_path> (defaults to
``docs/soak/table_chunking_real_datasheet_<YYYY-MM-DD>.md``) summarising:

- Counts of detected tables / table_group_ids / summary chunks / row chunks
- section_path breadcrumb depth distribution
- Token-budget truncation events (V's table_summary_max_chars guard)
- Errors / exceptions encountered
- Spot-checks of three random table chunks (text excerpt, page_no, page_bbox)
- Cross-reparse determinism check on the ``table_group_id`` set

Skips cleanly (still writes a report) when Docling models cannot be made
ready in this environment.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import random
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _pdf_page_count(pdf_path: Path) -> int:
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]

        return len(PdfReader(str(pdf_path)).pages)
    except Exception:
        return -1


def _heading_depth(section_path: str) -> int:
    if not section_path:
        return 0
    return len([p for p in section_path.split(" > ") if p.strip()])


def _summary_chunks(chunks: list) -> list:
    out = []
    for c in chunks:
        meta = getattr(c, "extra_metadata", {}) or {}
        if meta.get("chunk_type") == "table_summary":
            out.append(c)
    return out


def _row_chunks(chunks: list) -> list:
    out = []
    for c in chunks:
        meta = getattr(c, "extra_metadata", {}) or {}
        if meta.get("chunk_type") == "table_row":
            out.append(c)
    return out


def _truncation_events(chunks: list) -> int:
    """Count summary chunks whose text contains V's truncation marker."""
    marker = "… [truncated"
    return sum(1 for c in chunks if marker in (getattr(c, "text", "") or ""))


def _page_no_of(chunk: Any) -> Any:
    pr = getattr(chunk, "page_ref", None)
    if pr is None:
        return None
    return getattr(pr, "page_no", None)


def _bbox_of(chunk: Any) -> Any:
    pr = getattr(chunk, "page_ref", None)
    if pr is None:
        return None
    return getattr(pr, "bbox", None)


# --------------------------------------------------------------------------- #
# Xref metrics                                                                #
# --------------------------------------------------------------------------- #


def _decode_xref_targets(raw: Any) -> list[dict]:
    """Decode an ``xref_targets`` payload into a list of ``{type, value}`` dicts.

    Tolerates already-decoded lists and bad/empty JSON. Mirrors the leniency
    of ``src.retrieval.xref_expansion._decode_targets`` but does not import
    it so the soak stays self-contained on the ingest-only path.
    """
    if not raw:
        return []
    if isinstance(raw, list):
        return raw
    try:
        decoded = json.loads(raw)
        return decoded if isinstance(decoded, list) else []
    except (TypeError, ValueError):
        return []


def _percentile_nearest_rank(sorted_values: list[int], pct: float) -> int:
    """Nearest-rank percentile over a pre-sorted list of ints.

    Uses ``ceil(pct * N)``-1 indexing on the sorted vector, which is the
    same convention as ``statistics.quantiles(method="inclusive")`` but
    spelled out to keep the math obvious for small N. Returns 0 on empty.
    """
    if not sorted_values:
        return 0
    n = len(sorted_values)
    idx = max(1, math.ceil(pct * n)) - 1
    idx = min(idx, n - 1)
    return int(sorted_values[idx])


def _xref_edge_metrics(chunks: list) -> dict:
    """Aggregate xref edge density across chunks.

    Counts JSON-decoded ``xref_targets`` on every chunk and reports total
    edges, chunks-with-edges, p50/p90 of edges-per-chunk (over the subset
    of chunks that have at least one edge), and a by-type breakdown.
    """
    by_type: Counter[str] = Counter()
    per_chunk_counts: list[int] = []
    total_edges = 0

    for c in chunks:
        meta = getattr(c, "extra_metadata", {}) or {}
        targets = _decode_xref_targets(meta.get("xref_targets"))
        n = 0
        for ref in targets:
            ref_type = str((ref or {}).get("type") or "")
            if not ref_type:
                continue
            by_type[ref_type] += 1
            n += 1
        if n > 0:
            per_chunk_counts.append(n)
            total_edges += n

    per_chunk_counts.sort()
    return {
        "total_chunks_with_edges": len(per_chunk_counts),
        "total_edges": total_edges,
        "edges_per_chunk_p50": _percentile_nearest_rank(per_chunk_counts, 0.5),
        "edges_per_chunk_p90": _percentile_nearest_rank(per_chunk_counts, 0.9),
        "by_type": dict(by_type),
    }


def _xref_resolvability(chunks: list, tables: list) -> dict:
    """Estimate how many xref edges resolve locally (no Weaviate query).

    - ``section`` / ``section_symbol``: resolvable if some chunk in the
      same ``document_id`` has a ``section_path`` containing the
      normalised section value (boundary-aware via
      ``src.retrieval.xref_expansion._section_value_matches_path`` —
      shared regex; ``"3.1"`` does NOT count ``"3.10"`` as a match).
    - ``table``: resolvable if some TableArtifact has ``caption_label``
      equal to the surface form (document-scoped when present).
    - ``figure`` / ``appendix``: always counted as unresolvable (no
      caption registry for figures/appendices yet).
    """
    from src.retrieval.xref_expansion import (
        _normalise_section_value,
        _section_value_matches_path,
    )

    # Index chunks by document_id → list of section_paths.
    #
    # ``section_path`` lives on the Chunk dataclass as a direct attribute —
    # only table chunks redundantly copy it into ``extra_metadata``. Prose
    # chunks do not. Prefer the direct attribute; keep the metadata fallback
    # so callers/tests that stuff section_path into metadata still work.
    # ``document_id`` is currently metadata-only on raw parser Chunks.
    section_paths_by_doc: dict[str, list[str]] = {}
    for c in chunks:
        meta = getattr(c, "extra_metadata", {}) or {}
        doc_id = str(meta.get("document_id") or "")
        sp = str(getattr(c, "section_path", "") or "") or str(meta.get("section_path") or "")
        if sp:
            section_paths_by_doc.setdefault(doc_id, []).append(sp)

    # Index table caption labels — both globally and per document.
    caption_labels_global: set[str] = set()
    caption_labels_by_doc: dict[str, set[str]] = {}
    for t in tables:
        label = str(getattr(t, "caption_label", "") or "")
        if not label:
            continue
        caption_labels_global.add(label)
        doc_id = str(getattr(t, "document_id", "") or "")
        caption_labels_by_doc.setdefault(doc_id, set()).add(label)

    counts = {
        "section_resolvable": 0,
        "table_resolvable": 0,
        "unresolvable_table_no_label": 0,
        "unresolvable_figure": 0,
        "unresolvable_appendix": 0,
    }

    for c in chunks:
        meta = getattr(c, "extra_metadata", {}) or {}
        doc_id = str(meta.get("document_id") or "")
        targets = _decode_xref_targets(meta.get("xref_targets"))
        for ref in targets:
            rt = str((ref or {}).get("type") or "")
            rv = str((ref or {}).get("value") or "")
            if not rt or not rv:
                continue
            if rt in ("section", "section_symbol"):
                norm = _normalise_section_value(rt, rv)
                if not norm:
                    continue
                candidates = section_paths_by_doc.get(doc_id, [])
                if any(_section_value_matches_path(norm, sp) for sp in candidates):
                    counts["section_resolvable"] += 1
            elif rt == "table":
                scoped = caption_labels_by_doc.get(doc_id, set())
                if rv in scoped or rv in caption_labels_global:
                    counts["table_resolvable"] += 1
                else:
                    counts["unresolvable_table_no_label"] += 1
            elif rt == "figure":
                counts["unresolvable_figure"] += 1
            elif rt == "appendix":
                counts["unresolvable_appendix"] += 1
            # Unknown ref types are silently ignored — they're not in the
            # extractor's vocabulary today.
    return counts


# --------------------------------------------------------------------------- #
# Figure-artifact metrics (FIG-3)                                             #
# --------------------------------------------------------------------------- #


def _figure_chunks(chunks: list) -> list:
    """Subset of chunks emitted as ``chunk_type="figure"``."""
    out = []
    for c in chunks:
        meta = getattr(c, "extra_metadata", {}) or {}
        if meta.get("chunk_type") == "figure":
            out.append(c)
    return out


def _figure_caption_label_rate(figures: list) -> dict:
    """Fraction of FigureArtifacts with a non-empty ``caption_label``.

    A low rate signals caption-label regex holes — figures whose captions
    don't start with a recognised ``Figure``/``Fig.`` prefix produce empty
    labels and are unreachable via xref expansion.
    """
    total = len(figures)
    with_label = sum(
        1 for f in figures if str(getattr(f, "caption_label", "") or "")
    )
    rate = (with_label / total) if total else 0.0
    return {
        "figures_total": int(total),
        "figures_with_label": int(with_label),
        "label_rate": float(rate),
    }


def _figure_image_uri_sanitised(figure_chunks: list) -> bool:
    """True when no figure CHUNK carries a ``data:`` URI in storage form.

    FIG-2's transformer coerces ``data:`` URIs to ``""`` to avoid bloating
    Weaviate; this check guards against a regression where a data URI
    sneaks through into the emitted chunk metadata.
    """
    for c in figure_chunks:
        meta = getattr(c, "extra_metadata", {}) or {}
        uri = str(meta.get("figure_image_uri") or "")
        if uri.startswith("data:"):
            return False
    return True


def _figure_chunk_idempotency(figures: list) -> dict:
    """Recompute figure chunk_ids twice and confirm they are stable.

    Mirrors the table-side determinism check. Uses
    :func:`build_figure_chunk_id` (same function ingest uses) so the
    soak compares apples to apples.
    """
    from src.vector_db.weaviate.store import build_figure_chunk_id

    def _ids(figs: list) -> list[str]:
        out: list[str] = []
        for f in figs or []:
            doc_id = str(getattr(f, "document_id", "") or "")
            self_ref = str(getattr(f, "self_ref", "") or "")
            if not doc_id or not self_ref:
                continue
            out.append(build_figure_chunk_id(doc_id, self_ref))
        return sorted(out)

    first = _ids(figures)
    second = _ids(figures)
    return {
        "ok": first == second,
        "first_count": len(first),
        "second_count": len(second),
        "duplicates_within_parse": len(first) - len(set(first)),
    }


class _FakeFigureObj:
    """Mimics a Weaviate object with ``.properties`` and ``.uuid``."""

    def __init__(self, props: dict, uuid: str = ""):
        self.properties = props
        self.uuid = uuid


class _FakeFigureCollection:
    """Resolves figure ref lookups against in-memory chunk metadata.

    Mirrors :func:`src.retrieval.xref_expansion._fetch_figure_chunks`'s
    side: a Weaviate filter is opaque to us, so we bypass it and resolve
    the (label, document_id) pair directly. We accept the kwargs the
    real client receives and discard the ``filters`` object.
    """

    def __init__(self, figure_chunks: list):
        # (caption_label, document_id) -> list[obj]
        self._by_key: dict[tuple[str, str], list[_FakeFigureObj]] = {}
        for c in figure_chunks:
            meta = getattr(c, "extra_metadata", {}) or {}
            label = str(meta.get("caption_label") or "")
            doc_id = str(meta.get("document_id") or "")
            if not label or not doc_id:
                continue
            props = {
                "text": getattr(c, "text", "") or "",
                "chunk_id": str(meta.get("chunk_id") or ""),
                "chunk_type": "figure",
                "section_path": getattr(c, "section_path", "") or "",
                "heading": getattr(c, "heading", "") or "",
                "document_id": doc_id,
                "caption_label": label,
                "page_no": int(meta.get("page_no") or 0),
            }
            self._by_key.setdefault((label, doc_id), []).append(
                _FakeFigureObj(props, uuid=props["chunk_id"])
            )

        # Track (label, doc_id) lookups so the soak can report per-ref
        # resolution without inspecting the expanded payload.
        self.last_request: tuple[str, str] | None = None

    class _Q:
        def __init__(self, parent: "_FakeFigureCollection"):
            self._parent = parent

        def fetch_objects(self, *, filters=None, limit=None, return_properties=None):
            # The filter is opaque here. We rely on the caller (the soak)
            # to invoke us via expand_xref_hits which passes
            # (label, document_id) into the filter object. We instead
            # snoop the last-resolved key off the parent.
            class _Resp:
                def __init__(self, objs):
                    self.objects = objs

            label, doc_id = self._parent.last_request or ("", "")
            return _Resp(self._parent._by_key.get((label, doc_id), [])[: limit or 10])

    @property
    def query(self):
        return _FakeFigureCollection._Q(self)


class _FakeFigureClient:
    """Top-level fake matching the ``client.collections.get(name)`` shape."""

    def __init__(self, figure_chunks: list):
        self._col = _FakeFigureCollection(figure_chunks)

    class _CollectionsAccessor:
        def __init__(self, col):
            self._col = col

        def get(self, name):
            return self._col

    @property
    def collections(self):
        return _FakeFigureClient._CollectionsAccessor(self._col)


def _figure_resolvability(chunks: list, figure_chunks: list, document_id: str) -> dict:
    """Compute figure_resolvable_rate via the real ``expand_xref_hits``.

    Builds prose-chunk "hits" carrying their ``xref_targets`` and the
    parser-derived ``document_id``, then routes each unique figure ref
    through :func:`src.retrieval.xref_expansion.expand_xref_hits`
    backed by a fake client that resolves on (caption_label, doc_id).

    Returns ``{"figure_refs_emitted": int, "unique_targets": int,
    "resolved": int, "resolvable_rate": float}``.
    """
    from src.retrieval import xref_expansion as _xx

    # Monkey-patch the figure fetcher to record the (label, doc_id) the
    # expander asks for. The real filter object is opaque so we cannot
    # match against it inside our fake — easier to intercept here.
    fake = _FakeFigureClient(figure_chunks)
    col = fake._col
    original_fetch = _xx._fetch_figure_chunks

    def _patched_fetch(*, client, collection, label, document_id, limit):
        col.last_request = (label, document_id)
        return original_fetch(
            client=client, collection=collection, label=label,
            document_id=document_id, limit=limit,
        )

    _xx._fetch_figure_chunks = _patched_fetch  # type: ignore[assignment]
    try:
        # Build hits from prose chunks that emitted at least one figure ref.
        hits: list[dict] = []
        unique_targets: set[tuple[str, str]] = set()
        figure_refs_emitted = 0
        for c in chunks:
            meta = getattr(c, "extra_metadata", {}) or {}
            if meta.get("chunk_type") == "figure":
                continue
            raw = meta.get("xref_targets")
            if not raw:
                continue
            try:
                decoded = json.loads(raw) if isinstance(raw, str) else list(raw)
            except (TypeError, ValueError):
                decoded = []
            fig_refs = [
                r for r in decoded
                if isinstance(r, dict) and str(r.get("type") or "") == "figure"
            ]
            if not fig_refs:
                continue
            figure_refs_emitted += len(fig_refs)
            for r in fig_refs:
                unique_targets.add(("figure", str(r.get("value") or "")))
            hits.append({
                "text": getattr(c, "text", "") or "",
                "score": 0.0,
                "uuid": "",
                "metadata": {
                    "xref_targets": raw,
                    "document_id": document_id,
                    "chunk_type": meta.get("chunk_type", ""),
                    "section_path": getattr(c, "section_path", "") or "",
                },
            })

        if not unique_targets:
            return {
                "figure_refs_emitted": int(figure_refs_emitted),
                "unique_targets": 0,
                "resolved": 0,
                "resolvable_rate": 0.0,
            }

        # Drive resolution one unique target at a time so the
        # ``last_request`` snoop stays unambiguous (the real expander
        # iterates many refs in a single call; we want a clean count).
        resolved_count = 0
        for ref_type, ref_value in unique_targets:
            synth_hit = {
                "text": f"see {ref_value}",
                "score": 0.0,
                "uuid": "",
                "metadata": {
                    "xref_targets": json.dumps([{"type": ref_type, "value": ref_value}]),
                    "document_id": document_id,
                },
            }
            expanded = _xx.expand_xref_hits(
                [synth_hit], client=fake, collection="ignored",
                enabled=True, max_per_hit=2, max_total=2,
            )
            if len(expanded) > 1:
                resolved_count += 1

        return {
            "figure_refs_emitted": int(figure_refs_emitted),
            "unique_targets": int(len(unique_targets)),
            "resolved": int(resolved_count),
            "resolvable_rate": float(resolved_count) / float(len(unique_targets)),
        }
    finally:
        _xx._fetch_figure_chunks = original_fetch  # type: ignore[assignment]


def _figure_section_distribution(figures: list, top_n: int = 5) -> list[tuple[str, int]]:
    """Top ``top_n`` section_paths by figure count — qualitative anchor for
    the transcript ("most figures live in section X").
    """
    counter: Counter[str] = Counter()
    for f in figures or []:
        sp = str(getattr(f, "section_path", "") or "(no section)")
        counter[sp] += 1
    return counter.most_common(top_n)


# --------------------------------------------------------------------------- #
# Original table-side helpers                                                 #
# --------------------------------------------------------------------------- #


def _caption_label_coverage(tables: list) -> dict:
    """Fraction of TableArtifacts that got a normalised ``caption_label``.

    Catches caption-parser holes early — a low rate means
    ``_extract_caption_label`` is missing label prefixes on real captions.
    """
    total = len(tables)
    with_label = sum(1 for t in tables if str(getattr(t, "caption_label", "") or ""))
    rate = (with_label / total) if total else 0.0
    return {
        "tables_total": int(total),
        "tables_with_label": int(with_label),
        "label_rate": float(rate),
    }


# --------------------------------------------------------------------------- #
# Soak                                                                        #
# --------------------------------------------------------------------------- #


def _rechunk_under_flag(parser: Any, parse_result: Any, *, emit_figure_refs: bool) -> list:
    """Re-run ``parser.chunk(parse_result)`` with the figure-ref flag set.

    Mutates ``config.settings.RAG_XREF_EXTRACT_FIGURE_REFS`` for the duration
    of the call so ``_stamp_xref_targets`` (which reads the flag fresh per
    call) emits / suppresses figure refs accordingly.
    """
    from config import settings as _settings

    prev = getattr(_settings, "RAG_XREF_EXTRACT_FIGURE_REFS", False)
    _settings.RAG_XREF_EXTRACT_FIGURE_REFS = bool(emit_figure_refs)
    try:
        return parser.chunk(parse_result)
    finally:
        _settings.RAG_XREF_EXTRACT_FIGURE_REFS = prev


def _stamp_document_id(chunks: list, document_id: str) -> None:
    """Stamp ``document_id`` on prose-chunk metadata.

    Production stamps this downstream in chunk_enrichment; the soak runs
    only parser.chunk() so we mirror the stamping here to enable
    xref expansion's document-scoped figure resolver.
    """
    for c in chunks:
        meta = getattr(c, "extra_metadata", None)
        if meta is None:
            continue
        if meta.get("chunk_type") == "figure":
            continue  # figure chunks already carry document_id
        meta.setdefault("document_id", document_id)


def _figure_soak_verdict(soak: dict) -> dict:
    """Apply the FIG-3 acceptance criteria. Returns per-criterion PASS/FAIL."""
    counts = soak.get("counts") or {}
    figs_total = counts.get("figure_count", 0) or 0
    cap_rate = (soak.get("caption_coverage") or {}).get("label_rate", 0.0) or 0.0
    img_ok = bool(soak.get("figure_image_uri_sanitized", False))
    idem_ok = bool((soak.get("idempotency") or {}).get("ok", False))
    mode_b = soak.get("mode_b") or {}
    mode_b_rate = mode_b.get("resolvable_rate", 0.0) or 0.0
    mode_a = soak.get("mode_a") or {}
    mode_a_emitted = mode_a.get("figure_refs_emitted", 0) or 0

    criteria = {
        "figure_count_gt_0": {
            "ok": figs_total > 0,
            "value": figs_total,
            "threshold": "> 0",
        },
        "figure_caption_label_rate_ge_0_30": {
            "ok": cap_rate >= 0.30,
            "value": round(float(cap_rate), 4),
            "threshold": ">= 0.30",
        },
        "figure_image_uri_sanitized": {
            "ok": img_ok,
            "value": img_ok,
            "threshold": "true",
        },
        "figure_chunk_idempotent": {
            "ok": idem_ok,
            "value": idem_ok,
            "threshold": "true",
        },
        "mode_a_emits_zero_figure_refs": {
            "ok": mode_a_emitted == 0,
            "value": mode_a_emitted,
            "threshold": "== 0",
        },
        "mode_b_resolvable_rate_ge_0_30": {
            "ok": mode_b_rate >= 0.30,
            "value": round(float(mode_b_rate), 4),
            "threshold": ">= 0.30",
        },
    }
    all_ok = all(v["ok"] for v in criteria.values())
    return {"criteria": criteria, "all_pass": bool(all_ok)}


def _run_figure_soak(*, parser: Any, parse_result: Any, document_id: str) -> dict:
    """Compute FIG-3 figure-artifact metrics over the parsed document.

    Two modes (per the FIG-3 brief):
      - mode_a: ``RAG_XREF_EXTRACT_FIGURE_REFS=False`` — must emit 0 figure
        refs and therefore resolve none.
      - mode_b: ``RAG_XREF_EXTRACT_FIGURE_REFS=True`` — exercises the
        full emit+resolve loop end-to-end.
    """
    figures = list(getattr(parse_result, "figures", []) or [])

    out: dict[str, Any] = {
        "counts": {
            "figure_count": len(figures),
        },
        "caption_coverage": _figure_caption_label_rate(figures),
        "section_distribution_top5": [],
        "idempotency": {},
        "figure_image_uri_sanitized": True,
        "mode_a": {},
        "mode_b": {},
    }

    # Section distribution (qualitative anchor).
    out["section_distribution_top5"] = [
        {"section_path": sp, "count": int(n)}
        for sp, n in _figure_section_distribution(figures, top_n=5)
    ]

    # Idempotency over FigureArtifact chunk-id derivation.
    out["idempotency"] = _figure_chunk_idempotency(figures)

    # Mode A — flag off.
    chunks_a = _rechunk_under_flag(parser, parse_result, emit_figure_refs=False)
    _stamp_document_id(chunks_a, document_id)
    fig_chunks_a = _figure_chunks(chunks_a)
    out["figure_image_uri_sanitized"] = _figure_image_uri_sanitised(fig_chunks_a)
    out["counts"]["figure_chunks_emitted"] = len(fig_chunks_a)
    out["mode_a"] = _figure_resolvability(chunks_a, fig_chunks_a, document_id)

    # Mode B — flag on.
    chunks_b = _rechunk_under_flag(parser, parse_result, emit_figure_refs=True)
    _stamp_document_id(chunks_b, document_id)
    fig_chunks_b = _figure_chunks(chunks_b)
    # Re-check sanitisation under mode_b too (defensive).
    out["figure_image_uri_sanitized"] = bool(
        out["figure_image_uri_sanitized"]
        and _figure_image_uri_sanitised(fig_chunks_b)
    )
    out["mode_b"] = _figure_resolvability(chunks_b, fig_chunks_b, document_id)

    out["verdict"] = _figure_soak_verdict(out)
    return out


def run_soak(pdf_path: Path) -> dict:
    """Run the soak end-to-end. Returns a structured result dict.

    Result schema (partial):
      models_ready: bool, models_error: str|None
      pdf: {path, size_bytes, page_count}
      parse_error: str|None
      counts: {tables, summary_chunks, row_chunks, total_chunks, group_ids}
      heading_depth_distribution: dict[int, int]
      truncation_events: int
      table_samples: list[dict]
      determinism: {ok: bool, first: list, second: list}
    """
    from src.ingest.common.types import IngestionConfig
    from src.ingest.support.docling import DoclingParser

    result: dict[str, Any] = {
        "pdf": {
            "path": str(pdf_path.resolve()),
            "size_bytes": pdf_path.stat().st_size if pdf_path.exists() else 0,
            "page_count": _pdf_page_count(pdf_path),
        },
        "models_ready": False,
        "models_error": None,
        "parse_error": None,
        "counts": {},
        "heading_depth_distribution": {},
        "truncation_events": 0,
        "table_samples": [],
        "determinism": {"ok": False, "first": [], "second": []},
        "xref_edges": {},
        "xref_resolvability": {},
        "caption_label_coverage": {},
        "figure_soak": {},
        "errors": [],
    }

    config = IngestionConfig()
    try:
        config.docling_auto_download = False  # rely on local cache
    except Exception:
        pass

    try:
        DoclingParser.ensure_ready(config)
        result["models_ready"] = True
    except Exception as exc:
        result["models_error"] = repr(exc)
        return result

    # --- First parse --------------------------------------------------------
    parser = DoclingParser()
    try:
        parse_result = parser.parse(pdf_path, config)
        chunks = parser.chunk(parse_result)
    except Exception as exc:
        result["parse_error"] = f"{exc!r}\n{traceback.format_exc()}"
        return result

    tables = list(getattr(parse_result, "tables", []) or [])
    summaries = _summary_chunks(chunks)
    rows = _row_chunks(chunks)
    group_ids = sorted(
        {(c.extra_metadata or {}).get("table_group_id", "") for c in summaries}
    )

    result["counts"] = {
        "total_chunks": len(chunks),
        "tables": len(tables),
        "summary_chunks": len(summaries),
        "row_chunks": len(rows),
        "group_ids": len(group_ids),
    }

    # Heading depth distribution across detected tables.
    depths = Counter(_heading_depth(getattr(t, "section_path", "") or "") for t in tables)
    result["heading_depth_distribution"] = {int(k): int(v) for k, v in sorted(depths.items())}

    result["truncation_events"] = _truncation_events(summaries)

    # --- Xref-extraction soak metrics (parse + chunk only) ------------------
    try:
        result["xref_edges"] = _xref_edge_metrics(chunks)
        result["xref_resolvability"] = _xref_resolvability(chunks, tables)
        result["caption_label_coverage"] = _caption_label_coverage(tables)
    except Exception as exc:  # pragma: no cover - defensive
        result["errors"].append(f"xref metrics failed: {exc!r}")

    # --- Spot-check three random table summary chunks -----------------------
    rng = random.Random(42)
    pick = rng.sample(summaries, min(3, len(summaries))) if summaries else []
    for c in pick:
        meta = c.extra_metadata or {}
        text = getattr(c, "text", "") or ""
        md = meta.get("table_markdown", "") or ""
        result["table_samples"].append(
            {
                "table_group_id": meta.get("table_group_id", ""),
                "section_path": getattr(c, "section_path", "") or "",
                "table_caption": meta.get("table_caption", ""),
                "page_no": _page_no_of(c),
                "page_bbox": _bbox_of(c),
                "table_num_rows": meta.get("table_num_rows"),
                "table_num_cols": meta.get("table_num_cols"),
                "text_head": text[:240],
                "markdown_head": md[:240],
                "markdown_len": len(md),
            }
        )

    # --- Figure-artifact soak (FIG-3) --------------------------------------
    try:
        result["figure_soak"] = _run_figure_soak(
            parser=parser,
            parse_result=parse_result,
            document_id=pdf_path.stem,
        )
    except Exception as exc:  # pragma: no cover - defensive
        result["errors"].append(f"figure soak failed: {exc!r}")
        result["figure_soak"] = {"error": repr(exc)}

    # --- Second parse for determinism --------------------------------------
    try:
        parser2 = DoclingParser()
        parse_result2 = parser2.parse(pdf_path, config)
        chunks2 = parser2.chunk(parse_result2)
        summaries2 = _summary_chunks(chunks2)
        gids2 = sorted(
            {(c.extra_metadata or {}).get("table_group_id", "") for c in summaries2}
        )
        result["determinism"] = {
            "ok": group_ids == gids2,
            "first_count": len(group_ids),
            "second_count": len(gids2),
            "diff_first_minus_second": sorted(set(group_ids) - set(gids2)),
            "diff_second_minus_first": sorted(set(gids2) - set(group_ids)),
        }
    except Exception as exc:
        result["errors"].append(f"reparse failed: {exc!r}")

    return result


# --------------------------------------------------------------------------- #
# Report                                                                      #
# --------------------------------------------------------------------------- #


def _render_report(pdf_path: Path, data: dict, *, today: str) -> str:
    pdf = data["pdf"]
    counts = data.get("counts") or {}
    depth_dist = data.get("heading_depth_distribution") or {}
    samples = data.get("table_samples") or []
    det = data.get("determinism") or {}

    lines: list[str] = []
    lines.append(f"# Table-chunking soak — real datasheet ({today})")
    lines.append("")
    lines.append("## Dataset")
    lines.append("")
    lines.append(f"- PDF: `{pdf['path']}`")
    lines.append(f"- Source: Espressif ESP32-S3 datasheet (public, redistributable)")
    lines.append(f"- Size: {pdf['size_bytes']:,} bytes")
    lines.append(f"- Pages: {pdf['page_count']}")
    lines.append("")

    lines.append("## Environment")
    lines.append("")
    lines.append(f"- Docling models ready: **{data['models_ready']}**")
    if data.get("models_error"):
        lines.append(f"- Models error: `{data['models_error']}`")
    if data.get("parse_error"):
        lines.append("")
        lines.append("### Parse error")
        lines.append("")
        lines.append("```")
        lines.append(str(data["parse_error"])[:4000])
        lines.append("```")
    lines.append("")

    if counts:
        lines.append("## Counts")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        lines.append(f"| total chunks emitted | {counts.get('total_chunks', 0)} |")
        lines.append(f"| TableArtifact entries | {counts.get('tables', 0)} |")
        lines.append(f"| distinct `table_group_id`s | {counts.get('group_ids', 0)} |")
        lines.append(f"| `table_summary` chunks | {counts.get('summary_chunks', 0)} |")
        lines.append(f"| `table_row` chunks | {counts.get('row_chunks', 0)} |")
        lines.append(f"| truncation events (V's guard) | {data.get('truncation_events', 0)} |")
        lines.append("")

    if depth_dist:
        lines.append("## section_path breadcrumb depth distribution (over TableArtifacts)")
        lines.append("")
        lines.append("| heading levels | tables |")
        lines.append("|---|---|")
        for depth, n in sorted(depth_dist.items()):
            lines.append(f"| {depth} | {n} |")
        lines.append("")

    if samples:
        lines.append("## Spot-checks (3 random `table_summary` chunks)")
        lines.append("")
        for i, s in enumerate(samples, 1):
            lines.append(f"### Sample {i} — `table_group_id={s.get('table_group_id', '')!r}`")
            lines.append("")
            lines.append(f"- section_path: `{s.get('section_path', '')}`")
            lines.append(f"- caption: `{s.get('table_caption', '')!r}`")
            lines.append(f"- page_no: `{s.get('page_no')}`")
            lines.append(f"- page_bbox: `{s.get('page_bbox')}`")
            lines.append(f"- rows × cols: `{s.get('table_num_rows')} × {s.get('table_num_cols')}`")
            lines.append(f"- table_markdown length: `{s.get('markdown_len')}` chars")
            lines.append("")
            lines.append("text excerpt:")
            lines.append("")
            lines.append("```")
            lines.append((s.get("text_head") or "")[:400])
            lines.append("```")
            lines.append("")
            lines.append("markdown excerpt:")
            lines.append("")
            lines.append("```")
            lines.append((s.get("markdown_head") or "")[:400])
            lines.append("```")
            lines.append("")

    if det:
        lines.append("## Determinism (cross-reparse)")
        lines.append("")
        ok = det.get("ok")
        lines.append(f"- Identical `table_group_id` set across two parses: **{ok}**")
        lines.append(f"- first parse: {det.get('first_count', '?')} group_ids")
        lines.append(f"- second parse: {det.get('second_count', '?')} group_ids")
        df = det.get("diff_first_minus_second") or []
        ds = det.get("diff_second_minus_first") or []
        if df:
            lines.append(f"- only in first: `{df}`")
        if ds:
            lines.append(f"- only in second: `{ds}`")
        lines.append("")

    xref_edges = data.get("xref_edges") or {}
    if xref_edges:
        lines.append("## Xref edges")
        lines.append("")
        lines.append(f"- chunks with edges: {xref_edges.get('total_chunks_with_edges', 0)}")
        lines.append(f"- total edges: {xref_edges.get('total_edges', 0)}")
        lines.append(f"- edges per chunk p50: {xref_edges.get('edges_per_chunk_p50', 0)}")
        lines.append(f"- edges per chunk p90: {xref_edges.get('edges_per_chunk_p90', 0)}")
        by_type = xref_edges.get("by_type") or {}
        if by_type:
            parts = ", ".join(f"{k}={v}" for k, v in sorted(by_type.items()))
            lines.append(f"- by type: {parts}")
        else:
            lines.append("- by type: (none)")
        lines.append("")

    xref_res = data.get("xref_resolvability") or {}
    if xref_res:
        lines.append("## Xref resolvability")
        lines.append("")
        lines.append(f"- section resolvable: {xref_res.get('section_resolvable', 0)}")
        lines.append(f"- table resolvable: {xref_res.get('table_resolvable', 0)}")
        lines.append(
            f"- unresolvable table (no matching caption_label): "
            f"{xref_res.get('unresolvable_table_no_label', 0)}"
        )
        lines.append(f"- unresolvable figure: {xref_res.get('unresolvable_figure', 0)}")
        lines.append(f"- unresolvable appendix: {xref_res.get('unresolvable_appendix', 0)}")
        lines.append("")

    cap_cov = data.get("caption_label_coverage") or {}
    if cap_cov:
        lines.append("## Caption label coverage")
        lines.append("")
        lines.append(f"- tables total: {cap_cov.get('tables_total', 0)}")
        lines.append(f"- tables with caption_label: {cap_cov.get('tables_with_label', 0)}")
        rate = cap_cov.get("label_rate", 0.0)
        try:
            rate_str = f"{float(rate):.2%}"
        except (TypeError, ValueError):
            rate_str = str(rate)
        lines.append(f"- label rate: {rate_str}")
        lines.append("")

    fig_soak = data.get("figure_soak") or {}
    if fig_soak:
        lines.append("## Figure-artifact soak (FIG-3)")
        lines.append("")
        if "error" in fig_soak:
            lines.append(f"- FAILED: `{fig_soak['error']}`")
            lines.append("")
        else:
            fcounts = fig_soak.get("counts") or {}
            cap = fig_soak.get("caption_coverage") or {}
            idem = fig_soak.get("idempotency") or {}
            ma = fig_soak.get("mode_a") or {}
            mb = fig_soak.get("mode_b") or {}
            lines.append("| Metric | Value |")
            lines.append("|---|---|")
            lines.append(f"| figure_count | {fcounts.get('figure_count', 0)} |")
            lines.append(f"| figure_chunks_emitted | {fcounts.get('figure_chunks_emitted', 0)} |")
            lines.append(f"| figures_with_caption_label | {cap.get('figures_with_label', 0)} |")
            try:
                rate_str = f"{float(cap.get('label_rate', 0.0)):.2%}"
            except (TypeError, ValueError):
                rate_str = str(cap.get("label_rate", 0.0))
            lines.append(f"| figure_caption_label_rate | {rate_str} |")
            lines.append(
                f"| figure_image_uri_sanitized | {fig_soak.get('figure_image_uri_sanitized')} |"
            )
            lines.append(
                f"| figure_chunk_idempotent | {idem.get('ok')} "
                f"({idem.get('first_count', 0)} ids, "
                f"{idem.get('duplicates_within_parse', 0)} dupes) |"
            )
            lines.append(
                f"| mode_a figure_refs_emitted (flag off) | {ma.get('figure_refs_emitted', 0)} |"
            )
            lines.append(
                f"| mode_b figure_refs_emitted (flag on) | {mb.get('figure_refs_emitted', 0)} |"
            )
            lines.append(
                f"| mode_b unique_targets | {mb.get('unique_targets', 0)} |"
            )
            lines.append(
                f"| mode_b resolved | {mb.get('resolved', 0)} |"
            )
            try:
                rb_str = f"{float(mb.get('resolvable_rate', 0.0)):.2%}"
            except (TypeError, ValueError):
                rb_str = str(mb.get("resolvable_rate", 0.0))
            lines.append(f"| mode_b resolvable_rate | {rb_str} |")
            lines.append("")
            top = fig_soak.get("section_distribution_top5") or []
            if top:
                lines.append("### Top section_paths by figure count")
                lines.append("")
                lines.append("| section_path | figures |")
                lines.append("|---|---|")
                for row in top:
                    lines.append(f"| `{row['section_path']}` | {row['count']} |")
                lines.append("")
            verdict = fig_soak.get("verdict") or {}
            crits = verdict.get("criteria") or {}
            if crits:
                lines.append("### Verdict criteria")
                lines.append("")
                lines.append("| criterion | threshold | actual | result |")
                lines.append("|---|---|---|---|")
                for k, v in crits.items():
                    marker = "PASS" if v.get("ok") else "FAIL"
                    lines.append(
                        f"| {k} | {v.get('threshold')} | {v.get('value')} | **{marker}** |"
                    )
                lines.append("")
            overall = "PASS" if verdict.get("all_pass") else "FAIL"
            lines.append(f"**FIG-3 verdict: {overall}** — see criteria table above.")
            lines.append("")

    errs = data.get("errors") or []
    if errs:
        lines.append("## Other errors")
        lines.append("")
        for e in errs:
            lines.append(f"- `{e}`")
        lines.append("")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pdf", type=Path, help="Path to a datasheet PDF")
    ap.add_argument(
        "report",
        type=Path,
        nargs="?",
        default=None,
        help="Output markdown report path (default: docs/soak/table_chunking_real_datasheet_<today>.md)",
    )
    ap.add_argument(
        "--raw-json",
        type=Path,
        default=None,
        help="Optional path to dump the raw result dict as JSON.",
    )
    args = ap.parse_args(argv)

    today = _dt.date.today().isoformat()
    report_path = args.report or Path("docs/soak") / f"table_chunking_real_datasheet_{today}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.pdf.exists():
        print(f"PDF not found: {args.pdf}", file=sys.stderr)
        return 2

    # Ensure project src is importable when invoked outside `uv run`.
    here = Path(__file__).resolve().parent.parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))

    data = run_soak(args.pdf)

    rendered = _render_report(args.pdf, data, today=today)
    report_path.write_text(rendered, encoding="utf-8")
    print(f"wrote {report_path}")

    if args.raw_json is not None:
        args.raw_json.parent.mkdir(parents=True, exist_ok=True)
        args.raw_json.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        print(f"wrote {args.raw_json}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
