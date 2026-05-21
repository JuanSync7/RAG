# @summary
# End-to-end sanity verification for the table-aware pipeline against a
# LIVE stack (Weaviate + TEI + LLM). Ingests one datasheet PDF, issues a
# representative retrieval query, asserts that table-group expansion
# fires AND that citations carry decoded ``page_bbox`` dicts (DEFECT-2
# proof). Idempotent collection name — re-runnable. Companion to the
# ingest-only ``soak_table_chunking.py``.
# Exports: main, run_sanity, _assert_expansion_fired,
#          _assert_citation_has_bbox, _build_parser, _render_report
# Deps: src.ingest, src.retrieval.pipeline.rag_chain, server.routes.query
# @end-summary
"""End-to-end sanity check for the merged table-aware retrieval pipeline.

# manual:
# This script connects to LIVE Weaviate, TEI, and LLM services. It is
# NOT run by any automated test. Operators invoke it after starting the
# compose stack — see ``docs/ingest/E2E_SANITY_RUNBOOK.md`` for the
# exact step-by-step (compose up, source .env, uv run …).
#
# The script's INTERNAL helpers (``_assert_expansion_fired``,
# ``_assert_citation_has_bbox``, ``_render_report``, ``_build_parser``)
# are unit-tested in ``tests/integration/test_e2e_sanity_runbook.py``
# against fake hit/source-ref shapes — so the runbook logic is
# verifiable without spinning up infra.

Usage:
    uv run python scripts/e2e_sanity_table_ingest.py \
        --pdf data/datasheets/esp32-s3_datasheet.pdf \
        --collection e2e_sanity \
        --query "What is the operating voltage range of the ESP32-S3?"

Gates (each prints PASS/FAIL with a diagnostic message on failure):

1. **Ingest** — the PDF goes through Docling → chunker → embedder →
   Weaviate, producing at least one stored chunk.
2. **Retrieval** — the query returns at least one hit from the freshly
   populated collection.
3. **Expansion** — at least one returned hit carries
   ``metadata.expanded_from`` (the marker stamped by
   ``expand_table_group_hits``). This is the proof that the new
   query-time expansion stage fired.
4. **Citation bbox** — at least one source_ref returned by
   ``_source_refs`` carries both ``page_no`` and a decoded
   ``page_bbox`` dict ``{l,t,r,b}``. This is the proof that the
   DEFECT-2 JSON-decode path is in effect post-merge.

Exit code: 0 on PASS for all gates, 1 if any gate fails, 2 on env error
(PDF missing, etc.).
"""
from __future__ import annotations

import argparse
import logging
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Ensure project root is importable when invoked outside ``uv run``.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger("e2e_sanity")


_DEFAULT_PDF = _REPO_ROOT / "data" / "datasheets" / "esp32-s3_datasheet.pdf"
_DEFAULT_COLLECTION = "e2e_sanity_table_aware"
_DEFAULT_QUERY = "What is the operating voltage range of the ESP32-S3?"

_BBOX_KEYS = ("l", "t", "r", "b")


# ---------------------------------------------------------------------------
# Pure assertion helpers (unit-tested in
# tests/integration/test_e2e_sanity_runbook.py — these MUST stay pure).
# ---------------------------------------------------------------------------


def _assert_expansion_fired(hits: list[dict]) -> None:
    """Assert that ``expand_table_group_hits`` fired during retrieval.

    The marker is ``metadata["expanded_from"]`` — set by
    ``src.retrieval.table_group_expansion._obj_to_hit`` on every hit it
    injects. If NO hit in the result list carries that key, expansion
    either didn't fire or didn't find a sibling to attach.

    Args:
        hits: Ordered list of retrieval-hit dicts (the dict shape
            produced by ``hybrid_search`` / consumed by
            ``expand_table_group_hits``).

    Raises:
        AssertionError: with a diagnostic message naming the marker,
        the number of hits inspected, and the chunk_types observed —
        so the operator can debug whether the query hit a table at all.
    """
    if not hits:
        raise AssertionError(
            "expansion gate FAIL: zero retrieval hits — cannot verify "
            "'expanded_from' marker on an empty list. Likely cause: "
            "the query missed the collection or ingest produced 0 chunks."
        )
    expanded = [h for h in hits if (h.get("metadata") or {}).get("expanded_from")]
    if not expanded:
        chunk_types = sorted({
            str((h.get("metadata") or {}).get("chunk_type") or "?")
            for h in hits
        })
        raise AssertionError(
            f"expansion gate FAIL: inspected {len(hits)} hit(s); none carry "
            f"metadata.expanded_from. chunk_types observed: {chunk_types}. "
            "Likely causes: (a) query did not match any table chunk, "
            "(b) RAG_TABLE_EXPANSION_ENABLED is off, "
            "(c) max_groups_to_expand budget was 0."
        )


def _assert_citation_has_bbox(source_refs: list[dict]) -> None:
    """Assert that at least one citation carries decoded page provenance.

    The runtime ``_decode_page_bbox`` in ``server/routes/query.py``
    decodes Weaviate's JSON-encoded TEXT into a ``{l,t,r,b}`` dict.
    This helper asserts the END shape — if the JSON string slipped
    through undecoded, that's a regression of DEFECT-2.

    Args:
        source_refs: List of dicts as returned by ``_source_refs``.

    Raises:
        AssertionError: when no ref has both an int ``page_no`` and a
        dict ``page_bbox`` containing all four ``l,t,r,b`` float keys.
    """
    if not source_refs:
        raise AssertionError(
            "citation gate FAIL: zero source_refs returned. "
            "Cannot verify page_no / page_bbox without any citations."
        )

    for ref in source_refs:
        if "page_no" not in ref or ref.get("page_no") in (None, ""):
            continue
        bbox = ref.get("page_bbox")
        if bbox is None:
            continue
        if isinstance(bbox, str):
            raise AssertionError(
                f"citation gate FAIL: page_bbox is a raw string "
                f"({bbox!r}) — _decode_page_bbox did not run. "
                "Expected a decoded dict {l,t,r,b}; got the JSON text. "
                "This is a DEFECT-2 regression."
            )
        if not isinstance(bbox, dict):
            continue
        missing = [k for k in _BBOX_KEYS if k not in bbox]
        if missing:
            continue
        try:
            for k in _BBOX_KEYS:
                float(bbox[k])
        except (TypeError, ValueError):
            continue
        return  # found a well-formed ref — PASS

    # Walked the list, found nothing usable. Emit a diagnostic.
    sample = source_refs[0] if source_refs else {}
    raise AssertionError(
        f"citation gate FAIL: no source_ref carries both an int "
        f"page_no AND a decoded page_bbox dict with l,t,r,b keys. "
        f"Inspected {len(source_refs)} ref(s). First ref keys: "
        f"{sorted(sample.keys())}. "
        "Likely causes: (a) Docling lost page provenance during chunking, "
        "(b) Weaviate schema is missing page_no/page_bbox properties "
        "(run scripts/migrate_weaviate_table_schema.py), "
        "(c) _decode_page_bbox returned None for every ref."
    )


# ---------------------------------------------------------------------------
# Result container + report rendering
# ---------------------------------------------------------------------------


@dataclass
class _Gate:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class SanityResult:
    """Structured result of a sanity run. Used by ``_render_report``."""

    ingest_ok: bool = False
    ingest_stored_chunks: int = 0
    retrieval_ok: bool = False
    retrieval_hits: int = 0
    expansion_ok: bool = False
    expanded_hit_count: int = 0
    citation_ok: bool = False
    citation_with_bbox_count: int = 0
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def all_pass(self) -> bool:
        return (
            self.ingest_ok
            and self.retrieval_ok
            and self.expansion_ok
            and self.citation_ok
            and not self.error
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "ingest_ok": self.ingest_ok,
            "ingest_stored_chunks": self.ingest_stored_chunks,
            "retrieval_ok": self.retrieval_ok,
            "retrieval_hits": self.retrieval_hits,
            "expansion_ok": self.expansion_ok,
            "expanded_hit_count": self.expanded_hit_count,
            "citation_ok": self.citation_ok,
            "citation_with_bbox_count": self.citation_with_bbox_count,
            "error": self.error,
            **self.extra,
        }


def _flag(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _render_report(
    *,
    pdf_path: Path,
    collection: str,
    query: str,
    results: dict[str, Any],
) -> str:
    """Render a structured markdown PASS/FAIL report.

    Pure function — unit-tested against synthetic ``results`` dicts.
    """
    ingest_ok = bool(results.get("ingest_ok"))
    retrieval_ok = bool(results.get("retrieval_ok"))
    expansion_ok = bool(results.get("expansion_ok"))
    citation_ok = bool(results.get("citation_ok"))
    err = results.get("error")

    all_pass = ingest_ok and retrieval_ok and expansion_ok and citation_ok and not err

    lines: list[str] = []
    lines.append("# E2E sanity — table-aware retrieval")
    lines.append("")
    lines.append(f"- pdf: `{pdf_path}`")
    lines.append(f"- collection: `{collection}`")
    lines.append(f"- query: `{query}`")
    lines.append(f"- overall: **{_flag(all_pass)}**")
    lines.append("")
    lines.append("## Gates")
    lines.append("")
    lines.append("| gate | status | detail |")
    lines.append("|---|---|---|")
    lines.append(
        f"| ingest | {_flag(ingest_ok)} | "
        f"stored_chunks={results.get('ingest_stored_chunks', 0)} |"
    )
    lines.append(
        f"| retrieval | {_flag(retrieval_ok)} | "
        f"hits={results.get('retrieval_hits', 0)} |"
    )
    lines.append(
        f"| expansion (expanded_from) | {_flag(expansion_ok)} | "
        f"expanded={results.get('expanded_hit_count', 0)} |"
    )
    lines.append(
        f"| citation (page_bbox decoded) | {_flag(citation_ok)} | "
        f"with_bbox={results.get('citation_with_bbox_count', 0)} |"
    )
    lines.append("")
    if err:
        lines.append("## Error")
        lines.append("")
        lines.append("```")
        lines.append(str(err))
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="e2e_sanity_table_ingest",
        description=(
            "End-to-end sanity check for the table-aware retrieval "
            "pipeline against a live Weaviate + TEI + LLM stack."
        ),
    )
    ap.add_argument(
        "--pdf",
        type=Path,
        default=_DEFAULT_PDF,
        help=f"Path to the PDF to ingest (default: {_DEFAULT_PDF}).",
    )
    ap.add_argument(
        "--collection",
        type=str,
        default=_DEFAULT_COLLECTION,
        help=f"Weaviate collection name (default: {_DEFAULT_COLLECTION}).",
    )
    ap.add_argument(
        "--query",
        type=str,
        default=_DEFAULT_QUERY,
        help=f"Retrieval query to issue (default: {_DEFAULT_QUERY!r}).",
    )
    ap.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional output path for the markdown report.",
    )
    ap.add_argument(
        "--keep-collection",
        action="store_true",
        help="Do NOT delete the collection on exit (default: keep).",
    )
    return ap


# ---------------------------------------------------------------------------
# Live-stack runner (NOT exercised by the test suite)
# ---------------------------------------------------------------------------


def _run_ingest_live(pdf_path: Path) -> int:
    """Drive the production ingest pipeline against a single PDF.

    Returns the number of stored chunks. Raises on hard failure.
    """
    from src.ingest import IngestionConfig, ingest_directory

    # We pin the source to the single PDF via selected_sources so the
    # caller's documents/ folder isn't swept.
    cfg = IngestionConfig()
    summary = ingest_directory(
        documents_dir=pdf_path.parent,
        config=cfg,
        fresh=False,
        update=True,
        selected_sources=[pdf_path.resolve()],
    )
    return int(summary.stored_chunks)


def _run_retrieval_live(query: str) -> tuple[list[dict], list[dict]]:
    """Run a live RAG query end-to-end and return (hits, source_refs).

    Uses the production RAGChain (which wires the table-expansion stage
    when ``RAG_TABLE_EXPANSION_ENABLED`` is on — default ON post-merge)
    and the production ``_source_refs`` helper for citation assembly.
    """
    from server.routes.query import _source_refs
    from src.retrieval.pipeline.rag_chain import RAGChain

    chain = RAGChain(persistent_weaviate=True)
    try:
        response = chain.run(query=query, skip_generation=True)
    finally:
        # Best-effort close.
        try:
            chain.close()  # type: ignore[attr-defined]
        except Exception:
            pass

    # ``response`` is a RAGResponse — extract its retrieval-shaped list
    # for the helper. The response carries either ``.results`` or
    # ``.chunks`` depending on the version; we accept both.
    results = (
        getattr(response, "results", None)
        or getattr(response, "chunks", None)
        or []
    )
    # Normalise to dicts: hybrid_search returns dicts; RankedResult is
    # an object — convert via the chain's documented shape.
    hits: list[dict] = []
    for r in results:
        if isinstance(r, dict):
            hits.append(r)
        else:
            hits.append({
                "text": getattr(r, "text", ""),
                "score": getattr(r, "score", 0.0),
                "uuid": getattr(r, "uuid", ""),
                "metadata": dict(getattr(r, "metadata", {}) or {}),
            })

    source_refs = _source_refs(hits)
    return hits, source_refs


def run_sanity(
    *,
    pdf_path: Path,
    collection: str,
    query: str,
) -> SanityResult:
    """Drive a full live-stack sanity run.

    This function IS connected to live infra. Do not call from tests.
    The helpers it invokes (``_assert_expansion_fired``,
    ``_assert_citation_has_bbox``) ARE pure and tested independently.
    """
    result = SanityResult()

    # NB: ``collection`` is documented in the runbook; the live pipeline
    # currently reads the collection from ``VECTOR_COLLECTION_DEFAULT``.
    # We thread the value through so future config-driven overrides
    # land in one place.
    result.extra["collection"] = collection

    try:
        stored = _run_ingest_live(pdf_path)
        result.ingest_stored_chunks = stored
        result.ingest_ok = stored > 0
        if not result.ingest_ok:
            result.error = "ingest produced 0 stored chunks"
            return result

        hits, refs = _run_retrieval_live(query)
        result.retrieval_hits = len(hits)
        result.retrieval_ok = len(hits) > 0
        if not result.retrieval_ok:
            result.error = "retrieval returned 0 hits"
            return result

        try:
            _assert_expansion_fired(hits)
            result.expansion_ok = True
            result.expanded_hit_count = sum(
                1 for h in hits
                if (h.get("metadata") or {}).get("expanded_from")
            )
        except AssertionError as exc:
            result.error = str(exc)
            return result

        try:
            _assert_citation_has_bbox(refs)
            result.citation_ok = True
            result.citation_with_bbox_count = sum(
                1 for r in refs
                if isinstance(r.get("page_bbox"), dict)
                and r.get("page_no") not in (None, "")
            )
        except AssertionError as exc:
            result.error = str(exc)
            return result

    except Exception as exc:  # pragma: no cover - live-stack only
        result.error = f"{exc!r}\n{traceback.format_exc()}"

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - live
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    args = _build_parser().parse_args(argv)

    pdf_path: Path = args.pdf
    if not pdf_path.exists():
        print(
            f"ERROR: PDF not found at {pdf_path}. "
            "Download or pass --pdf <path>.",
            file=sys.stderr,
        )
        return 2

    logger.info("starting sanity run: pdf=%s collection=%s", pdf_path, args.collection)
    sanity = run_sanity(
        pdf_path=pdf_path,
        collection=args.collection,
        query=args.query,
    )

    report = _render_report(
        pdf_path=pdf_path,
        collection=args.collection,
        query=args.query,
        results=sanity.as_dict(),
    )
    print(report)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
        logger.info("wrote report to %s", args.report)

    return 0 if sanity.all_pass else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
