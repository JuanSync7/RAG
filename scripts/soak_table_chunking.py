# @summary
# Offline soak harness for the adaptive table-chunking pipeline.
# Runs DoclingParser end-to-end on a real datasheet PDF twice (for
# cross-reparse determinism), aggregates table / chunk / heading metrics,
# and emits a markdown quality report.
# Exports: main, run_soak
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
# Soak                                                                        #
# --------------------------------------------------------------------------- #


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
