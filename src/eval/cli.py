# @summary
# P6b — Eval CLI runner. Orchestrates load_pack → plan_pack_ingest →
# execute_plan → retrieve_for_goldens → score_goldens → build_eval_report
# → validate_eval_report and translates the GateResult into an exit code.
# Heavy modules (execute_plan, retrieve, judge) are LAZY-imported inside
# run_cli so the parser stays cheap and tests can monkeypatch the runner
# package symbols without ordering hazards.
# Exit codes: 0=pass, 1=gate fail, 2=pack load/validation error,
# 3=runtime/infra error.
# Exports: _build_parser, run_cli, main
# Deps: argparse, json, logging, sys; src.eval.pack (errors, loader);
#       src.eval.runner (lazy) — execute_plan, retrieve_for_goldens,
#       score_goldens, build_judge_client, build_eval_report,
#       aggregate_recall_by_qtype, validate_eval_report, plan_pack_ingest,
#       read_samples_per_claim, read_max_parallel_judges.
# @end-summary
"""Eval CLI runner (P6b).

Single subcommand ``run`` that chains the full eval pipeline against a
loaded eval_pack and exits with a deterministic code reflecting the
gate outcome. The CLI is a thin orchestrator — no business logic.

Lazy imports for heavy modules keep argparse-time work to a minimum and
sidestep langchain_core stub-timing hazards in the test environment.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("rag.eval.cli")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the eval CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m src.eval",
        description="Run an eval_pack: ingest, retrieve, judge, gate.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Run the full eval pipeline against an eval_pack directory.",
    )
    run_parser.add_argument(
        "pack_path",
        type=str,
        help="Path to the eval_pack directory (containing pack.yaml).",
    )
    run_parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Top-k retrieval cutoff per query (default: 5).",
    )
    run_parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format (default: text).",
    )
    run_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    run_parser.add_argument(
        "--show-samples",
        action="store_true",
        help=(
            "Include per-sample judge scores and reasoning in JSON output "
            "(default: off; silently ignored under --format text)."
        ),
    )
    run_parser.add_argument(
        "--samples-per-claim",
        type=int,
        default=None,
        help=(
            "Override pack.meta.judge.samples_per_claim — number of judge "
            "invocations per golden (mean score, best reasoning)."
        ),
    )
    run_parser.add_argument(
        "--max-parallel-judges",
        type=int,
        default=None,
        help=(
            "Override pack.meta.judge.max_parallel_judges — bound on "
            "concurrent judge calls (1 = sequential; >1 fans all "
            "(golden, sample) calls into a bounded thread pool)."
        ),
    )
    fresh_group = run_parser.add_mutually_exclusive_group()
    fresh_group.add_argument(
        "--fresh",
        dest="fresh",
        action="store_true",
        help="Recreate the target collection before ingesting (default).",
    )
    fresh_group.add_argument(
        "--no-fresh",
        dest="fresh",
        action="store_false",
        help="Incremental update mode; do not drop the collection.",
    )
    run_parser.set_defaults(fresh=True)
    return parser


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _emit_text(report, gate, *, stdout, stderr) -> None:
    """Print a human-readable per-qtype table + PASS/FAIL summary."""
    print(f"Collection: {report.collection_name}", file=stdout)
    print(f"k:          {report.k}", file=stdout)
    print(
        f"Queries:    scored={report.total_queries_scored} "
        f"skipped={report.total_queries_skipped} "
        f"judged={report.total_queries_judged}",
        file=stdout,
    )
    print("", file=stdout)
    print(f"{'qtype':<20} {'recall@k':>10} {'faithfulness':>14}", file=stdout)
    print("-" * 46, file=stdout)
    qtypes = sorted(
        set(report.recall_by_qtype) | set(report.faithfulness_by_qtype)
    )
    for qt in qtypes:
        r = report.recall_by_qtype.get(qt)
        f = report.faithfulness_by_qtype.get(qt)
        r_s = f"{r:.3f}" if r is not None else "  -  "
        f_s = f"{f:.3f}" if f is not None else "  -  "
        print(f"{qt:<20} {r_s:>10} {f_s:>14}", file=stdout)
    print("", file=stdout)
    print(f"Gate: {'PASS' if gate.passed else 'FAIL'}", file=stdout)
    if not gate.passed:
        for failure in gate.failures:
            print(
                f"  FAIL qtype={failure.qtype} metric={failure.metric} "
                f"expected>={failure.expected:.3f} actual={failure.actual:.3f}",
                file=stderr,
            )


def _emit_json(
    report,
    gate,
    *,
    stdout,
    faithfulness_results=None,
    show_samples: bool = False,
) -> None:
    """Emit a single-line JSON payload with the report + gate outcome.

    When *show_samples* is True AND *faithfulness_results* is provided,
    the payload also carries ``per_query_samples``: a mapping from qid
    to a list of ``{score, reasoning}`` dicts (one per judge sample).
    """
    payload = {
        "collection_name": report.collection_name,
        "k": report.k,
        "passed": gate.passed,
        "failures": [
            {
                "qtype": f.qtype,
                "metric": f.metric,
                "expected": f.expected,
                "actual": f.actual,
            }
            for f in gate.failures
        ],
        "recall_by_qtype": dict(report.recall_by_qtype),
        "faithfulness_by_qtype": dict(report.faithfulness_by_qtype),
        "total_queries_scored": report.total_queries_scored,
        "total_queries_skipped": report.total_queries_skipped,
        "total_queries_judged": report.total_queries_judged,
    }
    if show_samples and faithfulness_results is not None:
        payload["per_query_samples"] = {
            qid: [
                {"score": float(s.score), "reasoning": s.reasoning}
                for s in qres.samples
            ]
            for qid, qres in faithfulness_results.per_query.items()
        }
    print(json.dumps(payload), file=stdout)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_cli(
    pack_path: str,
    *,
    k: int = 5,
    format: str = "text",
    verbose: bool = False,
    fresh: bool = True,
    samples_per_claim: int | None = None,
    max_parallel_judges: int | None = None,
    show_samples: bool = False,
    chat_model_factory: Callable[..., Any] | None = None,
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    """Run the full eval pipeline against ``pack_path`` and return an exit code.

    Exit codes:
      * ``0`` — gate passed
      * ``1`` — gate failed (at least one threshold breach)
      * ``2`` — pack load/validation error
      * ``3`` — any other runtime / infra error
    """
    if stdout is None:
        stdout = sys.stdout
    if stderr is None:
        stderr = sys.stderr

    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    # --- Phase 1: load (errors → exit 2) ---
    from src.eval.pack import load_pack
    from src.eval.pack.errors import PackValidationError

    try:
        pack = load_pack(pack_path)
    except PackValidationError as exc:
        print(f"pack load error: {exc}", file=stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"pack load error: {exc}", file=stderr)
        return 2

    pack_dir = Path(pack_path)

    # --- Phase 2+: ingest → retrieve → judge → report → gate (errors → exit 3) ---
    try:
        # Lazy import: avoid pulling LLM stubs at module-top.
        from src.eval import runner as runner_pkg
        from src.eval.runner import (
            aggregate_recall_by_qtype,
            build_eval_report,
            plan_pack_ingest,
            read_max_parallel_judges,
            read_samples_per_claim,
            retrieve_for_goldens,
            score_goldens,
            validate_eval_report,
        )

        effective_samples = (
            samples_per_claim
            if samples_per_claim is not None
            else read_samples_per_claim(pack)
        )
        effective_parallel = (
            max_parallel_judges
            if max_parallel_judges is not None
            else read_max_parallel_judges(pack)
        )

        plan = plan_pack_ingest(pack, pack_dir)
        ingest_report = runner_pkg.execute_plan(plan, fresh=fresh)
        logger.info(
            "Ingest complete: collection=%s docs=%d chunks=%d",
            ingest_report.collection_name,
            ingest_report.document_count,
            ingest_report.stored_chunks,
        )

        retrieval_results = retrieve_for_goldens(
            ingest_report.collection_name, pack.goldens, k=k
        )

        judge_client = runner_pkg.build_judge_client(
            pack,
            chat_model_factory=chat_model_factory,
            pack_dir=pack_dir,
        )
        faithfulness_results = score_goldens(
            retrieval_results,
            pack.goldens,
            judge_client,
            samples_per_claim=effective_samples,
            max_parallel_judges=effective_parallel,
        )

        recall_by_qtype = aggregate_recall_by_qtype(
            retrieval_results, pack.goldens
        )
        # Per-query recall: pure recall, computed in-line to keep build_eval_report happy.
        from src.eval.runner.metrics import recall_at_k as _recall_at_k

        per_query_recall: dict[str, float] = {}
        for qtype, golden_list in pack.goldens.items():
            for golden in golden_list:
                if not golden.expected_source_docs:
                    continue
                qr = retrieval_results.per_query.get(golden.qid)
                if qr is None:
                    continue
                per_query_recall[golden.qid] = _recall_at_k(
                    qr.retrieved_sources, golden.expected_source_docs
                )

        report = build_eval_report(
            retrieval_results,
            faithfulness_results,
            recall_by_qtype,
            per_query_recall,
        )
        gate = validate_eval_report(pack, report)
    except Exception as exc:  # noqa: BLE001 — infra failures are bucketed to exit 3.
        print(f"eval runtime error: {type(exc).__name__}: {exc}", file=stderr)
        if verbose:
            import traceback

            traceback.print_exc(file=stderr)
        return 3

    # --- Phase 3: emit + gate-to-exit-code ---
    if format == "json":
        _emit_json(
            report,
            gate,
            stdout=stdout,
            faithfulness_results=faithfulness_results,
            show_samples=show_samples,
        )
    else:
        _emit_text(report, gate, stdout=stdout, stderr=stderr)

    return 0 if gate.passed else 1


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Parse argv, dispatch to ``run_cli``, return its exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return run_cli(
            args.pack_path,
            k=args.k,
            format=args.format,
            verbose=args.verbose,
            fresh=args.fresh,
            samples_per_claim=args.samples_per_claim,
            max_parallel_judges=args.max_parallel_judges,
            show_samples=getattr(args, "show_samples", False),
        )
    parser.error(f"unknown command: {args.command}")
    return 2  # pragma: no cover — parser.error() exits.


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
