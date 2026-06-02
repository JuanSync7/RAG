# @summary
# P6b — Eval CLI runner. Orchestrates load_pack → plan_pack_ingest →
# execute_plan → retrieve_for_goldens → score_goldens → build_eval_report
# → validate_eval_report and translates the GateResult into an exit code.
# Heavy modules (execute_plan, retrieve, judge) are LAZY-imported inside
# run_cli so the parser stays cheap and tests can monkeypatch the runner
# package symbols without ordering hazards.
# Exit codes: 0=pass, 1=gate fail, 2=pack load/validation error,
# 3=runtime/infra error.
# P7d adds the CI summary write seam: --summary-file / $GITHUB_STEP_SUMMARY
# auto-detect, rendering via src.eval.runner.render_ci_summary on BOTH the
# gate-pass and gate-fail paths (never on exit-2/exit-3 paths).
# P8a.0 extracts a reusable, return-the-objects core: run_eval(...) performs
# Phase 1+2 and RETURNS an EvalRunResult (pack/report/gate/collection_name/
# faithfulness_results); run_cli is now a thin wrapper that maps run_eval's
# raised errors to exit codes (2/3) and runs Phase 3 (emit + CI summary)
# unchanged. _report_payload single-sources the JSON shape shared by
# _emit_json and runner.persistence.persist_eval_report.
# P8b adds the promote-baseline subcommand: promote_baseline(pack, ...) writes
# the latest (or given) report JSON to <packs_root>/<pack>/baseline.json (the
# git-tracked baseline that runner.baseline diffs against); no git-commit.
# P10 adds three default-preserving DI seams to run_eval: execute_fn /
# retrieve_fn / judge_client_factory (each defaulting to None and resolved to
# the live runner symbol at CALL TIME after the lazy import). These let the
# offline CI smoke (src.eval.smoke) drive the FULL loop with no live infra and
# no monkeypatch; run_cli is unchanged (live CLI keeps the live defaults).
# --judge-backend {litellm,claude-cli} selects the LLM-as-judge backend on the
# run subcommand: litellm (default, in-process chat model) or claude-cli (shells
# out to the headless `claude` CLI via runner.build_claude_cli_judge_client).
# The default path is unchanged; selection is resolved at CALL TIME in run_eval
# after the lazy runner import (an explicit judge_client_factory still wins).
# P15: --judge-retries (default 1) threads through run_cli → run_eval →
# score_goldens(judge_retries=) for bounded per-golden judge retries; a golden
# still failing after retries is skipped + counted, not run-aborting.
# show-trend adds a READ-ONLY run-history inspection subcommand:
# show_trend(pack, ...) calls load_run_history → render_trend_table and prints a
# markdown per-(qtype, metric) trend table. It performs NO writes/ingest/
# retrieve/judge; missing/empty history → friendly message at exit 0.
# Exports: _build_parser, _report_payload, run_eval, EvalRunResult, run_cli,
#          promote_baseline, show_trend, main
# Deps: argparse, json, logging, os, sys; src.eval.pack (errors, loader);
#       src.eval.runner (lazy) — execute_plan, retrieve_for_goldens,
#       score_goldens, build_judge_client, build_eval_report,
#       aggregate_recall_by_qtype, aggregate_mrr_by_qtype,
#       validate_eval_report, plan_pack_ingest,
#       read_samples_per_claim, read_max_parallel_judges.
# @end-summary
"""Eval CLI runner (P6b).

The ``run`` subcommand chains the full eval pipeline against a loaded
eval_pack and exits with a deterministic code reflecting the gate outcome.
The ``promote-baseline`` subcommand (P8b) writes a report to the pack's
committed ``baseline.json``. The ``show-trend`` subcommand is a READ-ONLY
inspection surface that loads a pack's append-only run-history index and prints
a markdown per-``(qtype, metric)`` trend table (no ingest/retrieve/judge, no
writes). The CLI is a thin orchestrator — no business logic.

Lazy imports for heavy modules keep argparse-time work to a minimum and
sidestep langchain_core stub-timing hazards in the test environment.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
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
        "--summary-file",
        type=str,
        default=None,
        help=(
            "Write a GitHub-flavored-markdown PASS/FAIL summary to PATH. "
            "When omitted, falls back to $GITHUB_STEP_SUMMARY if set; "
            "otherwise no summary is written. Written on both gate-pass "
            "and gate-fail."
        ),
    )
    run_parser.add_argument(
        "--judge-backend",
        choices=("litellm", "claude-cli"),
        default="litellm",
        help=(
            "LLM-as-judge backend. 'litellm' (default) uses the in-process "
            "litellm chat model; 'claude-cli' shells out to the local headless "
            "`claude` CLI (constrained: no tools, clean-room settings)."
        ),
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
    run_parser.add_argument(
        "--judge-retries",
        type=int,
        default=1,
        help=(
            "Bounded per-golden judge retries (default: 1 → up to 2 attempts). "
            "A golden whose judge call still fails after retries is skipped + "
            "counted (not silently dropped) and the run continues on survivors."
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

    # --- promote-baseline subcommand (P8b) ---
    promote_parser = subparsers.add_parser(
        "promote-baseline",
        help="Promote an eval report to the pack's committed baseline.json.",
    )
    promote_parser.add_argument(
        "pack",
        help="Pack name (directory under --packs-root).",
    )
    promote_parser.add_argument(
        "--report-json",
        default=None,
        help="Report JSON to promote. If omitted, the latest "
        "<reports-dir>/<pack>_*.json (by filename) is used.",
    )
    promote_parser.add_argument(
        "--packs-root",
        default="evals/packs",
        help="Root directory containing pack directories (default: evals/packs).",
    )
    promote_parser.add_argument(
        "--reports-dir",
        default="eval_reports",
        help="Directory of persisted reports for latest-discovery "
        "(default: eval_reports).",
    )

    # --- show-trend subcommand (read-only run-history inspection) ---
    trend_parser = subparsers.add_parser(
        "show-trend",
        help="Print a markdown per-(qtype, metric) trend table from a pack's "
        "run-history index (read-only; no ingest/retrieve/judge).",
    )
    trend_parser.add_argument(
        "pack",
        help="Pack name whose <pack>.history.jsonl lives under --reports-dir.",
    )
    trend_parser.add_argument(
        "--reports-dir",
        default="eval_reports",
        help="Directory containing <pack>.history.jsonl (default: eval_reports).",
    )
    trend_parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="How many most-recent runs to show as delta columns (default: 3).",
    )
    trend_parser.add_argument(
        "--window",
        type=int,
        default=3,
        help="Detector lookback: recent runs inspected for sustained "
        "regression (default: 3).",
    )
    trend_parser.add_argument(
        "--min-regressed",
        type=int,
        default=2,
        help="Min regressed runs in the window to flag a pair sustained "
        "(default: 2).",
    )
    trend_parser.add_argument(
        "--epsilon",
        type=float,
        default=0.05,
        help="Regression tolerance band; predicate is strict delta < -epsilon "
        "(default: 0.05).",
    )
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


def _report_payload(report, gate) -> dict[str, Any]:
    """Build the canonical report+gate JSON payload dict (single source).

    Both :func:`_emit_json` and :func:`persist_eval_report
    <src.eval.runner.persistence.persist_eval_report>` consume this so the
    serialized eval shape is single-sourced and cannot drift between the
    stdout emission and the persisted report file.
    """
    return {
        "collection_name": report.collection_name,
        "k": report.k,
        "passed": gate.passed,
        "failures": [
            {
                "qtype": f.qtype,
                "metric": f.metric,
                "expected": f.expected,
                "actual": f.actual,
                "qid": f.qid,
            }
            for f in gate.failures
        ],
        "recall_by_qtype": dict(report.recall_by_qtype),
        "faithfulness_by_qtype": dict(report.faithfulness_by_qtype),
        "mrr_by_qtype": dict(report.mrr_by_qtype),
        "per_query_recall": dict(report.per_query_recall),
        "per_query_mrr": dict(report.per_query_mrr),
        "per_query_faithfulness": dict(report.per_query_faithfulness),
        "total_queries_scored": report.total_queries_scored,
        "total_queries_skipped": report.total_queries_skipped,
        "total_queries_judged": report.total_queries_judged,
    }


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
    payload = _report_payload(report, gate)
    if show_samples and faithfulness_results is not None:
        payload["per_query_samples"] = {
            qid: [
                {"score": float(s.score), "reasoning": s.reasoning}
                for s in qres.samples
            ]
            for qid, qres in faithfulness_results.per_query.items()
        }
    print(json.dumps(payload), file=stdout)


def _write_ci_summary(report, gate, summary_file, *, stderr) -> None:
    """Resolve the effective summary path and write the CI markdown there.

    The effective path is ``summary_file`` if not None, else the value of
    ``$GITHUB_STEP_SUMMARY`` (read at call time so tests can monkeypatch it),
    else None (no-op). The render is appended (GitHub Step Summary is
    append-semantics) with a trailing newline; the file is created if absent.

    A write failure is logged to *stderr* and swallowed so it never masks the
    gate exit code.
    """
    effective = summary_file if summary_file is not None else os.environ.get(
        "GITHUB_STEP_SUMMARY"
    )
    if not effective:
        return
    from src.eval.runner import render_ci_summary

    try:
        markdown = render_ci_summary(report, gate)
        with open(effective, "a", encoding="utf-8") as fh:
            fh.write(markdown)
            fh.write("\n")
    except OSError as exc:  # pragma: no cover — defensive: never mask gate code.
        print(f"warning: failed to write CI summary to {effective}: {exc}", file=stderr)


# ---------------------------------------------------------------------------
# Reusable eval core (library surface)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalRunResult:
    """Immutable bundle of everything a single eval run produces.

    Returned by :func:`run_eval` so callers (the CLI, and the P8a
    orchestrator) can persist, alert on, or render the result objects
    without re-deriving them. Fields are additive/trailing-friendly:
    ``faithfulness_results`` is whatever ``score_goldens`` returns and is
    carried so ``--show-samples`` can render per-sample judge scores.
    """

    pack: Any
    report: Any
    gate: Any
    collection_name: str
    faithfulness_results: Any


def run_eval(
    pack_path: str,
    *,
    k: int = 5,
    fresh: bool = True,
    samples_per_claim: int | None = None,
    max_parallel_judges: int | None = None,
    judge_retries: int = 1,
    chat_model_factory: Callable[..., Any] | None = None,
    judge_backend: str = "litellm",
    verbose: bool = False,
    stderr: Any = None,
    execute_fn: Callable[..., Any] | None = None,
    retrieve_fn: Callable[..., Any] | None = None,
    judge_client_factory: Callable[..., Any] | None = None,
) -> "EvalRunResult":
    """Run the full eval pipeline against ``pack_path`` and RETURN the results.

    This is the reusable library core extracted from :func:`run_cli`. It
    performs Phase 1 (``load_pack``) and Phase 2 (plan → execute → retrieve →
    judge → aggregate → build report → gate) and returns the populated
    :class:`EvalRunResult`.

    Unlike :func:`run_cli`, this function does NOT map errors to exit codes:

    * ``PackValidationError`` / ``FileNotFoundError`` propagate for pack
      load/validation errors (the CLI maps these to exit 2).
    * Any other exception propagates for runtime/infra errors (the CLI maps
      these to exit 3).

    Emission, CI-summary writing, and exit-code translation are the CLI's
    responsibility and live in :func:`run_cli`.

    **Dependency-injection seams (P10).** The three external touchpoints —
    ingest, retrieval, and judge-client construction — are injectable for the
    offline CI smoke (:mod:`src.eval.smoke`) so the full loop can run with NO
    live Weaviate / embeddings / LLM and NO monkeypatching:

    * ``execute_fn`` — defaults to ``runner.execute_plan``; called as
      ``execute_fn(plan, fresh=fresh)``.
    * ``retrieve_fn`` — defaults to ``runner.retrieve_for_goldens``; called as
      ``retrieve_fn(collection_name, pack.goldens, k=k)``.
    * ``judge_client_factory`` — defaults to ``runner.build_judge_client``;
      called as ``judge_client_factory(pack, chat_model_factory=..., pack_dir=...)``.

    Each defaults to ``None`` and is resolved to the live implementation at
    CALL TIME (after the lazy ``runner`` import) so existing tests that
    monkeypatch ``src.eval.runner.*`` symbols keep working unchanged.
    """
    if stderr is None:
        stderr = sys.stderr

    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    # --- Phase 1: load (PackValidationError / FileNotFoundError propagate) ---
    from src.eval.pack import load_pack

    pack = load_pack(pack_path)
    pack_dir = Path(pack_path)

    # --- Phase 2: ingest → retrieve → judge → report → gate (errors propagate) ---
    # Lazy import: avoid pulling LLM stubs at module-top.
    from src.eval import runner as runner_pkg
    from src.eval.runner import (
        aggregate_mrr_by_qtype,
        aggregate_recall_by_qtype,
        build_eval_report,
        plan_pack_ingest,
        read_max_parallel_judges,
        read_samples_per_claim,
        retrieve_for_goldens,
        score_goldens,
        validate_eval_report,
    )

    # Resolve the DI seams to their live defaults at CALL TIME (after the lazy
    # ``runner`` import) so monkeypatched ``runner.*`` symbols still take effect.
    execute_fn = execute_fn or runner_pkg.execute_plan
    retrieve_fn = retrieve_fn or retrieve_for_goldens
    # Backend selection: an explicit ``judge_client_factory`` (DI seam / tests)
    # always wins; otherwise pick the litellm default or the headless claude-cli
    # factory. Resolved at CALL TIME so monkeypatched runner symbols take effect.
    if judge_client_factory is None:
        if judge_backend == "claude-cli":
            judge_client_factory = runner_pkg.build_claude_cli_judge_client
        else:
            judge_client_factory = runner_pkg.build_judge_client

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
    ingest_report = execute_fn(plan, fresh=fresh)
    logger.info(
        "Ingest complete: collection=%s docs=%d chunks=%d",
        ingest_report.collection_name,
        ingest_report.document_count,
        ingest_report.stored_chunks,
    )

    retrieval_results = retrieve_fn(
        ingest_report.collection_name, pack.goldens, k=k
    )

    judge_client = judge_client_factory(
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
        judge_retries=judge_retries,
    )

    recall_by_qtype = aggregate_recall_by_qtype(retrieval_results, pack.goldens)
    mrr_by_qtype = aggregate_mrr_by_qtype(retrieval_results, pack.goldens)
    # Per-query recall + mrr: computed in-line (mirror each other) for the
    # per-qid threshold-override checks in the gate (P7f).
    from src.eval.runner.metrics import recall_at_k as _recall_at_k
    from src.eval.runner.metrics import reciprocal_rank as _reciprocal_rank

    per_query_recall: dict[str, float] = {}
    per_query_mrr: dict[str, float] = {}
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
            per_query_mrr[golden.qid] = _reciprocal_rank(
                qr.retrieved_sources, golden.expected_source_docs
            )

    report = build_eval_report(
        retrieval_results,
        faithfulness_results,
        recall_by_qtype,
        per_query_recall,
        mrr_by_qtype=mrr_by_qtype,
        per_query_mrr=per_query_mrr,
    )
    gate = validate_eval_report(pack, report)

    return EvalRunResult(
        pack=pack,
        report=report,
        gate=gate,
        collection_name=report.collection_name,
        faithfulness_results=faithfulness_results,
    )


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
    judge_retries: int = 1,
    show_samples: bool = False,
    summary_file: str | None = None,
    chat_model_factory: Callable[..., Any] | None = None,
    judge_backend: str = "litellm",
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

    # --- Phase 1+2: delegate to the reusable core, mapping errors to codes. ---
    # Pack load/validation errors → exit 2; any other runtime/infra error → 3.
    # The byte-identical stderr messages + return values are preserved here.
    from src.eval.pack.errors import PackValidationError

    try:
        result = run_eval(
            pack_path,
            k=k,
            fresh=fresh,
            samples_per_claim=samples_per_claim,
            max_parallel_judges=max_parallel_judges,
            judge_retries=judge_retries,
            chat_model_factory=chat_model_factory,
            judge_backend=judge_backend,
            verbose=verbose,
            stderr=stderr,
        )
    except (PackValidationError, FileNotFoundError) as exc:
        print(f"pack load error: {exc}", file=stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 — infra failures are bucketed to exit 3.
        print(f"eval runtime error: {type(exc).__name__}: {exc}", file=stderr)
        if verbose:
            import traceback

            traceback.print_exc(file=stderr)
        return 3

    report = result.report
    gate = result.gate
    faithfulness_results = result.faithfulness_results

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

    # CI summary write seam: on BOTH gate-pass and gate-fail (a report+gate
    # exist here). A write failure must not mask the gate exit code.
    _write_ci_summary(report, gate, summary_file, stderr=stderr)

    return 0 if gate.passed else 1


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def promote_baseline(
    pack: str,
    *,
    report_json: str | None = None,
    packs_root: str = "evals/packs",
    reports_dir: str = "eval_reports",
    stdout: Any = None,
    stderr: Any = None,
) -> int:
    """Promote an eval report to ``<packs_root>/<pack>/baseline.json``.

    Resolves the report to promote (explicit ``report_json`` or the latest
    ``<reports_dir>/<pack>_*.json`` by sorted filename — ISO timestamps in the
    filename sort chronologically), then writes it verbatim to the pack's
    committed ``baseline.json``. Does NOT git-commit — versioning the baseline
    is the caller's deliberate step.

    Returns 0 on success; 2 on a pack/report resolution error (no git side
    effects). Deterministic and side-effect-light beyond the single write.
    """
    out = stdout if stdout is not None else sys.stdout
    err = stderr if stderr is not None else sys.stderr

    pack_dir = Path(packs_root) / pack
    if not pack_dir.is_dir():
        print(f"promote-baseline: pack not found: {pack_dir}", file=err)
        return 2

    if report_json is not None:
        report_path = Path(report_json)
        if not report_path.is_file():
            print(
                f"promote-baseline: report not found: {report_path}", file=err
            )
            return 2
    else:
        candidates = sorted(Path(reports_dir).glob(f"{pack}_*.json"))
        if not candidates:
            print(
                f"promote-baseline: no reports found for {pack!r} under "
                f"{reports_dir}",
                file=err,
            )
            return 2
        report_path = candidates[-1]  # latest by sorted (ISO) filename

    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"promote-baseline: cannot read report: {exc}", file=err)
        return 2
    if not isinstance(payload, dict) or "passed" not in payload:
        print(
            f"promote-baseline: malformed report (missing keys): {report_path}",
            file=err,
        )
        return 2

    baseline_path = pack_dir / "baseline.json"
    baseline_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Promoted baseline -> {baseline_path}", file=out)
    return 0


def show_trend(
    pack: str,
    *,
    reports_dir: str = "eval_reports",
    runs: int = 3,
    window: int = 3,
    min_regressed: int = 2,
    epsilon: float = 0.05,
    stdout: Any = None,
) -> int:
    """Print a markdown trend table from ``pack``'s run-history index.

    READ-ONLY inspection surface: loads the append-only history index via
    :func:`~src.eval.runner.run_history.load_run_history` (missing file → ``[]``),
    renders it with the pure
    :func:`~src.eval.runner.show_trend.render_trend_table`, prints to stdout, and
    returns ``0``. Does NOT write anything, load the pack, ingest, retrieve,
    judge, or run the loop. An empty/missing history renders a friendly one-line
    message — still exit ``0`` (a pack with no runs yet is normal, not an error).

    :param pack: pack name whose ``<pack>.history.jsonl`` lives under
        ``reports_dir``.
    :param reports_dir: directory holding the history index (default
        ``eval_reports``; maps to ``history_dir``).
    :param runs: how many most-recent runs to show as delta columns.
    :param window: detector lookback (distinct from ``runs``).
    :param min_regressed: min regressed runs in the window to flag a pair.
    :param epsilon: regression tolerance band (strict ``delta < -epsilon``).
    :returns: ``0`` always (including empty/missing history and no-sustained).
    """
    out = stdout if stdout is not None else sys.stdout

    from src.eval.runner.run_history import load_run_history
    from src.eval.runner.show_trend import render_trend_table

    history = load_run_history(pack, reports_dir)
    table = render_trend_table(
        history,
        runs=runs,
        window=window,
        min_regressed=min_regressed,
        epsilon=epsilon,
    )
    print(table, file=out)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse argv, dispatch to a subcommand, return its exit code."""
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
            judge_retries=getattr(args, "judge_retries", 1),
            show_samples=getattr(args, "show_samples", False),
            summary_file=getattr(args, "summary_file", None),
            judge_backend=getattr(args, "judge_backend", "litellm"),
        )
    if args.command == "promote-baseline":
        return promote_baseline(
            args.pack,
            report_json=args.report_json,
            packs_root=args.packs_root,
            reports_dir=args.reports_dir,
        )
    if args.command == "show-trend":
        return show_trend(
            args.pack,
            reports_dir=args.reports_dir,
            runs=args.runs,
            window=args.window,
            min_regressed=args.min_regressed,
            epsilon=args.epsilon,
        )
    parser.error(f"unknown command: {args.command}")
    return 2  # pragma: no cover — parser.error() exits.


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
