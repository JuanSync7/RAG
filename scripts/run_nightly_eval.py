#!/usr/bin/env python3
# @summary
# P8a — thin CLI shim for the nightly eval orchestrator. Parses argv (pack-root,
# pack glob, output-dir, k, fresh, verbose), resolves packs via
# discover_packs, and delegates to run_nightly. NO business logic — everything
# lives in src.eval.orchestrator. Exit codes: run_nightly's 0/1, or 2 when no
# packs are discovered (pack error).
# Exports: main
# Deps: stdlib (argparse, logging, sys); src.eval.orchestrator.
# @end-summary
"""Nightly eval orchestrator CLI shim (P8a).

Usage::

    python scripts/run_nightly_eval.py --pack opentitan_riscv --output-dir eval_reports

This is a thin argparse adapter over :mod:`src.eval.orchestrator`. It discovers
the requested pack(s), runs the nightly loop, and returns the orchestrator's
exit code (0 = all passed, 1 = a gate failed or a pack errored). When no packs
are discovered it prints to stderr and returns 2 (pack error).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make the repo importable when this file is run directly
# (``python scripts/run_nightly_eval.py``); harmless when imported as a module.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.eval.orchestrator import discover_packs, run_nightly  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Parse argv, resolve packs, run the nightly loop, return its exit code."""
    parser = argparse.ArgumentParser(
        prog="run_nightly_eval",
        description="Run the nightly eval loop over discovered eval pack(s).",
    )
    parser.add_argument(
        "--pack-root",
        default="evals/packs",
        help="Directory containing eval pack subdirectories (default: evals/packs).",
    )
    parser.add_argument(
        "--pack",
        default="*",
        help="Pack name or glob to select (default: '*' = all packs).",
    )
    parser.add_argument(
        "--output-dir",
        default="eval_reports",
        help="Directory to write persisted report JSON files (default: eval_reports).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Top-k retrieval cutoff per query (default: 5).",
    )
    parser.add_argument(
        "--no-fresh",
        dest="fresh",
        action="store_false",
        help="Incremental update mode; do not drop the collection before ingest.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )
    parser.set_defaults(fresh=True)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    pack_paths = discover_packs(args.pack_root, args.pack)
    if not pack_paths:
        print(
            f"no eval packs found under {args.pack_root!r} matching {args.pack!r}",
            file=sys.stderr,
        )
        return 2

    return run_nightly(
        pack_paths,
        output_dir=args.output_dir,
        k=args.k,
        fresh=args.fresh,
    )


if __name__ == "__main__":
    raise SystemExit(main())
