#!/usr/bin/env python3
# @summary
# Eval_pack ingest entrypoint — loads a pack via the P0.5 loader and either
# emits a dry-run JSON envelope (no heavy imports) or invokes the existing
# ingest CLI per-document with the per-pack collection-name override (P0
# plumbing). Designed to keep `src/ingest/`, `src/retrieval/`, and
# `src/vector_db/` untouched: this is content-on-loop separation.
# Exports: main
# Deps: argparse, json, subprocess, sys, pathlib, src.eval.pack
# @end-summary
"""CLI that ingests an eval_pack's corpus into an isolated Weaviate collection.

Usage:
    python scripts/eval_ingest.py --pack opentitan_riscv [--dry-run] [--pack-root PATH]

In ``--dry-run`` mode the script emits a single JSON envelope on stdout with
the keys ``pack``, ``collection``, ``manifest_pin``, ``files_to_ingest`` and
exits 0 WITHOUT importing the Weaviate client or any ingest engine module —
this is the anti-coupling guard the fast tests assert on.

In live mode the script shells out to ``python -m src.ingest.cli --file <doc>
--collection <target>`` once per manifest entry, threading the per-pack
``collection_name`` (resolved by P0.5 template substitution) through P0's
``--collection`` plumbing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow imports from project root regardless of CWD.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.eval.pack import PackValidationError, load_pack  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest an eval_pack's corpus into an isolated Weaviate collection.",
    )
    parser.add_argument(
        "--pack",
        required=True,
        help="Pack name (directory name under --pack-root).",
    )
    parser.add_argument(
        "--pack-root",
        type=Path,
        default=Path("evals/packs"),
        help="Root directory containing pack directories (default: evals/packs).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print a JSON envelope describing what would be ingested; perform no writes.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entrypoint. Returns the process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    pack_root: Path = args.pack_root
    if not pack_root.is_absolute():
        pack_root = (_REPO_ROOT / pack_root).resolve()
    pack_dir = pack_root / args.pack

    try:
        pack = load_pack(pack_dir)
    except PackValidationError as exc:
        print(f"PackValidationError: {exc}", file=sys.stderr)
        return 2

    target_collection = pack.collection_name
    files_to_ingest = [str(pack_dir / "corpus" / entry.path) for entry in pack.manifest]

    envelope = {
        "pack": pack.meta.name,
        "collection": target_collection,
        "manifest_pin": pack.meta.corpus_pin,
        "files_to_ingest": files_to_ingest,
    }

    if args.dry_run:
        # Dry-run path: emit envelope and exit. NO heavy imports above this gate.
        print(json.dumps(envelope))
        return 0

    # Live mode below: deferred imports keep the dry-run path lightweight.
    if not args.dry_run:
        import subprocess  # noqa: PLC0415 - deferred behind the dry-run gate

        failures: list[tuple[str, int, str]] = []
        for doc_path in files_to_ingest:
            cmd = [
                sys.executable,
                "-m",
                "src.ingest.cli",
                "--file",
                doc_path,
                "--collection",
                target_collection,
            ]
            proc = subprocess.run(
                cmd,
                cwd=str(_REPO_ROOT),
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                failures.append((doc_path, proc.returncode, proc.stderr or proc.stdout))

        if failures:
            print(
                f"eval_ingest: {len(failures)}/{len(files_to_ingest)} ingest invocations failed:",
                file=sys.stderr,
            )
            for doc_path, rc, err in failures:
                print(f"  - {doc_path}: rc={rc}", file=sys.stderr)
                print(f"    stderr: {err.strip()[:500]}", file=sys.stderr)
            return 1

        # Success envelope on stdout for live mode as well.
        print(json.dumps({**envelope, "ingested": len(files_to_ingest)}))
        return 0

    return 0  # pragma: no cover - unreachable


if __name__ == "__main__":
    sys.exit(main())
