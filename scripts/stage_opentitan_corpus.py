"""Materialize an ingestion-ready mirror of the OpenTitan corpus under
``documents/opentitan/`` using symlinks (no file copies).

The existing ingest CLI (``python -m src.ingest.cli --dir <path>``) walks a
directory tree and ingests every supported file. Rather than fork a manifest
loader, we just stage a curated tree of symlinks — preserves source paths
(so ``metadata.source`` reads as ``hw/ip/aes/doc/theory_of_operation.md``)
and stays cheap to refresh.

Reads ``evals/retrieval/deep_research/fixtures/opentitan/ingestion_manifest.json``
(must be built first via ``scripts/build_opentitan_manifest.py``).

Run with::

    python scripts/stage_opentitan_corpus.py
    python -m src.ingest.cli --dir documents/opentitan --update
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

DEFAULT_MANIFEST = Path("evals/retrieval/deep_research/fixtures/opentitan/ingestion_manifest.json")
DEFAULT_DEST = Path("documents/opentitan")


def stage(manifest_path: Path, dest: Path, force: bool) -> tuple[int, int]:
    if not manifest_path.exists():
        raise SystemExit(
            f"Manifest not found: {manifest_path}. "
            "Run scripts/build_opentitan_manifest.py first."
        )
    payload = json.loads(manifest_path.read_text())
    files = payload["files"]

    if force and dest.exists():
        for root, _dirs, filenames in os.walk(dest, topdown=False):
            root_p = Path(root)
            for fn in filenames:
                (root_p / fn).unlink(missing_ok=True)
            try:
                root_p.rmdir()
            except OSError:
                pass

    dest.mkdir(parents=True, exist_ok=True)
    linked = 0
    skipped = 0
    for entry in files:
        src = Path(entry["path"])
        rel = entry["rel_path"]
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            try:
                if target.is_symlink() and target.resolve() == src.resolve():
                    skipped += 1
                    continue
                target.unlink()
            except OSError:
                skipped += 1
                continue
        target.symlink_to(src)
        linked += 1
    return linked, skipped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    ap.add_argument(
        "--force",
        action="store_true",
        help="Wipe the destination tree before re-staging",
    )
    args = ap.parse_args()

    linked, skipped = stage(args.manifest, args.dest, args.force)
    print(f"Staged {linked} new symlinks under {args.dest} ({skipped} unchanged)")
    print()
    print("Next steps:")
    print(f"  python -m src.ingest.cli --dir {args.dest} --update")


if __name__ == "__main__":
    main()
