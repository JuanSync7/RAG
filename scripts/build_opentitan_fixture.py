"""Build a tree-retrieval eval fixture from OpenTitan IP markdown docs.

Usage::

    python scripts/build_opentitan_fixture.py \
        --ip uart \
        --src opentitan_data/hw/ip/uart/doc \
        --out tests/fixtures/opentitan_uart

The script emits ``corpus.json`` containing both leaves (one per markdown
paragraph or code block) and section nodes (one per unique heading-path
prefix), keyed by the IP name as ``document_id``. Queries and gold labels
are authored separately at ``queries.json`` / ``gold.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as a script from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.markdown_chunker import (  # noqa: E402
    chunks_from_markdown,
    emit_section_nodes,
)


# Files we extract from. Order matters: it determines chunk_index ordering
# (which the eval insertion-order assumptions depend on).
_DEFAULT_FILES = [
    "theory_of_operation.md",
    "registers.md",
    "programmers_guide.md",
    "interfaces.md",
]


def build_corpus(ip: str, src: Path, files: list[str]) -> dict:
    leaves: list[dict] = []
    for fname in files:
        path = src / fname
        if not path.exists():
            print(f"  skip {fname} (not found)", file=sys.stderr)
            continue
        text = path.read_text(encoding="utf-8")
        # Document id = "<ip>__<file_stem>" so each file is a separate doc;
        # tree retrieval still spans them via the cross-doc descent.
        doc_id = f"{ip}__{path.stem}"
        new_leaves = chunks_from_markdown(text, document_id=doc_id)
        leaves.extend(new_leaves)
        print(f"  {fname}: {len(new_leaves)} leaves", file=sys.stderr)

    # One section-node set per document_id so heading_paths don't collide
    # across files (each file has its own '# ...' top heading).
    by_doc: dict[str, list[dict]] = {}
    for leaf in leaves:
        by_doc.setdefault(leaf["document_id"], []).append(leaf)

    documents = []
    for doc_id, doc_leaves in by_doc.items():
        sections = emit_section_nodes(doc_leaves, document_id=doc_id)
        # Sections first, then leaves, for deterministic insertion-order
        # behaviour in the simulated retriever.
        documents.append({
            "document_id": doc_id,
            "chunks": sections + doc_leaves,
        })
        print(
            f"  {doc_id}: {len(sections)} sections, {len(doc_leaves)} leaves",
            file=sys.stderr,
        )

    return {"documents": documents}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", required=True)
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--files", nargs="*", default=_DEFAULT_FILES)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    corpus = build_corpus(args.ip, args.src, args.files)
    (args.out / "corpus.json").write_text(json.dumps(corpus, indent=2))

    # Don't overwrite queries/gold — they are hand-authored.
    for stub in ("queries.json", "gold.json"):
        target = args.out / stub
        if not target.exists():
            target.write_text("[]" if stub == "queries.json" else "{}")
            print(f"  created empty {stub} (hand-author next)", file=sys.stderr)

    print(f"\nWrote {args.out / 'corpus.json'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
