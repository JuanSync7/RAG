# @summary
# Pure-logic pack ingest planner — maps an EvalPack into an IngestPlan
# whose field names match the real ingest_directory entrypoint, so P3
# can wire ingest in without re-deriving paths or collection names.
# Exports: IngestPlan, plan_pack_ingest
# Deps: stdlib (dataclasses, pathlib); src.eval.pack (EvalPack only).
# @end-summary
"""Pure-logic helper that turns a loaded ``EvalPack`` into an ``IngestPlan``.

The planner has no awareness of the vector store, the ingest pipeline,
or any side-effecting infrastructure. It joins paths and reads pack
properties. That is the entire contract — P3 layers the actual ingest
call on top.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..pack import EvalPack


@dataclass(frozen=True)
class IngestPlan:
    """Frozen, hashable description of how to ingest a pack.

    Field names intentionally mirror ``ingest_directory``'s parameter
    shape (``documents_dir``, ``collection_name`` via target collection)
    so callers can route fields directly into the ingest entrypoint.
    """

    pack_name: str
    collection_name: str
    documents_dir: Path
    doc_paths: tuple[Path, ...]
    expected_doc_count: int
    corpus_pin: str


def plan_pack_ingest(pack: EvalPack, pack_dir: Path) -> IngestPlan:
    """Map a loaded ``EvalPack`` to an ``IngestPlan``.

    ``pack_dir`` is the directory containing ``pack.yaml``; the corpus
    lives at ``pack_dir / "corpus"`` and every manifest entry's
    ``.path`` is joined under that directory in manifest order. No
    on-disk validation is performed — that is ``validate_pack``'s job.
    """

    documents_dir = pack_dir / "corpus"
    doc_paths = tuple(documents_dir / entry.path for entry in pack.manifest)
    return IngestPlan(
        pack_name=pack.meta.name,
        collection_name=pack.collection_name,
        documents_dir=documents_dir,
        doc_paths=doc_paths,
        expected_doc_count=len(pack.manifest),
        corpus_pin=pack.meta.corpus_pin,
    )
