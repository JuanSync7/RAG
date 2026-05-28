"""Tests for the P3.0 pure-logic pack ingest planner.

The planner maps a loaded ``EvalPack`` into an ``IngestPlan`` whose field
names line up with the ``ingest_directory`` entrypoint, so P3 can call
ingest cleanly. The planner itself imports nothing from ``src/ingest``,
``src/vector_db``, or any weaviate-related module — this is enforced by
``test_planner_does_not_import_ingest_or_weaviate``.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from src.eval.pack import load_pack
from src.eval.runner import IngestPlan, plan_pack_ingest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPENTITAN_DIR = PROJECT_ROOT / "evals" / "packs" / "opentitan_riscv"
EXAMPLE_DIR = PROJECT_ROOT / "evals" / "packs" / "_example_"


def _load_opentitan():
    return load_pack(OPENTITAN_DIR), OPENTITAN_DIR


def _load_example():
    return load_pack(EXAMPLE_DIR), EXAMPLE_DIR


def test_plan_resolves_opentitan_pack():
    pack, pack_dir = _load_opentitan()
    plan = plan_pack_ingest(pack, pack_dir)

    assert isinstance(plan, IngestPlan)
    assert plan.pack_name == "opentitan_riscv"
    assert plan.collection_name == pack.collection_name
    assert plan.documents_dir == pack_dir / "corpus"
    assert len(plan.doc_paths) == len(pack.manifest) == 5
    assert plan.expected_doc_count == 5
    assert plan.corpus_pin == pack.meta.corpus_pin


def test_plan_doc_paths_preserve_manifest_order():
    pack, pack_dir = _load_opentitan()
    plan = plan_pack_ingest(pack, pack_dir)

    for i, entry in enumerate(pack.manifest):
        assert plan.doc_paths[i] == pack_dir / "corpus" / entry.path


def test_plan_works_for_example_pack():
    pack, pack_dir = _load_example()
    plan = plan_pack_ingest(pack, pack_dir)

    assert plan.pack_name == "_example_"
    assert plan.expected_doc_count == 1
    assert len(plan.doc_paths) == 1


def test_plan_is_frozen():
    pack, pack_dir = _load_opentitan()
    plan = plan_pack_ingest(pack, pack_dir)

    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.collection_name = "x"  # type: ignore[misc]


def test_plan_collection_name_template_resolved():
    pack, pack_dir = _load_opentitan()
    plan = plan_pack_ingest(pack, pack_dir)

    expected = pack.meta.collection_name_template.format(
        name=pack.meta.name,
        corpus_pin_short=pack.meta.corpus_pin[:8],
    )
    assert plan.collection_name == expected


def test_plan_doc_paths_all_exist():
    pack, pack_dir = _load_opentitan()
    plan = plan_pack_ingest(pack, pack_dir)

    for p in plan.doc_paths:
        assert p.exists(), f"manifest doc missing on disk: {p}"


def test_planner_does_not_import_ingest_or_weaviate():
    """Anti-gaming guard: planner must stay pure-logic.

    Source-string inspection is intentional — even a lazy/local import of
    ``src.ingest`` or anything weaviate-shaped would show up here.
    """
    plan_src = (PROJECT_ROOT / "src" / "eval" / "runner" / "plan.py").read_text()
    assert "src.ingest" not in plan_src
    assert "weaviate" not in plan_src.lower()
