# @summary
# Tests for the opentitan_riscv eval_pack: P0.5 loader validation,
# dry-run script behavior, and isolated live ingest into Weaviate.
# Exports: test_pack_validates_under_p0_5_loader, test_eval_ingest_script_dry_run,
#          test_ingest_into_isolated_collection
# Deps: src.eval.pack, scripts/eval_ingest.py, config.settings
# @end-summary
"""P1 — OpenTitan/Ibex/RISC-V reference pack ingest tests."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PACK_DIR = REPO_ROOT / "evals" / "packs" / "opentitan_riscv"
EVAL_INGEST_SCRIPT = REPO_ROOT / "scripts" / "eval_ingest.py"


def _recompute_corpus_pin(manifest_entries: list[dict]) -> str:
    lines = sorted(f"{e['path']}:{e['sha256']}" for e in manifest_entries)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def test_pack_validates_under_p0_5_loader() -> None:
    """The opentitan_riscv pack loads under the P0.5 loader with the expected
    meta fields, manifest size, and template-resolved collection name."""
    from src.eval.pack import EvalPack, load_pack

    pack = load_pack(PACK_DIR)
    assert isinstance(pack, EvalPack)

    assert pack.meta.name == "opentitan_riscv"
    assert pack.meta.profile == "asic_riscv_soc"
    assert pack.meta.version == 1

    assert len(pack.manifest) >= 3
    assert len(pack.manifest) == 5  # five-doc corpus per slice spec

    # Recompute corpus_pin from the manifest on disk (anti-gaming: no hardcoded hex).
    raw = json.loads((PACK_DIR / "corpus" / "manifest.json").read_text())
    recomputed = _recompute_corpus_pin(raw["entries"])
    assert pack.meta.corpus_pin == recomputed

    # P0.5 loader requires at least one goldens/*.jsonl file; we ship an empty
    # placeholder, so the rows list is empty.  P2 owns golden authoring.
    for qtype, rows in pack.goldens.items():
        assert rows == [], f"P1 declares no goldens; qtype={qtype} unexpectedly populated"

    short = pack.meta.corpus_pin[:8]
    assert pack.collection_name == f"ragweave_test_opentitan_riscv_{short}"


def test_eval_ingest_script_dry_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """`python scripts/eval_ingest.py --pack opentitan_riscv --dry-run` emits a
    JSON envelope on stdout, exits 0, and does NOT import the Weaviate/ingest
    engine along its dry-run path (structural import-check)."""
    from src.eval.pack import load_pack

    pack = load_pack(PACK_DIR)
    expected_collection = pack.collection_name

    # Run the script as a subprocess so we exercise the real entrypoint.
    proc = subprocess.run(
        [sys.executable, str(EVAL_INGEST_SCRIPT), "--pack", "opentitan_riscv", "--dry-run"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"dry-run exit={proc.returncode}; stderr={proc.stderr}"

    envelope = json.loads(proc.stdout)
    assert set(envelope.keys()) >= {"pack", "collection", "manifest_pin", "files_to_ingest"}
    assert envelope["pack"] == "opentitan_riscv"
    assert envelope["collection"] == expected_collection
    assert envelope["manifest_pin"] == pack.meta.corpus_pin
    assert isinstance(envelope["files_to_ingest"], list)
    assert len(envelope["files_to_ingest"]) == len(pack.manifest)

    # Anti-gaming structural import-check: ensure the script source defers the
    # heavy ingest/weaviate imports behind the dry-run gate.  We assert the
    # gating pattern appears in the script source and only AFTER that gate do
    # we see any reference to weaviate/src.vector_db/src.ingest.impl.
    src = EVAL_INGEST_SCRIPT.read_text()
    assert "if not args.dry_run:" in src, "expected dry-run gate `if not args.dry_run:`"
    gate_index = src.index("if not args.dry_run:")
    head, tail = src[:gate_index], src[gate_index:]
    forbidden_in_head = ("import weaviate", "from weaviate", "src.vector_db", "src.ingest.impl")
    for token in forbidden_in_head:
        assert token not in head, (
            f"forbidden token {token!r} found before the dry-run gate "
            f"(must be deferred behind `if not args.dry_run:`)"
        )


@pytest.mark.slow
@pytest.mark.integration
def test_ingest_into_isolated_collection(tmp_path: Path) -> None:
    """Live test: ingest the pack into an isolated Weaviate collection via
    `scripts/eval_ingest.py`, asserting (a) the target collection exists with
    chunks, (b) `VECTOR_COLLECTION_DEFAULT` is unaffected (zero leakage), and
    (c) the target collection is cleaned up in try/finally."""
    pytest.importorskip("weaviate")
    import socket

    from config.settings import MINIO_ENDPOINT, VECTOR_COLLECTION_DEFAULT
    from src.eval.pack import load_pack
    from src.vector_db import close_client, create_persistent_client

    try:
        client = create_persistent_client()
    except Exception as exc:  # pragma: no cover - infra-dependent
        pytest.skip(f"Live Weaviate not reachable: {exc}")

    # The ingest CLI also depends on MinIO (object store) and Temporal.
    # Probe MinIO with a short TCP connect; skip cleanly on infra absence.
    try:
        host, _, port = MINIO_ENDPOINT.partition(":")
        with socket.create_connection((host or "localhost", int(port or "9000")), timeout=2):
            pass
    except Exception as exc:  # pragma: no cover - infra-dependent
        try:
            close_client(client)
        except Exception:
            pass
        pytest.skip(f"Live MinIO ({MINIO_ENDPOINT}) not reachable: {exc}")

    # Copy the pack into a temporary location INSIDE the worktree so that
    # `src.ingest.cli` `validate_documents_dir` (project-root containment
    # check) accepts the per-document --file path. Cleaned up in finally.
    unique_suffix = uuid.uuid4().hex[:8]
    unique_name = f"opentitan_riscv_test_{unique_suffix}"
    packs_root = REPO_ROOT / "evals" / "packs" / f"_p1_live_test_{unique_suffix}"
    target_pack_dir = packs_root / unique_name
    shutil.copytree(PACK_DIR, target_pack_dir)

    pack_yaml_path = target_pack_dir / "pack.yaml"
    pack_data = yaml.safe_load(pack_yaml_path.read_text())
    pack_data["name"] = unique_name
    pack_yaml_path.write_text(yaml.safe_dump(pack_data, sort_keys=False))

    # Resolve the expected collection name through the loader.
    reloaded = load_pack(target_pack_dir)
    target_collection = reloaded.collection_name

    def _count(coll: str) -> int:
        if not client.collections.exists(coll):
            return 0
        return client.collections.get(coll).aggregate.over_all(total_count=True).total_count

    default_count_before = _count(VECTOR_COLLECTION_DEFAULT)

    try:
        proc = subprocess.run(
            [
                sys.executable,
                str(EVAL_INGEST_SCRIPT),
                "--pack",
                unique_name,
                "--pack-root",
                str(packs_root),
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, (
            f"live ingest exit={proc.returncode}\n"
            f"stdout={proc.stdout}\nstderr={proc.stderr}"
        )

        # 5s settle loop for indexer consistency.
        target_count = 0
        for _ in range(10):
            if client.collections.exists(target_collection):
                target_count = (
                    client.collections.get(target_collection)
                    .aggregate.over_all(total_count=True)
                    .total_count
                )
                if target_count > 0:
                    break
            time.sleep(0.5)

        assert client.collections.exists(target_collection), (
            f"expected target collection {target_collection!r} to exist after ingest"
        )
        assert target_count > 0, (
            f"expected target collection {target_collection!r} to have chunks; got 0"
        )

        default_count_after = _count(VECTOR_COLLECTION_DEFAULT)
        assert default_count_after == default_count_before, (
            f"prod collection leakage: {VECTOR_COLLECTION_DEFAULT} "
            f"went {default_count_before} -> {default_count_after}"
        )
    finally:
        try:
            if client.collections.exists(target_collection):
                client.collections.delete(target_collection)
        finally:
            try:
                close_client(client)
            except Exception:
                pass
            # Always clean up the temporary pack copy (project-root location).
            shutil.rmtree(packs_root, ignore_errors=True)
