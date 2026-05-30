"""Tests for the eval_pack loader (P0.5 keystone).

Tests observe behaviour through the public loader API only:
    from src.eval.pack import EvalPack, load_pack, validate_pack, PackValidationError
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from src.eval.pack import EvalPack, PackValidationError, load_pack, validate_pack

EXAMPLE_PACK = Path("evals/packs/_example_")


def _compute_corpus_pin(manifest_path: Path) -> str:
    """Recompute corpus_pin from manifest.json — sha256 of sorted '{path}:{sha256}' lines."""
    data = json.loads(manifest_path.read_text())
    entries = data["entries"]
    lines = sorted(f"{e['path']}:{e['sha256']}" for e in entries)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _copy_pack(tmp_path: Path) -> Path:
    dst = tmp_path / "pack"
    shutil.copytree(EXAMPLE_PACK, dst)
    return dst


def test_load_valid_pack() -> None:
    pack = load_pack(EXAMPLE_PACK)

    assert isinstance(pack, EvalPack)
    assert pack.meta.name == "_example_"
    assert pack.meta.version == 1
    assert pack.meta.profile == "asic_riscv_soc"

    expected_pin = _compute_corpus_pin(EXAMPLE_PACK / "corpus" / "manifest.json")
    assert pack.meta.corpus_pin == expected_pin

    factoids = pack.goldens["factoid"]
    assert isinstance(factoids, list) and len(factoids) >= 1
    first = factoids[0]
    # Typed Golden record — required fields surface as attributes
    assert first.qid
    assert first.qtype == "factoid"
    assert first.query
    assert first.expected_answer_span

    r5 = pack.thresholds.defaults["factoid"]["recall_at_5"]
    assert isinstance(r5, float)
    assert 0.0 <= r5 <= 1.0

    expected_collection = f"ragweave_test__example__{expected_pin[:8]}"
    assert pack.collection_name == expected_collection


def test_rejects_missing_manifest(tmp_path: Path) -> None:
    pack_dir = _copy_pack(tmp_path)
    (pack_dir / "corpus" / "manifest.json").unlink()

    with pytest.raises(PackValidationError) as exc_info:
        load_pack(pack_dir)

    msg = str(exc_info.value)
    assert "corpus/manifest.json" in msg
    # Must not be FileNotFoundError or KeyError leaking through
    assert not isinstance(exc_info.value, FileNotFoundError)
    assert not isinstance(exc_info.value, KeyError)


def test_rejects_schema_invalid_golden(tmp_path: Path) -> None:
    pack_dir = _copy_pack(tmp_path)
    goldens_path = pack_dir / "goldens" / "factoid.jsonl"
    bad_line = json.dumps(
        {
            "qid": "f-bad-001",
            "qtype": "qfs",  # mismatch — file is factoid.jsonl
            "query": "What is broken?",
            # NOTE: expected_answer_span deliberately missing
            "expected_source_docs": ["docs/intro.md"],
        }
    )
    goldens_path.write_text(bad_line + "\n")

    with pytest.raises(PackValidationError) as exc_info:
        load_pack(pack_dir)

    msg = str(exc_info.value)
    assert "f-bad-001" in msg, f"error message must reference the offending qid; got: {msg}"
    assert "expected_answer_span" in msg or "qtype" in msg, (
        f"error message must reference the failing field; got: {msg}"
    )
    # Must be re-raised as PackValidationError, not a bare pydantic ValidationError
    import pydantic

    assert not isinstance(exc_info.value, pydantic.ValidationError)


def test_rejects_unknown_profile(tmp_path: Path) -> None:
    pack_dir = _copy_pack(tmp_path)
    pack_yaml = pack_dir / "pack.yaml"
    text = pack_yaml.read_text()
    text = text.replace("profile: asic_riscv_soc", "profile: undeclared_profile")
    pack_yaml.write_text(text)

    # thresholds.yaml still says asic_riscv_soc — make it match the new profile
    # so that the profile-unknown check is what fires (not the profile-mismatch check).
    th_path = pack_dir / "thresholds.yaml"
    th_path.write_text(
        th_path.read_text().replace("profile: asic_riscv_soc", "profile: undeclared_profile")
    )

    with pytest.raises(PackValidationError) as exc_info:
        load_pack(pack_dir)

    msg = str(exc_info.value)
    assert "profile" in msg
    assert "undeclared_profile" in msg
