# @summary
# Structural validator for eval_pack directories.
# Exports: validate_pack, compute_corpus_pin, KNOWN_PROFILES.
# Deps: pyyaml, pydantic, hashlib, json, pathlib.
# @end-summary
"""Validate the structural contract of an eval_pack directory.

Every failure path raises :class:`PackValidationError` with a message that
name-checks the offending file/field. Pydantic ``ValidationError`` is
caught and re-raised so the public API surfaces a single exception type.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .errors import PackValidationError
from .schema import KNOWN_PROFILES, Golden, ManifestEntry, PackMeta, Thresholds

REQUIRED_PACK_YAML_KEYS: tuple[str, ...] = (
    "name",
    "version",
    "profile",
    "corpus_pin",
    "judge",
    "collection_name_template",
)


def compute_corpus_pin(entries: list[dict[str, str]] | list[ManifestEntry]) -> str:
    """SHA-256 of sorted ``{path}:{sha256}`` lines joined by ``\\n``."""

    def _line(e: Any) -> str:
        if isinstance(e, ManifestEntry):
            return f"{e.path}:{e.sha256}"
        return f"{e['path']}:{e['sha256']}"

    lines = sorted(_line(e) for e in entries)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _read_yaml(path: Path, rel: str) -> dict[str, Any]:
    if not path.exists():
        raise PackValidationError(f"{rel}: file is missing")
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise PackValidationError(f"{rel}: failed to parse YAML — {exc}") from exc
    if not isinstance(data, dict):
        raise PackValidationError(f"{rel}: top-level YAML must be a mapping, got {type(data).__name__}")
    return data


def _read_json(path: Path, rel: str) -> Any:
    if not path.exists():
        raise PackValidationError(f"{rel}: file is missing")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise PackValidationError(f"{rel}: failed to parse JSON — {exc}") from exc


def validate_pack(path: Path | str) -> None:
    """Run structural checks against the pack at ``path``.

    Raises :class:`PackValidationError` on any failure.
    """
    root = Path(path)
    if not root.exists() or not root.is_dir():
        raise PackValidationError(f"pack directory does not exist or is not a directory: {root}")

    # --- pack.yaml ---
    pack_yaml_path = root / "pack.yaml"
    pack_data = _read_yaml(pack_yaml_path, "pack.yaml")
    missing = [k for k in REQUIRED_PACK_YAML_KEYS if k not in pack_data]
    if missing:
        raise PackValidationError(f"pack.yaml: missing required keys: {missing}")
    profile = pack_data["profile"]
    if profile not in KNOWN_PROFILES:
        raise PackValidationError(
            f"pack.yaml: profile {profile!r} (undeclared_profile is not allowed unless registered) "
            f"is not in the known profile set {sorted(KNOWN_PROFILES)}; "
            f"got profile={profile!r}"
        )
    try:
        meta = PackMeta(**pack_data)
    except ValidationError as exc:
        raise PackValidationError(f"pack.yaml: schema validation failed — {exc}") from exc

    # --- corpus/manifest.json ---
    manifest_path = root / "corpus" / "manifest.json"
    manifest_data = _read_json(manifest_path, "corpus/manifest.json")
    if not isinstance(manifest_data, dict) or "entries" not in manifest_data:
        raise PackValidationError(
            "corpus/manifest.json: must be a JSON object with an 'entries' list"
        )
    raw_entries = manifest_data["entries"]
    if not isinstance(raw_entries, list) or not raw_entries:
        raise PackValidationError("corpus/manifest.json: 'entries' must be a non-empty list")
    try:
        entries = [ManifestEntry(**e) for e in raw_entries]
    except ValidationError as exc:
        raise PackValidationError(
            f"corpus/manifest.json: entry schema validation failed — {exc}"
        ) from exc
    for e in entries:
        doc_path = root / "corpus" / e.path
        if not doc_path.exists():
            raise PackValidationError(
                f"corpus/manifest.json: declared document {e.path!r} does not exist on disk"
            )

    expected_pin = compute_corpus_pin(entries)
    if meta.corpus_pin != expected_pin:
        raise PackValidationError(
            f"pack.yaml: corpus_pin mismatch — declared {meta.corpus_pin!r}, "
            f"recomputed {expected_pin!r} from corpus/manifest.json"
        )

    # --- goldens/*.jsonl ---
    goldens_dir = root / "goldens"
    if not goldens_dir.exists() or not goldens_dir.is_dir():
        raise PackValidationError("goldens/: directory is missing")
    golden_files = sorted(goldens_dir.glob("*.jsonl"))
    if not golden_files:
        raise PackValidationError("goldens/: no *.jsonl files found")
    for gf in golden_files:
        qtype = gf.stem
        rel = f"goldens/{gf.name}"
        for line_no, line in enumerate(gf.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PackValidationError(
                    f"{rel}:line {line_no}: invalid JSON — {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise PackValidationError(
                    f"{rel}:line {line_no}: each line must be a JSON object"
                )
            qid = row.get("qid", "<missing-qid>")
            row_qtype = row.get("qtype")
            if row_qtype != qtype:
                raise PackValidationError(
                    f"{rel}:line {line_no}: golden qid={qid!r} has qtype={row_qtype!r} "
                    f"but the file's qtype (filename stem) is {qtype!r}; "
                    f"expected_answer_span and other per-qtype fields must match"
                )
            try:
                Golden(**{**row, "raw": row})
            except ValidationError as exc:
                # Surface the offending qid and field names in the message.
                fields = ", ".join(
                    ".".join(str(p) for p in err["loc"]) for err in exc.errors()
                )
                raise PackValidationError(
                    f"{rel}:line {line_no}: golden qid={qid!r} qtype={row_qtype!r} "
                    f"failed schema validation on fields [{fields}] — {exc}"
                ) from exc

    # --- thresholds.yaml ---
    thresholds_path = root / "thresholds.yaml"
    th_data = _read_yaml(thresholds_path, "thresholds.yaml")
    try:
        thresholds = Thresholds(**th_data)
    except ValidationError as exc:
        raise PackValidationError(
            f"thresholds.yaml: schema validation failed — {exc}"
        ) from exc
    if thresholds.profile != meta.profile:
        raise PackValidationError(
            f"thresholds.yaml: profile {thresholds.profile!r} does not match pack.yaml profile {meta.profile!r}"
        )
