# @summary
# Eval_pack loader — validates then materialises the typed EvalPack tree.
# Exports: load_pack.
# Deps: pyyaml, json, pathlib, .validate, .schema.
# @end-summary
"""Load an eval_pack into a typed :class:`EvalPack` aggregate.

``load_pack`` calls :func:`validate_pack` first so any structural failure
propagates as :class:`PackValidationError`. Only after validation passes
does it populate the dataclass tree.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from .schema import EvalPack, Golden, ManifestEntry, PackMeta, Thresholds
from .validate import validate_pack


def load_pack(path: Path | str) -> EvalPack:
    """Validate and load the eval_pack at ``path``.

    :raises PackValidationError: if validation fails for any reason.
    """
    root = Path(path)
    validate_pack(root)

    pack_data = yaml.safe_load((root / "pack.yaml").read_text())
    meta = PackMeta(**pack_data)

    manifest_raw = json.loads((root / "corpus" / "manifest.json").read_text())
    manifest = [ManifestEntry(**e) for e in manifest_raw["entries"]]

    goldens: dict[str, list[Golden]] = {}
    for gf in sorted((root / "goldens").glob("*.jsonl")):
        qtype = gf.stem
        rows: list[Golden] = []
        for line in gf.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(Golden(**{**row, "raw": row}))
        goldens[qtype] = rows

    thresholds = Thresholds(**yaml.safe_load((root / "thresholds.yaml").read_text()))

    return EvalPack(
        meta=meta,
        manifest=manifest,
        goldens=goldens,
        thresholds=thresholds,
    )
