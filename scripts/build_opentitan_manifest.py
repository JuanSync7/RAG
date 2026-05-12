"""Walk the local OpenTitan clone and emit a categorized ingestion manifest.

Produces ``evals/retrieval/deep_research/fixtures/opentitan/ingestion_manifest.json``
with one entry per file:

    {
      "path": "<absolute path in the clone>",
      "rel_path": "<path relative to the clone root>",
      "category": "<spec|microarch|programmers_guide|...>",
      "ip_block": "<aes|usbdev|...>" | null,
      "size": <bytes>,
    }

Filters applied:
  - Drops ``hw/vendor/`` (third-party vendored cores)
  - Drops files smaller than ``MIN_BYTES`` (placeholder READMEs)
  - Drops ``CHANGELOG`` / ``HISTORY`` / ``checklist.md`` (process noise)

Run with::

    python scripts/build_opentitan_manifest.py [--src ~/RagWeave/opentitan_data]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

DEFAULT_SRC = Path.home() / "RagWeave" / "opentitan_data"
DEFAULT_OUT = Path("evals/retrieval/deep_research/fixtures/opentitan/ingestion_manifest.json")

MIN_BYTES = 500
DROP_NAMES = {"CHANGELOG.md", "HISTORY.md", "checklist.md"}
DROP_PREFIXES = (
    "hw/vendor/",
    "sw/vendor/",
    "third_party/",
    "toolchain/",
    "rules/",
    "signing/",
)
SUFFIXES = (".md", ".rst")

IP_RE = re.compile(r"^hw/ip/([^/]+)/")
TOP_RE = re.compile(r"^hw/top_([^/]+)/")


def categorize(rel: str) -> tuple[str, str | None]:
    """Return ``(category, ip_block)`` for a path relative to the clone root."""
    m = IP_RE.match(rel)
    if m:
        ip = m.group(1)
        if rel.endswith("/theory_of_operation.md"):
            return "microarch", ip
        if rel.endswith("/programmers_guide.md"):
            return "programmers_guide", ip
        if rel.endswith("/registers.md") or rel.endswith("/interfaces.md"):
            return "spec", ip
        if "/dv/" in rel:
            return "dv_plan", ip
        return "ip_misc", ip

    m = TOP_RE.match(rel)
    if m:
        chip = m.group(1)
        if "/dv/" in rel:
            return "chip_dv", chip
        return "chip_arch", chip

    if rel.startswith("doc/security/"):
        return "security_arch", None
    if rel.startswith("doc/getting_started/"):
        return "getting_started", None
    if rel.startswith("doc/use_cases/"):
        return "use_case", None
    if rel.startswith("doc/contributing/") or rel.startswith("util/"):
        return "engineering_guide", None
    if rel.startswith("doc/project_governance/"):
        return "governance", None
    if rel.startswith("doc/"):
        return "doc_misc", None
    if rel.startswith("sw/device/"):
        return "embedded_sw", None
    if rel.startswith("sw/host/"):
        return "host_sw", None

    if rel.startswith("hw/dv/"):
        return "engineering_guide", None  # shared verification infra
    if "/" not in rel:
        return "doc_misc", None  # top-level CONTRIBUTING/README/SECURITY/SUMMARY
    if rel.startswith("hw/"):
        return "hw_misc", None

    return "other", None


def should_keep(rel: str, size: int) -> bool:
    if size < MIN_BYTES:
        return False
    if Path(rel).name in DROP_NAMES:
        return False
    if any(rel.startswith(p) for p in DROP_PREFIXES):
        return False
    if not rel.endswith(SUFFIXES):
        return False
    return True


def build(src: Path) -> list[dict]:
    entries: list[dict] = []
    for path in sorted(src.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(src).as_posix()
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if not should_keep(rel, size):
            continue
        category, ip = categorize(rel)
        entries.append({
            "path": str(path.resolve()),
            "rel_path": rel,
            "category": category,
            "ip_block": ip,
            "size": size,
        })
    return entries


def summarize(entries: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for e in entries:
        counts[e["category"]] = counts.get(e["category"], 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    if not args.src.exists():
        raise SystemExit(f"OpenTitan clone not found at {args.src}")

    entries = build(args.src)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_root": str(args.src.resolve()),
        "filters": {
            "min_bytes": MIN_BYTES,
            "drop_names": sorted(DROP_NAMES),
            "drop_prefixes": list(DROP_PREFIXES),
            "suffixes": list(SUFFIXES),
        },
        "summary": summarize(entries),
        "total": len(entries),
        "files": entries,
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.out} — {len(entries)} files")
    for cat, n in payload["summary"].items():
        print(f"  {cat:24s} {n}")


if __name__ == "__main__":
    main()
