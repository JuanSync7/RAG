# @summary
# Obsidian markdown vault export for the KG subsystem.
# Exports: export_obsidian
# Deps: re, pathlib, src.knowledge_graph.backend
# @end-summary
"""Obsidian markdown vault exporter.

Writes one ``.md`` file per entity into ``output_dir``, using
``[[wikilinks]]`` to cross-reference neighbours.  Each file includes the
entity type, aliases, mention count, sources, outgoing relationships, and
incoming back-references.
"""

from __future__ import annotations

import re
from pathlib import Path

from src.knowledge_graph.backend import GraphStorageBackend


__all__ = ["export_obsidian"]


def export_obsidian(backend: GraphStorageBackend, output_dir: Path) -> int:
    """Write one .md file per entity with [[wikilinks]] to neighbors.

    Args:
        backend: Graph storage backend to export from.
        output_dir: Directory to write markdown files to.

    Returns:
        Number of files written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for entity in backend.get_all_entities():
        name = entity.name
        safe_name = re.sub(r"[^\w\s\-]", "", name).strip()
        # Ensure safe_name is a bare filename with no directory components.
        safe_name = Path(safe_name).name if safe_name else "unnamed_node"
        if not safe_name:
            safe_name = "unnamed_node"

        lines = [f"# {name}"]
        lines.append(f"\n**Type**: {entity.type or 'unknown'}")
        if entity.aliases:
            lines.append(f"**Aliases**: {', '.join(entity.aliases)}")
        lines.append(f"**Mentions**: {entity.mention_count}")
        lines.append(f"**Sources**: {', '.join(entity.sources)}")

        out_edges = backend.get_outgoing_edges(name)
        if out_edges:
            lines.append("\n## Relationships")
            for triple in out_edges:
                lines.append(f"- {triple.predicate}: [[{triple.object}]]")

        in_edges = backend.get_incoming_edges(name)
        if in_edges:
            lines.append("\n## Referenced by")
            for triple in in_edges:
                lines.append(f"- [[{triple.subject}]] ({triple.predicate})")

        (output_dir / f"{safe_name}.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )
        count += 1

    return count
