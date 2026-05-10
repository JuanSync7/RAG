<!-- @summary
Graph export formats — currently Obsidian markdown vault.
@end-summary -->

# export/ — Graph Export

Converts the knowledge graph into external formats for visualization and browsing.

## Files

| File | Purpose |
|------|---------|
| `obsidian.py` | `export_obsidian(backend, output_dir)` — one `.md` file per entity with `[[wikilinks]]` |

## Obsidian Export

Accepts any `GraphStorageBackend` and writes a markdown vault:
- H1 heading with entity name
- Metadata: type, aliases, mention count, sources
- Outgoing relationships as `[[wikilinks]]`
- Incoming back-references with `[[wikilinks]]`
- Filenames sanitized for filesystem safety

Usage: `export_obsidian(backend, Path("obsidian_graph/"))` returns the number of files written.
