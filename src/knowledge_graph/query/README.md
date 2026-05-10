<!-- @summary
Query processing sub-package: entity matching, graph-based expansion, and sanitization.
@end-summary -->

# query/ — Query Processing

Handles the retrieval-side of the KG: matching user queries to graph entities,
expanding queries with graph context, and sanitizing input.

## Files

| File | Purpose |
|------|---------|
| `entity_matcher.py` | `EntityMatcher` — spaCy PhraseMatcher + substring fallback for entity lookup |
| `expander.py` | `GraphQueryExpander` — graph-walk expansion with optional community-aware global retrieval (Phase 2) |
| `sanitizer.py` | `QuerySanitizer` — query normalization, alias expansion |

## Query Expansion Flow

1. **Entity matching**: Find graph entities mentioned in the query
2. **Local expansion**: Walk graph neighbors up to `max_expansion_depth` hops
3. **Global expansion** (Phase 2, optional): If `enable_global_retrieval=True` and community detector is ready, add community member terms
4. **Term limiting**: Truncate to `max_expansion_terms`

The expander also provides `get_context_summary()` which returns a text block with entity relationships and community summaries for RAG prompt injection.
