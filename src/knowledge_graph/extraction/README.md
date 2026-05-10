<!-- @summary
Entity and relationship extraction sub-package with multiple extractor implementations.
@end-summary -->

# extraction/ — Entity and Relationship Extractors

All extractors implement the `EntityExtractor` protocol defined in `base.py`.

## Files

| File | Phase | Purpose |
|------|-------|---------|
| `base.py` | 1 | `EntityExtractor` protocol — `extract()`, `extract_entities()`, `extract_relations()` |
| `regex_extractor.py` | 1 | Rule-based entity extraction with YAML schema awareness |
| `gliner_extractor.py` | 1 | GLiNER zero-shot NER model extraction |
| `llm_extractor.py` | 1b | LLM structured-output extraction with JSON schema |
| `parser_extractor.py` | 1b | tree-sitter-verilog SystemVerilog parser |
| `python_parser.py` | 2 | Python AST-based extractor (classes, functions, imports, constants) |
| `bash_parser.py` | 2 | Regex-based Bash script extractor (functions, exported vars, sourced files) |

## Adding a New Extractor

1. Create `my_extractor.py` implementing the `EntityExtractor` protocol
2. Add a config toggle (`enable_my_extractor`) to `KGConfig` in `common/types.py`
3. Wire the env var in `config/settings.py`
4. Export from `extraction/__init__.py`
5. Register in `extractor_priority` list in `KGConfig`
