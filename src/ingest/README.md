<!-- @summary
This directory contains Python modules for processing documents and Markdown content. It includes functions for metadata extraction, text chunking, normalization, and cleaning.
@end-summary -->

# ingest

## Overview
This directory contains Python modules for processing documents and Markdown content. It includes functions for metadata extraction, text chunking, normalization, and cleaning.

## Files
| File | Purpose | Key Exports |
|------|---------|--------------|
| document_processor.py | Defines a document processing pipeline with key functions for metadata extraction, text chunking, normalization, and cleaning | process_document, metadata_to_dict, chunk_text; RecursiveCharacterTextSplitter, clean_text, re, unicodedata, dataclasses, typing, langchain_text_splitters, config.settings |
| markdown_processor.py | Preserves Markdown formatting while processing documents with key functions for metadata extraction, text chunking, normalization, and cleaning | process_document_markdown, chunk_markdown, clean_document; re, numpy, langchain_text_splitters, extract_metadata, metadata_to_dict, strip_boilerplate, normalize_unicode, clean_whitespace, strip_trailing_short_lines, _split_sentences, _semantic_split, _build_section_metadata |

## Internal Dependencies
- `document_processor.py` and `markdown_processor.py` depend on common utilities like `re`, `numpy`, `langchain_text_splitters`, `extract_metadata`, `metadata_to_dict`, `strip_boilerplate`, `normalize_unicode`, `clean_whitespace`, `strip_trailing_short_lines`, `_split_sentences`, `_semantic_split`, and `_build_section_metadata`.

## Subdirectories
- None