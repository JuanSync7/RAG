# @summary
# Node support libraries: document parsing (Docling), vision/VLM enrichment,
# LLM helpers, and text processing primitives. Includes pluggable parser
# abstraction (DoclingParser, PlainTextParser, CodeParser, ParserRegistry).
# Deps: src.ingest.support.docling, src.ingest.support.vision, src.ingest.support.llm,
#       src.ingest.support.document, src.ingest.support.markdown,
#       src.ingest.support.parser_registry, src.ingest.support.parser_text,
#       src.ingest.support.parser_code
# @end-summary
"""Support libraries for specialized ingestion processing.

This package contains optional helpers used by ingestion pipeline nodes, such as:

- Document parsing via Docling (when enabled)
- Vision/VLM extraction for figures and images (when enabled)
- LLM helpers for metadata enrichment (when enabled)
- Text normalization and markdown-aware chunking utilities
"""

# --- Parser base contracts ---
from src.ingest.support.parser_base import (
    Chunk,
    DocumentParser,
    ParseResult,
    chunk_with_markdown,
    validate_extra_metadata,
)

# --- Parser registry (stable import surface for pipeline nodes) ---
from src.ingest.support.parser_registry import ParserRegistry

# --- Auto-generated re-exports (fix_encapsulation.py) ---
from src.ingest.support.colqwen import (
    ColQwen2LoadError,
    ColQwen2PageEmbedding,
    VisualEmbeddingError,
    embed_page_images,
    embed_text_query,
    ensure_colqwen_ready,
    load_colqwen_model,
    unload_colqwen_model,
)
from src.ingest.support.docling import (
    ensure_docling_ready,
    parse_with_docling,
)
from src.ingest.support.document import (
    extract_metadata,
    metadata_to_dict,
)
from src.ingest.support.llm import _llm_json
from src.ingest.support.markdown import (
    _build_section_metadata,
    chunk_markdown,
    clean_document,
    normalize_headings_to_markdown,
)
from src.ingest.support.vision import (
    _IMAGE_REF_PATTERN,
    _describe_image,
    _extract_image_candidates,
    caption_markdown_images_inline,
    ensure_vision_ready,
    generate_vision_notes,
)
