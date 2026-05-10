# @summary
# LangGraph node for optional relation extraction to intermediate KG triples.
# Exports: knowledge_graph_extraction_node
# Deps: embedding.state
# @end-summary

"""Knowledge-graph extraction node implementation."""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger("rag.ingest.embedding.knowledge_graph_extraction")

from src.core import EntityExtractor
from src.ingest.common import append_processing_log
from src.ingest.embedding.state import EmbeddingPipelineState

_EXTRACTOR = EntityExtractor()


def knowledge_graph_extraction_node(state: EmbeddingPipelineState) -> dict[str, Any]:
    """Extract entity relations and stage triples for downstream KG storage.

    Args:
        state: Ingestion pipeline state.

    Returns:
        Partial state update containing ``kg_triples`` (when enabled) and an
        updated ``processing_log``. When disabled, returns only a skipped log
        entry.
    """
    t0 = time.monotonic()
    if not state["runtime"].config.enable_knowledge_graph_extraction:
        return {
            "processing_log": append_processing_log(
                state, "knowledge_graph_extraction:skipped"
            )
        }

    try:
        triples = []
        for chunk in state["chunks"]:
            entities = _EXTRACTOR.extract_entities(chunk.text)
            relations = _EXTRACTOR.extract_relations(chunk.text, entities)
            triples.extend(
                {
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj,
                    "source": state["source_name"],
                }
                for subject, predicate, obj in relations
            )
    except Exception as exc:
        logger.error("kg_extraction failed source=%s: %s", state.get("source_name", ""), exc, exc_info=True)
        return {
            "errors": state.get("errors", []) + [f"kg_extraction:{exc}"],
            "processing_log": append_processing_log(state, "knowledge_graph_extraction:error"),
        }

    logger.info("knowledge_graph_extraction complete: source=%s triples=%d", state["source_name"], len(triples))
    logger.debug("knowledge_graph_extraction_node completed in %.3fs", time.monotonic() - t0)
    return {
        "kg_triples": triples,
        "processing_log": append_processing_log(
            state, "knowledge_graph_extraction:ok"
        ),
    }
