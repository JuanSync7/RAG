# @summary
# LangGraph node: contextual chunking (Anthropic-style contextual retrieval).
# When enabled, generates a short per-chunk "situating context" via a batched LLM
# call (a document window + the chunk bodies) and prepends it to each chunk's
# EMBED text only (metadata["embed_text"]) — NOT the stored text — so a chunk is
# retrievable by its document context even when its own words don't match the
# query. embedding_storage_node embeds embed_text but stores enriched_content.
# Fail-open: any error / count mismatch leaves embed_text unset for that batch, so
# embedding falls back to the stored text (never worse than today's behavior).
# Exports: contextual_enrichment_node
# Deps: embedding.state, common (append_processing_log, parse_json_object),
#       platform.llm (get_llm_provider)
# @end-summary
"""Contextual-chunking enrichment node."""

from __future__ import annotations

import logging
import time
from typing import Any, Iterable

logger = logging.getLogger("rag.ingest.embedding.contextual_enrichment")

from src.ingest.common import append_processing_log, parse_json_object
from src.ingest.common.observability import node_span
from src.ingest.embedding.state import EmbeddingPipelineState
from src.platform.llm import get_llm_provider

# The situating context is short (one sentence per chunk); a batch of N chunks
# needs only a small JSON array back. Bounded so a large batch cannot request the
# full context window as output tokens.
_CTX_MAX_TOKENS = 512


def _batches(items: list, size: int) -> Iterable[list]:
    step = max(1, int(size))
    for i in range(0, len(items), step):
        yield items[i : i + step]


def _build_prompt(doc_window: str, bodies: list[str]) -> str:
    """Generic contextualization prompt (CLAUDE.md §0 — no vendor/corpus terms):
    situate ANY chunk within ANY document by section/topic so a query using the
    document's terminology can retrieve it."""
    numbered = "\n\n".join(f"[CHUNK {i}]\n{b}" for i, b in enumerate(bodies))
    return (
        "You are contextualizing chunks of ONE document to improve their "
        "retrievability. Given the DOCUMENT (possibly truncated) and a numbered "
        "list of CHUNKS taken from it, write for EACH chunk a single short "
        "sentence (<= 25 words) that situates the chunk within the document — "
        "which section/topic it belongs to and what it is about — using the "
        "document's own terminology (names, section titles, entities) so a search "
        "phrased in the document's terms can find this chunk. Do not summarize the "
        "whole document; describe only where THIS chunk fits.\n"
        'Return JSON ONLY: {"contexts": ["...", "..."]} with EXACTLY one string '
        "per chunk, in the same order as the CHUNKS.\n\n"
        f"DOCUMENT:\n{doc_window}\n\n"
        f"CHUNKS:\n{numbered}\n"
    )


@node_span("contextual_enrichment")
def contextual_enrichment_node(state: EmbeddingPipelineState) -> dict[str, Any]:
    """Attach a per-chunk situating context to each chunk's EMBED text.

    Runs one LLM call per batch of chunks (the document window is the shared
    context). Sets ``chunk.metadata["embed_text"] = context + "\\n\\n" +
    enriched_content`` — leaving ``chunk.text`` / ``enriched_content`` (the stored
    + generation text) untouched. Gated on ``config.enable_contextual_chunking``;
    fail-open on any error/shape mismatch (embed_text simply stays unset, so
    embedding falls back to the stored text).
    """
    t0 = time.monotonic()
    config = state["runtime"].config
    if not getattr(config, "enable_contextual_chunking", False):
        return {"processing_log": append_processing_log(state, "contextual_enrichment:skipped")}

    chunks = state.get("chunks", [])
    doc = state.get("cleaned_text", "") or ""
    if not chunks or not doc.strip():
        return {"processing_log": append_processing_log(state, "contextual_enrichment:skipped-empty")}

    doc_window = doc[: config.contextual_doc_max_chars]
    provider = get_llm_provider()
    enriched = 0

    for batch in _batches(chunks, config.contextual_batch_size):
        bodies = [c.metadata.get("enriched_content", c.text) for c in batch]
        try:
            resp = provider.json_completion(
                [
                    {"role": "system", "content": "Return JSON only."},
                    {"role": "user", "content": _build_prompt(doc_window, bodies)},
                ],
                temperature=config.llm_temperature,
                max_tokens=_CTX_MAX_TOKENS,
                timeout=config.llm_timeout_seconds,
            )
            obj = parse_json_object(resp.content)
            contexts = obj.get("contexts") if isinstance(obj, dict) else None
        except Exception:  # noqa: BLE001 — fail open: leave this batch un-contextualized
            logger.warning(
                "contextual_enrichment LLM call failed; batch left un-contextualized",
                exc_info=True,
            )
            contexts = None

        # Fail-open on any shape mismatch: only apply when we got exactly one
        # context string per chunk in the batch.
        if not isinstance(contexts, list) or len(contexts) != len(batch):
            continue
        for chunk, ctx in zip(batch, contexts):
            ctx = str(ctx or "").strip()
            if not ctx:
                continue
            base = chunk.metadata.get("enriched_content", chunk.text)
            chunk.metadata["embed_text"] = ctx + "\n\n" + base
            enriched += 1

    logger.info(
        "contextual_enrichment: %d/%d chunks contextualized source=%s in %.3fs",
        enriched, len(chunks), state.get("source_name", ""), time.monotonic() - t0,
    )
    return {
        "chunks": chunks,
        "processing_log": append_processing_log(state, f"contextual_enrichment:ok:{enriched}"),
    }
