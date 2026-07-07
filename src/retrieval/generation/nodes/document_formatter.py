# @summary
# Document formatting stage for structured context presentation and version conflict detection.
# Exports: format_context, FormattedContext, VersionConflict
# Deps: src.retrieval.generation.schemas, pathlib
# @end-summary
"""Document formatting for the generation stage.

Transforms raw RankedResult objects into a structured context string with
metadata headers and version conflict detection. This stage sits between
reranking (stage 5) and generation (stage 6) in the RAG pipeline.

All functions are pure/deterministic — no I/O, no LLM calls.

Requirements references: REQ-501 (metadata attachment), REQ-502 (version
conflict detection), REQ-503 (deterministic formatting).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import PurePosixPath
from typing import Any

from src.retrieval.generation.schemas import FormattedContext, VersionConflict


def format_context(
    results: list[Any],
    include_scores: bool = True,
) -> FormattedContext:
    """Format ranked results into a structured context string.

    Each chunk is presented with a metadata header extracted from the
    RankedResult.metadata dict. Version conflicts are detected and
    prepended as a warning block.

    Args:
        results: List of RankedResult objects (must have .text, .score,
            .metadata attributes).
        include_scores: Whether to include relevance scores in headers.

    Returns:
        FormattedContext with the formatted string and conflict metadata.
    """
    if not results:
        return FormattedContext(context_string="", chunk_count=0)

    conflicts = _detect_version_conflicts(results)

    formatted_chunks = []
    for i, result in enumerate(results):
        # Stamp the 1-based citation index onto the result's metadata so the API
        # response exposes the SAME number the model cites ([N]). This function is
        # the single numbering authority; the response returns these same objects,
        # so source N in the answer maps unambiguously to the Nth returned source.
        meta = getattr(result, "metadata", None)
        if isinstance(meta, dict):
            meta["citation_index"] = i + 1
        chunk_str = _format_chunk(i + 1, result, include_scores)
        formatted_chunks.append(chunk_str)

    parts = []

    if conflicts:
        warning_lines = ["--- VERSION CONFLICT WARNING ---"]
        for c in conflicts:
            versions_str = ", ".join(c.versions)
            warning_lines.append(
                f"Documents for \"{c.spec_stem}\" were retrieved in "
                f"versions [{versions_str}]. Information may be "
                f"inconsistent. Cite specific versions in your answer."
            )
        warning_lines.append("---")
        parts.append("\n".join(warning_lines))

    parts.append("\n\n".join(formatted_chunks))

    return FormattedContext(
        context_string="\n\n".join(parts),
        chunk_count=len(results),
        version_conflicts=conflicts,
    )


def _format_chunk(
    index: int,
    result: Any,
    include_scores: bool = True,
) -> str:
    meta = getattr(result, "metadata", {}) or {}
    score = getattr(result, "score", 0.0)
    text = getattr(result, "text", "")

    header_parts = []
    if include_scores:
        header_parts.append(f"[{index}] (relevance: {score:.0%})")
    else:
        header_parts.append(f"[{index}]")

    meta_header = _extract_metadata_header(meta)
    if meta_header:
        header_parts.append(meta_header)

    header = " | ".join(header_parts)
    return f"{header}\n{text}"


def _extract_metadata_header(metadata: dict[str, Any]) -> str:
    parts = []

    source = metadata.get("source") or metadata.get("filename")
    if source:
        parts.append(f"Source: {source}")

    version = metadata.get("source_version") or metadata.get("version")
    if version:
        parts.append(f"Version: {version}")

    date = metadata.get("date") or metadata.get("source_date")
    if date:
        parts.append(f"Date: {date}")

    section = (
        metadata.get("heading")
        or metadata.get("section_path")
        or metadata.get("section")
    )
    if section:
        parts.append(f"Section: {section}")

    domain = metadata.get("domain") or metadata.get("tags")
    if domain:
        if isinstance(domain, list):
            domain = ", ".join(str(d) for d in domain)
        parts.append(f"Domain: {domain}")

    return " | ".join(parts)


def _detect_version_conflicts(results: list[Any]) -> list[VersionConflict]:
    stem_versions: dict[str, set] = defaultdict(set)

    for result in results:
        meta = getattr(result, "metadata", {}) or {}

        spec_id = meta.get("source_id") or meta.get("spec_id")
        if spec_id:
            stem = str(spec_id)
        else:
            source = meta.get("source") or meta.get("filename") or ""
            if source:
                stem = PurePosixPath(source).stem
            else:
                continue

        version = meta.get("source_version") or meta.get("version")
        if version:
            stem_versions[stem].add(str(version))

    conflicts = []
    for stem, versions in stem_versions.items():
        if len(versions) > 1:
            conflicts.append(
                VersionConflict(
                    spec_stem=stem,
                    versions=sorted(versions),
                )
            )

    return conflicts
