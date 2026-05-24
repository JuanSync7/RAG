# @summary
# Idempotency tests for the figure-artifact → chunk transformer. Re-running
# with identical FigureArtifact inputs MUST yield identical chunk_ids so
# Weaviate upserts in place rather than orphaning prior vectors. Mirrors the
# table_group_id discipline (memory ``project_chunk_id_idempotency``).
# Exports: (pytest test functions)
# Deps: pytest, src.ingest.support.docling, src.ingest.support.parser_base,
#       src.vector_db.weaviate.store
# @end-summary
"""Figure chunk-id idempotency."""
from __future__ import annotations

from src.ingest.support.parser_base import FigureArtifact
from src.ingest.support.docling import _figure_artifacts_to_chunks
from src.vector_db.weaviate.store import build_figure_chunk_id


def _fig(**overrides) -> FigureArtifact:
    base = dict(
        document_id="esp32_s3_ds",
        caption="Figure 4-1: SoC block diagram",
        caption_label="Figure 4-1",
        section_path="Chapter 4 > Architecture",
        page_no=37,
        self_ref="#/pictures/0",
        image_uri="file:///tmp/pic-0.png",
    )
    base.update(overrides)
    return FigureArtifact(**base)


def test_build_figure_chunk_id_is_deterministic():
    a = build_figure_chunk_id("esp32_s3_ds", "#/pictures/0")
    b = build_figure_chunk_id("esp32_s3_ds", "#/pictures/0")
    assert a == b


def test_build_figure_chunk_id_distinct_per_self_ref():
    a = build_figure_chunk_id("esp32_s3_ds", "#/pictures/0")
    b = build_figure_chunk_id("esp32_s3_ds", "#/pictures/1")
    assert a != b


def test_build_figure_chunk_id_distinct_per_document():
    a = build_figure_chunk_id("esp32_s3_ds", "#/pictures/0")
    b = build_figure_chunk_id("msp430_ds", "#/pictures/0")
    assert a != b


def test_figure_artifacts_to_chunks_idempotent():
    figs = [_fig(), _fig(self_ref="#/pictures/1", caption_label="Figure 4-2",
                         caption="Figure 4-2: ADC")]
    first = _figure_artifacts_to_chunks(figs)
    second = _figure_artifacts_to_chunks(figs)
    assert [c.extra_metadata.get("chunk_id") for c in first] == [
        c.extra_metadata.get("chunk_id") for c in second
    ]
    # All chunk_ids are non-empty.
    assert all(c.extra_metadata.get("chunk_id") for c in first)
