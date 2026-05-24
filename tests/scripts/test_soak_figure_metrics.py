"""Unit tests for FIG-3 figure-artifact soak metric helpers.

These tests do NOT invoke ``run_soak`` / ``_run_figure_soak`` directly —
the latter requires DoclingParser. They target the pure-function helpers
added for the FIG-3 figure-artifact soak pass:

- ``_figure_chunks(chunks)``
- ``_figure_caption_label_rate(figures)``
- ``_figure_image_uri_sanitised(figure_chunks)``
- ``_figure_chunk_idempotency(figures)``
- ``_figure_resolvability(chunks, figure_chunks, document_id)``
- ``_figure_section_distribution(figures)``
- ``_figure_soak_verdict(soak)``
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from scripts.soak_table_chunking import (
    _figure_caption_label_rate,
    _figure_chunk_idempotency,
    _figure_chunks,
    _figure_image_uri_sanitised,
    _figure_resolvability,
    _figure_section_distribution,
    _figure_soak_verdict,
)


# --------------------------------------------------------------------------- #
# Lightweight fakes                                                           #
# --------------------------------------------------------------------------- #


@dataclass
class FakeChunk:
    text: str = ""
    extra_metadata: dict[str, Any] = field(default_factory=dict)
    section_path: str = ""
    heading: str = ""


@dataclass
class FakeFigure:
    document_id: str = ""
    caption: str = ""
    caption_label: str = ""
    section_path: str = ""
    page_no: int = 0
    self_ref: str = ""
    image_uri: str = ""


def _mk_prose_chunk(targets: list[dict] | None = None, **meta: Any) -> FakeChunk:
    em: dict[str, Any] = dict(meta)
    if targets is not None:
        em["xref_targets"] = json.dumps(targets)
    sp = str(em.pop("section_path", "") or "")
    return FakeChunk(extra_metadata=em, section_path=sp)


def _mk_figure_chunk(label: str, doc_id: str, *, image_uri: str = "") -> FakeChunk:
    return FakeChunk(
        text=label,
        extra_metadata={
            "chunk_type": "figure",
            "caption_label": label,
            "document_id": doc_id,
            "figure_image_uri": image_uri,
            "chunk_id": f"chunk-{label}-{doc_id}",
            "page_no": 1,
        },
        section_path="Section X",
    )


# --------------------------------------------------------------------------- #
# _figure_chunks                                                              #
# --------------------------------------------------------------------------- #


class TestFigureChunks:
    def test_empty(self):
        assert _figure_chunks([]) == []

    def test_filters_by_chunk_type(self):
        prose = _mk_prose_chunk()
        fig = _mk_figure_chunk("Figure 1", "docA")
        out = _figure_chunks([prose, fig, _mk_prose_chunk()])
        assert len(out) == 1
        assert out[0] is fig


# --------------------------------------------------------------------------- #
# _figure_caption_label_rate                                                  #
# --------------------------------------------------------------------------- #


class TestFigureCaptionLabelRate:
    def test_empty(self):
        assert _figure_caption_label_rate([]) == {
            "figures_total": 0,
            "figures_with_label": 0,
            "label_rate": 0.0,
        }

    def test_all_labelled(self):
        figs = [FakeFigure(caption_label="Figure 1"), FakeFigure(caption_label="Figure 2-3")]
        out = _figure_caption_label_rate(figs)
        assert out == {"figures_total": 2, "figures_with_label": 2, "label_rate": 1.0}

    def test_mixed(self):
        figs = [
            FakeFigure(caption_label="Figure 1"),
            FakeFigure(caption_label=""),
            FakeFigure(caption_label="Figure 3"),
            FakeFigure(caption_label=""),
        ]
        out = _figure_caption_label_rate(figs)
        assert out["figures_total"] == 4
        assert out["figures_with_label"] == 2
        assert out["label_rate"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# _figure_image_uri_sanitised                                                 #
# --------------------------------------------------------------------------- #


class TestFigureImageUriSanitised:
    def test_empty(self):
        assert _figure_image_uri_sanitised([]) is True

    def test_clean_uris(self):
        chunks = [
            _mk_figure_chunk("Figure 1", "docA", image_uri=""),
            _mk_figure_chunk("Figure 2", "docA", image_uri="file:///x.png"),
            _mk_figure_chunk("Figure 3", "docA", image_uri="https://x/y.png"),
        ]
        assert _figure_image_uri_sanitised(chunks) is True

    def test_data_uri_fails_sanitisation(self):
        chunks = [
            _mk_figure_chunk("Figure 1", "docA", image_uri=""),
            _mk_figure_chunk("Figure 2", "docA", image_uri="data:image/png;base64,xxxx"),
        ]
        assert _figure_image_uri_sanitised(chunks) is False


# --------------------------------------------------------------------------- #
# _figure_chunk_idempotency                                                   #
# --------------------------------------------------------------------------- #


class TestFigureChunkIdempotency:
    def test_empty(self):
        out = _figure_chunk_idempotency([])
        assert out["ok"] is True
        assert out["first_count"] == 0
        assert out["second_count"] == 0
        assert out["duplicates_within_parse"] == 0

    def test_distinct_self_refs_no_dupes(self):
        figs = [
            FakeFigure(document_id="docA", self_ref="#/pictures/0"),
            FakeFigure(document_id="docA", self_ref="#/pictures/1"),
        ]
        out = _figure_chunk_idempotency(figs)
        assert out["ok"] is True
        assert out["first_count"] == 2
        assert out["duplicates_within_parse"] == 0

    def test_skips_figures_missing_required_fields(self):
        figs = [
            FakeFigure(document_id="docA", self_ref=""),
            FakeFigure(document_id="", self_ref="#/pictures/1"),
            FakeFigure(document_id="docA", self_ref="#/pictures/2"),
        ]
        out = _figure_chunk_idempotency(figs)
        assert out["first_count"] == 1

    def test_repeated_self_ref_shows_as_dupe(self):
        figs = [
            FakeFigure(document_id="docA", self_ref="#/pictures/0"),
            FakeFigure(document_id="docA", self_ref="#/pictures/0"),
        ]
        out = _figure_chunk_idempotency(figs)
        assert out["duplicates_within_parse"] == 1


# --------------------------------------------------------------------------- #
# _figure_section_distribution                                                #
# --------------------------------------------------------------------------- #


class TestFigureSectionDistribution:
    def test_empty(self):
        assert _figure_section_distribution([]) == []

    def test_top_n_ordering(self):
        figs = [
            FakeFigure(section_path="A"),
            FakeFigure(section_path="A"),
            FakeFigure(section_path="B"),
            FakeFigure(section_path="C"),
            FakeFigure(section_path="A"),
            FakeFigure(section_path="B"),
        ]
        top = _figure_section_distribution(figs, top_n=2)
        assert top[0] == ("A", 3)
        assert top[1] == ("B", 2)

    def test_empty_section_path_bucketed(self):
        figs = [FakeFigure(section_path=""), FakeFigure(section_path="")]
        top = _figure_section_distribution(figs)
        assert top == [("(no section)", 2)]


# --------------------------------------------------------------------------- #
# _figure_resolvability                                                       #
# --------------------------------------------------------------------------- #


class TestFigureResolvability:
    def test_zero_emissions_when_no_figure_refs(self):
        chunks = [
            _mk_prose_chunk(targets=[{"type": "section", "value": "Section 3"}], document_id="docA"),
        ]
        out = _figure_resolvability(chunks, figure_chunks=[], document_id="docA")
        assert out["figure_refs_emitted"] == 0
        assert out["unique_targets"] == 0
        assert out["resolved"] == 0
        assert out["resolvable_rate"] == 0.0

    def test_resolves_via_caption_label(self):
        chunks = [
            _mk_prose_chunk(
                targets=[{"type": "figure", "value": "Figure 4-1"}],
                document_id="docA",
            ),
        ]
        fig_chunks = [_mk_figure_chunk("Figure 4-1", "docA")]
        out = _figure_resolvability(chunks, fig_chunks, document_id="docA")
        assert out["figure_refs_emitted"] == 1
        assert out["unique_targets"] == 1
        assert out["resolved"] == 1
        assert out["resolvable_rate"] == pytest.approx(1.0)

    def test_unresolvable_when_no_matching_label(self):
        chunks = [
            _mk_prose_chunk(
                targets=[{"type": "figure", "value": "Figure 99"}],
                document_id="docA",
            ),
        ]
        fig_chunks = [_mk_figure_chunk("Figure 4-1", "docA")]
        out = _figure_resolvability(chunks, fig_chunks, document_id="docA")
        assert out["unique_targets"] == 1
        assert out["resolved"] == 0
        assert out["resolvable_rate"] == 0.0

    def test_unique_targets_dedupes(self):
        chunks = [
            _mk_prose_chunk(
                targets=[{"type": "figure", "value": "Figure 4-1"}],
                document_id="docA",
            ),
            _mk_prose_chunk(
                targets=[{"type": "figure", "value": "Figure 4-1"}],
                document_id="docA",
            ),
        ]
        fig_chunks = [_mk_figure_chunk("Figure 4-1", "docA")]
        out = _figure_resolvability(chunks, fig_chunks, document_id="docA")
        assert out["figure_refs_emitted"] == 2
        assert out["unique_targets"] == 1
        assert out["resolved"] == 1


# --------------------------------------------------------------------------- #
# _figure_soak_verdict                                                        #
# --------------------------------------------------------------------------- #


def _good_soak() -> dict:
    return {
        "counts": {"figure_count": 10},
        "caption_coverage": {"label_rate": 0.85},
        "figure_image_uri_sanitized": True,
        "idempotency": {"ok": True},
        "mode_a": {"figure_refs_emitted": 0},
        "mode_b": {"resolvable_rate": 0.55},
    }


class TestFigureSoakVerdict:
    def test_all_pass(self):
        v = _figure_soak_verdict(_good_soak())
        assert v["all_pass"] is True
        for c in v["criteria"].values():
            assert c["ok"] is True

    def test_zero_figure_count_fails(self):
        soak = _good_soak()
        soak["counts"]["figure_count"] = 0
        v = _figure_soak_verdict(soak)
        assert v["all_pass"] is False
        assert v["criteria"]["figure_count_gt_0"]["ok"] is False

    def test_low_caption_rate_fails(self):
        soak = _good_soak()
        soak["caption_coverage"]["label_rate"] = 0.10
        v = _figure_soak_verdict(soak)
        assert v["criteria"]["figure_caption_label_rate_ge_0_30"]["ok"] is False
        assert v["all_pass"] is False

    def test_data_uri_fails(self):
        soak = _good_soak()
        soak["figure_image_uri_sanitized"] = False
        v = _figure_soak_verdict(soak)
        assert v["criteria"]["figure_image_uri_sanitized"]["ok"] is False
        assert v["all_pass"] is False

    def test_mode_a_emits_figure_refs_fails(self):
        soak = _good_soak()
        soak["mode_a"]["figure_refs_emitted"] = 3
        v = _figure_soak_verdict(soak)
        assert v["criteria"]["mode_a_emits_zero_figure_refs"]["ok"] is False

    def test_low_mode_b_rate_fails(self):
        soak = _good_soak()
        soak["mode_b"]["resolvable_rate"] = 0.1
        v = _figure_soak_verdict(soak)
        assert v["criteria"]["mode_b_resolvable_rate_ge_0_30"]["ok"] is False

    def test_idempotency_fail(self):
        soak = _good_soak()
        soak["idempotency"]["ok"] = False
        v = _figure_soak_verdict(soak)
        assert v["criteria"]["figure_chunk_idempotent"]["ok"] is False
