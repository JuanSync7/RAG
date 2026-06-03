"""Unit tests for :mod:`src.platform.cli_log_formatting`.

Every formatting branch is exercised through the public surface
(:func:`build_logger_style`, :func:`build_level_badges`,
:func:`style_log_message`). The private stylers are reached only via
:func:`style_log_message` dispatch, and :func:`_format_stage_path` is
reached via the ``skipped``/``done`` pipeline branches.

Color codes are made assertable with a **sentinel palette**: each style
key maps to a readable ``<KEY>`` token rather than a real ANSI escape.
Assertions then check for structure and parsed content (e.g. a confidence
of 0.70 must emit ``<B_GREEN>``), independent of terminal escape syntax.
"""

from __future__ import annotations

import pytest

from src.platform import cli_log_formatting as clf

# Every palette key referenced by the module under test.
_PALETTE_KEYS = [
    "B_MAGENTA",
    "RESET",
    "MAGENTA",
    "B_CYAN",
    "CYAN",
    "B_BLUE",
    "BLUE",
    "B_GREEN",
    "GREEN",
    "B_YELLOW",
    "YELLOW",
    "B_WHITE",
    "WHITE",
    "DIM",
    "B_RED",
]


@pytest.fixture
def palette() -> dict[str, str]:
    """Sentinel palette mapping each style key to a readable ``<KEY>`` token."""
    return {key: f"<{key}>" for key in _PALETTE_KEYS}


# --------------------------------------------------------------------------- #
# build_logger_style
# --------------------------------------------------------------------------- #
def test_build_logger_style_returns_eight_expected_logger_keys(palette):
    """All eight logger names are present as keys in the style map."""
    style = clf.build_logger_style(palette)
    assert set(style) == {
        "rag.query_processor",
        "rag.rag_chain",
        "rag.vector_store",
        "rag.generator",
        "rag.knowledge_graph",
        "rag.ingest.pipeline",
        "rag.ingest.pipeline.stage",
        "rag.query_cli",
    }


def test_build_logger_style_each_value_is_label_color_tuple(palette):
    """Every entry is a 2-tuple of (label, color) strings."""
    style = clf.build_logger_style(palette)
    for label, color in style.values():
        assert isinstance(label, str)
        assert isinstance(color, str)


def test_build_logger_style_query_processor_label_and_color(palette):
    """Query-processor label wraps the B_MAGENTA badge + RESET; color is MAGENTA."""
    label, color = clf.build_logger_style(palette)["rag.query_processor"]
    assert label == "<B_MAGENTA>⟡ Query<RESET>"
    assert color == "<MAGENTA>"


# --------------------------------------------------------------------------- #
# build_level_badges
# --------------------------------------------------------------------------- #
def test_build_level_badges_returns_five_levels(palette):
    """All five log-level names are present as keys."""
    badges = clf.build_level_badges(palette)
    assert set(badges) == {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def test_build_level_badges_info_wraps_color_and_reset(palette):
    """INFO badge wraps its B_CYAN color and a trailing RESET."""
    badges = clf.build_level_badges(palette)
    assert badges["INFO"] == "<B_CYAN>ℹ<RESET>"


def test_build_level_badges_error_uses_single_cross(palette):
    """ERROR badge uses a single ✗ glyph."""
    assert clf.build_level_badges(palette)["ERROR"] == "<B_RED>✗<RESET>"


def test_build_level_badges_critical_uses_double_cross(palette):
    """CRITICAL badge uses the doubled ✗✗ glyph."""
    assert clf.build_level_badges(palette)["CRITICAL"] == "<B_RED>✗✗<RESET>"


# --------------------------------------------------------------------------- #
# style_log_message — dispatch
# --------------------------------------------------------------------------- #
def test_dispatch_query_processor_routes_to_query_styler(palette):
    """A query-processor reformulation message is styled, not returned raw."""
    msg = "Iteration 2: reformulated 'a' -> 'b'"
    out = clf.style_log_message("rag.query_processor", msg, palette)
    assert out == "Reformulation #2: <RESET>b"
    assert out != msg


def test_dispatch_ingest_stage_routes_to_stage_styler(palette):
    """An ingest-stage message is styled, not returned raw."""
    msg = "source=doc.pdf stage=table_chunk:ok"
    out = clf.style_log_message("rag.ingest.pipeline.stage", msg, palette)
    assert "<B_GREEN>ok" in out
    assert out != msg


def test_dispatch_ingest_pipeline_routes_to_pipeline_styler(palette):
    """An ingest-pipeline message is styled, not returned raw."""
    msg = "ingestion_start source=/path/to/doc.pdf"
    out = clf.style_log_message("rag.ingest.pipeline", msg, palette)
    assert "doc.pdf" in out
    assert "<B_CYAN>start" in out
    assert out != msg


def test_dispatch_unknown_logger_returns_msg_unchanged(palette):
    """An unknown logger name passes the message through byte-for-byte."""
    msg = "Iteration 2: reformulated 'a' -> 'b'"
    out = clf.style_log_message("rag.totally_unknown", msg, palette)
    assert out == msg


# --------------------------------------------------------------------------- #
# query-processor styler (via public style_log_message)
# --------------------------------------------------------------------------- #
def _query(msg: str, palette: dict[str, str]) -> str:
    return clf.style_log_message("rag.query_processor", msg, palette)


def test_query_reformulation_formats_iteration_and_new_query(palette):
    """Reformulation emits 'Reformulation #N: <RESET><new query>'."""
    out = _query("Iteration 2: reformulated 'old text' -> 'new text'", palette)
    assert out == "Reformulation #2: <RESET>new text"


def test_query_confidence_070_is_green(palette):
    """Confidence of exactly 0.70 falls in the green tier and renders 70%."""
    out = _query("Iteration 1: confidence=0.70 reasoning='r'", palette)
    assert "<B_GREEN>" in out
    assert "70%" in out
    assert "<B_YELLOW>" not in out
    assert "<B_RED>" not in out


def test_query_confidence_069_is_yellow(palette):
    """Confidence of 0.69 (one below the green boundary) is yellow."""
    out = _query("Iteration 1: confidence=0.69 reasoning='r'", palette)
    assert "<B_YELLOW>" in out
    assert "69%" in out
    assert "<B_GREEN>" not in out


def test_query_confidence_040_is_yellow(palette):
    """Confidence of exactly 0.40 is on the yellow side of the lower boundary."""
    out = _query("Iteration 1: confidence=0.40 reasoning='r'", palette)
    assert "<B_YELLOW>" in out
    assert "40%" in out
    assert "<B_RED>" not in out


def test_query_confidence_039_is_red(palette):
    """Confidence of 0.39 (one below the yellow boundary) is red."""
    out = _query("Iteration 1: confidence=0.39 reasoning='r'", palette)
    assert "<B_RED>" in out
    assert "39%" in out
    assert "<B_YELLOW>" not in out


def test_query_complete_shows_query_action_and_iters(palette):
    """Completion line surfaces the final query, action and iteration count."""
    out = _query(
        "Query processing complete: action=RETURN confidence=0.9 "
        "iterations=3 query='hi'",
        palette,
    )
    assert out.startswith("Final: ")
    assert "hi" in out
    assert "RETURN" in out
    assert "3 iters" in out


def test_query_processing_formats_query(palette):
    """Processing line emits 'Processing: <RESET><query>'."""
    out = _query("Processing query: 'hello'", palette)
    assert out == "Processing: <RESET>hello"


def test_query_non_matching_message_unchanged(palette):
    """A query message matching no pattern is returned unchanged."""
    msg = "some unrelated query-processor log line"
    assert _query(msg, palette) == msg


# --------------------------------------------------------------------------- #
# ingest-stage styler (via public style_log_message)
# --------------------------------------------------------------------------- #
def _stage(msg: str, palette: dict[str, str]) -> str:
    return clf.style_log_message("rag.ingest.pipeline.stage", msg, palette)


def test_stage_ok_shows_source_stage_and_green_status(palette):
    """Stage line shows source, underscore-to-space stage, and green ok status."""
    out = _stage("source=doc.pdf stage=table_chunk:ok", palette)
    assert "doc.pdf" in out
    assert "table chunk" in out
    assert "<B_GREEN>ok" in out


def test_stage_skipped_status_is_yellow(palette):
    """A skipped stage renders the skipped status in yellow."""
    out = _stage("source=doc.pdf stage=embed:skipped", palette)
    assert "<B_YELLOW>skipped" in out


def test_stage_failed_status_is_red(palette):
    """A failed stage renders the failed status in red."""
    out = _stage("source=doc.pdf stage=embed:failed", palette)
    assert "<B_RED>failed" in out


def test_stage_other_status_is_cyan(palette):
    """An unrecognized status (running) renders in the cyan fallback."""
    out = _stage("source=doc.pdf stage=embed:running", palette)
    assert "<B_CYAN>running" in out


def test_stage_non_matching_message_unchanged(palette):
    """A stage message matching no pattern is returned unchanged."""
    msg = "stage log without the expected shape"
    assert _stage(msg, palette) == msg


# --------------------------------------------------------------------------- #
# ingest-pipeline styler (via public style_log_message)
# --------------------------------------------------------------------------- #
def _pipeline(msg: str, palette: dict[str, str]) -> str:
    return clf.style_log_message("rag.ingest.pipeline", msg, palette)


def test_pipeline_start_uses_basename_and_start(palette):
    """ingestion_start reduces the source path to its basename and shows start."""
    out = _pipeline("ingestion_start source=/path/to/doc.pdf", palette)
    assert "doc.pdf" in out
    assert "/path/to/" not in out
    assert "<B_CYAN>start" in out


def test_pipeline_failed_shows_source_and_error_text(palette):
    """ingestion_failed surfaces the source and the failure error text."""
    out = _pipeline("ingestion_failed source=x errors=boom", palette)
    assert "x" in out
    assert "<B_RED>failed" in out
    assert "boom" in out


def test_pipeline_skipped_shows_reason_and_multistage_path(palette):
    """ingestion_skipped shows underscore-to-space reason and a multi-stage path."""
    out = _pipeline(
        "ingestion_skipped source=x reason=already_done stages=a:ok>b:skipped",
        palette,
    )
    assert "already done" in out
    assert "<B_GREEN>ok" in out
    assert "<B_YELLOW>skipped" in out


def test_pipeline_done_shows_chunk_and_stored_counts(palette):
    """ingestion_done surfaces chunk and stored counts and a stage path."""
    out = _pipeline("ingestion_done source=x chunks=5 stored=4 stages=a:ok", palette)
    assert "chunks <B_CYAN>5" in out
    assert "stored <B_CYAN>4" in out
    assert "<B_GREEN>ok" in out


def test_pipeline_non_matching_message_unchanged(palette):
    """A pipeline message matching no pattern is returned unchanged."""
    msg = "ingestion happened, vaguely"
    assert _pipeline(msg, palette) == msg


# --------------------------------------------------------------------------- #
# _format_stage_path (reached via skipped/done branches)
# --------------------------------------------------------------------------- #
def test_format_stage_path_multistage_produces_two_formatted_items(palette):
    """A two-stage path formats each item (space-name, green ok, red failed)."""
    out = clf._format_stage_path("a_b:ok>c:failed", palette)
    assert "a b" in out  # underscore -> space on stage name
    assert "<B_GREEN>ok" in out
    assert "<B_RED>failed" in out
    assert "<B_CYAN>c" in out


def test_format_stage_path_unmatched_item_passed_through_verbatim(palette):
    """An item not matching name:status is emitted verbatim alongside a match."""
    out = clf._format_stage_path("a:ok>weirditem", palette)
    assert "weirditem" in out
    assert "<B_GREEN>ok" in out
