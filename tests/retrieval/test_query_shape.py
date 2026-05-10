# @summary
# Unit tests for the deep-research suggestion engine.
# Covers compound-marker detection, topic-spread heuristic, edge cases.
# Deps: pytest
# @end-summary
"""Unit tests for ``src.retrieval.pipeline.query_shape``."""

from __future__ import annotations

import pytest

from src.retrieval.common.schemas import RankedResult
from src.retrieval.pipeline.query_shape import (
    COMPOUND_MARKERS,
    MAX_UNIQUE_SOURCES_FOR_SUGGESTION,
    should_suggest_deep_research,
)


def _result(text: str, source: str, score: float = 0.5) -> RankedResult:
    return RankedResult(text=text, score=score, metadata={"source": source})


class TestCompoundMarkers:
    """Signal 1: query string must contain a compound marker."""

    def test_simple_query_returns_false(self) -> None:
        results = [_result("foo", f"doc{i}.md") for i in range(5)]
        suggest, _ = should_suggest_deep_research("what is gpio", results)
        assert suggest is False

    def test_single_question_mark_is_not_compound(self) -> None:
        results = [_result("foo", "only.md") for _ in range(5)]
        suggest, _ = should_suggest_deep_research("what is gpio?", results)
        assert suggest is False

    def test_query_with_and_marker_compound(self) -> None:
        assert any(m in " verification methodology and lint flow " for m in COMPOUND_MARKERS)

    def test_query_with_vs_marker_compound(self) -> None:
        results = [_result("foo", "only.md") for _ in range(5)]
        suggest, reason = should_suggest_deep_research("axi vs ahb protocol", results)
        assert suggest is True
        assert reason and "compound" in reason.lower()

    def test_multiple_question_marks_compound(self) -> None:
        results = [_result("foo", "only.md") for _ in range(5)]
        suggest, _ = should_suggest_deep_research(
            "how does X work? and what about Y?", results
        )
        assert suggest is True


class TestTopicSpread:
    """Signal 2: low source diversity in baseline results."""

    def test_compound_query_low_diversity_suggests(self) -> None:
        results = [_result("foo", "only.md") for _ in range(5)]
        suggest, _ = should_suggest_deep_research(
            "verification methodology and lint flow", results
        )
        assert suggest is True

    def test_compound_query_high_diversity_does_not_suggest(self) -> None:
        results = [_result("foo", f"doc{i}.md") for i in range(5)]
        assert len({r.metadata["source"] for r in results}) > MAX_UNIQUE_SOURCES_FOR_SUGGESTION
        suggest, _ = should_suggest_deep_research(
            "verification methodology and lint flow", results
        )
        assert suggest is False

    def test_compound_query_two_sources_still_suggests(self) -> None:
        results = [_result("foo", "a.md"), _result("foo", "a.md"), _result("foo", "b.md")]
        suggest, _ = should_suggest_deep_research("a and b", results)
        assert suggest is True


class TestEdgeCases:
    def test_empty_results_returns_false(self) -> None:
        suggest, _ = should_suggest_deep_research("a and b", [])
        assert suggest is False

    def test_empty_query_returns_false(self) -> None:
        results = [_result("foo", "only.md") for _ in range(5)]
        suggest, _ = should_suggest_deep_research("", results)
        assert suggest is False

    def test_results_without_source_metadata_treated_as_distinct(self) -> None:
        # Missing source metadata cannot prove low diversity — be conservative.
        results = [
            RankedResult(text="t", score=0.5, metadata={})
            for _ in range(5)
        ]
        suggest, _ = should_suggest_deep_research("a and b", results)
        assert suggest is False


@pytest.mark.parametrize(
    "query",
    [
        "x and y",
        "x AND y",
        "x vs y",
        "x VS y",
        "x versus y",
        "compare x to y",
        "what is x? what is y?",
    ],
)
def test_recognized_compound_markers(query: str) -> None:
    results = [_result("foo", "only.md") for _ in range(5)]
    suggest, _ = should_suggest_deep_research(query, results)
    assert suggest is True, f"expected compound for: {query!r}"
