# @summary
# Pytest tests for PathMatcher and GraphContextFormatter.
# Covers: single/multi-hop patterns, cycle guards, deduplication, result
# structure, all marker styles, token-budget truncation, and edge cases.
# Exports: (pytest collected tests)
# Deps: pytest, src.knowledge_graph.backends.networkx_backend,
#       src.knowledge_graph.query.path_matcher,
#       src.knowledge_graph.query.context_formatter,
#       src.knowledge_graph.common, src.knowledge_graph.query.schemas
# @end-summary
"""Tests for PathMatcher and GraphContextFormatter.

Part 1 — PathMatcher: verifies single-hop and multi-hop traversal, multiple
pattern merging, validation errors, no-match cases, cycle guards, result
deduplication, and the shape of PathResult / PathHop objects.

Part 2 — GraphContextFormatter: verifies section assembly, entity
description fallbacks, predicate grouping, path narrative rendering,
marker-style variants, empty-section omission, and token-budget truncation.
"""
from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Fixtures — shared graph backend
# ---------------------------------------------------------------------------


@pytest.fixture
def backend():
    """NetworkX backend pre-loaded with a small test graph.

    Graph topology::

        Issue1 --design_decision_for--> Decision1 --specified_by--> Spec1
        Issue1 --design_decision_for--> Decision2 --specified_by--> Spec2
        CycleA --depends_on--> CycleB --depends_on--> CycleA  (cycle)
    """
    from src.knowledge_graph.backends.networkx_backend import NetworkXBackend
    from src.knowledge_graph.common import Entity, Triple

    b = NetworkXBackend()

    entities = [
        Entity(name="Issue1", type="KnownIssue"),
        Entity(name="Decision1", type="DesignDecision"),
        Entity(name="Spec1", type="Specification"),
        Entity(name="Decision2", type="DesignDecision"),
        Entity(name="Spec2", type="Specification"),
    ]
    b.upsert_entities(entities)

    triples = [
        Triple(subject="Issue1", predicate="design_decision_for", object="Decision1", source="test"),
        Triple(subject="Issue1", predicate="design_decision_for", object="Decision2", source="test"),
        Triple(subject="Decision1", predicate="specified_by", object="Spec1", source="test"),
        Triple(subject="Decision2", predicate="specified_by", object="Spec2", source="test"),
    ]
    b.upsert_triples(triples)

    # Cycle for cycle-guard test
    cycle_entities = [
        Entity(name="CycleA", type="Concept"),
        Entity(name="CycleB", type="Concept"),
    ]
    b.upsert_entities(cycle_entities)
    b.upsert_triples([
        Triple(subject="CycleA", predicate="depends_on", object="CycleB", source="test"),
        Triple(subject="CycleB", predicate="depends_on", object="CycleA", source="test"),
    ])

    return b


# ---------------------------------------------------------------------------
# Part 1: PathMatcher tests
# ---------------------------------------------------------------------------


class TestPathMatcherSingleHop:
    """Single-hop pattern traversal."""

    def test_single_hop_pattern(self, backend):
        """Pattern ['design_decision_for'] from Issue1 finds Decision1 and Decision2."""
        from src.knowledge_graph.query.path_matcher import PathMatcher

        matcher = PathMatcher(backend)
        results = matcher.evaluate("Issue1", [["design_decision_for"]])

        terminals = {r.terminal_entity for r in results}
        assert "Decision1" in terminals
        assert "Decision2" in terminals
        assert len(results) == 2


class TestPathMatcherTwoHop:
    """Two-hop pattern traversal."""

    def test_two_hop_pattern(self, backend):
        """Pattern ['design_decision_for', 'specified_by'] from Issue1 finds Spec1 and Spec2."""
        from src.knowledge_graph.query.path_matcher import PathMatcher

        matcher = PathMatcher(backend)
        results = matcher.evaluate("Issue1", [["design_decision_for", "specified_by"]])

        terminals = {r.terminal_entity for r in results}
        assert "Spec1" in terminals
        assert "Spec2" in terminals
        assert len(results) == 2

    def test_two_hop_pattern_label(self, backend):
        """Pattern label is the two edge types joined by '->'."""
        from src.knowledge_graph.query.path_matcher import PathMatcher

        matcher = PathMatcher(backend)
        results = matcher.evaluate("Issue1", [["design_decision_for", "specified_by"]])

        for r in results:
            assert r.pattern_label == "design_decision_for->specified_by"


class TestPathMatcherMultiplePatterns:
    """Multiple patterns evaluated and merged."""

    def test_multiple_patterns(self, backend):
        """Results from both patterns are present in the merged output."""
        from src.knowledge_graph.query.path_matcher import PathMatcher

        matcher = PathMatcher(backend)
        results = matcher.evaluate(
            "Issue1",
            [
                ["design_decision_for"],
                ["design_decision_for", "specified_by"],
            ],
        )

        terminals = {r.terminal_entity for r in results}
        # One-hop pattern yields Decision1, Decision2
        assert "Decision1" in terminals
        assert "Decision2" in terminals
        # Two-hop pattern yields Spec1, Spec2
        assert "Spec1" in terminals
        assert "Spec2" in terminals


class TestPathMatcherValidation:
    """Input validation — empty patterns raise ValueError."""

    def test_empty_patterns_raises(self, backend):
        """evaluate() with an empty patterns list raises ValueError."""
        from src.knowledge_graph.query.path_matcher import PathMatcher

        matcher = PathMatcher(backend)
        with pytest.raises(ValueError):
            matcher.evaluate("Issue1", [])

    def test_empty_pattern_element_raises(self, backend):
        """evaluate() with a pattern containing an empty list raises ValueError."""
        from src.knowledge_graph.query.path_matcher import PathMatcher

        matcher = PathMatcher(backend)
        with pytest.raises(ValueError):
            matcher.evaluate("Issue1", [[]])


class TestPathMatcherNoMatches:
    """Pattern with an edge type that exists but has no matching edges for the seed."""

    def test_no_matches(self, backend):
        """Pattern using 'specified_by' from Issue1 returns empty list (no direct edge)."""
        from src.knowledge_graph.query.path_matcher import PathMatcher

        matcher = PathMatcher(backend)
        results = matcher.evaluate("Issue1", [["specified_by"]])

        assert results == []


class TestPathMatcherCycleGuard:
    """Per-path cycle guard prevents infinite traversal."""

    def test_cycle_guard(self, backend):
        """Traversal on CycleA→CycleB→CycleA completes without looping."""
        from src.knowledge_graph.query.path_matcher import PathMatcher

        matcher = PathMatcher(backend)
        # This would loop forever without the cycle guard.
        results = matcher.evaluate(
            "CycleA", [["depends_on", "depends_on", "depends_on"]]
        )
        # Result may be empty or contain partial paths — key thing is it terminates.
        assert isinstance(results, list)


class TestPathMatcherDeduplication:
    """Same (seed, terminal, label) from different branches is deduplicated."""

    def test_deduplication(self):
        """Two patterns that both reach the same terminal produce one PathResult."""
        from src.knowledge_graph.backends.networkx_backend import NetworkXBackend
        from src.knowledge_graph.common import Entity, Triple
        from src.knowledge_graph.query.path_matcher import PathMatcher

        b = NetworkXBackend()
        b.upsert_entities([
            Entity(name="Root", type="Concept"),
            Entity(name="Middle", type="Concept"),
            Entity(name="Target", type="Concept"),
        ])
        b.upsert_triples([
            Triple(subject="Root", predicate="edge_a", object="Target", source="test"),
            Triple(subject="Root", predicate="edge_b", object="Middle", source="test"),
            Triple(subject="Middle", predicate="edge_a", object="Target", source="test"),
        ])

        matcher = PathMatcher(b)
        # Both patterns ultimately produce Root->...->Target with label "edge_a"
        results = matcher.evaluate("Root", [["edge_a"], ["edge_a"]])

        # Deduplication: same (seed, terminal, label) only once
        keys = [(r.seed_entity, r.terminal_entity, r.pattern_label) for r in results]
        assert len(keys) == len(set(keys)), "Duplicate (seed, terminal, label) tuples found"


class TestPathResultStructure:
    """PathResult and PathHop field contracts."""

    def test_path_result_structure(self, backend):
        """PathResult has correct seed_entity, terminal_entity, pattern_label, and hops."""
        from src.knowledge_graph.query.path_matcher import PathMatcher

        matcher = PathMatcher(backend)
        results = matcher.evaluate("Issue1", [["design_decision_for"]])

        assert len(results) > 0
        r = results[0]
        assert r.seed_entity == "Issue1"
        assert r.terminal_entity in {"Decision1", "Decision2"}
        assert r.pattern_label == "design_decision_for"
        assert len(r.hops) == 1

    def test_hop_structure(self, backend):
        """Each PathHop has from_entity, edge_type, and to_entity."""
        from src.knowledge_graph.query.path_matcher import PathMatcher

        matcher = PathMatcher(backend)
        results = matcher.evaluate("Issue1", [["design_decision_for"]])

        hop = results[0].hops[0]
        assert hop.from_entity == "Issue1"
        assert hop.edge_type == "design_decision_for"
        assert hop.to_entity in {"Decision1", "Decision2"}


# ---------------------------------------------------------------------------
# Part 2: GraphContextFormatter tests
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_entities():
    """Two entities: one with a summary, one without."""
    from src.knowledge_graph.common import Entity

    return [
        Entity(name="EntityA", type="KnownIssue", current_summary="A timing violation issue"),
        Entity(name="EntityB", type="DesignDecision"),  # no summary, no mentions
    ]


@pytest.fixture
def sample_triples():
    """Two triples with different predicates."""
    from src.knowledge_graph.common import Triple

    return [
        Triple(subject="EntityA", predicate="design_decision_for", object="EntityB", source="test"),
        Triple(subject="EntityA", predicate="depends_on", object="EntityB", source="test"),
    ]


@pytest.fixture
def sample_paths():
    """One two-hop PathResult."""
    from src.knowledge_graph.query.schemas import PathHop, PathResult

    return [
        PathResult(
            pattern_label="design_decision_for->specified_by",
            seed_entity="EntityA",
            hops=[
                PathHop("EntityA", "design_decision_for", "EntityB"),
                PathHop("EntityB", "specified_by", "EntityC"),
            ],
            terminal_entity="EntityC",
        )
    ]


class TestGraphContextFormatterSections:
    """Section presence and absence."""

    def test_all_sections_present(self, sample_entities, sample_triples, sample_paths):
        """Output contains all three section markers when all inputs are non-empty."""
        from src.knowledge_graph.query.context_formatter import GraphContextFormatter

        fmt = GraphContextFormatter()
        output = fmt.format(sample_entities, sample_triples, sample_paths)

        assert "## Graph Context" in output
        assert "### Entities" in output
        assert "### Relationships" in output
        assert "### Paths" in output

    def test_empty_inputs_returns_empty(self):
        """format([], [], []) returns an empty string."""
        from src.knowledge_graph.query.context_formatter import GraphContextFormatter

        fmt = GraphContextFormatter()
        assert fmt.format([], [], []) == ""

    def test_empty_section_omitted(self, sample_entities):
        """Relationships heading is absent when no triples are provided."""
        from src.knowledge_graph.query.context_formatter import GraphContextFormatter

        fmt = GraphContextFormatter()
        output = fmt.format(sample_entities, [], [])

        assert "### Relationships" not in output
        assert "### Paths" not in output
        assert "### Entities" in output


class TestGraphContextFormatterEntities:
    """Entity summary and fallback description rendering."""

    def test_entity_with_summary(self, sample_entities):
        """Entity's current_summary appears in the output."""
        from src.knowledge_graph.query.context_formatter import GraphContextFormatter

        fmt = GraphContextFormatter()
        output = fmt.format(sample_entities, [], [])

        assert "A timing violation issue" in output

    def test_entity_no_description(self, sample_entities):
        """'[No description available]' appears for entity with no summary or mentions."""
        from src.knowledge_graph.query.context_formatter import GraphContextFormatter

        fmt = GraphContextFormatter()
        output = fmt.format(sample_entities, [], [])

        assert "[No description available]" in output


class TestGraphContextFormatterTriples:
    """Relationship triples section rendering."""

    def test_triples_grouped_by_predicate(self, sample_entities, sample_triples):
        """Two distinct predicates produce two group headings."""
        from src.knowledge_graph.query.context_formatter import GraphContextFormatter

        fmt = GraphContextFormatter()
        output = fmt.format(sample_entities, sample_triples, [])

        assert "**design_decision_for**" in output
        assert "**depends_on**" in output


class TestGraphContextFormatterPaths:
    """Path narrative rendering."""

    def test_path_narrative_two_hop(self, sample_entities, sample_paths):
        """Two-hop path narrative contains 'which' connector."""
        from src.knowledge_graph.query.context_formatter import GraphContextFormatter

        fmt = GraphContextFormatter()
        output = fmt.format(sample_entities, [], sample_paths)

        assert "which" in output

    def test_path_narrative_underscore_replacement(self, sample_entities, sample_paths):
        """Underscores in predicate names are replaced with spaces in narratives."""
        from src.knowledge_graph.query.context_formatter import GraphContextFormatter

        fmt = GraphContextFormatter()
        output = fmt.format(sample_entities, [], sample_paths)

        # "design_decision_for" must appear as "design decision for"
        assert "design decision for" in output
        # Raw underscore form must NOT appear in the paths section
        # (it may appear in relationship group headings, so narrow the check)
        paths_section_start = output.find("### Paths")
        if paths_section_start != -1:
            paths_text = output[paths_section_start:]
            assert "design_decision_for" not in paths_text


class TestGraphContextFormatterMarkerStyles:
    """Section marker style variants."""

    def test_markdown_markers(self, sample_entities):
        """Default markdown style uses ATX headings."""
        from src.knowledge_graph.query.context_formatter import GraphContextFormatter

        fmt = GraphContextFormatter(marker_style="markdown")
        output = fmt.format(sample_entities, [], [])

        assert "## Graph Context" in output
        assert "### Entities" in output

    def test_xml_markers(self, sample_entities):
        """XML style wraps sections in XML-style tags."""
        from src.knowledge_graph.query.context_formatter import GraphContextFormatter

        fmt = GraphContextFormatter(marker_style="xml")
        output = fmt.format(sample_entities, [], [])

        assert "<graph_context>" in output
        assert "<entities>" in output
        assert "</entities>" in output
        assert "</graph_context>" in output

    def test_plain_markers(self, sample_entities):
        """Plain style uses ASCII ruler headings."""
        from src.knowledge_graph.query.context_formatter import GraphContextFormatter

        fmt = GraphContextFormatter(marker_style="plain")
        output = fmt.format(sample_entities, [], [])

        assert "=== GRAPH CONTEXT ===" in output
        assert "--- ENTITIES ---" in output

    def test_invalid_marker_style_raises(self):
        """Constructing with marker_style='html' raises ValueError."""
        from src.knowledge_graph.query.context_formatter import GraphContextFormatter

        with pytest.raises(ValueError):
            GraphContextFormatter(marker_style="html")


class TestGraphContextFormatterTokenBudget:
    """Token-budget truncation behaviour."""

    def test_token_budget_truncation(self, sample_entities, sample_triples):
        """A very small budget (50 tokens) truncates some content."""
        from src.knowledge_graph.query.context_formatter import GraphContextFormatter

        fmt = GraphContextFormatter(token_budget=50)
        output = fmt.format(sample_entities, sample_triples, [])

        assert "[Context truncated:" in output

    def test_token_budget_zero_unlimited(self, sample_entities, sample_triples):
        """Budget=0 produces no truncation annotation."""
        from src.knowledge_graph.query.context_formatter import GraphContextFormatter

        fmt = GraphContextFormatter(token_budget=0)
        output = fmt.format(sample_entities, sample_triples, [])

        assert "[Context truncated:" not in output

    def test_seed_name_preserved_under_truncation(self):
        """Seed entity name+type stub survives even with a tiny token budget."""
        from src.knowledge_graph.common import Entity
        from src.knowledge_graph.query.context_formatter import GraphContextFormatter

        # A budget of 5 tokens (20 chars) forces description truncation.
        fmt = GraphContextFormatter(token_budget=5)
        entities = [
            Entity(
                name="MySeed",
                type="KnownIssue",
                current_summary="Very long description that should be truncated away",
            )
        ]
        output = fmt.format(entities, [], [], seed_entity_names=["MySeed"])

        # The name+type stub must still appear.
        assert "MySeed" in output
        assert "KnownIssue" in output
