# @summary
# Comprehensive tests for three KG retrieval modules:
#   Part 1 — ExpansionResult and PathResult from query/schemas.py
#   Part 2 — validate_edge_types and validate_path_patterns from common/validation.py
#   Part 3 — NetworkXBackend.query_neighbors_typed typed traversal
# Exports: TestExpansionResult, TestPathResult, TestValidateEdgeTypes,
#          TestValidatePathPatterns, TestNetworkXBackendTypedTraversal
# Deps: pytest, yaml, src.knowledge_graph.query.schemas,
#       src.knowledge_graph.common.validation,
#       src.knowledge_graph.backends.networkx_backend,
#       src.knowledge_graph.common (Entity, Triple)
# @end-summary
"""Tests for KG retrieval schemas, schema validation, and typed graph traversal."""

from __future__ import annotations

import pytest
import yaml

from src.knowledge_graph.query.schemas import ExpansionResult, PathHop, PathResult
from src.knowledge_graph.common.validation import (
    KGConfigValidationError,
    PatternWarning,
    validate_edge_types,
    validate_path_patterns,
)


# ---------------------------------------------------------------------------
# Shared YAML schema fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def schema_file(tmp_path):
    """Write a minimal but load_schema-compatible kg_schema.yaml and return its path.

    ``load_schema`` requires ``description``, ``category``, and ``phase`` on every
    node type and edge type entry.  The nested-dict format
    ``{category: {TypeName: {description: ..., category: ..., phase: ...}}}``
    is used here because that is what the production schema uses.
    """
    schema = {
        "version": "1.0",
        "description": "Test schema fixture",
        "node_types": {
            "structural": {
                "RTL_Module": {
                    "description": "An RTL module",
                    "category": "structural",
                    "phase": "phase_1",
                },
                "Port": {
                    "description": "A port on a module",
                    "category": "structural",
                    "phase": "phase_1",
                },
                "Signal": {
                    "description": "A signal",
                    "category": "structural",
                    "phase": "phase_1",
                },
            },
            "semantic": {
                "DesignDecision": {
                    "description": "A design decision",
                    "category": "semantic",
                    "phase": "phase_1",
                },
                "Specification": {
                    "description": "A specification document",
                    "category": "semantic",
                    "phase": "phase_1",
                },
                "KnownIssue": {
                    "description": "A known issue or bug",
                    "category": "semantic",
                    "phase": "phase_1",
                },
            },
        },
        "edge_types": {
            "structural": {
                "depends_on": {
                    "description": "Module depends on another module",
                    "category": "structural",
                    "phase": "phase_1",
                    "source_types": ["RTL_Module"],
                    "target_types": ["RTL_Module"],
                },
                "connects_to": {
                    "description": "Port or signal connects to another",
                    "category": "structural",
                    "phase": "phase_1",
                    "source_types": ["Port", "Signal"],
                    "target_types": ["Port", "Signal"],
                },
            },
            "semantic": {
                "specified_by": {
                    "description": "Decision specified by a specification",
                    "category": "semantic",
                    "phase": "phase_1",
                    "source_types": ["DesignDecision"],
                    "target_types": ["Specification"],
                },
                "design_decision_for": {
                    "description": "Decision made for a module or issue",
                    "category": "semantic",
                    "phase": "phase_1",
                    "source_types": ["DesignDecision"],
                    "target_types": ["KnownIssue", "RTL_Module"],
                },
            },
        },
    }
    path = tmp_path / "test_schema.yaml"
    path.write_text(yaml.dump(schema))
    return str(path)


# ---------------------------------------------------------------------------
# Part 1: ExpansionResult
# ---------------------------------------------------------------------------

class TestExpansionResult:
    """Tests for ExpansionResult dataclass — schema contract and sequence protocol."""

    def test_iteration(self):
        er = ExpansionResult(terms=["a", "b", "c"])
        assert list(er) == ["a", "b", "c"]

    def test_len(self):
        er = ExpansionResult(terms=["a", "b"])
        assert len(er) == 2

    def test_getitem(self):
        er = ExpansionResult(terms=["a", "b"])
        assert er[0] == "a"
        assert er[1] == "b"

    def test_default_graph_context(self):
        er = ExpansionResult(terms=[])
        assert er.graph_context == ""

    def test_explicit_graph_context(self):
        er = ExpansionResult(terms=["x"], graph_context="some context")
        assert er.graph_context == "some context"

    def test_empty_terms(self):
        er = ExpansionResult(terms=[])
        assert list(er) == []
        assert len(er) == 0

    def test_terms_slice(self):
        er = ExpansionResult(terms=["a", "b", "c"])
        assert er.terms[:2] == ["a", "b"]

    def test_single_term(self):
        er = ExpansionResult(terms=["only"])
        assert len(er) == 1
        assert er[0] == "only"

    def test_terms_field_is_mutable(self):
        """Mutating .terms is reflected by __len__ and __iter__."""
        er = ExpansionResult(terms=["a"])
        er.terms.append("b")
        assert len(er) == 2
        assert list(er) == ["a", "b"]


# ---------------------------------------------------------------------------
# Part 1: PathResult / PathHop
# ---------------------------------------------------------------------------

class TestPathResult:
    """Tests for PathResult.length and PathHop construction."""

    def test_length_two_hops(self):
        pr = PathResult(
            pattern_label="test",
            seed_entity="A",
            hops=[PathHop("A", "edge", "B"), PathHop("B", "edge", "C")],
            terminal_entity="C",
        )
        assert pr.length == 2

    def test_length_one_hop(self):
        pr = PathResult(
            pattern_label="single",
            seed_entity="X",
            hops=[PathHop("X", "connects_to", "Y")],
            terminal_entity="Y",
        )
        assert pr.length == 1

    def test_empty_hops(self):
        pr = PathResult(
            pattern_label="empty",
            seed_entity="A",
            hops=[],
            terminal_entity="A",
        )
        assert pr.length == 0

    def test_path_hop_fields(self):
        hop = PathHop(from_entity="A", edge_type="depends_on", to_entity="B")
        assert hop.from_entity == "A"
        assert hop.edge_type == "depends_on"
        assert hop.to_entity == "B"

    def test_length_property_reflects_hops_list(self):
        """length is derived from hops, so appending a hop updates it."""
        hops = [PathHop("A", "e", "B")]
        pr = PathResult(pattern_label="p", seed_entity="A", hops=hops, terminal_entity="B")
        assert pr.length == 1
        pr.hops.append(PathHop("B", "e", "C"))
        assert pr.length == 2


# ---------------------------------------------------------------------------
# Part 2: validate_edge_types
# ---------------------------------------------------------------------------

class TestValidateEdgeTypes:
    """Tests for validate_edge_types against a YAML schema fixture."""

    def test_valid_types_pass(self, schema_file):
        """All known types should pass without raising."""
        validate_edge_types(["depends_on", "specified_by"], schema_file)

    def test_single_valid_type_passes(self, schema_file):
        validate_edge_types(["connects_to"], schema_file)

    def test_empty_list_passes(self, schema_file):
        """An empty list has nothing to validate — should not raise."""
        validate_edge_types([], schema_file)

    def test_unknown_type_raises(self, schema_file):
        with pytest.raises(KGConfigValidationError) as exc_info:
            validate_edge_types(["nonexistent_edge"], schema_file)
        err = exc_info.value
        assert any("nonexistent_edge" in e for e in err.errors)

    def test_unknown_type_message_contains_valid_types(self, schema_file):
        with pytest.raises(KGConfigValidationError) as exc_info:
            validate_edge_types(["bad_edge"], schema_file)
        full_msg = str(exc_info.value)
        # The error message should list some valid alternatives
        assert "depends_on" in full_msg or "specified_by" in full_msg

    def test_multiple_unknown_types_all_accumulated(self, schema_file):
        """All unknown types must appear in a single KGConfigValidationError."""
        with pytest.raises(KGConfigValidationError) as exc_info:
            validate_edge_types(["bad_a", "bad_b", "bad_c"], schema_file)
        err = exc_info.value
        assert len(err.errors) == 3
        unknown_mentioned = {e for e in err.errors if "bad_a" in e or "bad_b" in e or "bad_c" in e}
        assert len(unknown_mentioned) == 3

    def test_mixed_valid_and_invalid_raises(self, schema_file):
        """Valid types alongside invalid ones still raise; valid types are not flagged."""
        with pytest.raises(KGConfigValidationError) as exc_info:
            validate_edge_types(["depends_on", "ghost_edge"], schema_file)
        err = exc_info.value
        assert len(err.errors) == 1
        assert "ghost_edge" in err.errors[0]

    def test_file_not_found_raises(self, tmp_path):
        missing = str(tmp_path / "no_such_schema.yaml")
        with pytest.raises(FileNotFoundError):
            validate_edge_types(["depends_on"], missing)


# ---------------------------------------------------------------------------
# Part 2: validate_path_patterns
# ---------------------------------------------------------------------------

class TestValidatePathPatterns:
    """Tests for validate_path_patterns against a YAML schema fixture."""

    def test_valid_patterns_return_empty_warnings(self, schema_file):
        """A compatible hop sequence returns an empty warning list."""
        # depends_on: RTL_Module → RTL_Module; connects_to: Port/Signal → Port/Signal
        # These two are from different categories, but each hop is individually valid.
        warnings = validate_path_patterns(
            [["depends_on", "depends_on"]], schema_file
        )
        assert warnings == []

    def test_empty_patterns_return_empty_warnings(self, schema_file):
        warnings = validate_path_patterns([], schema_file)
        assert warnings == []

    def test_single_hop_pattern_no_warnings(self, schema_file):
        warnings = validate_path_patterns([["depends_on"]], schema_file)
        assert warnings == []

    def test_unknown_edge_raises(self, schema_file):
        with pytest.raises(KGConfigValidationError) as exc_info:
            validate_path_patterns([["depends_on", "unknown_hop"]], schema_file)
        err = exc_info.value
        assert any("unknown_hop" in e for e in err.errors)

    def test_multiple_unknown_edges_all_accumulated(self, schema_file):
        with pytest.raises(KGConfigValidationError) as exc_info:
            validate_path_patterns(
                [["bad_x", "bad_y"], ["bad_z"]], schema_file
            )
        err = exc_info.value
        # Each unknown type generates its own error entry
        assert len(err.errors) >= 3

    def test_incompatible_hops_return_pattern_warning(self, schema_file):
        """Hops with no shared boundary type produce a PatternWarning (not an error)."""
        # depends_on targets RTL_Module; specified_by expects source DesignDecision
        # RTL_Module ∩ DesignDecision = ∅ → warning
        warnings = validate_path_patterns(
            [["depends_on", "specified_by"]], schema_file
        )
        assert len(warnings) == 1
        w = warnings[0]
        assert isinstance(w, PatternWarning)
        assert w.pattern_index == 0
        assert w.hop_index == 0
        assert w.edge_type_a == "depends_on"
        assert w.edge_type_b == "specified_by"

    def test_incompatible_hops_warning_message_describes_boundary(self, schema_file):
        warnings = validate_path_patterns(
            [["depends_on", "specified_by"]], schema_file
        )
        assert len(warnings) == 1
        assert "depends_on" in warnings[0].message
        assert "specified_by" in warnings[0].message

    def test_strict_true_with_warnings_raises(self, schema_file):
        """strict=True converts PatternWarnings into a KGConfigValidationError."""
        with pytest.raises(KGConfigValidationError) as exc_info:
            validate_path_patterns(
                [["depends_on", "specified_by"]],
                schema_file,
                strict=True,
            )
        err = exc_info.value
        assert len(err.errors) == 1
        assert "depends_on" in err.errors[0]

    def test_strict_false_with_warnings_returns_list(self, schema_file):
        """strict=False (default) returns warnings without raising."""
        warnings = validate_path_patterns(
            [["depends_on", "specified_by"]],
            schema_file,
            strict=False,
        )
        assert len(warnings) == 1

    def test_compatible_hops_no_warnings(self, schema_file):
        """design_decision_for targets RTL_Module; depends_on sources RTL_Module → compatible."""
        warnings = validate_path_patterns(
            [["design_decision_for", "depends_on"]], schema_file
        )
        assert warnings == []

    def test_file_not_found_raises(self, tmp_path):
        missing = str(tmp_path / "ghost_schema.yaml")
        with pytest.raises(FileNotFoundError):
            validate_path_patterns([["depends_on"]], missing)


# ---------------------------------------------------------------------------
# Part 3: NetworkXBackend typed traversal
# ---------------------------------------------------------------------------

@pytest.fixture
def backend_with_graph():
    """Small in-memory graph:  A -design_decision_for-> B -specified_by-> C
                                A -authored_by-> D                          """
    from src.knowledge_graph.backends.networkx_backend import NetworkXBackend
    from src.knowledge_graph.common import Entity, Triple

    backend = NetworkXBackend()
    entities = [
        Entity(name="A", type="KnownIssue"),
        Entity(name="B", type="DesignDecision"),
        Entity(name="C", type="Specification"),
        Entity(name="D", type="Concept"),
    ]
    backend.upsert_entities(entities)
    triples = [
        Triple(subject="A", predicate="design_decision_for", object="B", source="test"),
        Triple(subject="B", predicate="specified_by", object="C", source="test"),
        Triple(subject="A", predicate="authored_by", object="D", source="test"),
    ]
    backend.upsert_triples(triples)
    return backend


class TestNetworkXBackendTypedTraversal:
    """Tests for NetworkXBackend.query_neighbors_typed (REQ-KG-760)."""

    def test_matching_edge_type_returns_neighbor(self, backend_with_graph):
        results = backend_with_graph.query_neighbors_typed("A", ["design_decision_for"])
        names = {e.name for e in results}
        assert "B" in names

    def test_matching_edge_type_excludes_non_matching(self, backend_with_graph):
        """authored_by should not be returned when filtering on design_decision_for only."""
        results = backend_with_graph.query_neighbors_typed("A", ["design_decision_for"])
        names = {e.name for e in results}
        assert "D" not in names

    def test_non_matching_edge_type_returns_empty(self, backend_with_graph):
        results = backend_with_graph.query_neighbors_typed("A", ["specified_by"])
        # A has no specified_by edges (outgoing or incoming)
        assert results == []

    def test_non_existent_entity_returns_empty(self, backend_with_graph):
        results = backend_with_graph.query_neighbors_typed("GHOST", ["design_decision_for"])
        assert results == []

    def test_empty_edge_types_raises_value_error(self, backend_with_graph):
        with pytest.raises(ValueError, match="non-empty"):
            backend_with_graph.query_neighbors_typed("A", [])

    def test_depth_less_than_one_raises_value_error(self, backend_with_graph):
        with pytest.raises(ValueError, match="depth"):
            backend_with_graph.query_neighbors_typed("A", ["design_decision_for"], depth=0)

    def test_depth_two_follows_chain(self, backend_with_graph):
        """With depth=2, starting from A via design_decision_for → B → specified_by → C."""
        results = backend_with_graph.query_neighbors_typed(
            "A", ["design_decision_for", "specified_by"], depth=2
        )
        names = {e.name for e in results}
        assert "B" in names
        assert "C" in names

    def test_depth_one_does_not_follow_chain(self, backend_with_graph):
        """With depth=1, C should not be reachable from A via a two-hop chain."""
        results = backend_with_graph.query_neighbors_typed(
            "A", ["design_decision_for", "specified_by"], depth=1
        )
        names = {e.name for e in results}
        assert "C" not in names

    def test_deduplication_multiple_paths(self):
        """Entity reachable by two paths appears only once in the result."""
        from src.knowledge_graph.backends.networkx_backend import NetworkXBackend
        from src.knowledge_graph.common import Entity, Triple

        backend = NetworkXBackend()
        entities = [
            Entity(name="Start", type="Concept"),
            Entity(name="Mid1", type="Concept"),
            Entity(name="Mid2", type="Concept"),
            Entity(name="Shared", type="Concept"),
        ]
        backend.upsert_entities(entities)
        triples = [
            Triple(subject="Start", predicate="rel", object="Mid1", source="t"),
            Triple(subject="Start", predicate="rel", object="Mid2", source="t"),
            Triple(subject="Mid1", predicate="rel", object="Shared", source="t"),
            Triple(subject="Mid2", predicate="rel", object="Shared", source="t"),
        ]
        backend.upsert_triples(triples)

        results = backend.query_neighbors_typed("Start", ["rel"], depth=2)
        names = [e.name for e in results]
        # "Shared" should appear at most once despite two paths reaching it
        assert names.count("Shared") == 1

    def test_multiple_matching_edge_types_in_whitelist(self, backend_with_graph):
        """Passing multiple edge types in the whitelist traverses both."""
        results = backend_with_graph.query_neighbors_typed(
            "A", ["design_decision_for", "authored_by"]
        )
        names = {e.name for e in results}
        assert "B" in names
        assert "D" in names

    def test_seed_entity_not_included_in_results(self, backend_with_graph):
        """The seed entity itself must never appear in the returned neighbor list."""
        results = backend_with_graph.query_neighbors_typed("A", ["design_decision_for"])
        names = {e.name for e in results}
        assert "A" not in names

    def test_incoming_edge_traversal(self, backend_with_graph):
        """Typed traversal follows incoming edges too — B should find A via design_decision_for."""
        results = backend_with_graph.query_neighbors_typed("B", ["design_decision_for"])
        names = {e.name for e in results}
        assert "A" in names
