# @summary
# Pytest tests for GraphQueryExpander retrieval enhancement and integration.
# Covers: typed dispatch, untyped fallback, graceful degradation on backend/formatter
# errors, no-seed early return, backward-compat iteration, and end-to-end integration
# with a real NetworkXBackend.
# Exports: (test module — no public exports)
# Deps: pytest, unittest.mock, src.knowledge_graph.query.expander,
#       src.knowledge_graph.query.schemas, src.knowledge_graph.common,
#       src.knowledge_graph.backends.networkx_backend
# @end-summary
"""Tests for GraphQueryExpander typed dispatch, degradation, and integration paths."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.knowledge_graph.common.schemas import Entity, Triple
from src.knowledge_graph.query.expander import GraphQueryExpander
from src.knowledge_graph.query.schemas import ExpansionResult


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


# Canonical entity names used in mock tests.  No underscores so that the
# QuerySanitizer's underscore→space normalisation does not break substring
# matching (e.g. "EntityAlpha".lower() == "entityalpha" survives intact).
_MOCK_ENTITY_ALPHA = "EntityAlpha"
_MOCK_ENTITY_BETA = "EntityBeta"


@pytest.fixture
def mock_backend():
    """Mock GraphStorageBackend with a minimal entity index pre-loaded.

    Entity names intentionally have no underscores so they survive the
    QuerySanitizer's underscore-to-space normalisation without becoming
    non-matchable substrings.
    """
    backend = MagicMock()
    backend.get_all_node_names_and_aliases.return_value = {
        "entityalpha": _MOCK_ENTITY_ALPHA,
        "entitybeta": _MOCK_ENTITY_BETA,
    }
    backend.query_neighbors.return_value = [
        Entity(name=_MOCK_ENTITY_BETA, type="Concept"),
    ]
    backend.query_neighbors_typed.return_value = [
        Entity(name=_MOCK_ENTITY_BETA, type="Concept"),
    ]
    backend.get_outgoing_edges.return_value = [
        Triple(
            subject=_MOCK_ENTITY_ALPHA,
            predicate="depends_on",
            object=_MOCK_ENTITY_BETA,
            source="test",
        ),
    ]
    backend.get_entity.return_value = Entity(name=_MOCK_ENTITY_ALPHA, type="KnownIssue")
    backend.get_predecessors.return_value = []
    backend.get_incoming_edges.return_value = []
    return backend


@pytest.fixture
def mock_config():
    """KGConfig with graph context injection enabled and edge types set."""
    from src.knowledge_graph.common.types import KGConfig

    return KGConfig(
        enable_graph_context_injection=True,
        retrieval_edge_types=["depends_on", "specified_by"],
        retrieval_path_patterns=[["depends_on"]],
        graph_context_token_budget=500,
    )


@pytest.fixture
def disabled_config():
    """KGConfig with graph context injection disabled."""
    from src.knowledge_graph.common.types import KGConfig

    return KGConfig(
        enable_graph_context_injection=False,
    )


# ---------------------------------------------------------------------------
# Part 1: Expander typed dispatch + degradation tests
# ---------------------------------------------------------------------------


class TestExpandReturnsExpansionResult:
    """expand() return type contract."""

    def test_expand_returns_expansion_result(self, mock_backend, mock_config):
        """Return type is always ExpansionResult with list terms and str context."""
        expander = GraphQueryExpander(backend=mock_backend, config=mock_config)
        result = expander.expand(f"test query about {_MOCK_ENTITY_ALPHA}")

        assert isinstance(result, ExpansionResult)
        assert isinstance(result.terms, list)
        assert isinstance(result.graph_context, str)


class TestTypedDispatch:
    """Typed vs untyped neighbour traversal dispatch (REQ-KG-762)."""

    def test_typed_dispatch_calls_typed(self, mock_backend, mock_config):
        """When injection enabled and edge_types set, query_neighbors_typed is called."""
        expander = GraphQueryExpander(backend=mock_backend, config=mock_config)
        expander.expand(f"test query about {_MOCK_ENTITY_ALPHA}")

        mock_backend.query_neighbors_typed.assert_called()

    def test_untyped_fallback_when_disabled(self, mock_backend, disabled_config):
        """Feature disabled: typed method never called, context is empty string."""
        expander = GraphQueryExpander(backend=mock_backend, config=disabled_config)
        result = expander.expand(f"test query about {_MOCK_ENTITY_ALPHA}")

        mock_backend.query_neighbors_typed.assert_not_called()
        assert result.graph_context == ""

    def test_untyped_fallback_empty_edge_types(self, mock_backend):
        """retrieval_edge_types=[] → untyped traversal, typed method never called."""
        from src.knowledge_graph.common.types import KGConfig

        config = KGConfig(
            enable_graph_context_injection=True,
            retrieval_edge_types=[],
        )
        expander = GraphQueryExpander(backend=mock_backend, config=config)
        expander.expand(f"test query about {_MOCK_ENTITY_ALPHA}")

        mock_backend.query_neighbors_typed.assert_not_called()


class TestDegradation:
    """Graceful degradation for typed traversal and formatter errors (REQ-KG-1214)."""

    def test_typed_traversal_degradation(self, mock_backend, mock_config):
        """Typed backend raises → falls back to untyped, does not propagate exception."""
        mock_backend.query_neighbors_typed.side_effect = RuntimeError("Backend timeout")

        expander = GraphQueryExpander(backend=mock_backend, config=mock_config)
        result = expander.expand(f"test query about {_MOCK_ENTITY_ALPHA}")

        # Must not raise; result is still a valid ExpansionResult
        assert isinstance(result, ExpansionResult)
        # Untyped fallback was used
        mock_backend.query_neighbors.assert_called()

    def test_formatting_degradation(self, mock_backend, mock_config):
        """Formatter raises → graph_context is empty, result still valid."""
        expander = GraphQueryExpander(backend=mock_backend, config=mock_config)

        if expander._formatter is not None:
            expander._formatter.format = MagicMock(
                side_effect=RuntimeError("Format error")
            )

        result = expander.expand(f"test query about {_MOCK_ENTITY_ALPHA}")

        assert isinstance(result, ExpansionResult)
        assert result.graph_context == ""


class TestEdgeCases:
    """Edge cases: no seed entities, backward-compat iteration."""

    def test_no_seed_entities(self, mock_backend, mock_config):
        """No entity index entries → early return with empty terms and context."""
        mock_backend.get_all_node_names_and_aliases.return_value = {}
        expander = GraphQueryExpander(backend=mock_backend, config=mock_config)
        result = expander.expand("completely unrelated query")

        assert result.terms == []
        assert result.graph_context == ""

    def test_backward_compat_iteration(self, mock_backend, mock_config):
        """ExpansionResult can be iterated like a list (backward-compat contract)."""
        expander = GraphQueryExpander(backend=mock_backend, config=mock_config)
        result = expander.expand(f"test query about {_MOCK_ENTITY_ALPHA}")

        terms_list = list(result)
        assert isinstance(terms_list, list)


# ---------------------------------------------------------------------------
# Part 2: End-to-end integration with a real NetworkXBackend
# ---------------------------------------------------------------------------


class TestIntegrationEndToEnd:
    """Full pipeline integration tests using NetworkXBackend with fixture data."""

    # Entity names for the integration fixture are free of underscores so
    # QuerySanitizer normalisation (underscore → space) does not break
    # substring matching at query time.
    _TIMING_ISSUE = "TimingIssue"
    _CLOCK_FIX = "ClockGatingFix"
    _UART_SPEC = "UARTSpec"

    @pytest.fixture
    def real_backend(self):
        """NetworkXBackend pre-loaded with a small 3-entity, 2-edge graph."""
        from src.knowledge_graph.backends.networkx_backend import NetworkXBackend

        backend = NetworkXBackend()
        entities = [
            Entity(name=self._TIMING_ISSUE, type="KnownIssue"),
            Entity(name=self._CLOCK_FIX, type="DesignDecision"),
            Entity(name=self._UART_SPEC, type="Specification"),
        ]
        backend.upsert_entities(entities)
        triples = [
            Triple(
                subject=self._TIMING_ISSUE,
                predicate="design_decision_for",
                object=self._CLOCK_FIX,
                source="test",
            ),
            Triple(
                subject=self._CLOCK_FIX,
                predicate="specified_by",
                object=self._UART_SPEC,
                source="test",
            ),
        ]
        backend.upsert_triples(triples)
        return backend

    def test_end_to_end_typed_with_paths(self, real_backend):
        """Full pipeline: typed traversal + path matching + context formatting."""
        from src.knowledge_graph.common.types import KGConfig

        config = KGConfig(
            enable_graph_context_injection=True,
            retrieval_edge_types=["design_decision_for", "specified_by"],
            retrieval_path_patterns=[["design_decision_for", "specified_by"]],
            graph_context_token_budget=500,
        )
        expander = GraphQueryExpander(backend=real_backend, config=config)
        result = expander.expand(self._TIMING_ISSUE)

        assert isinstance(result, ExpansionResult)
        # At least one expansion term should be produced
        assert len(result.terms) > 0
        # Graph context must be populated when injection is on and entities matched
        assert result.graph_context != ""
        # Context block should contain a section header of some kind
        ctx_lower = result.graph_context.lower()
        assert (
            "graph context" in ctx_lower
            or "entity" in ctx_lower
            or "relationship" in ctx_lower
        ), f"Expected graph context header, got: {result.graph_context[:200]!r}"

    def test_end_to_end_disabled(self, real_backend):
        """Feature disabled: no graph context returned; backward-compat iteration works."""
        from src.knowledge_graph.common.types import KGConfig

        config = KGConfig(enable_graph_context_injection=False)
        expander = GraphQueryExpander(backend=real_backend, config=config)
        result = expander.expand(self._TIMING_ISSUE)

        assert isinstance(result, ExpansionResult)
        assert result.graph_context == ""
        # Backward-compat: result is iterable
        terms = list(result)
        assert isinstance(terms, list)
