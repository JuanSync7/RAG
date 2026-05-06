# @summary
# Smoke test that the kgweave Python package is installed and exposes the
# read-side API surface RagWeave needs after Step 14. If this fails the
# local-path dep in pyproject.toml is broken or KGWeave drifted.
# Deps: pytest, kgweave (installed dep)
# @end-summary
"""Verify kgweave package is consumable as a normal Python dependency."""

from __future__ import annotations


def test_kgweave_package_importable() -> None:
    import kgweave  # noqa: F401


def test_contracts_surface_available() -> None:
    from kgweave.contracts import (  # noqa: F401
        CONTRACT_VERSION,
        KG_TASK_QUEUE,
        KG_PHASE2B_ACTIVITY,
        KGIngestRequest,
        KGIngestResult,
    )


def test_retrieval_side_symbols_available() -> None:
    """Exact symbols RagWeave's retrieval pipeline calls today."""
    from kgweave.knowledge_graph import (  # noqa: F401
        get_term_index,
        get_graph_backend,
        get_query_expander,
    )
    from kgweave.knowledge_graph.common import KGConfig  # noqa: F401
    from kgweave.core.knowledge_graph import KnowledgeGraphBuilder  # noqa: F401
