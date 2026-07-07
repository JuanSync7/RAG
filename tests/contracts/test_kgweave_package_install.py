# @summary
# Smoke test that the kgweave Python package is installed and exposes only
# the public façade surface RagWeave is allowed to consume:
#   - kgweave (top-level: set_default_llm_factory)
#   - kgweave.client (queries)
#   - kgweave.admin (lifecycle)
#   - kgweave.contracts (Temporal handles + Pydantic types)
# Internal modules (kgweave.core, kgweave.knowledge_graph, kgweave.service)
# are NOT validated here — they're internal implementation and may change.
# Deps: pytest, kgweave (installed dep)
# @end-summary
"""Verify kgweave package is consumable as a normal Python dependency."""

from __future__ import annotations


def test_kgweave_package_importable() -> None:
    import kgweave  # noqa: F401


def test_top_level_facade_surface() -> None:
    """Top-level dependency-injection hook for the host application."""
    from kgweave import set_default_llm_factory  # noqa: F401


def test_contracts_surface_available() -> None:
    from kgweave.contracts import (  # noqa: F401
        CONTRACT_VERSION,
        KG_TASK_QUEUE,
        KG_PHASE2B_ACTIVITY,
        KGIngestRequest,
        KGIngestResult,
    )


def test_client_facade_available() -> None:
    """Public read-side façade — the only KG entry point for retrieval code."""
    from kgweave.client import (  # noqa: F401
        get_client,
        KGQueryService,
        HTTPKGQueryService,
    )


def test_admin_facade_available() -> None:
    """Public lifecycle façade — used by RagWeave's GC and validation paths."""
    from kgweave.admin import (  # noqa: F401
        get_admin_backend,
        GraphStorageBackend,
    )
