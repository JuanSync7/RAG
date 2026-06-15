# @summary
# Stable public facade for the RAPTOR-lite document-routing package.
# Re-exports the shared protocol glossary, the lexical comparison-intent
# detector, term canonicalization, the typed routing contracts, the C3 gated
# decomposition orchestrator (decompose_query) with its failure signal
# (DecompositionError), and the B1 Stage-1 card-index router (route_documents).
# Exports: PROTOCOL_GLOSSARY, detect_comparison_intent, canonicalize_terms,
#          RoutingResult, DecompositionResult, DocCard, decompose_query,
#          DecompositionError, route_documents
# Deps: src.retrieval.routing.glossary, src.retrieval.routing.schemas,
#       src.retrieval.routing.decomposition, src.retrieval.routing.router
# @end-summary
"""Public import surface for ``src.retrieval.routing``.

This is a thin facade (per project convention): it imports nothing heavy and
exposes only stable names. Downstream slices import from here, not from the
internal modules, so the internal layout can evolve.

Currently exported:

* :data:`PROTOCOL_GLOSSARY`, :func:`detect_comparison_intent`,
  :func:`canonicalize_terms` (from :mod:`.glossary`);
* :class:`RoutingResult`, :class:`DecompositionResult`, :class:`DocCard`
  (from :mod:`.schemas`);
* :func:`decompose_query`, :class:`DecompositionError` (the C3 gated
  decomposition orchestrator and its failure signal, from :mod:`.decomposition`);
* :func:`route_documents` (the B1 Stage-1 card-index router that turns a query
  embedding into a soft set of routed document ids, from :mod:`.router`).
"""
from __future__ import annotations

from src.retrieval.routing.decomposition import (
    DecompositionError,
    decompose_query,
)
from src.retrieval.routing.glossary import (
    PROTOCOL_GLOSSARY,
    canonicalize_terms,
    detect_comparison_intent,
)
from src.retrieval.routing.router import route_documents
from src.retrieval.routing.schemas import (
    DecompositionResult,
    DocCard,
    RoutingResult,
)

__all__ = [
    "PROTOCOL_GLOSSARY",
    "detect_comparison_intent",
    "canonicalize_terms",
    "RoutingResult",
    "DecompositionResult",
    "DocCard",
    "decompose_query",
    "DecompositionError",
    "route_documents",
]
