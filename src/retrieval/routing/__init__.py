# @summary
# Stable public facade for the RAPTOR-lite document-routing package.
# Re-exports the shared protocol glossary, the lexical comparison-intent
# detector, term canonicalization, the typed routing contracts, and the C3
# gated decomposition orchestrator (decompose_query) with its failure signal
# (DecompositionError). Later slices add route_documents — the facade is
# intentionally additive and tolerant of names not yet existing.
# Exports: PROTOCOL_GLOSSARY, detect_comparison_intent, canonicalize_terms,
#          RoutingResult, DecompositionResult, DocCard, decompose_query,
#          DecompositionError
# Deps: src.retrieval.routing.glossary, src.retrieval.routing.schemas,
#       src.retrieval.routing.decomposition
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
  decomposition orchestrator and its failure signal, from :mod:`.decomposition`).

Later slices (Pillar B router) will *add* ``route_documents`` here; keep this
facade additive.
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
]
