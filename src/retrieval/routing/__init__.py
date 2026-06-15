# @summary
# Stable public facade for the RAPTOR-lite document-routing package.
# Re-exports the shared protocol glossary, the lexical comparison-intent
# detector, term canonicalization, and the typed routing contracts. Later
# slices add route_documents / decompose_query — the facade is intentionally
# additive and tolerant of those not yet existing.
# Exports: PROTOCOL_GLOSSARY, detect_comparison_intent, canonicalize_terms,
#          RoutingResult, DecompositionResult, DocCard
# Deps: src.retrieval.routing.glossary, src.retrieval.routing.schemas
# @end-summary
"""Public import surface for ``src.retrieval.routing``.

This is a thin facade (per project convention): it imports nothing heavy and
exposes only stable names. Downstream slices import from here, not from the
internal modules, so the internal layout can evolve.

Currently exported (S0c):

* :data:`PROTOCOL_GLOSSARY`, :func:`detect_comparison_intent`,
  :func:`canonicalize_terms` (from :mod:`.glossary`);
* :class:`RoutingResult`, :class:`DecompositionResult`, :class:`DocCard`
  (from :mod:`.schemas`).

Later slices (Pillar B router, Pillar C decomposition) will *add*
``route_documents`` and ``decompose_query`` here; keep this facade additive.
"""
from __future__ import annotations

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
]
