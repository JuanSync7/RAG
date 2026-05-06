# @summary
# Phase 2b sibling KG ingest package — independent consumer of CleanDocumentStore.
# Exports: KGPhase2bRunner
# Deps: src.ingest.kg.phase2b
# @end-summary
"""Phase 2b KG ingest: sibling consumer of CleanDocumentStore.

The embedding pipeline (Phase 2a) and the KG ingest (Phase 2b) both read
from the same Phase 1 → Phase 2 boundary (``CleanDocumentStore``) and write
to disjoint sinks. Splitting them here is the architectural prep for the
KGWeave repo extraction.
"""

from src.ingest.kg.phase2b import KGPhase2bRunner

__all__ = ["KGPhase2bRunner"]
