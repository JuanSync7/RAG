# @summary
# Public surface of the eval runner subpackage.
# Exports: IngestPlan, plan_pack_ingest
# Deps: stdlib only (re-exports from .plan).
# @end-summary
"""Eval runner — pack-to-ingest orchestration helpers.

Currently exposes the pure-logic ``plan_pack_ingest`` helper (P3.0).
The live ingest invocation lands in P3.
"""
from __future__ import annotations

from .plan import IngestPlan, plan_pack_ingest

__all__ = ["IngestPlan", "plan_pack_ingest"]
