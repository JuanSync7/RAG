# @summary
# Public surface of the eval runner subpackage.
# Exports: IngestPlan, IngestReport, execute_plan, plan_pack_ingest
# Deps: .plan (pure), .report (frozen dataclass), .execute (ingest +
#       vector_db; first live-Weaviate code path in the runner).
# @end-summary
"""Eval runner — pack-to-ingest orchestration helpers.

P3.0 contributes the pure-logic ``plan_pack_ingest`` / ``IngestPlan``.
P3 layers on ``execute_plan`` + ``IngestReport`` for the first live
Weaviate-backed run.
"""
from __future__ import annotations

from .execute import execute_plan
from .plan import IngestPlan, plan_pack_ingest
from .report import IngestReport

__all__ = [
    "IngestPlan",
    "IngestReport",
    "execute_plan",
    "plan_pack_ingest",
]
