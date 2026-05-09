# @summary
# Temporal custom search attribute keys for the RAG query workflow. Centralised
# so worker, workflow, and registration script use the same names/types. These
# keys must be registered with the Temporal cluster (see
# ops/temporal/register_search_attributes.py) before workflows can upsert them.
# Exports: DR_ITERATIONS, DR_LLM_CALLS, DR_EARLY_STOPPED, DR_TOPIC_COUNT,
# DR_ENABLED, ALL_DR_KEYS
# Deps: temporalio
# @end-summary
"""Temporal custom search attribute keys for Deep Research observability.

Operators query DR runs in the Temporal UI / ``tctl`` via these attributes,
e.g. ``DRIterations > 5`` to find runs that looped a lot.
"""

from __future__ import annotations

from temporalio.common import SearchAttributeKey


# Number of sufficiency-check iterations the orchestrator drove.
DR_ITERATIONS = SearchAttributeKey.for_int("DRIterations")

# Total LLM calls (sufficiency + decomposition) consumed by the run.
DR_LLM_CALLS = SearchAttributeKey.for_int("DRLLMCalls")

# True when the orchestrator returned without running a multi-topic
# decomposition (e.g. sufficient on round 0, or DR was requested but the
# sanitizer / early-exit prevented the loop from starting).
DR_EARLY_STOPPED = SearchAttributeKey.for_bool("DREarlyStopped")

# Number of topic pools in the result. >=1 always; >1 indicates a fan-out.
DR_TOPIC_COUNT = SearchAttributeKey.for_int("DRTopicCount")

# True when the request had ``deep_research=True``. Lets operators filter out
# baseline runs without inspecting the full input.
DR_ENABLED = SearchAttributeKey.for_bool("DREnabled")


ALL_DR_KEYS = (
    DR_ITERATIONS,
    DR_LLM_CALLS,
    DR_EARLY_STOPPED,
    DR_TOPIC_COUNT,
    DR_ENABLED,
)


__all__ = [
    "DR_ITERATIONS",
    "DR_LLM_CALLS",
    "DR_EARLY_STOPPED",
    "DR_TOPIC_COUNT",
    "DR_ENABLED",
    "ALL_DR_KEYS",
]
