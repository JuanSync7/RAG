"""Contract tests for Temporal custom search attribute keys.

Locks the canonical names and types so a typo or reordering shows up as a
test failure rather than a silent UI regression. The names below MUST match
what the registration script (ops/temporal/register_search_attributes.py)
declares to the cluster.
"""

from __future__ import annotations

from temporalio.common import SearchAttributeIndexedValueType as IVT

from server.search_attributes import (
    ALL_DR_KEYS,
    DR_EARLY_STOPPED,
    DR_ENABLED,
    DR_ITERATIONS,
    DR_LLM_CALLS,
    DR_TOPIC_COUNT,
)


def test_dr_iterations_is_int():
    assert DR_ITERATIONS.name == "DRIterations"
    assert DR_ITERATIONS.indexed_value_type == IVT.INT


def test_dr_llm_calls_is_int():
    assert DR_LLM_CALLS.name == "DRLLMCalls"
    assert DR_LLM_CALLS.indexed_value_type == IVT.INT


def test_dr_early_stopped_is_bool():
    assert DR_EARLY_STOPPED.name == "DREarlyStopped"
    assert DR_EARLY_STOPPED.indexed_value_type == IVT.BOOL


def test_dr_topic_count_is_int():
    assert DR_TOPIC_COUNT.name == "DRTopicCount"
    assert DR_TOPIC_COUNT.indexed_value_type == IVT.INT


def test_dr_enabled_is_bool():
    assert DR_ENABLED.name == "DREnabled"
    assert DR_ENABLED.indexed_value_type == IVT.BOOL


def test_all_keys_collected():
    assert len(ALL_DR_KEYS) == 5
    names = {k.name for k in ALL_DR_KEYS}
    assert names == {
        "DRIterations", "DRLLMCalls", "DREarlyStopped",
        "DRTopicCount", "DREnabled",
    }
