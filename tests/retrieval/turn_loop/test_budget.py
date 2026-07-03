# @summary
# Tests for TurnBudget.from_settings effort scaling: balanced (default) is
# byte-for-byte the raw settings budget; fast shrinks and thorough grows
# max_actions/max_llm_calls via the RAG_TURN_LOOP_EFFORT_*_SCALE keys while the
# wall clock (a fixed safety ceiling) is never scaled; floors keep the budget
# runnable at aggressive scales.
# @end-summary
"""Tests for ``TurnBudget.from_settings(effort=...)`` (effort dial)."""

from __future__ import annotations

import config.settings as settings
from src.retrieval.pipeline.turn_loop.schemas import TurnBudget


def _base(monkeypatch, **overrides):
    """Pin the budget-relevant settings to known values for scaling math."""
    values = {
        "RAG_TURN_LOOP_MAX_ACTIONS": 6,
        "RAG_TURN_LOOP_MAX_LLM_CALLS": 16,
        "RAG_TURN_LOOP_EFFORT_FAST_SCALE": 0.5,
        "RAG_TURN_LOOP_EFFORT_THOROUGH_SCALE": 1.5,
    }
    values.update(overrides)
    for key, val in values.items():
        monkeypatch.setattr(settings, key, val, raising=False)


def test_balanced_default_matches_raw_settings(monkeypatch):
    _base(monkeypatch)
    balanced = TurnBudget.from_settings()  # default effort
    explicit = TurnBudget.from_settings(effort="balanced")
    assert balanced.max_actions == 6
    assert balanced.max_llm_calls == 16
    assert balanced == explicit  # frozen dataclass value equality


def test_fast_effort_shrinks_work_budgets(monkeypatch):
    _base(monkeypatch)
    fast = TurnBudget.from_settings(effort="fast")
    assert fast.max_actions == 3  # round(6 * 0.5)
    assert fast.max_llm_calls == 8  # round(16 * 0.5)


def test_thorough_effort_grows_work_budgets(monkeypatch):
    _base(monkeypatch)
    thorough = TurnBudget.from_settings(effort="thorough")
    assert thorough.max_actions == 9  # round(6 * 1.5)
    assert thorough.max_llm_calls == 24  # round(16 * 1.5)


def test_wall_clock_is_never_scaled_by_effort(monkeypatch):
    _base(monkeypatch)
    base_wall = TurnBudget.from_settings(effort="balanced").wall_clock_ms
    assert TurnBudget.from_settings(effort="fast").wall_clock_ms == base_wall
    assert TurnBudget.from_settings(effort="thorough").wall_clock_ms == base_wall


def test_unknown_effort_is_treated_as_balanced(monkeypatch):
    _base(monkeypatch)
    assert (
        TurnBudget.from_settings(effort="turbo")
        == TurnBudget.from_settings(effort="balanced")
    )


def test_aggressive_fast_scale_floors_keep_budget_runnable(monkeypatch):
    # A tiny scale must not zero out the ceilings (loop would never act).
    _base(monkeypatch, RAG_TURN_LOOP_MAX_ACTIONS=1, RAG_TURN_LOOP_EFFORT_FAST_SCALE=0.1)
    fast = TurnBudget.from_settings(effort="fast")
    assert fast.max_actions >= 1
    assert fast.max_llm_calls >= 2
