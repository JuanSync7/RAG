"""Unit tests for ``src.retrieval.common.terminal.decide_terminal``.

These tests cover the centralized ask_user decision function. The intent is
exhaustive coverage of the (outcomes x deep_research x has_results x
sanitizer_verdict x confidence_ask_user) cube — every branch the previous
8 scattered early-return sites used to encode is one parametrized case
here.
"""
from __future__ import annotations

import pytest

from src.retrieval.common import (
    AskUserReason,
    StageOutcome,
    StageStatus,
    TerminalDecision,
    decide_terminal,
)


def test_imports_resolve_and_enum_has_five_members():
    assert {r.value for r in AskUserReason} == {
        "sanitizer_reject",
        "injection_blocked",
        "vague_query",
        "budget_exhausted",
        "no_results",
    }


def test_sanitizer_reject_wins_over_everything_else():
    d = decide_terminal(
        outcomes=[],
        deep_research=True,            # DR cannot override
        has_results=True,
        sanitizer_verdict="reject",
    )
    assert d.action == "ask_user"
    assert d.ask_user_reason == AskUserReason.SANITIZER_REJECT


def test_injection_blocked_when_input_rails_errored():
    d = decide_terminal(
        outcomes=[StageOutcome("input_rails", StageStatus.ERROR, "injection")],
        deep_research=False,
        has_results=False,
        sanitizer_verdict="reject",
    )
    assert d.action == "ask_user"
    assert d.ask_user_reason == AskUserReason.INJECTION_BLOCKED


def test_canned_passes_through():
    d = decide_terminal(
        outcomes=[],
        deep_research=False,
        has_results=False,
        sanitizer_verdict="canned",
    )
    assert d.action == "canned"
    assert d.ask_user_reason is None


def test_vague_query_off_dr_asks_user():
    d = decide_terminal(
        outcomes=[],
        deep_research=False,
        has_results=False,
        confidence_ask_user=True,
    )
    assert d.action == "ask_user"
    assert d.ask_user_reason == AskUserReason.VAGUE_QUERY


def test_vague_query_under_dr_is_bypassed():
    d = decide_terminal(
        outcomes=[],
        deep_research=True,
        has_results=True,
        confidence_ask_user=True,
    )
    assert d.action == "answer"


def test_budget_exhausted_with_results_returns_search():
    d = decide_terminal(
        outcomes=[StageOutcome("hybrid_search", StageStatus.BUDGET_EXHAUSTED)],
        deep_research=False,
        has_results=True,
    )
    assert d.action == "search"
    assert d.ask_user_reason is None


def test_budget_exhausted_no_results_off_dr_asks_user():
    d = decide_terminal(
        outcomes=[StageOutcome("query_processing", StageStatus.BUDGET_EXHAUSTED)],
        deep_research=False,
        has_results=False,
    )
    assert d.action == "ask_user"
    assert d.ask_user_reason == AskUserReason.BUDGET_EXHAUSTED


def test_budget_exhausted_under_dr_is_bypassed():
    d = decide_terminal(
        outcomes=[StageOutcome("kg_expansion", StageStatus.BUDGET_EXHAUSTED)],
        deep_research=True,
        has_results=True,
    )
    assert d.action == "answer"


def test_no_results_off_dr_asks_user():
    d = decide_terminal(
        outcomes=[StageOutcome("hybrid_search", StageStatus.EMPTY)],
        deep_research=False,
        has_results=False,
    )
    assert d.action == "ask_user"
    assert d.ask_user_reason == AskUserReason.NO_RESULTS


def test_no_results_under_dr_is_bypassed():
    # DR has its own pools; trust them.
    d = decide_terminal(
        outcomes=[],
        deep_research=True,
        has_results=False,
    )
    assert d.action == "answer"


def test_happy_path_returns_answer():
    d = decide_terminal(
        outcomes=[StageOutcome("hybrid_search", StageStatus.OK)],
        deep_research=False,
        has_results=True,
    )
    assert d.action == "answer"
    assert d.ask_user_reason is None


def test_degraded_flag_propagates():
    d = decide_terminal(
        outcomes=[],
        deep_research=False,
        has_results=True,
        degraded=True,
    )
    assert d.action == "answer"
    assert d.degraded is True


@pytest.mark.parametrize(
    "verdict,reason",
    [
        ("reject", AskUserReason.SANITIZER_REJECT),
    ],
)
def test_sanitizer_reasons_parametrized(verdict, reason):
    d = decide_terminal(
        outcomes=[],
        deep_research=False,
        has_results=True,
        sanitizer_verdict=verdict,
    )
    assert d.action == "ask_user"
    assert d.ask_user_reason == reason
