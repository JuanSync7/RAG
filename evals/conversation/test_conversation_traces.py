# @summary
# Pytest entry for the multi-turn conversation eval (marker: eval_conversation,
# offline mode). Asserts the golden-fixture expectations against the real
# run_turn_loop driven through scripted fakes: per-turn action containment,
# terminal accuracy, anchor-doc retention/drift, expected-chunk recall,
# clarify hints, genuine cross-turn context transfer, and the
# skip-if-unpopulated convention for live-only fields (clarify quality).
# Exports: (tests only)
# Deps: pytest, evals.conversation.conftest fixtures,
#       evals.conversation.harness (report contracts)
# @end-summary
"""Offline multi-turn conversation eval: golden traces through the real loop.

Run with::

    uv run --extra dev python -m pytest evals/conversation -m eval_conversation -q

The offline drive mode needs NO infrastructure — the fixtures script every
LLM/retrieval response, while ``run_turn_loop`` itself (imported from
``src.retrieval.pipeline.turn_loop``) supplies all control flow, budgeting,
gating and context consumption under test.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.eval_conversation


def _report_for(offline_report, conversation_id: str):
    """Fetch one conversation's report from the shared suite run."""
    for conversation in offline_report.conversations:
        if conversation.id == conversation_id:
            return conversation
    pytest.fail(f"conversation {conversation_id!r} missing from the suite report")


def _fail_detail(report) -> str:
    """Render a compact failure context: per-turn outcomes + unscripted calls."""
    lines = []
    for trace, evaluation in zip(report.turns, report.evaluations):
        lines.append(
            f"turn {trace.turn_index}: terminal={trace.terminal!r} "
            f"actions={trace.actions_taken} stop={trace.stop_reason!r} "
            f"error={trace.error!r} unscripted={trace.unscripted_calls} "
            f"eval={evaluation.details}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Suite-level metrics
# ---------------------------------------------------------------------------


def test_no_turn_errors(offline_report):
    """Every scripted turn must complete without an orchestrator exception."""
    errors = [
        (conversation.id, trace.turn_index, trace.error)
        for conversation in offline_report.conversations
        for trace in conversation.turns
        if trace.error is not None
    ]
    assert not errors, f"turn errors: {errors}"


def test_action_accuracy(offline_report):
    """All actions each turn takes are within its expected.actions_allowed."""
    assert offline_report.aggregate["action_accuracy"] == 1.0, "\n".join(
        _fail_detail(c) for c in offline_report.conversations
    )


def test_terminal_accuracy(offline_report):
    """Each turn ends in the expected terminal (answer vs clarify)."""
    assert offline_report.aggregate["terminal_accuracy"] == 1.0, "\n".join(
        _fail_detail(c) for c in offline_report.conversations
    )


def test_anchor_retention_no_drift(offline_report):
    """Anchor-doc retention holds on every turn that declares it (drift 0)."""
    assert offline_report.aggregate["anchor_retention_rate"] == 1.0
    assert offline_report.aggregate["anchor_drift_rate"] == 0.0


def test_expected_chunk_recall(offline_report):
    """Turns with expected-chunk matchers surface all of them in the pool."""
    assert offline_report.aggregate["avg_chunk_recall"] == 1.0


def test_trace_events_populated(offline_report):
    """Every turn carries a typed event trace (100% observability, design 8)."""
    for conversation in offline_report.conversations:
        for trace in conversation.turns:
            assert trace.events, (
                f"{conversation.id} turn {trace.turn_index} emitted no events:\n"
                f"{_fail_detail(conversation)}"
            )


# ---------------------------------------------------------------------------
# Conversation 1 — verification-environment deepen (the live 2026-06-30 case)
# ---------------------------------------------------------------------------


def test_verif_env_turn2_retains_anchor(offline_report):
    """'tell me more' must keep the turn-1 anchor document in its evidence."""
    report = _report_for(offline_report, "verif-env-deepen")
    turn2 = report.turns[1]
    assert turn2.anchor_docs_retained is True, _fail_detail(report)
    assert "doc-verif-env-setup" in turn2.document_ids()


def test_verif_env_context_transfer(offline_report):
    """Turn 2 must START with the turn-1 chunk refs in its TurnContext.

    This is the memory-transfer contract itself (design section 7): the
    snapshot is taken from the TurnContext object handed to run_turn_loop,
    so a passing check proves the loop was GIVEN the prior grounding — not
    that the harness merely remembered it.
    """
    report = _report_for(offline_report, "verif-env-deepen")
    snapshot = report.turns[1].context_snapshot
    assert snapshot["recent_turns"] >= 1
    assert snapshot["chunk_refs"] >= 1
    assert "doc-verif-env-setup" in snapshot["chunk_ref_document_ids"]


# ---------------------------------------------------------------------------
# Conversation 2 — clarify path
# ---------------------------------------------------------------------------


def test_clarify_turn_emits_question_and_hints(offline_report):
    """The underspecified turn terminates in a clarification with hints."""
    report = _report_for(offline_report, "clarify-underspecified")
    turn1 = report.turns[0]
    assert turn1.terminal == "clarify", _fail_detail(report)
    assert turn1.clarification is not None
    assert turn1.clarification["question"]
    assert len(turn1.clarification["hints"]) >= 1


def test_clarify_resolution_turn_answers(offline_report):
    """Picking a hint on the next turn resolves to a terminal answer."""
    report = _report_for(offline_report, "clarify-underspecified")
    turn2 = report.turns[1]
    assert turn2.terminal == "answer", _fail_detail(report)
    assert turn2.answer


def test_pending_clarification_transfers_to_next_turn(offline_report):
    """The clarification is pending in the TurnContext the next turn starts with."""
    report = _report_for(offline_report, "clarify-underspecified")
    assert report.turns[1].context_snapshot["pending_clarification"] is True


# ---------------------------------------------------------------------------
# Conversation 3 — refine with anchor
# ---------------------------------------------------------------------------


def test_refine_runs_fresh_retrieve_and_keeps_anchor(offline_report):
    """The narrowing turn retrieves FRESH while retaining the named anchor."""
    report = _report_for(offline_report, "refine-with-anchor")
    turn2 = report.turns[1]
    assert "RETRIEVE" in turn2.actions_taken, _fail_detail(report)
    assert turn2.anchor_docs_retained is True
    assert "doc-signoff-overview" in turn2.document_ids()


# ---------------------------------------------------------------------------
# Live-only fields — skip-if-unpopulated convention
# ---------------------------------------------------------------------------


def test_clarify_quality_judge_scored(offline_report):
    """Clarify quality is judge-scored in live mode only; N/A offline."""
    report = _report_for(offline_report, "clarify-underspecified")
    quality = report.turns[0].clarify_quality
    if quality is None:
        pytest.skip("clarify_quality is live/judge-scored only (N/A offline)")
    assert 0.0 <= quality <= 1.0
