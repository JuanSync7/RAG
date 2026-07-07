"""Unit tests for RailMergeGate.merge (priority-ordered routing decisions).

Covers the five-tier priority order: injection reject (1) > toxicity reject (2)
> topic off-topic canned (3) > intent canned (4) > search (5). Includes
priority-ordering teeth (higher priority wins when multiple would route) and
the redacted_query-vs-processed_query fallback for the search path.

No langchain import path is exercised here.
"""
from __future__ import annotations

from types import SimpleNamespace

from src.guardrails.common.merge_gate import RailMergeGate
from src.guardrails.common.schemas import InputRailResult, RailVerdict
from src.guardrails.shared import INTENT_RESPONSES
from src.guardrails.shared.topic_safety import REJECTION_MESSAGE as TOPIC_REJECTION_MESSAGE

INJECTION_MESSAGE = "Your query could not be processed. Please rephrase your question."
TOXICITY_MESSAGE = (
    "Your query contains content that violates our usage policy. Please rephrase."
)


def _qr(processed_query="raw query"):
    return SimpleNamespace(processed_query=processed_query)


def test_injection_reject_wins_over_everything():
    """Priority 1: injection REJECT overrides toxicity reject and a canned intent."""
    rail = InputRailResult(
        injection_verdict=RailVerdict.REJECT,
        toxicity_verdict=RailVerdict.REJECT,
        topic_off_topic=True,
        intent="greeting",
    )
    result = RailMergeGate().merge(_qr(), rail)

    assert result == {"action": "reject", "message": INJECTION_MESSAGE}


def test_toxicity_reject_when_injection_passes():
    """Priority 2: toxicity REJECT (injection PASS) yields the toxicity reject message."""
    rail = InputRailResult(
        injection_verdict=RailVerdict.PASS,
        toxicity_verdict=RailVerdict.REJECT,
        topic_off_topic=True,
        intent="greeting",
    )
    result = RailMergeGate().merge(_qr(), rail)

    assert result == {"action": "reject", "message": TOXICITY_MESSAGE}


def test_topic_off_topic_canned():
    """Priority 3: off-topic (no rejects) returns the topic_safety REJECTION_MESSAGE."""
    rail = InputRailResult(
        injection_verdict=RailVerdict.PASS,
        toxicity_verdict=RailVerdict.PASS,
        topic_off_topic=True,
        intent="rag_search",
    )
    result = RailMergeGate().merge(_qr(), rail)

    assert result == {"action": "canned", "message": TOPIC_REJECTION_MESSAGE}


def test_intent_canned_for_non_rag_search_known_intent():
    """Priority 4: a non-rag_search intent present in INTENT_RESPONSES is canned."""
    rail = InputRailResult(
        injection_verdict=RailVerdict.PASS,
        toxicity_verdict=RailVerdict.PASS,
        topic_off_topic=False,
        intent="greeting",
    )
    result = RailMergeGate().merge(_qr(), rail)

    assert result == {"action": "canned", "message": INTENT_RESPONSES["greeting"]}


def test_rag_search_intent_falls_through_to_search():
    """intent == 'rag_search' is not canned; it falls through to the search path."""
    rail = InputRailResult(
        injection_verdict=RailVerdict.PASS,
        toxicity_verdict=RailVerdict.PASS,
        topic_off_topic=False,
        intent="rag_search",
        redacted_query=None,
    )
    result = RailMergeGate().merge(_qr("the processed q"), rail)

    assert result == {"action": "search", "query": "the processed q"}


def test_search_uses_redacted_query_when_present():
    """Search path prefers redacted_query when it is truthy."""
    rail = InputRailResult(
        injection_verdict=RailVerdict.PASS,
        toxicity_verdict=RailVerdict.PASS,
        topic_off_topic=False,
        intent="rag_search",
        redacted_query="REDACTED text",
    )
    result = RailMergeGate().merge(_qr("original processed"), rail)

    assert result == {"action": "search", "query": "REDACTED text"}


def test_search_falls_back_to_processed_query_when_redacted_falsy():
    """Search path falls back to processed_query when redacted_query is falsy."""
    rail = InputRailResult(
        injection_verdict=RailVerdict.PASS,
        toxicity_verdict=RailVerdict.PASS,
        topic_off_topic=False,
        intent="rag_search",
        redacted_query="",  # falsy
    )
    result = RailMergeGate().merge(_qr("fallback processed"), rail)

    assert result == {"action": "search", "query": "fallback processed"}


def test_topic_wins_over_canned_intent():
    """Priority teeth: topic off-topic (3) wins over a canned intent (4)."""
    rail = InputRailResult(
        injection_verdict=RailVerdict.PASS,
        toxicity_verdict=RailVerdict.PASS,
        topic_off_topic=True,
        intent="greeting",  # would be canned-intent if topic did not win
    )
    result = RailMergeGate().merge(_qr(), rail)

    assert result == {"action": "canned", "message": TOPIC_REJECTION_MESSAGE}
    assert result["message"] != INTENT_RESPONSES["greeting"]
