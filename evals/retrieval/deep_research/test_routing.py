"""Routing tests for the confidence/ask_user gate when deep_research is on.

Asserts that messy-but-answerable queries (terse, ungrammatical, no
punctuation) do NOT false-positive into ``action="ask_user"`` when DR is on,
while genuinely under-specified queries still do.

Run with::

    pytest evals/retrieval/deep_research/test_routing.py -m eval_deep_research -v

Marker keeps it out of default CI; the suite needs Weaviate + Ollama up.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

from evals.conftest import load_json_fixture

pytestmark = pytest.mark.eval_deep_research


_FIXTURES_ROOT = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def ask_user_routing_fixture() -> Dict[str, Any]:
    path = _FIXTURES_ROOT / "asic" / "ask_user_routing.json"
    if not path.exists():
        pytest.skip(f"Routing fixture not present: {path}")
    data = load_json_fixture(path)
    if not data.get("queries"):
        pytest.skip("Routing fixture has no queries")
    return data


@pytest.mark.parametrize("query_idx", [0, 1, 2, 3])
def test_dr_routing_per_query(rag_chain, ask_user_routing_fixture, query_idx):
    """Per-query routing assertion.

    Each fixture entry declares ``expected_action`` ("retrieve"|"ask_user").
    With ``deep_research=True``, the actual ``response.action`` must match.
    """
    queries = ask_user_routing_fixture["queries"]
    if query_idx >= len(queries):
        pytest.skip("fixture index out of range")
    q = queries[query_idx]
    expected = q.get("expected_action")
    if not expected:
        pytest.skip(f"{q['id']} declares no expected_action")

    from evals.retrieval.deep_research.harness import _run_single

    _, _, _, action = _run_single(
        rag_chain, q["query"], deep_research=True, top_k=10, routing_mode=True
    )

    if expected == "ask_user":
        assert action == "ask_user", (
            f"{q['id']}: expected ask_user but got action={action!r}. "
            f"Vague-control should still route to ask_user even with DR on."
        )
    else:
        assert action != "ask_user", (
            f"{q['id']}: messy-but-answerable query false-positived into "
            f"ask_user with deep_research=True. Query: {q['query']!r}. "
            f"DR opt-in should bypass the LLM-confidence-loop ask_user gate."
        )


def test_routing_aggregate_action_match_rate(rag_chain, ask_user_routing_fixture):
    """Aggregate gate: deep_research action_match_rate must be 1.0 on this suite."""
    from evals.retrieval.deep_research.harness import run_suite

    report = run_suite(rag_chain, ask_user_routing_fixture, top_k=10)
    rate = (report.aggregate.get("action_match_rate") or {}).get("deep_research")
    assert rate is not None, "no routing samples were recorded"
    assert rate == 1.0, (
        f"deep_research action_match_rate={rate}; some queries routed to the "
        f"wrong action. See report.results for per-query breakdown."
    )
