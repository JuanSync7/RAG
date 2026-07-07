"""Offline validation of the turn_loop query basket.

Keeps evals/turn_loop/query_basket.yaml well-formed and schema-consistent so it
can be trusted as the end-to-end test source across the turn_loop phases. No live
stack required — pure load + schema checks.
"""
from __future__ import annotations

import pytest

pytest.importorskip("yaml")

from evals.turn_loop.run_basket import (  # noqa: E402
    VALID_QTYPES,
    iter_queries,
    load_basket,
    validate_basket,
)


@pytest.fixture(scope="module")
def basket() -> dict:
    return load_basket()


def test_basket_loads_and_is_nonempty(basket: dict) -> None:
    assert basket.get("meta", {}).get("name") == "turn_loop_query_basket"
    assert len(list(iter_queries(basket))) >= 20


def test_basket_schema_valid(basket: dict) -> None:
    problems = validate_basket(basket)
    assert not problems, "basket schema problems:\n" + "\n".join(problems)


def test_qids_unique(basket: dict) -> None:
    qids = [q["qid"] for q in iter_queries(basket)]
    assert len(qids) == len(set(qids))


def test_core_routing_categories_present(basket: dict) -> None:
    """The basket must exercise every routing path the unified orchestrator owns."""
    qtypes = {q["qtype"] for q in iter_queries(basket)}
    # fast-lane, DECOMPOSE, clarify, refuse, and multi-turn must all be represented.
    for required in ("factoid", "multi_aspect", "clarify", "out_of_corpus", "multi_turn"):
        assert required in qtypes, f"basket missing {required} coverage"
    assert qtypes <= VALID_QTYPES


def test_ground_truth_entries_have_anchor_terms(basket: dict) -> None:
    """Any query we assert a ground-truth answer for must also carry anchor terms
    (so a retrieval-hit check is possible), keeping the two honest together."""
    for q in iter_queries(basket):
        if q.get("ground_truth"):
            assert q.get("anchor_terms"), f"{q['qid']} has ground_truth but no anchor_terms"
