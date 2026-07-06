"""Offline tests for the multi-mode basket harness (Phase 4 gate tooling).

Hermetic: no network, no models. Covers (1) the mode->request overlays produce
valid QueryRequests (the mutual-exclusion validators do not trip), (2) the
collector rejects unknown modes before any I/O, and (3) the pure scoring /
aggregation / gate logic with deterministic injected scorers.
"""
from __future__ import annotations

import pytest

from evals.turn_loop import run_basket, score_basket


# --------------------------------------------------------------------------- #
# Mode overlays are valid, unambiguous requests
# --------------------------------------------------------------------------- #

def test_every_mode_overlay_builds_a_valid_request():
    """Each MODES overlay must construct a QueryRequest without tripping the
    mutual-exclusion validators — otherwise the live run would 422."""
    from server.schemas import QueryRequest

    for mode, overlay in run_basket.MODES.items():
        req = QueryRequest(query="what is MBIST?", **overlay)
        assert req.query
        # The overlay must force exactly the intended orchestrator on.
        if mode == "turn_loop":
            assert req.turn_loop is True
        if mode == "linear":
            assert req.turn_loop is False and req.agentic_retrieval is False
            assert req.deep_research is False and req.tree_retrieval is False
        if mode == "deep_research":
            assert req.deep_research is True
        if mode == "tree":
            assert req.tree_retrieval is True
        if mode == "agentic":
            assert req.agentic_retrieval is True


def test_default_compare_modes_are_known():
    assert set(run_basket.DEFAULT_COMPARE_MODES) <= set(run_basket.MODES)
    assert "turn_loop" in run_basket.DEFAULT_COMPARE_MODES


def test_collect_rejects_unknown_mode_before_io():
    with pytest.raises(ValueError, match="unknown mode"):
        run_basket.collect("http://unused", modes=["bogus"], basket={"queries": []})


# --------------------------------------------------------------------------- #
# Per-row scoring picks the axis the query implies
# --------------------------------------------------------------------------- #

def _row(qtype, mode, behavior, *, answer="some answer", route=None,
         anchors=None, ttd=5.0, error=None):
    return {
        "qid": f"{qtype}-{mode}", "qtype": qtype, "mode": mode, "query": "q",
        "route": route, "ttd_s": ttd, "answer": answer, "error": error,
        "spec": {"anchor_terms": anchors or [], "expected_behavior": behavior},
    }


def test_answer_row_uses_quality_fn():
    rows = [_row("factoid", "turn_loop", "answer", anchors=["x"])]
    scored = score_basket.score_rows(
        rows, quality_fn=lambda q, a, k: 0.73, refusal_fn=lambda q, a: True)
    assert scored[0]["kind"] == "answer"
    assert scored[0]["metric"] == 0.73


def test_refuse_row_uses_refusal_fn():
    rows = [_row("adversarial", "linear", "refuse_or_state_unknown")]
    good = score_basket.score_rows(rows, quality_fn=lambda *a: 0.0,
                                   refusal_fn=lambda q, a: True)
    bad = score_basket.score_rows(rows, quality_fn=lambda *a: 0.0,
                                  refusal_fn=lambda q, a: False)
    assert good[0]["kind"] == "refuse" and good[0]["metric"] == 1.0
    assert bad[0]["metric"] == 0.0


def test_clarify_row_scored_on_question_shape():
    by_route = _row("clarify", "turn_loop", "clarify", answer="anything", route="clarify")
    by_qmark = _row("clarify", "linear", "clarify", answer="Which block do you mean?")
    neither = _row("clarify", "agentic", "clarify", answer="The setup is done via X.")
    scored = score_basket.score_rows([by_route, by_qmark, neither],
                                     quality_fn=lambda *a: 0.0,
                                     refusal_fn=lambda *a: False)
    assert [s["metric"] for s in scored] == [1.0, 1.0, 0.0]


# --------------------------------------------------------------------------- #
# Aggregation + gate (pure)
# --------------------------------------------------------------------------- #

def _scored(mode, qtype, metric, ttd=5.0, errored=False):
    return {"qid": f"{qtype}-{mode}", "qtype": qtype, "mode": mode,
            "kind": "answer", "metric": metric, "answered": True,
            "errored": errored, "ttd_s": ttd}


def test_aggregate_means_per_mode_and_qtype():
    scored = [
        _scored("turn_loop", "factoid", 0.8), _scored("turn_loop", "factoid", 0.6),
        _scored("linear", "factoid", 0.5),
    ]
    agg = score_basket.aggregate(scored)
    assert agg["by_mode_qtype"]["turn_loop"]["factoid"] == pytest.approx(0.7)
    assert agg["by_mode_qtype"]["linear"]["factoid"] == pytest.approx(0.5)
    assert agg["per_mode"]["turn_loop"]["overall"] == pytest.approx(0.7)


def test_gate_pass_when_turn_loop_at_parity():
    scored = [_scored("turn_loop", "factoid", 0.80), _scored("linear", "factoid", 0.82)]
    verdict = score_basket.gate(score_basket.aggregate(scored))
    # 0.80 >= 0.82 - 0.10 noise floor -> PASS
    assert verdict["verdict"] == "PASS"
    assert verdict["regressions"] == []


def test_gate_fails_on_real_regression():
    scored = [_scored("turn_loop", "howto", 0.40), _scored("deep_research", "howto", 0.90)]
    verdict = score_basket.gate(score_basket.aggregate(scored))
    assert verdict["verdict"] == "FAIL"
    assert verdict["regressions"][0]["qtype"] == "howto"
    assert verdict["regressions"][0]["best_legacy_mode"] == "deep_research"


def test_gate_latency_warning_does_not_fail_quality():
    scored = [
        _scored("turn_loop", "factoid", 0.9, ttd=30.0),
        _scored("linear", "factoid", 0.9, ttd=5.0),
    ]
    verdict = score_basket.gate(score_basket.aggregate(scored))
    assert verdict["verdict"] == "PASS"          # quality parity holds
    assert verdict["latency_warnings"]           # but 30s vs 5s is flagged
    assert verdict["latency_warnings"][0]["factor"] == pytest.approx(6.0)


def test_gate_na_without_turn_loop_rows():
    scored = [_scored("linear", "factoid", 0.5)]
    assert score_basket.gate(score_basket.aggregate(scored))["verdict"] == "N/A"


def test_render_smoke():
    scored = [_scored("turn_loop", "factoid", 0.8), _scored("linear", "factoid", 0.7)]
    agg = score_basket.aggregate(scored)
    out = score_basket.render(agg, score_basket.gate(agg))
    assert "GATE" in out and "factoid" in out and "turn_loop" in out
