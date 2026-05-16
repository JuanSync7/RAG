# @summary
# E2E gate test exercising the eval harness with rerank fusion on/off.
# Confirms fusion changes pool→post-rerank ranking on the OpenTitan UART
# fixture in the expected directions (QFS lift, factoid no regression).
# Deps: pytest, src.eval.fixtures, src.eval.tree_retrieval_eval
# @end-summary
"""End-to-end gate for R1 fusion in the tree-retrieval simulator."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.eval.fixtures import load_fixture
from src.eval.tree_retrieval_eval import build_simulated_retriever, run_eval


FIXTURE = Path("tests/fixtures/opentitan_uart")


@pytest.fixture(scope="module")
def fx():
    if not FIXTURE.exists():
        pytest.skip(f"Fixture {FIXTURE} not present in this checkout")
    return load_fixture(FIXTURE)


def _eval(fx, retriever, qtype: str, tree_enabled: bool):
    qs = [q for q in fx.queries if q.qtype == qtype]
    gold = {q.qid: fx.gold[q.qid] for q in qs}
    return run_eval(
        retriever=retriever,
        queries=[q.__dict__ for q in qs],
        gold=gold,
        tree_enabled=tree_enabled,
        k_recall=5, k_ndcg=10,
        compute_pool=False,
    )


def test_fusion_off_matches_legacy_lexical(fx):
    """Fusion-off must reproduce the pre-R1 lexical baseline (regression gate)."""
    r = build_simulated_retriever(fx, apply_fusion=False)
    qfs_off = _eval(fx, r, "qfs", tree_enabled=False)
    assert qfs_off.recall_at_k == pytest.approx(0.304, abs=1e-3)
    fac_off = _eval(fx, r, "factoid", tree_enabled=False)
    assert fac_off.recall_at_k == pytest.approx(0.0, abs=1e-9)


def test_fusion_on_lifts_factoid_recall_via_heading(fx):
    """R1.2: heading-aware boost should lift factoid tree-on recall@5."""
    r_off = build_simulated_retriever(fx, apply_fusion=False)
    r_on = build_simulated_retriever(
        fx, apply_fusion=True,
        fusion_weights={"rrf": 1.0, "heading": 0.15, "anchor": 0.10},
    )
    base = _eval(fx, r_off, "factoid", tree_enabled=True)
    fused = _eval(fx, r_on, "factoid", tree_enabled=True)
    assert fused.recall_at_k > base.recall_at_k, (
        f"factoid recall did not improve with fusion on: "
        f"{base.recall_at_k:.3f} -> {fused.recall_at_k:.3f}"
    )


def test_fusion_on_does_not_regress_qfs_post_rerank(fx):
    """R1 contract: QFS recall@5 must not regress more than 2pp."""
    r_off = build_simulated_retriever(fx, apply_fusion=False)
    r_on = build_simulated_retriever(
        fx, apply_fusion=True,
        fusion_weights={"rrf": 1.0, "heading": 0.15, "anchor": 0.10},
    )
    qfs_off = _eval(fx, r_off, "qfs", tree_enabled=True)
    qfs_on = _eval(fx, r_on, "qfs", tree_enabled=True)
    assert qfs_on.recall_at_k >= qfs_off.recall_at_k - 0.02


def test_fusion_changes_ranking(fx):
    """Sanity: enabling fusion changes the ranked output for at least one
    query (proves the wiring is live, not silent passthrough)."""
    r_off = build_simulated_retriever(fx, apply_fusion=False)
    r_on = build_simulated_retriever(
        fx, apply_fusion=True,
        fusion_weights={"rrf": 1.0, "heading": 0.15, "anchor": 0.10},
    )
    differs = False
    for q in fx.queries:
        a = list(r_off(q.text, tree_enabled=True))
        b = list(r_on(q.text, tree_enabled=True))
        if a != b:
            differs = True
            break
    assert differs, "fusion did not alter ranking for any fixture query"
