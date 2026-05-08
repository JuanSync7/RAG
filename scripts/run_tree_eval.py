"""Run the tree-retrieval eval against a fixture, optionally with a real reranker.

Usage::

    # Lexical fallback (fast, no external deps)
    python scripts/run_tree_eval.py --fixture tests/fixtures/opentitan_uart

    # Real cross-encoder via TEI HTTP endpoint
    python scripts/run_tree_eval.py \
        --fixture tests/fixtures/opentitan_uart \
        --reranker tei \
        --tei-url http://localhost:8082

The script prints a per-qtype gate report (recall@5 / nDCG@10 with deltas)
plus per-query traces. Returns 0 if the design-doc gates pass (QFS Δ ≥ 5pp,
factoid no regression), 1 otherwise.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a script from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.fixtures import load_fixture  # noqa: E402
from src.eval.tree_retrieval_eval import (  # noqa: E402
    TEIRerankerAdapter,
    build_simulated_retriever,
    run_eval,
)


GATE_QFS_RECALL_DELTA = 0.05
GATE_QFS_NDCG_DELTA = 0.05
GATE_FACTOID_FLOOR_DELTA = -1e-9


def _print_qtype_block(qtype: str, fx, retriever, k_recall: int, k_ndcg: int):
    queries = [q for q in fx.queries if q.qtype == qtype]
    if not queries:
        return None, None
    gold = {q.qid: fx.gold[q.qid] for q in queries}
    qdicts = [q.__dict__ for q in queries]
    off = run_eval(retriever=retriever, queries=qdicts, gold=gold,
                   tree_enabled=False, k_recall=k_recall, k_ndcg=k_ndcg)
    on = run_eval(retriever=retriever, queries=qdicts, gold=gold,
                  tree_enabled=True, k_recall=k_recall, k_ndcg=k_ndcg)
    print(f"\n  {qtype.upper():>8} | recall@{k_recall}  off={off.recall_at_k:.3f}"
          f"  on={on.recall_at_k:.3f}  Δ={on.recall_at_k-off.recall_at_k:+.3f}")
    print(f"  {qtype.upper():>8} | nDCG@{k_ndcg}  off={off.ndcg_at_k:.3f}"
          f"  on={on.ndcg_at_k:.3f}  Δ={on.ndcg_at_k-off.ndcg_at_k:+.3f}")
    for qid in sorted(off.per_query):
        r_off, r_on = off.per_query[qid]["recall"], on.per_query[qid]["recall"]
        n_off, n_on = off.per_query[qid]["ndcg"], on.per_query[qid]["ndcg"]
        marker = "✓" if r_on > r_off else (" " if r_on == r_off else "✗")
        print(f"           {marker} {qid:<28} "
              f"recall: {r_off:.3f}→{r_on:.3f}  nDCG: {n_off:.3f}→{n_on:.3f}")
    return off, on


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", required=True, type=Path)
    ap.add_argument("--reranker", choices=["lexical", "tei"], default="lexical")
    ap.add_argument("--tei-url", default="http://localhost:8082")
    ap.add_argument("--k-recall", type=int, default=5)
    ap.add_argument("--k-ndcg", type=int, default=10)
    args = ap.parse_args()

    fx = load_fixture(args.fixture)
    n_chunks = sum(1 for _ in fx.all_chunks())
    print(f"Fixture: {args.fixture} — {n_chunks} chunks across {len(fx.documents)} docs"
          f"  ({len(fx.queries)} queries)")

    rer = None
    rer_label = "lexical (text_jaccard + 0.5*heading_jaccard)"
    if args.reranker == "tei":
        rer = TEIRerankerAdapter(base_url=args.tei_url)
        rer_label = f"TEI cross-encoder ({args.tei_url}/rerank)"
    print(f"Reranker: {rer_label}")

    retriever = build_simulated_retriever(fx, reranker=rer)

    qfs_off, qfs_on = _print_qtype_block("qfs", fx, retriever, args.k_recall, args.k_ndcg)
    fac_off, fac_on = _print_qtype_block("factoid", fx, retriever, args.k_recall, args.k_ndcg)

    print("\nGate (per TREE_RETRIEVAL_DESIGN.md §8):")
    fail = False
    if qfs_off is not None:
        d_recall = qfs_on.recall_at_k - qfs_off.recall_at_k
        d_ndcg = qfs_on.ndcg_at_k - qfs_off.ndcg_at_k
        ok_r = d_recall >= GATE_QFS_RECALL_DELTA
        ok_n = d_ndcg >= GATE_QFS_NDCG_DELTA
        print(f"  QFS recall@{args.k_recall} Δ={d_recall:+.3f} "
              f"(gate ≥ +{GATE_QFS_RECALL_DELTA:.2f}) {'PASS' if ok_r else 'FAIL'}")
        print(f"  QFS nDCG@{args.k_ndcg}   Δ={d_ndcg:+.3f} "
              f"(gate ≥ +{GATE_QFS_NDCG_DELTA:.2f}) {'PASS' if ok_n else 'FAIL'}")
        fail = fail or not (ok_r and ok_n)
    if fac_off is not None:
        d_recall = fac_on.recall_at_k - fac_off.recall_at_k
        ok_r = d_recall >= GATE_FACTOID_FLOOR_DELTA
        print(f"  Factoid recall@{args.k_recall} Δ={d_recall:+.3f} "
              f"(gate ≥ {GATE_FACTOID_FLOOR_DELTA:.0e}) {'PASS' if ok_r else 'FAIL'}")
        fail = fail or not ok_r

    if rer is not None:
        rer.close()
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
