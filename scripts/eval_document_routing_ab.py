#!/usr/bin/env python3
# @summary
# Live A/B harness for RAPTOR-lite document routing (design §10 validation set).
# Runs each validation query through RAGChain.run() TWICE — once with routing OFF
# and once with routing ON — by toggling the module-level gate constants the
# pipeline reads at call time (mirroring tests/integration/test_document_routing_e2e.py,
# but against the LIVE dev corpus). Compares the RETRIEVED documents (skip_generation),
# which is exactly what the §10 claims are about, so it needs only the embed + rerank
# backends, not generation.
# Intended to be run INSIDE rag-worker-dev (has the routing code, weaviate access, and
# the TEI embed/rerank tunnels):  podman exec rag-worker-dev python scripts/eval_document_routing_ab.py
# Exports: main, run_query, summarize_results, doc_set, evaluate_case
# Deps: src.retrieval.pipeline.rag_chain, config.settings
# @end-summary
"""ON-vs-OFF A/B for document routing on the live corpus (design §10).

This is operational/eval tooling, not part of the request path. It flips
``src.retrieval.pipeline.rag_chain.RAG_DOCUMENT_ROUTING_ENABLED`` and
``config.settings.RAG_DECOMPOSITION_ENABLED`` per run (the same constants the
pipeline consults), so no compose edit / container recreate is needed — the
podman-cp'd code stays in place.
"""
from __future__ import annotations

import argparse
import json
import sys

# Resolved real corpus document_ids (from the dev-stack recon). Each validation
# case lists "expectation groups": the case passes its retrieval bar when, with
# routing ON, at least one document_id from EACH group appears in the results.
AXI_IHI0022E = "53857d43-6686-544b-9f36-bc07323b920d"
CHI_IHI0098B = "ef1ee2a7-0440-5624-8e3b-1d66e42faf2e"
CHI_IHI0050H = "10f27a1b-18ef-57ad-8410-232622129687"
NEEDLE_ARC = "39401a93-937c-55b2-a941-1ac016d07b6d"
PMU_SPEC = "46949dd3-1da2-53c0-86d3-f18e13052826"
VITEC_G60 = "b9f72da9-e462-53f8-9314-327b5dfef126"

# kind: "improve" => routing ON should surface the authoritative docs (and ideally
#                    more of the expectation groups than OFF).
#       "no_regress" => the target doc must remain retrieved ON just as OFF.
DEFAULT_CASES = [
    {
        "id": "axi4_vs_chi",
        "query": "Compare AMBA AXI4 vs CHI coherency",
        "kind": "improve",
        "groups": [[AXI_IHI0022E], [CHI_IHI0098B, CHI_IHI0050H]],
    },
    {
        "id": "mbist_steps",
        "query": "What are the steps before inserting MBIST for a block?",
        "kind": "improve",
        "groups": [],  # authoritative doc id not pre-resolved; report-only
    },
    {
        "id": "regression_setup",
        "query": "Step-by-step guide to set up regression for a new project",
        "kind": "improve",
        "groups": [],  # report-only
    },
    {
        "id": "needle_filters_status",
        "query": "What is the reset value of Filters_N_Status at offset 0x0064?",
        "kind": "no_regress",
        "groups": [[NEEDLE_ARC]],
    },
    {
        "id": "single_doc_pmu",
        "query": "Vitec PMU power domains",
        "kind": "no_regress",
        "groups": [[PMU_SPEC, VITEC_G60]],
    },
]


def summarize_results(resp) -> list[dict]:
    """Extract (rank, score, source, document_id, heading) from a RAGResponse."""
    rows = []
    for i, r in enumerate(getattr(resp, "results", []) or []):
        meta = getattr(r, "metadata", {}) or {}
        rows.append(
            {
                "rank": i,
                "score": round(float(getattr(r, "score", 0.0) or 0.0), 4),
                "source": meta.get("source"),
                "document_id": meta.get("document_id"),
                "heading": meta.get("heading"),
            }
        )
    return rows


def doc_set(summary: list[dict]) -> set:
    """Distinct document_ids present in a summarized result list."""
    return {row["document_id"] for row in summary if row.get("document_id")}


def evaluate_case(case: dict, off_summary: list[dict], on_summary: list[dict]) -> dict:
    """Compute the §10 verdict for one case from its OFF and ON summaries (pure)."""
    off_docs, on_docs = doc_set(off_summary), doc_set(on_summary)
    groups = case.get("groups") or []

    def groups_hit(docs: set) -> int:
        return sum(1 for g in groups if docs & set(g))

    on_groups, off_groups = groups_hit(on_docs), groups_hit(off_docs)
    verdict = {
        "id": case["id"],
        "kind": case["kind"],
        "query": case["query"],
        "off_docs": sorted(off_docs),
        "on_docs": sorted(on_docs),
        "gained_on": sorted(on_docs - off_docs),
        "lost_on": sorted(off_docs - on_docs),
        "groups_total": len(groups),
        "on_groups_hit": on_groups,
        "off_groups_hit": off_groups,
    }
    if not groups:
        verdict["status"] = "report_only"
    elif case["kind"] == "improve":
        # All expectation groups must surface ON, and ON must be >= OFF on coverage.
        verdict["status"] = "PASS" if (on_groups == len(groups) and on_groups >= off_groups) else "FAIL"
    else:  # no_regress: every group present OFF must still be present ON
        verdict["status"] = "PASS" if on_groups >= off_groups and on_groups == len(groups) else "FAIL"
    return verdict


def run_query(chain, rc, settings, query, *, routing, search_limit, rerank_top_k, llm_decomp):
    """Run one query at a chosen routing mode by toggling the gate constants."""
    rc.RAG_DOCUMENT_ROUTING_ENABLED = bool(routing)
    settings.RAG_DECOMPOSITION_ENABLED = bool(routing)
    settings.RAG_DECOMPOSITION_LLM_PRIMARY = bool(llm_decomp)
    resp = chain.run(
        query,
        search_limit=search_limit,
        rerank_top_k=rerank_top_k,
        skip_generation=True,
    )
    return summarize_results(resp)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Document-routing ON-vs-OFF A/B on the live corpus (§10).")
    ap.add_argument("--search-limit", type=int, default=40)
    ap.add_argument("--rerank-top-k", type=int, default=12)
    ap.add_argument("--llm-decomp", action="store_true",
                    help="Use the LLM decomposition tier (needs gen backend); default is deterministic regex.")
    ap.add_argument("--json", help="Write the full report (per-case OFF/ON summaries + verdicts) to this path.")
    args = ap.parse_args(argv)

    import src.retrieval.pipeline.rag_chain as rc  # heavy import deferred to run time
    from config import settings

    chain = rc.RAGChain()
    report = []
    n_pass = n_fail = 0
    for case in DEFAULT_CASES:
        off = run_query(chain, rc, settings, case["query"], routing=False,
                        search_limit=args.search_limit, rerank_top_k=args.rerank_top_k,
                        llm_decomp=args.llm_decomp)
        on = run_query(chain, rc, settings, case["query"], routing=True,
                       search_limit=args.search_limit, rerank_top_k=args.rerank_top_k,
                       llm_decomp=args.llm_decomp)
        v = evaluate_case(case, off, on)
        v["off_summary"], v["on_summary"] = off, on
        report.append(v)
        if v["status"] == "PASS":
            n_pass += 1
        elif v["status"] == "FAIL":
            n_fail += 1
        print(f"\n=== [{v['status']}] {case['id']} ({case['kind']}) ===")
        print(f"  Q: {case['query']}")
        print(f"  groups hit  OFF={v['off_groups_hit']}/{v['groups_total']}  ON={v['on_groups_hit']}/{v['groups_total']}")
        print(f"  gained ON: {v['gained_on']}")
        print(f"  lost   ON: {v['lost_on']}")
        print(f"  OFF top sources: {[r['source'] for r in off[:5]]}")
        print(f"  ON  top sources: {[r['source'] for r in on[:5]]}")

    print(f"\n=== SUMMARY: {n_pass} pass, {n_fail} fail, "
          f"{len(report) - n_pass - n_fail} report-only ===")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"wrote {args.json}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
