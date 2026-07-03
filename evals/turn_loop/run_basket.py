# @summary
# Loader / validator / live driver for the turn_loop query basket
# (evals/turn_loop/query_basket.yaml). Offline: load + schema-validate the basket
# (used by the pytest so the basket "runs" in CI). Live: drive each query against
# a running stack over SSE and record the route taken, the outcome
# (answer/clarify/refuse), latency, and a soft anchor-term retrieval hit — the
# end-to-end smoke test used across the turn_loop unification phases.
# NB: uses urllib (never curl/wget) per the EDR constraint on these hosts.
# Exports: load_basket, iter_queries, validate_basket, run_live, VALID_QTYPES, VALID_ROUTES
# Deps: yaml (offline); urllib (live only)
# @end-summary
"""Turn-loop query-basket loader, validator, and live SSE driver.

Usage:
    python -m evals.turn_loop.run_basket validate
    python -m evals.turn_loop.run_basket run --target http://localhost:8102 [--qid f001,c001]
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterator, Optional

BASKET_PATH = Path(__file__).resolve().parent / "query_basket.yaml"

VALID_QTYPES = {
    "factoid", "disambiguation", "howto", "multi_aspect", "multi_topic",
    "qfs", "messy", "clarify", "multi_turn", "adversarial", "out_of_corpus",
}
VALID_ROUTES = {"fast_lane", "retrieve", "decompose", "deep_study", "clarify", "refuse"}
VALID_BEHAVIORS = {"answer", "clarify", "refuse_or_state_unknown"}


def load_basket(path: Path = BASKET_PATH) -> dict:
    """Parse the basket YAML into a dict (raises on malformed YAML)."""
    import yaml  # local import so the live driver has no hard yaml dep

    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def iter_queries(basket: dict) -> Iterator[dict]:
    """Yield each query entry (single-turn entries and multi_turn entries alike)."""
    yield from basket.get("queries", [])


def validate_basket(basket: dict) -> list[str]:
    """Return a list of schema problems (empty ⇒ valid).

    Checks: unique qids; known qtype/route/behavior; single-turn entries carry a
    ``query``; multi_turn entries carry a non-empty ``turns`` list whose steps each
    carry a ``query``; anchor_terms is a list when present.
    """
    problems: list[str] = []
    seen: set[str] = set()
    for i, q in enumerate(iter_queries(basket)):
        where = f"entry[{i}] qid={q.get('qid', '?')}"
        qid = q.get("qid")
        if not qid:
            problems.append(f"{where}: missing qid")
        elif qid in seen:
            problems.append(f"{where}: duplicate qid")
        else:
            seen.add(qid)

        qtype = q.get("qtype")
        if qtype not in VALID_QTYPES:
            problems.append(f"{where}: bad qtype {qtype!r}")

        beh = q.get("expected_behavior")
        if beh is not None and beh not in VALID_BEHAVIORS:
            problems.append(f"{where}: bad expected_behavior {beh!r}")

        if q.get("qtype") == "multi_turn":
            turns = q.get("turns")
            if not isinstance(turns, list) or not turns:
                problems.append(f"{where}: multi_turn needs a non-empty 'turns' list")
            else:
                for j, t in enumerate(turns):
                    if not t.get("query"):
                        problems.append(f"{where}: turns[{j}] missing query")
                    r = t.get("expected_route")
                    if r is not None and r not in VALID_ROUTES:
                        problems.append(f"{where}: turns[{j}] bad route {r!r}")
        else:
            if not q.get("query"):
                problems.append(f"{where}: missing query")
            r = q.get("expected_route")
            if r is not None and r not in VALID_ROUTES:
                problems.append(f"{where}: bad expected_route {r!r}")

        at = q.get("anchor_terms")
        if at is not None and not isinstance(at, list):
            problems.append(f"{where}: anchor_terms must be a list")
    return problems


# --------------------------------------------------------------------------- #
# Live driver (SSE over urllib — no curl/wget per the EDR constraint)
# --------------------------------------------------------------------------- #

def _stream_query(target: str, body: dict, timeout: float = 200.0) -> dict:
    """POST one query to <target>/query/stream and collect the outcome.

    Returns {ttfr, ttd, route, metadata, answer, results_n, error}. ``route`` is
    inferred from the retrieval-event metadata (turn_loop / deep_research /
    agentic / linear).
    """
    import urllib.request

    data = json.dumps(body).encode()
    req = urllib.request.Request(
        target.rstrip("/") + "/query/stream", data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    t0 = time.monotonic()
    out: dict[str, Any] = {"ttfr": None, "ttd": None, "route": None,
                           "metadata": {}, "answer": "", "results_n": 0, "error": None}
    ev, dl, answer_parts = None, [], []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").rstrip("\n")
                if line.startswith("event: "):
                    ev = line[7:]
                elif line.startswith("data: "):
                    dl.append(line[6:])
                elif line == "":
                    payload = "".join(dl)
                    if ev == "retrieval" and payload and out["ttfr"] is None:
                        out["ttfr"] = round(time.monotonic() - t0, 2)
                        d = json.loads(payload)
                        out["metadata"] = d.get("metadata") or {}
                        out["results_n"] = len(d.get("results") or [])
                        out["route"] = _infer_route(out["metadata"], d)
                    elif ev == "token" and payload:
                        answer_parts.append(json.loads(payload).get("token", ""))
                    elif ev == "error" and payload:
                        out["error"] = json.loads(payload).get("message", "error")
                    elif ev == "done":
                        out["ttd"] = round(time.monotonic() - t0, 2)
                    ev, dl = None, []
    except Exception as exc:  # noqa: BLE001 — a live probe failure is data, not fatal
        out["error"] = repr(exc)
    out["answer"] = "".join(answer_parts)
    return out


def _infer_route(metadata: dict, retrieval_payload: dict) -> str:
    """Best-effort classification of which orchestrator handled the query."""
    md = metadata or {}
    if md.get("turn_loop"):
        return "turn_loop"
    dr = md.get("deep_research") or {}
    if dr.get("decomposed") or dr.get("topic_count"):
        return "deep_research"
    if md.get("agentic_retrieval"):
        return "agentic"
    if retrieval_payload.get("clarification_message"):
        return "clarify"
    return "linear"


def run_live(target: str, qids: Optional[set[str]] = None,
             basket: Optional[dict] = None) -> list[dict]:
    """Drive basket queries against a live stack; return per-query result rows.

    Multi-turn entries are run as a sequence sharing one conversation_id so the
    later turns exercise context carry.
    """
    basket = basket or load_basket()
    rows: list[dict] = []
    for q in iter_queries(basket):
        if qids and q.get("qid") not in qids:
            continue
        if q.get("qtype") == "multi_turn":
            conv_id = None
            for j, t in enumerate(q["turns"]):
                body = {"query": t["query"], "search_limit": 12, "rerank_top_k": 6}
                if conv_id:
                    body["conversation_id"] = conv_id
                res = _stream_query(target, body)
                conv_id = (res["metadata"] or {}).get("conversation_id") or conv_id
                rows.append({"qid": f"{q['qid']}.t{j+1}", "qtype": q["qtype"],
                             "query": t["query"], **_row(res, t)})
        else:
            body = {"query": q["query"], "search_limit": 12, "rerank_top_k": 6}
            res = _stream_query(target, body)
            rows.append({"qid": q["qid"], "qtype": q["qtype"],
                         "query": q["query"], **_row(res, q)})
    return rows


def _row(res: dict, spec: dict) -> dict:
    """Fold a raw stream result + its spec into a compact reportable row."""
    ans = (res.get("answer") or "").lower()
    anchors = [a.lower() for a in (spec.get("anchor_terms") or [])]
    hit = sum(1 for a in anchors if a in ans)
    return {
        "route": res.get("route"),
        "ttfr_s": res.get("ttfr"),
        "ttd_s": res.get("ttd"),
        "results_n": res.get("results_n"),
        "anchor_hits": f"{hit}/{len(anchors)}" if anchors else "-",
        "answer_chars": len(res.get("answer") or ""),
        "error": res.get("error"),
    }


def _main(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Turn-loop query basket driver")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate", help="schema-validate the basket (offline)")
    rp = sub.add_parser("run", help="drive the basket against a live stack")
    rp.add_argument("--target", default="http://localhost:8102")
    rp.add_argument("--qid", default="", help="comma-separated qids to run (default: all)")
    args = ap.parse_args(argv)

    basket = load_basket()
    if args.cmd == "validate":
        problems = validate_basket(basket)
        if problems:
            print("INVALID basket:")
            for p in problems:
                print("  -", p)
            return 1
        n = sum(1 for _ in iter_queries(basket))
        print(f"OK: {n} basket entries valid")
        return 0

    qids = {x for x in args.qid.split(",") if x} or None
    rows = run_live(args.target, qids=qids, basket=basket)
    print(f"{'qid':<10} {'qtype':<13} {'route':<13} {'ttfr':>6} {'ttd':>6} "
          f"{'res':>4} {'anchors':>8}  error")
    for r in rows:
        print(f"{r['qid']:<10} {r['qtype']:<13} {str(r['route']):<13} "
              f"{str(r['ttfr_s']):>6} {str(r['ttd_s']):>6} {str(r['results_n']):>4} "
              f"{r['anchor_hits']:>8}  {r['error'] or ''}")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(_main(sys.argv[1:]))
