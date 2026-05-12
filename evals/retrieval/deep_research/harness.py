"""A/B harness: run a golden query set through baseline vs deep_research mode.

Computes per-query and aggregate metrics. Designed to be invoked from pytest
(`test_compare.py`) or as a CLI for ad-hoc reports.

CLI:
    python -m evals.retrieval.deep_research.harness \
        --fixtures evals/retrieval/deep_research/fixtures/asic/golden_queries.json \
        --output /tmp/dr_eval_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)


@dataclass
class ChunkMatcher:
    """Predicate over a returned chunk — matches against expected_chunks entries.

    Hybrid matching:
      - ``source``: exact match against ``chunk.source`` (typo-loud; the source
        path is stable post-ingest, so an exact spec is what we want).
      - ``heading_contains`` / ``text_contains``: case-insensitive substring
        (chunkers may shift content around within a heading; substring tolerates
        that without forcing the golden set to track every wording change).
    """
    source: str | None = None
    heading_contains: str | None = None
    text_contains: str | None = None

    def matches(self, chunk: dict[str, Any]) -> bool:
        if self.source is not None and self.source != (chunk.get("source") or ""):
            return False
        if self.heading_contains and self.heading_contains.lower() not in (chunk.get("heading") or "").lower():
            return False
        if self.text_contains and self.text_contains.lower() not in (chunk.get("text") or "").lower():
            return False
        # Reject "match nothing" matchers — at least one criterion must be set.
        return any([self.source, self.heading_contains, self.text_contains])


@dataclass
class ModeResult:
    mode: str  # "baseline" | "deep_research"
    chunk_count: int
    latency_ms: float
    recall_at_k: float
    topic_coverage: float
    llm_calls: int | None = None
    error: str | None = None
    # Routing assertion support: the chain returns ``response.action`` as
    # one of "search"/"ask_user"/"canned". Surface it so test_routing.py can
    # assert messy-but-answerable queries do not false-positive into
    # ask_user when DR is on. None means the field was not populated.
    action: str | None = None
    action_expected: str | None = None  # "retrieve" | "ask_user" (from fixture)
    action_match: bool | None = None    # None when fixture sets no expectation
    # Typed ask_user reason (additive). Populated only when the chain's
    # central terminal-decision step chose action="ask_user". Mirrors the
    # ``RAGResponse.ask_user_reason`` field for harness-level inspection
    # and downstream debugging dashboards.
    ask_user_reason: str | None = None


@dataclass
class QueryResult:
    id: str
    query: str
    category: str
    baseline: ModeResult | None = None
    deep_research: ModeResult | None = None
    skipped_reason: str | None = None


@dataclass
class SuiteReport:
    domain: str
    query_count: int
    results: list[QueryResult] = field(default_factory=list)
    aggregate: dict[str, Any] = field(default_factory=dict)


def _chunk_to_dict(chunk: Any) -> dict[str, Any]:
    """Normalize a returned chunk (RankedResult / SearchResult / dict) to a flat
    dict with `source`, `heading`, `text` for matching. Source and heading live
    inside `metadata` on RankedResult; surface them at the top level here.
    """
    if isinstance(chunk, dict):
        meta = chunk.get("metadata") or {}
        return {
            "source": chunk.get("source") or meta.get("source"),
            "heading": chunk.get("heading") or meta.get("heading"),
            "text": chunk.get("text"),
        }
    text = getattr(chunk, "text", None)
    meta = getattr(chunk, "metadata", None) or {}
    if not isinstance(meta, dict):
        meta = {}
    return {
        "source": getattr(chunk, "source", None) or meta.get("source"),
        "heading": getattr(chunk, "heading", None) or meta.get("heading"),
        "text": text or str(chunk),
    }


def _compute_recall(returned: Iterable[dict[str, Any]], expected: list[dict[str, Any]]) -> float:
    if not expected:
        return float("nan")
    matchers = [ChunkMatcher(**e) for e in expected]
    returned_list = list(returned)
    hits = sum(1 for m in matchers if any(m.matches(c) for c in returned_list))
    return hits / len(matchers)


def _compute_topic_coverage(
    returned: Iterable[dict[str, Any]], expected_topics: list[str]
) -> float:
    if not expected_topics:
        return float("nan")
    blob = " ".join(
        ((c.get("heading") or "") + " " + (c.get("text") or "")).lower()
        for c in returned
    )
    hits = sum(1 for t in expected_topics if t.lower() in blob)
    return hits / len(expected_topics)


def _run_single(
    rag: Any,
    query: str,
    *,
    deep_research: bool,
    top_k: int,
    routing_mode: bool = False,
) -> tuple[list[dict], float, int | None, str | None]:
    """Invoke RAGChain.run once.

    Returns ``(chunks, latency_ms, llm_calls, action)``.

    ``routing_mode=True`` switches the chain into ``mode="query"`` so the
    LLM-confidence-loop ask_user gate is reachable. Recall fixtures leave
    it False and stay in ``mode="retrieval"`` (raw retrieval, no gate).

    Note on tuple shape: extended additively (action appended at the end)
    so concurrent harness extensions can keep their existing return-tuple
    positions stable.
    """
    t0 = time.perf_counter()
    # Baseline uses retrieval_sub_mode="hard" to bypass the LLM query rewriter
    # so it measures raw retrieval, not query-rewriting quality. Deep research
    # uses "auto" since its own decomposer is the LLM-driven part.
    # Deep research alone runs 70-90s; the default 30s overall_timeout_ms
    # would trip post-DR stage budget checks and mask the result with
    # action=ask_user. Pass a 3-min ceiling when deep research is on.
    if routing_mode:
        # mode="query" runs the full confidence loop (where the ask_user gate
        # actually lives). retrieval_sub_mode is ignored outside mode="retrieval"
        # so we omit it. Headroom on overall_timeout: confidence loop + DR.
        response = rag.run(
            query=query,
            rerank_top_k=top_k,
            skip_generation=True,
            mode="query",
            deep_research=deep_research,
            overall_timeout_ms=240_000 if deep_research else 60_000,
        )
    else:
        response = rag.run(
            query=query,
            rerank_top_k=top_k,
            skip_generation=True,
            mode="retrieval",
            retrieval_sub_mode="hard" if not deep_research else "auto",
            deep_research=deep_research,
            overall_timeout_ms=180_000 if deep_research else 30_000,
        )
    elapsed = (time.perf_counter() - t0) * 1000
    chunks = [_chunk_to_dict(r) for r in (getattr(response, "results", None) or [])]
    # Best-effort: pull llm_calls from response metadata if surfaced.
    llm_calls = None
    meta = getattr(response, "metadata", None) or {}
    if isinstance(meta, dict):
        dr_meta = meta.get("deep_research") or {}
        llm_calls = dr_meta.get("llm_call_count")
    # Fallback: DR orchestrator exposes a side-channel ``last_result`` attribute
    # (RAGResponse does not yet carry a free-form metadata dict).
    if llm_calls is None and deep_research:
        try:
            from src.retrieval.pipeline.deep_research import DeepResearch
            last = DeepResearch.last_result
            if last is not None:
                llm_calls = last.llm_call_count
        except Exception:
            pass
    action = getattr(response, "action", None)
    ask_user_reason = getattr(response, "ask_user_reason", None)
    # Stuff into ``action`` tuple slot's neighbour via attribute hop —
    # callers that only unpack 4 values stay backward-compatible while
    # newer call sites can read ``response.ask_user_reason`` directly off
    # the response or pull it via the harness ModeResult below.
    _run_single._last_ask_user_reason = ask_user_reason  # type: ignore[attr-defined]
    return chunks, elapsed, llm_calls, action


def evaluate_query(
    rag: Any, q: dict[str, Any], *, top_k: int = 10
) -> QueryResult:
    expected_chunks = q.get("expected_chunks") or []
    expected_topics = q.get("expected_topics") or []
    expected_action = q.get("expected_action")  # "retrieve" | "ask_user" | None
    qr = QueryResult(id=q["id"], query=q["query"], category=q.get("category", "unknown"))

    # Routing-only fixtures may declare expected_action without expected_chunks
    # (the vague_control case). Skip the "no expected anything" guard for them.
    if not expected_chunks and not expected_topics and not expected_action:
        qr.skipped_reason = "no expected_chunks or expected_topics populated"
        return qr

    # Routing fixtures (any query in the suite carries expected_action) need
    # mode="query" so the confidence-loop ask_user gate is actually reachable.
    routing_mode = expected_action is not None

    for mode_name, deep in (("baseline", False), ("deep_research", True)):
        try:
            chunks, latency, llm_calls, action = _run_single(
                rag,
                q["query"],
                deep_research=deep,
                top_k=top_k,
                routing_mode=routing_mode,
            )
            action_match: bool | None = None
            if expected_action is not None:
                # "retrieve" expects anything that is not ask_user (search /
                # canned both surface chunks; canned is rare here).
                if expected_action == "ask_user":
                    action_match = action == "ask_user"
                else:
                    action_match = action != "ask_user"
            mr = ModeResult(
                mode=mode_name,
                chunk_count=len(chunks),
                latency_ms=round(latency, 1),
                recall_at_k=_compute_recall(chunks, expected_chunks),
                topic_coverage=_compute_topic_coverage(chunks, expected_topics),
                llm_calls=llm_calls,
                action=action,
                action_expected=expected_action,
                action_match=action_match,
                ask_user_reason=getattr(_run_single, "_last_ask_user_reason", None),
            )
        except Exception as exc:
            logger.exception("Mode %s failed for %s", mode_name, q["id"])
            mr = ModeResult(
                mode=mode_name, chunk_count=0, latency_ms=0.0,
                recall_at_k=float("nan"), topic_coverage=float("nan"),
                error=str(exc),
                action_expected=expected_action,
            )
        setattr(qr, mode_name, mr)
    return qr


def _aggregate(results: list[QueryResult]) -> dict[str, Any]:
    def _avg(vals: list[float]) -> float | None:
        clean = [v for v in vals if v == v]  # filter NaN
        return sum(clean) / len(clean) if clean else None

    by_mode = {"baseline": [], "deep_research": []}
    by_mode_latency = {"baseline": [], "deep_research": []}
    by_mode_topic = {"baseline": [], "deep_research": []}
    by_mode_llm_calls = {"baseline": [], "deep_research": []}
    by_mode_action_match = {"baseline": [], "deep_research": []}
    for r in results:
        for mode in ("baseline", "deep_research"):
            mr = getattr(r, mode)
            if mr and not mr.error:
                by_mode[mode].append(mr.recall_at_k)
                by_mode_latency[mode].append(mr.latency_ms)
                by_mode_topic[mode].append(mr.topic_coverage)
                if mr.llm_calls is not None:
                    by_mode_llm_calls[mode].append(float(mr.llm_calls))
                if mr.action_match is not None:
                    by_mode_action_match[mode].append(1.0 if mr.action_match else 0.0)

    def _rate(vals: list[float]) -> float | None:
        return sum(vals) / len(vals) if vals else None

    return {
        "avg_recall_at_k": {m: _avg(v) for m, v in by_mode.items()},
        "avg_topic_coverage": {m: _avg(v) for m, v in by_mode_topic.items()},
        "avg_latency_ms": {m: _avg(v) for m, v in by_mode_latency.items()},
        "avg_llm_calls": {m: _avg(v) for m, v in by_mode_llm_calls.items()},
        "action_match_rate": {m: _rate(v) for m, v in by_mode_action_match.items()},
        "evaluated_count": sum(1 for r in results if not r.skipped_reason),
        "skipped_count": sum(1 for r in results if r.skipped_reason),
    }


def run_suite(rag: Any, fixture: dict[str, Any], *, top_k: int = 10) -> SuiteReport:
    queries = fixture.get("queries") or []
    report = SuiteReport(domain=fixture.get("domain", "unknown"), query_count=len(queries))
    for q in queries:
        report.results.append(evaluate_query(rag, q, top_k=top_k))
    report.aggregate = _aggregate(report.results)
    return report


def _cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from src.retrieval.pipeline import RAGChain  # heavy import — only at CLI entry

    fixture = json.loads(args.fixtures.read_text())
    rag = RAGChain()
    try:
        report = run_suite(rag, fixture, top_k=args.top_k)
    finally:
        rag.close()

    args.output.write_text(json.dumps(asdict(report), indent=2, default=str))
    print(f"Report written to {args.output}")
    print(json.dumps(report.aggregate, indent=2, default=str))


if __name__ == "__main__":
    _cli()
