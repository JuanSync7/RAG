# @summary
# P5 faithfulness executor: scores each golden's retrieval result via a
# JudgeClient and aggregates per-qtype mean faithfulness scores.
# Exports: QueryFaithfulnessResult, FaithfulnessResults, score_goldens,
#          aggregate_faithfulness_by_qtype
# Deps: stdlib (dataclasses, statistics, typing); src.eval.pack.schema.Golden;
#       src.eval.runner.judge (JudgeClient, JudgeQuestion);
#       src.eval.runner.retrieve (RetrievalResults).
# @end-summary
"""P5 — faithfulness scoring executor.

Bridges P4's retrieval output (``RetrievalResults`` with
``retrieved_chunks``) into the P5.0 judge surface (``JudgeClient``).
Goldens whose retrieval returned zero chunks are skipped (no judge call,
counted in ``total_queries_skipped``). All other goldens are scored
exactly once via ``judge_client.score(JudgeQuestion(...))``.

The judge client is the sole LLM seam — this module imports nothing from
``src.common.llm.provider`` or ``src.platform.llm``.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Mapping

from src.eval.pack.schema import Golden

from .judge import JudgeClient, JudgeQuestion
from .retrieve import RetrievalResults


@dataclass(frozen=True)
class QueryFaithfulnessResult:
    """Per-query faithfulness score returned by the judge."""

    qid: str
    qtype: str
    score: float
    reasoning: str


@dataclass(frozen=True)
class FaithfulnessResults:
    """Aggregated faithfulness outcomes across all judged goldens."""

    collection_name: str
    per_query: Mapping[str, QueryFaithfulnessResult]
    total_queries_scored: int
    total_queries_skipped: int


def _index_goldens_by_qid(
    goldens: Mapping[str, list[Golden]],
) -> dict[str, Golden]:
    out: dict[str, Golden] = {}
    for _qtype, golden_list in goldens.items():
        for g in golden_list:
            out[g.qid] = g
    return out


def score_goldens(
    retrieval_results: RetrievalResults,
    goldens: Mapping[str, list[Golden]],
    judge_client: JudgeClient,
    *,
    samples_per_claim: int = 1,
) -> FaithfulnessResults:
    """Score each retrieved golden via the judge client.

    Goldens whose ``retrieved_chunks`` are empty are SKIPPED (no judge
    call). Every other golden is judged ``samples_per_claim`` times; the
    stored score is the mean across samples and the stored reasoning is
    taken from the highest-scored sample (first-wins on ties — Python
    ``max`` is stable).
    """
    if samples_per_claim < 1:
        raise ValueError(
            f"samples_per_claim must be >= 1, got {samples_per_claim}"
        )
    by_qid = _index_goldens_by_qid(goldens)
    per_query: dict[str, QueryFaithfulnessResult] = {}
    skipped = 0
    for qid, qresult in retrieval_results.per_query.items():
        if len(qresult.retrieved_chunks) == 0:
            skipped += 1
            continue
        golden = by_qid.get(qid)
        expected_span = golden.expected_answer_span if golden is not None else None
        question = JudgeQuestion(
            qid=qid,
            qtype=qresult.qtype,
            query=qresult.query,
            expected_answer_span=expected_span,
            chunk_texts=qresult.retrieved_chunks,
        )
        samples = [judge_client.score(question) for _ in range(samples_per_claim)]
        mean_score = statistics.mean(float(s.score) for s in samples)
        best = max(samples, key=lambda s: s.score)
        per_query[qid] = QueryFaithfulnessResult(
            qid=qid,
            qtype=qresult.qtype,
            score=float(mean_score),
            reasoning=best.reasoning,
        )
    return FaithfulnessResults(
        collection_name=retrieval_results.collection_name,
        per_query=per_query,
        total_queries_scored=len(per_query),
        total_queries_skipped=skipped,
    )


def aggregate_faithfulness_by_qtype(
    faithfulness_results: FaithfulnessResults,
    goldens: Mapping[str, list[Golden]],
) -> dict[str, float]:
    """Mean faithfulness score per qtype over scored goldens.

    Mirrors ``aggregate_recall_by_qtype``: qtypes with zero scored
    goldens are OMITTED from the result (no nan, no 0.0). Skipped
    goldens (those without a per-query entry in *faithfulness_results*)
    are excluded from the denominator.
    """
    out: dict[str, float] = {}
    for qtype, golden_list in goldens.items():
        scores: list[float] = []
        for golden in golden_list:
            qres = faithfulness_results.per_query.get(golden.qid)
            if qres is None:
                continue
            scores.append(qres.score)
        if scores:
            out[qtype] = statistics.mean(scores)
    return out
