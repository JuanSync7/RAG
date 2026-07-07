# @summary
# P5 faithfulness executor: scores each golden's retrieval result via a
# JudgeClient and aggregates per-qtype mean faithfulness scores. P7c adds an
# opt-in bounded-concurrency path (max_parallel_judges > 1) that fans every
# (golden, sample) judge call into one ThreadPoolExecutor and regroups by qid
# in submission order; N<=1 keeps the byte-identical sequential loop.
# P15 adds judge-call robustness: each golden's judge call is attempted up to
# 1+judge_retries times (_score_with_retry — the SINGLE retry site so clients
# stay thin), and is wrapped in a per-golden isolation boundary so a single
# golden's judge failure is SKIPPED+COUNTED (via the same total_queries_skipped
# accounting as empty-query goldens) instead of aborting the whole run.
# Exports: QueryFaithfulnessResult, FaithfulnessResults, score_goldens,
#          aggregate_faithfulness_by_qtype
# Deps: stdlib (concurrent.futures, dataclasses, logging, statistics, typing);
#       src.eval.pack.schema.Golden; src.eval.runner.judge (JudgeClient,
#       JudgeQuestion); src.eval.runner.retrieve (RetrievalResults);
#       src.platform.observability.concurrency.submit_with_context.
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

import logging
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Mapping

from src.eval.pack.schema import Golden
from src.platform.observability.concurrency import submit_with_context

from .judge import JudgeClient, JudgeQuestion, JudgmentScore
from .retrieve import RetrievalResults

logger = logging.getLogger("rag.eval.faithfulness")


@dataclass(frozen=True)
class QueryFaithfulnessResult:
    """Per-query faithfulness score returned by the judge.

    ``samples`` (P7b) carries the raw per-sample ``JudgmentScore`` tuple
    populated by ``score_goldens``. Defaults to ``()`` for manual
    construction without the kwarg (backwards-compat).
    """

    qid: str
    qtype: str
    score: float
    reasoning: str
    samples: tuple[JudgmentScore, ...] = ()


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


def _build_question(
    qid: str,
    qresult,
    by_qid: dict[str, Golden],
) -> JudgeQuestion:
    """Construct the JudgeQuestion for a scored golden (shared by both paths)."""
    golden = by_qid.get(qid)
    expected_span = golden.expected_answer_span if golden is not None else None
    return JudgeQuestion(
        qid=qid,
        qtype=qresult.qtype,
        query=qresult.query,
        expected_answer_span=expected_span,
        chunk_texts=qresult.retrieved_chunks,
    )


def _assemble_result(
    qid: str,
    qtype: str,
    samples: list[JudgmentScore],
) -> QueryFaithfulnessResult:
    """Aggregate per-sample scores into a QueryFaithfulnessResult.

    ``samples`` MUST already be in submission order (sample 0..N-1). The
    stored score is the mean; reasoning comes from the highest-scored sample
    (first-wins on ties — Python ``max`` is stable). Ordering is load-bearing:
    the P7b ``per_query_samples`` JSON serializes this tuple in order.
    """
    mean_score = statistics.mean(float(s.score) for s in samples)
    best = max(samples, key=lambda s: s.score)
    return QueryFaithfulnessResult(
        qid=qid,
        qtype=qtype,
        score=float(mean_score),
        reasoning=best.reasoning,
        samples=tuple(samples),
    )


def _score_with_retry(
    judge_client: JudgeClient,
    question: JudgeQuestion,
    *,
    retries: int,
) -> JudgmentScore:
    """Score *question* with bounded transport-resilience retries.

    Attempts ``judge_client.score(question)`` up to ``1 + retries`` times,
    returning the first successful :class:`JudgmentScore` and re-raising the
    last exception on exhaustion. This is the SINGLE retry site (P15): the
    judge clients themselves stay thin and just raise, so we never double-retry.
    """
    last_exc: Exception | None = None
    for _attempt in range(1 + retries):
        try:
            return judge_client.score(question)
        except Exception as exc:  # noqa: BLE001 — re-raised on exhaustion below.
            last_exc = exc
    assert last_exc is not None  # 1+retries >= 1, so the loop ran at least once.
    raise last_exc


def score_goldens(
    retrieval_results: RetrievalResults,
    goldens: Mapping[str, list[Golden]],
    judge_client: JudgeClient,
    *,
    samples_per_claim: int = 1,
    max_parallel_judges: int = 1,
    judge_retries: int = 1,
) -> FaithfulnessResults:
    """Score each retrieved golden via the judge client.

    Goldens whose ``retrieved_chunks`` are empty are SKIPPED (no judge
    call). Every other golden is judged ``samples_per_claim`` times; the
    stored score is the mean across samples and the stored reasoning is
    taken from the highest-scored sample (first-wins on ties — Python
    ``max`` is stable).

    When ``max_parallel_judges > 1`` (P7c), every ``(golden, sample)`` judge
    call across all goldens is fanned into ONE bounded
    :class:`~concurrent.futures.ThreadPoolExecutor`
    (``max_workers=max_parallel_judges``) via
    :func:`submit_with_context` so OTel span context propagates to workers.
    Results are regrouped by qid and each qid's samples are sorted back into
    submission order (0..N-1) before aggregation, so the output is identical
    to the sequential path for a deterministic judge regardless of completion
    order. ``max_parallel_judges == 1`` takes the plain sequential loop (no
    pool, no thread overhead) and is byte-identical to pre-P7c behavior.

    **Per-golden isolation (P15).** Each golden's judge call is attempted up to
    ``1 + judge_retries`` times (:func:`_score_with_retry`). If a golden still
    fails after retries (ANY sample exhausting its retries fails the WHOLE
    golden — no partial average), that golden is SKIPPED and COUNTED in
    ``total_queries_skipped`` (the same accounting as empty-query goldens) and
    excluded from aggregation, while every other golden is scored and the run
    completes. A single judge failure therefore no longer aborts the run.
    """
    if samples_per_claim < 1:
        raise ValueError(
            f"samples_per_claim must be >= 1, got {samples_per_claim}"
        )
    if max_parallel_judges < 1:
        raise ValueError(
            f"max_parallel_judges must be >= 1, got {max_parallel_judges}"
        )
    if judge_retries < 0:
        raise ValueError(f"judge_retries must be >= 0, got {judge_retries}")
    by_qid = _index_goldens_by_qid(goldens)

    if max_parallel_judges == 1:
        return _score_sequential(
            retrieval_results,
            by_qid,
            judge_client,
            samples_per_claim,
            judge_retries,
        )
    return _score_parallel(
        retrieval_results,
        by_qid,
        judge_client,
        samples_per_claim,
        max_parallel_judges,
        judge_retries,
    )


def _score_sequential(
    retrieval_results: RetrievalResults,
    by_qid: dict[str, Golden],
    judge_client: JudgeClient,
    samples_per_claim: int,
    judge_retries: int,
) -> FaithfulnessResults:
    """Sequential path — one plain loop with per-golden retry + isolation (P15)."""
    per_query: dict[str, QueryFaithfulnessResult] = {}
    skipped = 0
    for qid, qresult in retrieval_results.per_query.items():
        if len(qresult.retrieved_chunks) == 0:
            skipped += 1
            continue
        question = _build_question(qid, qresult, by_qid)
        # --- Per-golden isolation boundary (P15) ---------------------------
        # Like run_nightly's per-pack catch, this is a DELIBERATE broad
        # boundary: a single golden's judge failure (after bounded retries)
        # must not abort the whole run. The golden is SKIPPED + COUNTED via the
        # same skipped accounting as empty-query goldens, and we log LOUDLY
        # with the qid + exception type/message (no silent swallow). If ANY
        # sample exhausts its retries the whole golden is skipped (no partial
        # average — see score_goldens docstring).
        try:
            samples = [
                _score_with_retry(judge_client, question, retries=judge_retries)
                for _ in range(samples_per_claim)
            ]
        except Exception as exc:  # noqa: BLE001 — isolation boundary (see above).
            logger.warning(
                "judge failed for qid=%s after %d attempt(s) (%s: %s) — "
                "skipping this golden",
                qid,
                1 + judge_retries,
                type(exc).__name__,
                exc,
            )
            skipped += 1
            continue
        per_query[qid] = _assemble_result(qid, qresult.qtype, samples)
    return FaithfulnessResults(
        collection_name=retrieval_results.collection_name,
        per_query=per_query,
        total_queries_scored=len(per_query),
        total_queries_skipped=skipped,
    )


def _score_parallel(
    retrieval_results: RetrievalResults,
    by_qid: dict[str, Golden],
    judge_client: JudgeClient,
    samples_per_claim: int,
    max_parallel_judges: int,
    judge_retries: int,
) -> FaithfulnessResults:
    """Parallel path — flat (golden × sample) fan-out into one bounded pool.

    Each future is tagged ``(qid, sample_index)`` and runs
    :func:`_score_with_retry` (bounded retry inside the worker). After all
    futures resolve we regroup by qid and sort each qid's samples by
    ``sample_index`` so the stored ``samples`` tuple is in submission order,
    not completion order.

    **Per-golden isolation (P15).** If ANY sample-future of a golden raises
    (after exhausting its retries) the WHOLE golden is SKIPPED + COUNTED (no
    partial average) and the others still resolve — the pool is not poisoned.
    Failures are collected per qid while draining (a broad catch at the
    isolation boundary), logged loudly with the qid, and applied after the
    drain so a deterministic failing judge yields deterministic skips.
    """
    scored_qtypes: dict[str, str] = {}
    skipped = 0
    # Tagged judge results: qid -> list[(sample_index, JudgmentScore)].
    raw: dict[str, list[tuple[int, JudgmentScore]]] = {}
    # qids whose judge call failed (after retries) — skipped whole-golden.
    failed: dict[str, Exception] = {}

    # future -> (qid, sample_index) tag, so results collected in COMPLETION
    # order can be regrouped and re-sorted into submission order below.
    tags: dict[Any, tuple[str, int]] = {}

    def _retrying_score(question: JudgeQuestion) -> JudgmentScore:
        return _score_with_retry(judge_client, question, retries=judge_retries)

    with ThreadPoolExecutor(max_workers=max_parallel_judges) as executor:
        for qid, qresult in retrieval_results.per_query.items():
            if len(qresult.retrieved_chunks) == 0:
                skipped += 1
                continue
            scored_qtypes[qid] = qresult.qtype
            raw[qid] = []
            question = _build_question(qid, qresult, by_qid)
            for sample_index in range(samples_per_claim):
                fut = submit_with_context(executor, _retrying_score, question)
                tags[fut] = (qid, sample_index)

        # Drain in completion order; the per-qid sort below restores submission
        # order (sample 0..N-1). Per-golden isolation boundary (P15): a future
        # that raised (after retries) marks its qid failed instead of
        # propagating, so one bad golden does not poison the pool.
        for fut in as_completed(tags):
            qid, sample_index = tags[fut]
            try:
                raw[qid].append((sample_index, fut.result()))
            except Exception as exc:  # noqa: BLE001 — isolation boundary.
                failed[qid] = exc

    per_query: dict[str, QueryFaithfulnessResult] = {}
    for qid in raw:
        if qid in failed:
            # A golden with ANY failed sample is skipped whole (no partial avg).
            logger.warning(
                "judge failed for qid=%s after %d attempt(s) (%s: %s) — "
                "skipping this golden",
                qid,
                1 + judge_retries,
                type(failed[qid]).__name__,
                failed[qid],
            )
            skipped += 1
            continue
        ordered = sorted(raw[qid], key=lambda pair: pair[0])
        samples = [score for _idx, score in ordered]
        per_query[qid] = _assemble_result(qid, scored_qtypes[qid], samples)

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
