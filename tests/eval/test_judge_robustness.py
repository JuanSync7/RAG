# @summary
# Tests for P15 — judge-call robustness (per-golden isolation + bounded retry).
# A single judge failure must NOT abort the run: the failing golden is retried
# (bounded), and if still failing is SKIPPED + COUNTED (via the existing
# total_queries_skipped accounting) while every other golden is scored and the
# run completes + gates on the survivors. Covers both the sequential and the
# parallel (max_parallel_judges>1) paths, the JudgeBackendError isolation tie to
# P14, plus two named-audit coverage backfills (corrupt baseline degrades to
# absent; ingest failure isolated per pack in run_nightly).
# Mutation targets:
#   remove per-golden try/except (either path) → test_one_golden_failure_*,
#       test_parallel_path_isolation, test_judge_backend_error_is_isolated
#   drop the retry loop                        → test_retry_then_succeed
#   off-by-one in retry bound                  → test_retry_exhausted_skips_not_crash
#   broaden the narrow corrupt-baseline catch  → test_corrupted_baseline_degrades_*
#   remove run_nightly per-pack try/except     → test_ingest_failure_isolated_per_pack
# Exports: test_one_golden_failure_does_not_abort_run,
#          test_one_golden_failure_run_eval_gates_on_survivors,
#          test_retry_then_succeed,
#          test_retry_exhausted_skips_not_crash,
#          test_parallel_path_isolation,
#          test_parallel_retry_then_succeed,
#          test_judge_backend_error_is_isolated,
#          test_partial_sample_failure_skips_golden,
#          test_corrupted_baseline_degrades_to_absent,
#          test_ingest_failure_isolated_per_pack
# Deps: src.eval.runner (score_goldens, JudgmentScore, JudgeQuestion,
#       RetrievalResults, QueryRetrievalResult), src.eval.runner.judge_cli
#       (JudgeBackendError), src.eval.orchestrator (run_nightly),
#       tests.eval.test_cli (_make_synthetic_pack, _patch_full_chain).
# @end-summary
"""P15 — judge-call robustness tests (per-golden isolation + bounded retry)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.eval.pack.schema import Golden
from src.eval.runner import (
    JudgeQuestion,
    JudgmentScore,
    QueryRetrievalResult,
    RetrievalResults,
    score_goldens,
)
from src.eval.runner.judge_cli import JudgeBackendError

FIXED_NOW = datetime(2026, 6, 2, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FailingJudgeClient:
    """Judge that RAISES for a configured set of qids and scores the rest.

    ``raise_qids`` maps a qid to the number of times .score should raise for
    that qid before succeeding; a value of -1 means "always raise". qids absent
    from the map score normally (``ok_score``). Records every call (in order)
    plus a per-qid attempt counter so tests can assert the retry bound.
    """

    def __init__(
        self,
        raise_qids: dict[str, int],
        *,
        ok_score: float = 0.9,
        exc_factory=None,
    ) -> None:
        self.raise_qids = dict(raise_qids)
        self.ok_score = ok_score
        self._exc_factory = exc_factory or (lambda qid: RuntimeError(f"judge boom {qid}"))
        self.calls: list[JudgeQuestion] = []
        self.attempts: dict[str, int] = {}

    def score(self, question: JudgeQuestion) -> JudgmentScore:
        self.calls.append(question)
        qid = question.qid
        self.attempts[qid] = self.attempts.get(qid, 0) + 1
        remaining = self.raise_qids.get(qid, 0)
        if remaining != 0:
            # Decrement a positive counter; -1 (always) stays sticky.
            if remaining > 0:
                self.raise_qids[qid] = remaining - 1
            raise self._exc_factory(qid)
        return JudgmentScore(score=self.ok_score, reasoning=f"ok-{qid}")


def _golden(qid: str, qtype: str = "factoid") -> Golden:
    return Golden(
        qid=qid,
        qtype=qtype,
        query=f"query-{qid}",
        expected_answer_span="span",
        expected_source_docs=[f"docs/{qid}.md"],
    )


def _qresult(qid: str, qtype: str = "factoid") -> QueryRetrievalResult:
    return QueryRetrievalResult(
        qid=qid,
        qtype=qtype,
        query=f"query-{qid}",
        retrieved_sources=(f"docs/{qid}.md",),
        retrieved_chunks=(f"chunk-{qid}",),
        k=5,
    )


def _three_goldens_results():
    """A pack with 3 factoid goldens f1/f2/f3, each with one retrieved chunk."""
    goldens = {"factoid": [_golden("f1"), _golden("f2"), _golden("f3")]}
    per_query = {
        "f1": _qresult("f1"),
        "f2": _qresult("f2"),
        "f3": _qresult("f3"),
    }
    results = RetrievalResults(collection_name="C", k=5, per_query=per_query)
    return goldens, results


# ---------------------------------------------------------------------------
# HEADLINE: one golden failure does not abort the run (sequential)
# ---------------------------------------------------------------------------


def test_one_golden_failure_does_not_abort_run() -> None:
    """Judge RAISES forever for golden f2 → f2 skipped+counted, f1/f3 scored.

    HEADLINE RED. Before the fix the raised RuntimeError escapes score_goldens
    and aborts the run. After the fix: the run COMPLETES; f2 lands in
    total_queries_skipped (NOT silently dropped); f1 and f3 are scored;
    total_queries_scored reflects only the survivors.
    """
    goldens, results = _three_goldens_results()
    judge = _FailingJudgeClient({"f2": -1}, ok_score=0.9)

    fres = score_goldens(goldens=goldens, retrieval_results=results, judge_client=judge)

    assert set(fres.per_query) == {"f1", "f3"}, (
        f"survivors must be scored; got {set(fres.per_query)}"
    )
    assert fres.total_queries_scored == 2
    # f2 flows through the SAME skipped accounting as empty-query goldens.
    assert fres.total_queries_skipped == 1, (
        f"failed golden must be counted as skipped; got {fres.total_queries_skipped}"
    )
    assert fres.per_query["f1"].score == pytest.approx(0.9)
    assert fres.per_query["f3"].score == pytest.approx(0.9)


def test_one_golden_failure_run_eval_gates_on_survivors(tmp_path: Path, monkeypatch) -> None:
    """End-to-end via run_eval: a failing golden is skipped, gate evals survivors.

    Proves the skip does not fabricate a spurious 0.0 faithfulness that would
    fail the gate. Single-golden pack whose only golden's judge raises → it is
    skipped → total_queries_judged == 0 → faithfulness gating is SKIPPED (gate
    passes on recall alone), and run_eval COMPLETES (no escaped exception).
    """
    from tests.eval.test_cli import _make_synthetic_pack
    from src.eval.cli import run_eval
    from src.eval.runner import execute as execute_mod
    from src.eval.runner import retrieve as retrieve_mod
    from src.eval.runner import report as report_mod
    import src.eval.runner as runner_pkg
    from src.vector_db.common.schemas import SearchResult

    pack_dir = _make_synthetic_pack(
        tmp_path, recall_floor=0.5, faithfulness_floor=0.5
    )

    def fake_execute_plan(plan, *, fresh: bool = True):
        return report_mod.IngestReport(
            collection_name=plan.collection_name,
            processed=1, skipped=0, failed=0, stored_chunks=1,
            document_count=1, plan=plan, errors=(),
        )

    def fake_search(client, query, query_embedding, alpha, limit, **kwargs):
        return [SearchResult(text="alpha", score=1.0, metadata={"source": "doc1.md"})]

    monkeypatch.setattr(runner_pkg, "execute_plan", fake_execute_plan)
    monkeypatch.setattr(retrieve_mod, "search", fake_search)
    monkeypatch.setattr(retrieve_mod, "get_embedding_provider", lambda: _Embedder())
    monkeypatch.setattr(retrieve_mod, "create_persistent_client", lambda: object())
    monkeypatch.setattr(retrieve_mod, "close_client", lambda _c: None)
    monkeypatch.setattr(
        runner_pkg, "build_judge_client",
        lambda pack, **kw: _FailingJudgeClient({"g1": -1}),
    )

    result = run_eval(str(pack_dir))

    assert result.report.total_queries_judged == 0, (
        "the only golden's judge failed → nothing judged"
    )
    assert result.report.total_queries_skipped == 1
    # Faithfulness gating is judged-guarded; recall=1.0 ≥ 0.5 → gate passes.
    assert result.gate.passed is True
    assert "factoid" not in result.report.faithfulness_by_qtype


class _Embedder:
    def embed_documents(self, texts):
        return [[0.0] * 8 for _ in texts]


# ---------------------------------------------------------------------------
# Bounded retry
# ---------------------------------------------------------------------------


def test_retry_then_succeed() -> None:
    """Judge raises ONCE for f2 then succeeds → with judge_retries=1, f2 SCORED.

    Discriminating: proves the retry actually retries (not just swallows). f2
    is in per_query (scored), nothing is skipped, and the judge was called
    twice for f2 (fail + success).
    """
    goldens, results = _three_goldens_results()
    judge = _FailingJudgeClient({"f2": 1}, ok_score=0.7)

    fres = score_goldens(
        goldens=goldens,
        retrieval_results=results,
        judge_client=judge,
        judge_retries=1,
    )

    assert set(fres.per_query) == {"f1", "f2", "f3"}, "f2 should recover on retry"
    assert fres.total_queries_skipped == 0
    assert fres.per_query["f2"].score == pytest.approx(0.7)
    assert judge.attempts["f2"] == 2, (
        f"f2 should be attempted exactly twice (fail+retry); got {judge.attempts['f2']}"
    )


def test_retry_exhausted_skips_not_crash() -> None:
    """Judge ALWAYS raises for f2 → with judge_retries=1 it is skipped after 2.

    Teeth on the bound: 1 + retries = 2 attempts, then skip. An off-by-one in
    the retry loop (e.g. 1+retries+1 or only 1) reds the attempt assertion.
    """
    goldens, results = _three_goldens_results()
    judge = _FailingJudgeClient({"f2": -1})

    fres = score_goldens(
        goldens=goldens,
        retrieval_results=results,
        judge_client=judge,
        judge_retries=1,
    )

    assert set(fres.per_query) == {"f1", "f3"}
    assert fres.total_queries_skipped == 1
    assert judge.attempts["f2"] == 2, (
        f"judge_retries=1 → exactly 2 attempts before skip; got {judge.attempts['f2']}"
    )


# ---------------------------------------------------------------------------
# Parallel path isolation
# ---------------------------------------------------------------------------


def test_parallel_path_isolation() -> None:
    """Parallel path (max_parallel_judges=3): one failing future doesn't poison.

    Same shape as the headline but through _score_parallel. f2's future raises;
    f1 and f3 still resolve and are scored; f2 is skipped+counted; the pool is
    not poisoned and the run completes.
    """
    goldens, results = _three_goldens_results()
    judge = _FailingJudgeClient({"f2": -1}, ok_score=0.8)

    fres = score_goldens(
        goldens=goldens,
        retrieval_results=results,
        judge_client=judge,
        max_parallel_judges=3,
    )

    assert set(fres.per_query) == {"f1", "f3"}
    assert fres.total_queries_scored == 2
    assert fres.total_queries_skipped == 1
    assert fres.per_query["f1"].score == pytest.approx(0.8)
    assert fres.per_query["f3"].score == pytest.approx(0.8)


def test_parallel_retry_then_succeed() -> None:
    """Parallel path also honors judge_retries (f2 fails once, recovers)."""
    goldens, results = _three_goldens_results()
    judge = _FailingJudgeClient({"f2": 1}, ok_score=0.6)

    fres = score_goldens(
        goldens=goldens,
        retrieval_results=results,
        judge_client=judge,
        max_parallel_judges=3,
        judge_retries=1,
    )

    assert set(fres.per_query) == {"f1", "f2", "f3"}
    assert fres.total_queries_skipped == 0
    assert fres.per_query["f2"].score == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# JudgeBackendError isolation (ties P15 to P14)
# ---------------------------------------------------------------------------


def test_judge_backend_error_is_isolated() -> None:
    """The claude-cli JudgeBackendError specifically is isolated (run completes).

    Ties P15 to the P14 headless backend: its typed failure must flow through
    the same per-golden isolation as any other exception.
    """
    goldens, results = _three_goldens_results()
    judge = _FailingJudgeClient(
        {"f2": -1},
        exc_factory=lambda qid: JudgeBackendError(f"claude cli judge failed {qid}"),
    )

    fres = score_goldens(goldens=goldens, retrieval_results=results, judge_client=judge)

    assert set(fres.per_query) == {"f1", "f3"}
    assert fres.total_queries_skipped == 1


def test_partial_sample_failure_skips_golden() -> None:
    """If ANY sample for a golden exhausts retries, the WHOLE golden is skipped.

    Per-sample semantics: a golden with samples_per_claim>1 whose 2nd sample
    always fails is skipped (no partial average over the 1 good sample). The
    judge here raises on every call for f2 EXCEPT it would succeed the first
    time — but because one sample fails, the golden is dropped, not averaged.
    """
    goldens = {"factoid": [_golden("f1")]}
    results = RetrievalResults(
        collection_name="C", k=5, per_query={"f1": _qresult("f1")}
    )

    # Raise on the SECOND call for f1 (first sample succeeds, second fails
    # through all retries). With judge_retries=0 that means: call1 ok, call2
    # raises → golden skipped.
    class _SecondSampleFails:
        def __init__(self) -> None:
            self.calls = 0

        def score(self, question):
            self.calls += 1
            if self.calls >= 2:
                raise RuntimeError("second sample boom")
            return JudgmentScore(score=0.9, reasoning="ok")

    judge = _SecondSampleFails()
    fres = score_goldens(
        goldens=goldens,
        retrieval_results=results,
        judge_client=judge,
        samples_per_claim=2,
        judge_retries=0,
    )

    assert fres.per_query == {}, "a partially-failed golden must be skipped, not averaged"
    assert fres.total_queries_skipped == 1


# ---------------------------------------------------------------------------
# Coverage backfill: corrupt baseline degrades to ABSENT (orchestrator)
# ---------------------------------------------------------------------------


def test_corrupted_baseline_degrades_to_absent(tmp_path: Path, monkeypatch) -> None:
    """A baseline.json with merge-conflict markers is treated as ABSENT.

    Backfills the narrow-catch degradation at orchestrator.py ~185-204: a
    corrupt/truncated baseline is a baseline problem, not an eval error, so the
    pack still runs, baseline_diff is None, and the run completes (rc reflects
    the gate only, even with fail_on_regression=True).
    """
    from tests.eval.test_cli import _make_synthetic_pack, _patch_full_chain
    from src.eval.orchestrator import run_nightly

    pack_dir = _make_synthetic_pack(
        tmp_path, recall_floor=0.5, faithfulness_floor=0.5
    )
    _patch_full_chain(monkeypatch, retrieved_source="doc1.md", judge_score=0.9)
    # Merge-conflict leftover + truncated JSON: json.loads cannot parse it.
    (Path(pack_dir) / "baseline.json").write_text(
        '<<<<<<< HEAD\n{"passed": tru', encoding="utf-8"
    )

    captured: list = []
    out_dir = tmp_path / "reports"

    rc = run_nightly(
        [pack_dir],
        output_dir=out_dir,
        now=FIXED_NOW,
        fail_on_regression=True,  # prove a corrupt baseline can't flip the exit
        alert_fn=captured.append,
    )

    assert rc == 0, "corrupt baseline must not error the pack nor flip the exit"
    assert len(captured) == 1, "the pack's alert must still fire"
    assert captured[0].baseline_diff is None, "unusable baseline → treated as absent"
    assert len(list(out_dir.glob("*.json"))) == 1, "the report was still persisted"


# ---------------------------------------------------------------------------
# Coverage backfill: ingest failure isolated per pack (run_nightly)
# ---------------------------------------------------------------------------


def test_ingest_failure_isolated_per_pack(tmp_path: Path, monkeypatch) -> None:
    """An execute_fn that raises for pack A → A marked failed, B still runs.

    Documents+tests the existing per-pack isolation for the ingest failure
    mode: run_nightly returns non-zero (A errored) but does NOT raise out, and
    healthy pack B in the same batch is still evaluated, persisted, and alerted.
    """
    from tests.eval.test_cli import _make_synthetic_pack, _patch_full_chain
    from src.eval.orchestrator import run_nightly

    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    pack_a = _make_synthetic_pack(dir_a, recall_floor=0.5)
    pack_b = _make_synthetic_pack(dir_b, recall_floor=0.5)
    _patch_full_chain(monkeypatch, retrieved_source="doc1.md", judge_score=0.9)

    # An execute_fn (ingest seam) that raises the FIRST time it is called
    # (pack A is first in the list) and succeeds thereafter (pack B). The
    # templated collection name does not encode the pack dir, so call-ordering
    # is the cleanest deterministic router here.
    from src.eval.runner import report as report_mod

    state = {"n": 0}

    def counting_execute(plan, *, fresh: bool = True):
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("ingest A blew up")
        return report_mod.IngestReport(
            collection_name=plan.collection_name,
            processed=1, skipped=0, failed=0, stored_chunks=1,
            document_count=1, plan=plan, errors=(),
        )

    captured: list = []
    out_dir = tmp_path / "reports"

    rc = run_nightly(
        [pack_a, pack_b],
        output_dir=out_dir,
        now=FIXED_NOW,
        alert_fn=captured.append,
        execute_fn=counting_execute,
    )

    assert rc == 1, "an errored (ingest-failed) pack must return non-zero"
    written = list(out_dir.glob("*.json"))
    assert len(written) == 1, f"only healthy pack B should persist, got {written!r}"
    assert json.loads(written[0].read_text())["pack_name"] == "synpack"
    assert len(captured) == 1, "pack B's alert still fires; pack A errored pre-payload"
    assert captured[0].gate_passed is True
