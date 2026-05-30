# @summary
# Tests for P7c — parallel sampling orchestration (bounded concurrent judge
# calls). Covers read_max_parallel_judges helper, score_goldens parallel-path
# call-count + result-equivalence + submission-order determinism, a non-flaky
# concurrency proof (max_concurrent >= 2), the sequential proof
# (max_concurrent == 1), ValueError guards, and CLI --max-parallel-judges
# threading. Fast tests only (no live LLM).
# Mutation A target: test_parallel_path_runs_concurrently
# Mutation B target: test_samples_preserve_submission_order_under_reordered_completion
# Mutation C target: test_score_goldens_raises_on_zero_parallel
# Exports: numerous test_* functions (see below).
# Deps: src.eval.runner, src.eval.cli
# @end-summary
"""P7c — parallel sampling orchestration tests (fast, non-flaky)."""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

import pytest

from src.eval.pack.schema import EvalPack, Golden, JudgeConfig, PackMeta, Thresholds
from src.eval.runner import (
    JudgeQuestion,
    JudgmentScore,
    QueryRetrievalResult,
    RetrievalResults,
    score_goldens,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _SequenceJudgeClient:
    """Returns a pre-recorded list of (score, reasoning) per call.

    Thread-safe call recording so it is usable from the parallel path.
    """

    def __init__(self, sequence: list[tuple[float, str]]) -> None:
        self.sequence = sequence
        self.calls: list[JudgeQuestion] = []
        self._lock = threading.Lock()

    def score(self, question: JudgeQuestion) -> JudgmentScore:
        with self._lock:
            idx = len(self.calls)
            self.calls.append(question)
        score, reasoning = self.sequence[idx % len(self.sequence)]
        return JudgmentScore(score=score, reasoning=reasoning)


class _PerSampleJudgeClient:
    """Deterministic per-(qid, submission-index) scores.

    Each qid has an ordered list ``per_qid[qid][i]`` consumed in submission
    order. Recording is thread-safe; the score returned for the i-th call of a
    given qid is the i-th element regardless of which thread runs it, so
    submission order is recoverable from the stored ``samples`` tuple.
    """

    def __init__(self, per_qid: dict[str, list[tuple[float, str]]]) -> None:
        self.per_qid = per_qid
        self.calls: list[JudgeQuestion] = []
        self._counters: dict[str, int] = {}
        self._lock = threading.Lock()

    def score(self, question: JudgeQuestion) -> JudgmentScore:
        with self._lock:
            self.calls.append(question)
            i = self._counters.get(question.qid, 0)
            self._counters[question.qid] = i + 1
        score, reasoning = self.per_qid[question.qid][i]
        return JudgmentScore(score=score, reasoning=reasoning)


class _ConcurrencyProbeJudgeClient:
    """Records peak in-flight concurrency under a lock.

    Each ``score`` call increments an ``active`` counter, updates
    ``max_concurrent``, then waits on a :class:`threading.Barrier` so that
    (when the pool allows it) at least ``parties`` threads are simultaneously
    in flight. A finite timeout means a too-small pool cannot deadlock — the
    barrier raises ``BrokenBarrierError`` after the timeout and the call
    proceeds, yielding ``max_concurrent < parties`` (which the sequential test
    asserts on).
    """

    def __init__(self, parties: int = 2, timeout: float = 2.0) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.max_concurrent = 0
        self.calls = 0
        self._barrier = threading.Barrier(parties, timeout=timeout)

    def score(self, question: JudgeQuestion) -> JudgmentScore:
        with self._lock:
            self.active += 1
            self.calls += 1
            if self.active > self.max_concurrent:
                self.max_concurrent = self.active
        try:
            self._barrier.wait()
        except threading.BrokenBarrierError:
            # Too few threads ever reached the barrier (sequential path or
            # pool smaller than ``parties``). Not an error for this probe.
            pass
        with self._lock:
            self.active -= 1
        return JudgmentScore(score=0.5, reasoning="probe")


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


def _make_pack(
    max_parallel_judges_value: Any = 1,
    *,
    drop_field: bool = False,
) -> EvalPack:
    """Minimal in-memory EvalPack with a configurable max_parallel_judges.

    ``drop_field=True`` constructs JudgeConfig without the field to exercise
    the read_max_parallel_judges defensive default.
    """
    if drop_field:
        judge = JudgeConfig.model_construct(
            tier1_model="fake",
            tier1_prompt_version="v1",
            temperature=0.0,
            samples_per_claim=1,
        )
    else:
        judge = JudgeConfig(
            tier1_model="fake",
            tier1_prompt_version="v1",
            temperature=0.0,
            samples_per_claim=1,
            max_parallel_judges=max_parallel_judges_value,
        )
    meta = PackMeta(
        name="syn",
        version=1,
        profile="generic",
        corpus_pin="0" * 64,
        judge=judge,
        collection_name_template="c_{name}_{corpus_pin_short}",
    )
    return EvalPack(
        meta=meta,
        manifest=[],
        goldens={"factoid": [_golden("f1")]},
        thresholds=Thresholds(profile="generic"),
    )


# ---------------------------------------------------------------------------
# read_max_parallel_judges
# ---------------------------------------------------------------------------


def test_read_max_parallel_judges_defaults_to_one_if_missing() -> None:
    """Pack without max_parallel_judges field → defaults to 1."""
    from src.eval.runner import read_max_parallel_judges

    pack = _make_pack(drop_field=True)
    assert read_max_parallel_judges(pack) == 1


def test_read_max_parallel_judges_returns_pack_value() -> None:
    """Pack with max_parallel_judges=4 → returns 4."""
    from src.eval.runner import read_max_parallel_judges

    pack = _make_pack(max_parallel_judges_value=4)
    assert read_max_parallel_judges(pack) == 4


def test_read_max_parallel_judges_none_maps_to_one() -> None:
    """Explicit None on the config → coerced to 1."""
    from src.eval.runner import read_max_parallel_judges

    pack = _make_pack(drop_field=True)
    pack.meta.judge.max_parallel_judges = None  # type: ignore[attr-defined]
    assert read_max_parallel_judges(pack) == 1


def test_read_max_parallel_judges_raises_on_zero_or_negative() -> None:
    """max_parallel_judges < 1 raises ValueError."""
    from src.eval.runner import read_max_parallel_judges

    pack = _make_pack(max_parallel_judges_value=0)
    with pytest.raises(ValueError, match="max_parallel_judges"):
        read_max_parallel_judges(pack)


# ---------------------------------------------------------------------------
# score_goldens — guards
# ---------------------------------------------------------------------------


def test_score_goldens_raises_on_zero_parallel() -> None:
    """max_parallel_judges=0 raises ValueError. (Mutation C target.)"""
    goldens = {"factoid": [_golden("f1")]}
    results = RetrievalResults(
        collection_name="C", k=5, per_query={"f1": _qresult("f1")}
    )
    judge = _SequenceJudgeClient([(0.5, "r")])
    with pytest.raises(ValueError, match="max_parallel_judges"):
        score_goldens(results, goldens, judge, max_parallel_judges=0)


def test_score_goldens_raises_on_negative_parallel() -> None:
    """max_parallel_judges=-2 raises ValueError."""
    goldens = {"factoid": [_golden("f1")]}
    results = RetrievalResults(
        collection_name="C", k=5, per_query={"f1": _qresult("f1")}
    )
    judge = _SequenceJudgeClient([(0.5, "r")])
    with pytest.raises(ValueError, match="max_parallel_judges"):
        score_goldens(results, goldens, judge, max_parallel_judges=-2)


# ---------------------------------------------------------------------------
# score_goldens — parallel-path call count + correctness
# ---------------------------------------------------------------------------


def test_parallel_path_call_count() -> None:
    """3 goldens × 2 samples → 6 judge calls under the parallel path."""
    goldens = {"factoid": [_golden("f1"), _golden("f2"), _golden("f3")]}
    per_query = {
        "f1": _qresult("f1"),
        "f2": _qresult("f2"),
        "f3": _qresult("f3"),
    }
    results = RetrievalResults(collection_name="C", k=5, per_query=per_query)
    judge = _SequenceJudgeClient([(0.5, "r")] * 6)

    fres = score_goldens(
        results, goldens, judge, samples_per_claim=2, max_parallel_judges=3
    )

    assert len(judge.calls) == 6, (
        f"expected 3 goldens × 2 samples = 6 calls; got {len(judge.calls)}"
    )
    qid_counts: dict[str, int] = {}
    for c in judge.calls:
        qid_counts[c.qid] = qid_counts.get(c.qid, 0) + 1
    assert qid_counts == {"f1": 2, "f2": 2, "f3": 2}
    assert fres.total_queries_scored == 3


def test_parallel_result_equals_sequential_result() -> None:
    """Parallel path produces identical per_query results to sequential.

    Scores are chosen so mean / first / max / last all differ, so any
    aggregation drift would surface. Both paths use a fresh deterministic
    fake keyed by (qid, submission-index).
    """
    goldens = {"factoid": [_golden("f1"), _golden("f2")]}
    per_query = {"f1": _qresult("f1"), "f2": _qresult("f2")}
    results = RetrievalResults(collection_name="C", k=5, per_query=per_query)

    scores = {
        "f1": [(0.2, "r1a"), (0.9, "r1b"), (0.5, "r1c")],
        "f2": [(0.3, "r2a"), (0.4, "r2b"), (0.8, "r2c")],
    }

    seq = score_goldens(
        results,
        goldens,
        _PerSampleJudgeClient({k: list(v) for k, v in scores.items()}),
        samples_per_claim=3,
        max_parallel_judges=1,
    )
    par = score_goldens(
        results,
        goldens,
        _PerSampleJudgeClient({k: list(v) for k, v in scores.items()}),
        samples_per_claim=3,
        max_parallel_judges=4,
    )

    assert set(seq.per_query) == set(par.per_query)
    for qid in seq.per_query:
        s = seq.per_query[qid]
        p = par.per_query[qid]
        assert p.score == pytest.approx(s.score), qid
        assert p.reasoning == s.reasoning, qid
        assert [x.score for x in p.samples] == [x.score for x in s.samples], qid
        assert [x.reasoning for x in p.samples] == [
            x.reasoning for x in s.samples
        ], qid
    assert seq.total_queries_scored == par.total_queries_scored
    assert seq.total_queries_skipped == par.total_queries_skipped


def test_samples_preserve_submission_order_under_reordered_completion() -> None:
    """samples tuple is in submission order even when completion order differs.

    (Mutation B target.) Submission-index 0 sleeps the LONGEST, so it
    completes LAST. If the regroup used completion order, samples[0] would
    carry the fast sample's score. We pin samples[0].score to the score
    assigned to submission-index 0. Scores chosen so first/max/last/mean all
    differ.
    """
    import time

    class _OrderingFake:
        def __init__(self) -> None:
            self._lock = threading.Lock()
            self._i = 0
            # submission-index -> (sleep_seconds, score, reasoning)
            self._plan = {
                0: (0.30, 0.10, "first"),
                1: (0.15, 0.90, "second"),
                2: (0.01, 0.55, "third"),
            }

        def score(self, question: JudgeQuestion) -> JudgmentScore:
            with self._lock:
                i = self._i
                self._i += 1
            sleep_s, sc, reason = self._plan[i]
            time.sleep(sleep_s)
            return JudgmentScore(score=sc, reasoning=reason)

    goldens = {"factoid": [_golden("f1")]}
    results = RetrievalResults(
        collection_name="C", k=5, per_query={"f1": _qresult("f1")}
    )
    judge = _OrderingFake()

    fres = score_goldens(
        results, goldens, judge, samples_per_claim=3, max_parallel_judges=3
    )

    samples = fres.per_query["f1"].samples
    assert [s.score for s in samples] == [0.10, 0.90, 0.55], (
        "samples must be in submission order (0,1,2), not completion order; "
        f"got {[s.score for s in samples]}"
    )
    assert [s.reasoning for s in samples] == ["first", "second", "third"]


# ---------------------------------------------------------------------------
# Concurrency / sequential proofs
# ---------------------------------------------------------------------------


def test_parallel_path_runs_concurrently() -> None:
    """max_parallel_judges=2 → at least 2 judge calls overlap in flight.

    (Mutation A target.) Non-flaky: uses a Barrier(2, timeout) latch, not a
    timing assertion. Two goldens × 1 sample = 2 calls; with a 2-wide pool
    both reach the barrier together and max_concurrent becomes 2.
    """
    goldens = {"factoid": [_golden("f1"), _golden("f2")]}
    per_query = {"f1": _qresult("f1"), "f2": _qresult("f2")}
    results = RetrievalResults(collection_name="C", k=5, per_query=per_query)
    probe = _ConcurrencyProbeJudgeClient(parties=2, timeout=5.0)

    score_goldens(results, goldens, probe, max_parallel_judges=2)

    assert probe.calls == 2
    assert probe.max_concurrent >= 2, (
        "expected >= 2 concurrent judge calls under max_parallel_judges=2; "
        f"observed peak {probe.max_concurrent} (the barrier timed out, "
        "meaning calls never overlapped → parallelism did not happen)"
    )


def test_sequential_path_is_not_concurrent() -> None:
    """max_parallel_judges=1 → calls never overlap (peak concurrency == 1)."""
    goldens = {"factoid": [_golden("f1"), _golden("f2")]}
    per_query = {"f1": _qresult("f1"), "f2": _qresult("f2")}
    results = RetrievalResults(collection_name="C", k=5, per_query=per_query)
    # Short timeout: the lone in-flight call will harmlessly time out at the
    # barrier (no second thread arrives) and proceed.
    probe = _ConcurrencyProbeJudgeClient(parties=2, timeout=0.3)

    score_goldens(results, goldens, probe, max_parallel_judges=1)

    assert probe.calls == 2
    assert probe.max_concurrent == 1, (
        f"sequential path must never overlap; peak was {probe.max_concurrent}"
    )


# ---------------------------------------------------------------------------
# CLI --max-parallel-judges threading
# ---------------------------------------------------------------------------


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _make_synthetic_pack_dir(
    tmp_path: Path,
    *,
    max_parallel_judges: int = 1,
) -> Path:
    """Mirror the multi-sample CLI test pack with a custom max_parallel_judges."""
    import yaml

    pack_dir = tmp_path / "synthetic_pack"
    pack_dir.mkdir()
    (pack_dir / "corpus").mkdir()
    (pack_dir / "goldens").mkdir()

    doc_text = "alpha beta gamma\n"
    doc_path = pack_dir / "corpus" / "doc1.md"
    doc_path.write_text(doc_text)
    doc_sha = _sha256_text(doc_text)
    manifest = {"entries": [{"path": "doc1.md", "sha256": doc_sha}]}
    (pack_dir / "corpus" / "manifest.json").write_text(json.dumps(manifest))
    corpus_pin = hashlib.sha256(f"doc1.md:{doc_sha}".encode("utf-8")).hexdigest()

    pack_yaml = {
        "name": "synpack",
        "version": 1,
        "profile": "generic",
        "corpus_pin": corpus_pin,
        "description": "synthetic test pack",
        "judge": {
            "tier1_model": "fake-model",
            "tier1_prompt_version": "v1",
            "temperature": 0.0,
            "samples_per_claim": 1,
            "max_parallel_judges": max_parallel_judges,
        },
        "collection_name_template": "test_{name}_{corpus_pin_short}",
    }
    (pack_dir / "pack.yaml").write_text(yaml.safe_dump(pack_yaml))

    golden = {
        "qid": "g1",
        "qtype": "factoid",
        "query": "what is alpha?",
        "expected_answer_span": "alpha",
        "expected_source_docs": ["doc1.md"],
    }
    (pack_dir / "goldens" / "factoid.jsonl").write_text(json.dumps(golden) + "\n")

    thresholds = {
        "profile": "generic",
        "defaults": {"factoid": {"recall_at_5": 0.5, "faithfulness": 0.5}},
        "min_goldens_per_qtype": {"factoid": 1},
    }
    (pack_dir / "thresholds.yaml").write_text(yaml.safe_dump(thresholds))
    return pack_dir


class _RecordingFakeJudgeClient:
    """Captures the max_parallel_judges value score_goldens applies.

    Records peak concurrency so the CLI test can verify the override reaches
    score_goldens' parallel path.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.max_concurrent = 0
        self.calls = 0
        self._barrier = threading.Barrier(2, timeout=5.0)

    def score(self, question):
        with self._lock:
            self.active += 1
            self.calls += 1
            if self.active > self.max_concurrent:
                self.max_concurrent = self.active
        try:
            self._barrier.wait()
        except threading.BrokenBarrierError:
            pass
        with self._lock:
            self.active -= 1
        return JudgmentScore(score=0.9, reasoning="ok")


class _FakeEmbedder:
    def embed_documents(self, texts):
        return [[0.0] * 8 for _ in texts]


def _wire_cli_stubs(monkeypatch, fake_judge):
    """Monkeypatch the heavy CLI seams (ingest/retrieve/judge)."""
    from src.eval.runner import retrieve as retrieve_mod
    from src.eval.runner import report as report_mod
    import src.eval.runner as runner_pkg
    from src.vector_db.common.schemas import SearchResult

    def fake_execute_plan(plan, *, fresh: bool = True):
        return report_mod.IngestReport(
            collection_name=plan.collection_name,
            processed=2, skipped=0, failed=0, stored_chunks=2,
            document_count=2, plan=plan, errors=(),
        )

    def fake_search(client, query, query_embedding, alpha, limit, **kwargs):
        return [SearchResult(text="alpha", score=1.0, metadata={"source": "doc1.md"})]

    monkeypatch.setattr(runner_pkg, "execute_plan", fake_execute_plan)
    monkeypatch.setattr(retrieve_mod, "search", fake_search)
    monkeypatch.setattr(retrieve_mod, "get_embedding_provider", lambda: _FakeEmbedder())
    monkeypatch.setattr(retrieve_mod, "create_persistent_client", lambda: object())
    monkeypatch.setattr(retrieve_mod, "close_client", lambda _c: None)
    monkeypatch.setattr(
        runner_pkg, "build_judge_client", lambda pack, **kw: fake_judge
    )


def _append_second_golden(pack_dir: Path) -> None:
    """Append a second factoid golden so two concurrent calls are possible."""
    g2 = {
        "qid": "g2",
        "qtype": "factoid",
        "query": "what is beta?",
        "expected_answer_span": "beta",
        "expected_source_docs": ["doc1.md"],
    }
    factoid_path = pack_dir / "goldens" / "factoid.jsonl"
    factoid_path.write_text(factoid_path.read_text() + json.dumps(g2) + "\n")


def test_cli_max_parallel_judges_flag_overrides_pack_meta(
    tmp_path: Path, monkeypatch
) -> None:
    """run_cli(max_parallel_judges=2) with pack default=1 → concurrency happens.

    Pack declares the SEQUENTIAL default (1); the two goldens would never
    overlap on that path (barrier times out, peak == 1). The CLI override of 2
    routes through the bounded pool so the barrier latches and peak
    concurrency reaches 2 — proving the override beats the pack value AND
    reaches score_goldens' parallel path.
    """
    from src.eval.cli import run_cli

    pack_dir = _make_synthetic_pack_dir(tmp_path, max_parallel_judges=1)
    _append_second_golden(pack_dir)
    fake_judge = _RecordingFakeJudgeClient()
    _wire_cli_stubs(monkeypatch, fake_judge)

    rc = run_cli(str(pack_dir), max_parallel_judges=2)
    assert rc == 0
    assert fake_judge.calls == 2
    assert fake_judge.max_concurrent >= 2, (
        "CLI override 2 should beat pack default 1 and drive the parallel "
        f"path; observed peak {fake_judge.max_concurrent}"
    )


def test_cli_max_parallel_judges_falls_back_to_pack_value(
    tmp_path: Path, monkeypatch
) -> None:
    """run_cli without the flag uses read_max_parallel_judges(pack).

    Pack declares max_parallel_judges=2 and two goldens are present, so the
    parallel path latches the barrier (peak concurrency == 2). If the CLI
    ignored the pack value (fell back to 1), the barrier would time out.
    """
    from src.eval.cli import run_cli

    pack_dir = _make_synthetic_pack_dir(tmp_path, max_parallel_judges=2)
    _append_second_golden(pack_dir)
    fake_judge = _RecordingFakeJudgeClient()
    _wire_cli_stubs(monkeypatch, fake_judge)

    rc = run_cli(str(pack_dir))  # no --max-parallel-judges override
    assert rc == 0
    assert fake_judge.calls == 2
    assert fake_judge.max_concurrent >= 2, (
        "pack max_parallel_judges=2 should drive the parallel path; "
        f"observed peak {fake_judge.max_concurrent}"
    )


def test_cli_max_parallel_judges_parses(monkeypatch) -> None:
    """--max-parallel-judges parses to an int on the run subcommand."""
    from src.eval.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["run", "/tmp/pack", "--max-parallel-judges", "4"])
    assert args.max_parallel_judges == 4
    # absent → None (so run_cli falls back to the pack value)
    args2 = parser.parse_args(["run", "/tmp/pack"])
    assert args2.max_parallel_judges is None
