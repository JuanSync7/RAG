# @summary
# Tests for P5 faithfulness scoring executor (7 fast + 1 dual-marker live).
# Asserts score_goldens call wiring, FaithfulnessResults shape,
# aggregate_faithfulness_by_qtype contract, QueryRetrievalResult additive
# extension, EvalReport additive extension, build_eval_report wiring,
# adversarial None-span handling, plus a live opentitan smoke test.
# Exports: test_score_goldens_calls_judge_per_query,
#          test_score_goldens_returns_frozen_faithfulness_results,
#          test_aggregate_faithfulness_by_qtype,
#          test_retrieve_for_goldens_now_includes_chunk_texts,
#          test_eval_report_extended_with_faithfulness,
#          test_build_eval_report_wires_recall_and_faithfulness,
#          test_judge_question_for_adversarial_carries_none_span,
#          test_score_goldens_opentitan_live
# Deps: src.eval.runner, src.eval.pack
# @end-summary
"""P5 — faithfulness scoring executor tests."""
from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any

import pytest

from src.eval.pack.schema import Golden
from src.eval.runner import (
    EvalReport,
    FaithfulnessResults,
    JudgeQuestion,
    JudgmentScore,
    QueryFaithfulnessResult,
    QueryRetrievalResult,
    RetrievalResults,
    aggregate_faithfulness_by_qtype,
    build_eval_report,
    score_goldens,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PACK_DIR = REPO_ROOT / "evals" / "packs" / "opentitan_riscv"


# ---------------------------------------------------------------------------
# Fake judge client (records calls, returns configurable scores per qid)
# ---------------------------------------------------------------------------


class _FakeJudgeClient:
    def __init__(self, scores: dict[str, float] | None = None,
                 default: float = 0.7) -> None:
        self.scores = scores or {}
        self.default = default
        self.calls: list[JudgeQuestion] = []

    def score(self, question: JudgeQuestion) -> JudgmentScore:
        self.calls.append(question)
        s = self.scores.get(question.qid, self.default)
        return JudgmentScore(score=s, reasoning=f"score-for-{question.qid}")


def _golden(
    qid: str,
    qtype: str,
    query: str,
    span: str | None = "x",
    sources: list[str] | None = None,
) -> Golden:
    return Golden(
        qid=qid,
        qtype=qtype,
        query=query,
        expected_answer_span=span,
        expected_source_docs=sources or [f"docs/{qid}.md"],
    )


def _qresult(
    qid: str,
    qtype: str,
    query: str,
    chunks: tuple[str, ...],
    sources: tuple[str, ...] = (),
    k: int = 5,
) -> QueryRetrievalResult:
    return QueryRetrievalResult(
        qid=qid,
        qtype=qtype,
        query=query,
        retrieved_sources=sources or tuple(f"docs/{qid}.md" for _ in chunks),
        retrieved_chunks=chunks,
        k=k,
    )


# ---------------------------------------------------------------------------
# Test 1 — score_goldens calls judge exactly once per non-empty result
# ---------------------------------------------------------------------------


def test_score_goldens_calls_judge_per_query() -> None:
    goldens = {
        "factoid": [
            _golden("f1", "factoid", "What is alpha?", span="alpha-span"),
            _golden("f2", "factoid", "What is beta?", span="beta-span"),
        ],
        "qfs": [
            _golden("q1", "qfs", "Summarize gamma", span=None),
        ],
    }
    per_query = {
        "f1": _qresult("f1", "factoid", "What is alpha?",
                       ("chunk-f1-A", "chunk-f1-B")),
        "f2": _qresult("f2", "factoid", "What is beta?",
                       ("chunk-f2-A",)),
        # q1 has empty retrieved_chunks — should be skipped.
        "q1": _qresult("q1", "qfs", "Summarize gamma", ()),
    }
    results = RetrievalResults(collection_name="C", k=5, per_query=per_query)
    judge = _FakeJudgeClient(scores={"f1": 0.9, "f2": 0.4})

    fres = score_goldens(results, goldens, judge)

    # Exactly one call per golden with non-empty retrieved_chunks.
    assert len(judge.calls) == 2
    by_qid = {c.qid: c for c in judge.calls}
    assert set(by_qid) == {"f1", "f2"}
    # JudgeQuestion fields match golden+retrieval.
    assert by_qid["f1"].qtype == "factoid"
    assert by_qid["f1"].query == "What is alpha?"
    assert by_qid["f1"].expected_answer_span == "alpha-span"
    assert by_qid["f1"].chunk_texts == ("chunk-f1-A", "chunk-f1-B")
    assert by_qid["f2"].chunk_texts == ("chunk-f2-A",)
    # Skip accounting.
    assert fres.total_queries_skipped == 1
    assert fres.total_queries_scored == 2
    assert set(fres.per_query) == {"f1", "f2"}


# ---------------------------------------------------------------------------
# Test 2 — FaithfulnessResults frozen, per-query shape
# ---------------------------------------------------------------------------


def test_score_goldens_returns_frozen_faithfulness_results() -> None:
    goldens = {
        "factoid": [_golden("f1", "factoid", "qa", span="s")],
    }
    per_query = {
        "f1": _qresult("f1", "factoid", "qa", ("chunk-A",)),
    }
    results = RetrievalResults(collection_name="Cn", k=5, per_query=per_query)
    judge = _FakeJudgeClient(scores={"f1": 0.55})

    fres = score_goldens(results, goldens, judge)

    assert dataclasses.is_dataclass(fres)
    assert fres.collection_name == "Cn"
    with pytest.raises(dataclasses.FrozenInstanceError):
        fres.total_queries_scored = 99  # type: ignore[misc]

    q1 = fres.per_query["f1"]
    assert dataclasses.is_dataclass(q1)
    assert q1.qid == "f1"
    assert q1.qtype == "factoid"
    assert q1.score == pytest.approx(0.55)
    assert q1.reasoning == "score-for-f1"
    with pytest.raises(dataclasses.FrozenInstanceError):
        q1.score = 0.1  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Test 3 — aggregate_faithfulness_by_qtype
# ---------------------------------------------------------------------------


def test_aggregate_faithfulness_by_qtype() -> None:
    goldens = {
        "factoid": [
            _golden("f1", "factoid", "qa", span="s"),
            _golden("f2", "factoid", "qb", span="s"),
        ],
        "qfs": [
            _golden("q1", "qfs", "qc", span="s"),
        ],
        "adversarial": [
            _golden("a1", "adversarial", "qd", span=None),
        ],
    }
    # adversarial a1 has empty chunks -> skipped.
    per_query_faith = {
        "f1": QueryFaithfulnessResult(qid="f1", qtype="factoid",
                                       score=0.9, reasoning="r"),
        "f2": QueryFaithfulnessResult(qid="f2", qtype="factoid",
                                       score=0.3, reasoning="r"),
        "q1": QueryFaithfulnessResult(qid="q1", qtype="qfs",
                                       score=0.5, reasoning="r"),
    }
    fres = FaithfulnessResults(
        collection_name="C",
        per_query=per_query_faith,
        total_queries_scored=3,
        total_queries_skipped=1,
    )

    out = aggregate_faithfulness_by_qtype(fres, goldens)

    assert out["factoid"] == pytest.approx(0.6)
    assert out["qfs"] == pytest.approx(0.5)
    # adversarial has zero scored goldens -> omitted.
    assert "adversarial" not in out


# ---------------------------------------------------------------------------
# Test 4 — retrieve_for_goldens now populates retrieved_chunks
# ---------------------------------------------------------------------------


def test_retrieve_for_goldens_now_includes_chunk_texts(monkeypatch) -> None:
    from src.eval.runner import retrieve as retrieve_mod
    from src.eval.runner import retrieve_for_goldens
    from src.vector_db.common.schemas import SearchResult

    class _FakeEmbedder:
        def embed_documents(self, texts):
            return [[0.0] * 1024 for _ in texts]

    def fake_search(client, query, query_embedding, alpha, limit, **kwargs):
        return [
            SearchResult(text="chunk-A", score=1.0,
                         metadata={"source": "docs/a.md"}),
            SearchResult(text="chunk-B", score=0.9,
                         metadata={"source": "docs/b.md"}),
        ]

    monkeypatch.setattr(retrieve_mod, "search", fake_search)
    monkeypatch.setattr(retrieve_mod, "get_embedding_provider",
                        lambda: _FakeEmbedder())
    monkeypatch.setattr(retrieve_mod, "create_persistent_client",
                        lambda: object())
    monkeypatch.setattr(retrieve_mod, "close_client", lambda _client: None)

    goldens = {
        "factoid": [
            _golden("f1", "factoid", "alpha", span="x"),
        ],
    }
    res = retrieve_for_goldens("C", goldens, k=5)

    q = res.per_query["f1"]
    # P5 additive contract — both fields populated.
    assert q.retrieved_chunks == ("chunk-A", "chunk-B")
    assert q.retrieved_sources == ("docs/a.md", "docs/b.md")


# ---------------------------------------------------------------------------
# Test 5 — EvalReport additive extension
# ---------------------------------------------------------------------------


def test_eval_report_extended_with_faithfulness() -> None:
    # Existing P4 construction (no faithfulness fields) still works.
    legacy = EvalReport(
        collection_name="C",
        k=5,
        per_query_recall={"f1": 1.0},
        recall_by_qtype={"factoid": 1.0},
        total_queries_scored=1,
        total_queries_skipped=0,
    )
    assert legacy.faithfulness_by_qtype == {}
    assert legacy.total_queries_judged == 0

    # New fields populated.
    extended = EvalReport(
        collection_name="C",
        k=5,
        per_query_recall={"f1": 1.0},
        recall_by_qtype={"factoid": 1.0},
        total_queries_scored=1,
        total_queries_skipped=0,
        faithfulness_by_qtype={"factoid": 0.8},
        total_queries_judged=1,
    )
    assert extended.faithfulness_by_qtype == {"factoid": 0.8}
    assert extended.total_queries_judged == 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        extended.faithfulness_by_qtype = {}  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Test 6 — build_eval_report wires recall + faithfulness coherently
# ---------------------------------------------------------------------------


def test_build_eval_report_wires_recall_and_faithfulness() -> None:
    goldens = {
        "factoid": [
            _golden("f1", "factoid", "qa", span="s"),
            _golden("f2", "factoid", "qb", span="s"),
        ],
    }
    per_query = {
        "f1": _qresult("f1", "factoid", "qa", ("chunk-A",)),
        "f2": _qresult("f2", "factoid", "qb", ("chunk-B",)),
    }
    rresults = RetrievalResults(collection_name="C", k=5, per_query=per_query)
    # Varied judge scores to give aggregation real signal (anti-gaming for
    # mutation probe B which hardcodes 0.5).
    judge = _FakeJudgeClient(scores={"f1": 0.9, "f2": 0.3})
    fres = score_goldens(rresults, goldens, judge)

    recall_by_qtype = {"factoid": 0.75}
    per_query_recall = {"f1": 1.0, "f2": 0.5}

    report = build_eval_report(
        rresults, fres, recall_by_qtype, per_query_recall
    )

    assert isinstance(report, EvalReport)
    assert report.collection_name == "C"
    assert report.k == 5
    assert report.recall_by_qtype == {"factoid": 0.75}
    assert report.per_query_recall == {"f1": 1.0, "f2": 0.5}
    # Faithfulness aggregation should reflect varied scores (mean(0.9, 0.3)=0.6).
    assert report.faithfulness_by_qtype["factoid"] == pytest.approx(0.6)
    assert report.total_queries_judged == fres.total_queries_scored == 2


# ---------------------------------------------------------------------------
# Test 7 — adversarial goldens with None span flow through
# ---------------------------------------------------------------------------


def test_judge_question_for_adversarial_carries_none_span() -> None:
    goldens = {
        "adversarial": [
            _golden("a1", "adversarial", "When did Z happen?", span=None),
        ],
    }
    per_query = {
        "a1": _qresult("a1", "adversarial", "When did Z happen?",
                       ("chunk-irrelevant",)),
    }
    rresults = RetrievalResults(collection_name="C", k=5, per_query=per_query)
    judge = _FakeJudgeClient(scores={"a1": 0.85})

    fres = score_goldens(rresults, goldens, judge)

    assert len(judge.calls) == 1
    call = judge.calls[0]
    assert call.qid == "a1"
    assert call.qtype == "adversarial"
    assert call.expected_answer_span is None
    assert call.chunk_texts == ("chunk-irrelevant",)
    assert fres.per_query["a1"].score == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# Test 8 — live opentitan smoke test (dual-marker)
# ---------------------------------------------------------------------------


def _skip_if_no_judge_llm() -> None:
    if not (
        os.environ.get("LLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
    ):
        pytest.skip("No LLM API key in env (LLM_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY)")


@pytest.mark.slow
@pytest.mark.integration
def test_score_goldens_opentitan_live(tmp_path: Path) -> None:
    """Live: faithfulness over opentitan_riscv pack (≤2 factoid goldens)."""
    pytest.importorskip("weaviate")
    import socket

    from config.settings import MINIO_ENDPOINT
    from src.eval.pack import load_pack
    from src.eval.runner import (
        build_judge_client,
        execute_plan,
        plan_pack_ingest,
        retrieve_for_goldens,
    )
    from src.vector_db import close_client, create_persistent_client

    try:
        client = create_persistent_client()
    except Exception as exc:  # pragma: no cover - infra-dependent
        pytest.skip(f"Live Weaviate not reachable: {exc}")

    try:
        host, _, port = MINIO_ENDPOINT.partition(":")
        with socket.create_connection(
            (host or "localhost", int(port or "9000")), timeout=2
        ):
            pass
    except Exception as exc:  # pragma: no cover - infra-dependent
        try:
            close_client(client)
        except Exception:
            pass
        pytest.skip(f"Live MinIO ({MINIO_ENDPOINT}) not reachable: {exc}")

    _skip_if_no_judge_llm()

    pack = load_pack(PACK_DIR)
    plan = plan_pack_ingest(pack, PACK_DIR)

    # Limit to first 2 factoid goldens to bound LLM cost.
    factoid_goldens = pack.goldens.get("factoid", [])[:2]
    bounded_goldens = {"factoid": factoid_goldens}

    report = None
    try:
        report = execute_plan(plan)
        assert report.failed == 0, f"unexpected ingest failures: {report.errors}"

        rresults = retrieve_for_goldens(plan.collection_name,
                                         bounded_goldens, k=5)
        judge_client = build_judge_client(pack, pack_dir=PACK_DIR)
        fres = score_goldens(rresults, bounded_goldens, judge_client)
        faith_by_qtype = aggregate_faithfulness_by_qtype(fres, bounded_goldens)
        print(f"faithfulness_by_qtype={faith_by_qtype}")

        assert fres.total_queries_scored >= 1, "no factoid was scored"
        assert faith_by_qtype.get("factoid", 0.0) >= 0.3, (
            f"factoid faithfulness < 0.3 floor; got {faith_by_qtype}"
        )
    finally:
        try:
            target = report.collection_name if report else plan.collection_name
            if client.collections.exists(target):
                client.collections.delete(target)
        except Exception:
            pass
        try:
            close_client(client)
        except Exception:
            pass
