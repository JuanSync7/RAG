# @summary
# Tests for P6c — multi-sample judge (averaged faithfulness scoring).
# Covers read_samples_per_claim helper, score_goldens looping/averaging/
# best-reasoning selection, backwards-compat default, and CLI override
# threading. Fast tests only (no live LLM).
# Mutation A target: test_score_goldens_calls_judge_N_times_per_golden
# Mutation B target: test_score_goldens_averages_scores
# Mutation C target: test_cli_samples_per_claim_flag_overrides_pack_meta
# Exports: test_read_samples_per_claim_defaults_to_one_if_missing,
#          test_read_samples_per_claim_returns_pack_value,
#          test_read_samples_per_claim_raises_on_zero_or_negative,
#          test_score_goldens_calls_judge_N_times_per_golden,
#          test_score_goldens_averages_scores,
#          test_score_goldens_picks_highest_scored_reasoning,
#          test_score_goldens_samples_per_claim_one_is_backwards_compat,
#          test_cli_samples_per_claim_flag_overrides_pack_meta
# Deps: src.eval.runner, src.eval.cli
# @end-summary
"""P6c — multi-sample judge tests (8 fast)."""
from __future__ import annotations

import hashlib
import json
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

    Cycles through the sequence in order across all goldens.
    """

    def __init__(self, sequence: list[tuple[float, str]]) -> None:
        self.sequence = sequence
        self.calls: list[JudgeQuestion] = []

    def score(self, question: JudgeQuestion) -> JudgmentScore:
        idx = len(self.calls)
        self.calls.append(question)
        score, reasoning = self.sequence[idx % len(self.sequence)]
        return JudgmentScore(score=score, reasoning=reasoning)


class _PerGoldenJudgeClient:
    """Returns a fresh sequence of (score, reasoning) restarting per qid."""

    def __init__(self, per_qid: dict[str, list[tuple[float, str]]]) -> None:
        self.per_qid = per_qid
        self.calls: list[JudgeQuestion] = []
        self._counters: dict[str, int] = {}

    def score(self, question: JudgeQuestion) -> JudgmentScore:
        self.calls.append(question)
        i = self._counters.get(question.qid, 0)
        self._counters[question.qid] = i + 1
        score, reasoning = self.per_qid[question.qid][i]
        return JudgmentScore(score=score, reasoning=reasoning)


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


def _make_pack(samples_per_claim_value: Any = 1, *, drop_field: bool = False) -> EvalPack:
    """Build a minimal in-memory EvalPack.

    When ``drop_field=True`` the JudgeConfig is constructed via
    ``model_construct`` without ``samples_per_claim`` to simulate a legacy/
    missing-field pack (read_samples_per_claim defensive default path).
    """
    if drop_field:
        judge = JudgeConfig.model_construct(
            tier1_model="fake",
            tier1_prompt_version="v1",
            temperature=0.0,
        )
    else:
        judge = JudgeConfig(
            tier1_model="fake",
            tier1_prompt_version="v1",
            temperature=0.0,
            samples_per_claim=samples_per_claim_value,
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
# read_samples_per_claim
# ---------------------------------------------------------------------------


def test_read_samples_per_claim_defaults_to_one_if_missing() -> None:
    """Pack without samples_per_claim field → defaults to 1."""
    from src.eval.runner import read_samples_per_claim

    pack = _make_pack(drop_field=True)
    assert read_samples_per_claim(pack) == 1


def test_read_samples_per_claim_returns_pack_value() -> None:
    """Pack with samples_per_claim=3 → returns 3."""
    from src.eval.runner import read_samples_per_claim

    pack = _make_pack(samples_per_claim_value=3)
    assert read_samples_per_claim(pack) == 3


def test_read_samples_per_claim_raises_on_zero_or_negative() -> None:
    """samples_per_claim=0 raises ValueError."""
    from src.eval.runner import read_samples_per_claim

    # JudgeConfig validates int type but not >=1; the helper enforces it.
    pack = _make_pack(samples_per_claim_value=0)
    with pytest.raises(ValueError, match="samples_per_claim"):
        read_samples_per_claim(pack)


# ---------------------------------------------------------------------------
# score_goldens with samples_per_claim
# ---------------------------------------------------------------------------


def test_score_goldens_calls_judge_N_times_per_golden() -> None:
    """2 goldens × N=3 samples → 6 total judge calls. (Mutation A target.)"""
    goldens = {
        "factoid": [_golden("f1"), _golden("f2")],
    }
    per_query = {
        "f1": _qresult("f1"),
        "f2": _qresult("f2"),
    }
    results = RetrievalResults(collection_name="C", k=5, per_query=per_query)
    judge = _SequenceJudgeClient([(0.5, "r")] * 6)

    fres = score_goldens(results, goldens, judge, samples_per_claim=3)

    assert len(judge.calls) == 6, (
        f"expected 2 goldens × 3 samples = 6 calls; got {len(judge.calls)}"
    )
    # Each qid contributes exactly 3 calls.
    qid_counts: dict[str, int] = {}
    for c in judge.calls:
        qid_counts[c.qid] = qid_counts.get(c.qid, 0) + 1
    assert qid_counts == {"f1": 3, "f2": 3}
    assert fres.total_queries_scored == 2


def test_score_goldens_averages_scores() -> None:
    """Judge returns [0.2, 0.8, 0.5] → stored score = mean = 0.5. (Mutation B target.)

    First sample (0.2) differs from the mean (0.5), so a "first-sample"
    mutation reds with q.score == 0.2 ≠ 0.5.
    """
    goldens = {"factoid": [_golden("f1")]}
    per_query = {"f1": _qresult("f1")}
    results = RetrievalResults(collection_name="C", k=5, per_query=per_query)
    judge = _PerGoldenJudgeClient({
        "f1": [(0.2, "r1"), (0.8, "r2"), (0.5, "r3")],
    })

    fres = score_goldens(results, goldens, judge, samples_per_claim=3)

    q = fres.per_query["f1"]
    assert q.score == pytest.approx(0.5), (
        f"expected mean(0.2,0.8,0.5)=0.5; got {q.score}"
    )


def test_score_goldens_picks_highest_scored_reasoning() -> None:
    """Reasoning comes from the highest-scored sample."""
    goldens = {"factoid": [_golden("f1")]}
    per_query = {"f1": _qresult("f1")}
    results = RetrievalResults(collection_name="C", k=5, per_query=per_query)
    judge = _PerGoldenJudgeClient({
        "f1": [(0.5, "bad"), (0.9, "good"), (0.7, "ok")],
    })

    fres = score_goldens(results, goldens, judge, samples_per_claim=3)

    q = fres.per_query["f1"]
    assert q.reasoning == "good", (
        f"expected reasoning from highest-scored sample ('good'); got {q.reasoning!r}"
    )


def test_score_goldens_samples_per_claim_one_is_backwards_compat() -> None:
    """N=1 (default) → exactly one call per golden, P5-identical behavior."""
    goldens = {"factoid": [_golden("f1"), _golden("f2")]}
    per_query = {"f1": _qresult("f1"), "f2": _qresult("f2")}
    results = RetrievalResults(collection_name="C", k=5, per_query=per_query)
    judge = _SequenceJudgeClient([(0.55, "r-a"), (0.77, "r-b")])

    # Default samples_per_claim should be 1 (no kwarg).
    fres = score_goldens(results, goldens, judge)

    assert len(judge.calls) == 2
    assert fres.per_query["f1"].score == pytest.approx(0.55)
    assert fres.per_query["f1"].reasoning == "r-a"
    assert fres.per_query["f2"].score == pytest.approx(0.77)
    assert fres.per_query["f2"].reasoning == "r-b"


# ---------------------------------------------------------------------------
# CLI --samples-per-claim threading
# ---------------------------------------------------------------------------


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _make_synthetic_pack_dir(
    tmp_path: Path,
    *,
    samples_per_claim: int = 5,
) -> Path:
    """Mirror the CLI test pack with a custom samples_per_claim."""
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
            "samples_per_claim": samples_per_claim,
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
    (pack_dir / "thresholds.yaml").write_text(__import__("yaml").safe_dump(thresholds))
    return pack_dir


class _FakeJudgeClient:
    def __init__(self, score_value: float = 0.9) -> None:
        self.score_value = score_value
        self.calls: list = []

    def score(self, question):
        self.calls.append(question)
        return JudgmentScore(score=self.score_value, reasoning="ok")


class _FakeEmbedder:
    def embed_documents(self, texts):
        return [[0.0] * 8 for _ in texts]


def test_cli_samples_per_claim_flag_overrides_pack_meta(
    tmp_path: Path, monkeypatch
) -> None:
    """run_cli(samples_per_claim=2) with pack default=5 → judge called 2× not 5×.

    MUTATION C TARGET: ignoring the kwarg falls back to pack default 5.
    """
    from src.eval.cli import run_cli
    from src.eval.runner import execute as execute_mod
    from src.eval.runner import retrieve as retrieve_mod
    from src.eval.runner import report as report_mod
    import src.eval.runner as runner_pkg
    from src.vector_db.common.schemas import SearchResult

    pack_dir = _make_synthetic_pack_dir(tmp_path, samples_per_claim=5)
    fake_judge = _FakeJudgeClient(score_value=0.9)

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
    monkeypatch.setattr(retrieve_mod, "get_embedding_provider", lambda: _FakeEmbedder())
    monkeypatch.setattr(retrieve_mod, "create_persistent_client", lambda: object())
    monkeypatch.setattr(retrieve_mod, "close_client", lambda _c: None)
    monkeypatch.setattr(
        runner_pkg, "build_judge_client", lambda pack, **kw: fake_judge
    )

    rc = run_cli(str(pack_dir), samples_per_claim=2)
    assert rc == 0
    # Single golden × 2 samples (override) — NOT 5 (pack default).
    assert len(fake_judge.calls) == 2, (
        f"expected CLI override 2 to win over pack 5; got {len(fake_judge.calls)} calls"
    )
