# @summary
# Tests for P7b — per-sample visibility for multi-sample judge.
# 8 fast tests covering QueryFaithfulnessResult.samples field population,
# default empty tuple, frozen dataclass invariant, and CLI --show-samples
# JSON injection (silently ignored under text format).
# Mutation A target: test_score_goldens_populates_samples_field_with_N_judgments
# Mutation B target: test_cli_show_samples_flag_includes_per_query_samples_in_json
# Exports: 8 test functions (see module body).
# Deps: src.eval.runner, src.eval.cli, src.eval.pack.schema.
# @end-summary
"""P7b — per-sample visibility tests (8 fast)."""
from __future__ import annotations

import dataclasses
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import pytest

from src.eval.pack.schema import EvalPack, Golden, JudgeConfig, PackMeta, Thresholds
from src.eval.runner import (
    JudgeQuestion,
    JudgmentScore,
    QueryFaithfulnessResult,
    QueryRetrievalResult,
    RetrievalResults,
    score_goldens,
)


# ---------------------------------------------------------------------------
# Fakes / fixtures (mirrored from test_multi_sample_judge.py / test_cli.py)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# QueryFaithfulnessResult.samples — schema-level tests
# ---------------------------------------------------------------------------


def test_score_goldens_populates_samples_field_with_N_judgments() -> None:
    """2 goldens × N=3 samples → each QueryFaithfulnessResult.samples is a
    3-tuple of JudgmentScore with the recorded scores in order. (Mutation A
    target — collapsing samples=tuple(samples) → samples=() reds this.)
    """
    goldens = {"factoid": [_golden("f1"), _golden("f2")]}
    per_query = {"f1": _qresult("f1"), "f2": _qresult("f2")}
    results = RetrievalResults(collection_name="C", k=5, per_query=per_query)
    judge = _PerGoldenJudgeClient({
        "f1": [(0.2, "r1a"), (0.8, "r1b"), (0.5, "r1c")],
        "f2": [(0.3, "r2a"), (0.6, "r2b"), (0.9, "r2c")],
    })

    fres = score_goldens(results, goldens, judge, samples_per_claim=3)

    q1 = fres.per_query["f1"]
    q2 = fres.per_query["f2"]
    assert isinstance(q1.samples, tuple), f"samples must be tuple, got {type(q1.samples)}"
    assert len(q1.samples) == 3, f"expected 3 samples for f1, got {len(q1.samples)}"
    assert len(q2.samples) == 3, f"expected 3 samples for f2, got {len(q2.samples)}"
    assert all(isinstance(s, JudgmentScore) for s in q1.samples), (
        f"f1 samples must all be JudgmentScore; got {[type(s) for s in q1.samples]}"
    )
    # In-order scores recorded.
    assert [s.score for s in q1.samples] == pytest.approx([0.2, 0.8, 0.5])
    assert [s.score for s in q2.samples] == pytest.approx([0.3, 0.6, 0.9])
    assert [s.reasoning for s in q1.samples] == ["r1a", "r1b", "r1c"]


def test_score_goldens_samples_is_one_tuple_for_default_N() -> None:
    """Default samples_per_claim=1 → samples tuple has length 1."""
    goldens = {"factoid": [_golden("f1")]}
    per_query = {"f1": _qresult("f1")}
    results = RetrievalResults(collection_name="C", k=5, per_query=per_query)
    judge = _PerGoldenJudgeClient({"f1": [(0.42, "lone-reasoning")]})

    fres = score_goldens(results, goldens, judge)

    q = fres.per_query["f1"]
    assert len(q.samples) == 1, f"expected 1 sample for default N=1, got {len(q.samples)}"
    assert q.samples[0].score == pytest.approx(0.42)
    assert q.samples[0].reasoning == "lone-reasoning"


def test_query_faithfulness_result_samples_defaults_to_empty_tuple() -> None:
    """Manual construction without samples kwarg → .samples == () (backwards-compat)."""
    q = QueryFaithfulnessResult(
        qid="x", qtype="factoid", score=0.5, reasoning="r"
    )
    assert q.samples == ()
    assert isinstance(q.samples, tuple)


def test_query_faithfulness_result_samples_frozen() -> None:
    """QueryFaithfulnessResult remains frozen — assigning .samples raises."""
    q = QueryFaithfulnessResult(
        qid="x", qtype="factoid", score=0.5, reasoning="r"
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        q.samples = (JudgmentScore(score=0.1, reasoning="nope"),)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CLI --show-samples
# ---------------------------------------------------------------------------


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _make_synthetic_pack(tmp_path: Path, *, samples_per_claim: int = 2) -> Path:
    """Minimal valid eval_pack with configurable samples_per_claim."""
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
    (pack_dir / "thresholds.yaml").write_text(yaml.safe_dump(thresholds))
    return pack_dir


class _FakeJudgeClient:
    def __init__(self, score_value: float = 0.9) -> None:
        self.score_value = score_value
        self.calls: list = []

    def score(self, question):
        self.calls.append(question)
        return JudgmentScore(score=self.score_value, reasoning="ok-reasoning")


class _FakeEmbedder:
    def embed_documents(self, texts):
        return [[0.0] * 8 for _ in texts]


def _patch_full_chain(monkeypatch, *, judge_score: float = 0.9):
    """Patch the runner pipeline for CLI tests; return the fake judge."""
    from src.eval.runner import execute as execute_mod
    from src.eval.runner import retrieve as retrieve_mod
    from src.eval.runner import report as report_mod
    from src.vector_db.common.schemas import SearchResult
    import src.eval.runner as runner_pkg

    fake_judge = _FakeJudgeClient(score_value=judge_score)

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
    return fake_judge


def test_cli_show_samples_off_by_default_in_json(tmp_path: Path, monkeypatch, capsys) -> None:
    """Without --show-samples, per_query_samples MUST NOT appear in JSON payload."""
    from src.eval.cli import run_cli

    pack_dir = _make_synthetic_pack(tmp_path, samples_per_claim=2)
    _patch_full_chain(monkeypatch, judge_score=0.9)

    rc = run_cli(str(pack_dir), format="json")
    assert rc == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert "per_query_samples" not in payload, (
        f"per_query_samples should be omitted by default; got payload keys {sorted(payload)}"
    )


def test_cli_show_samples_flag_includes_per_query_samples_in_json(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """With --show-samples and JSON format, per_query_samples is present with
    per-qid lists of {score, reasoning} dicts. (Mutation B target — ignoring
    show_samples in _emit_json reds this.)
    """
    from src.eval.cli import run_cli

    pack_dir = _make_synthetic_pack(tmp_path, samples_per_claim=2)
    _patch_full_chain(monkeypatch, judge_score=0.77)

    rc = run_cli(str(pack_dir), format="json", show_samples=True)
    assert rc == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert "per_query_samples" in payload, (
        f"expected per_query_samples in payload; got keys {sorted(payload)}"
    )
    assert "g1" in payload["per_query_samples"], (
        f"expected qid 'g1' in per_query_samples; got {list(payload['per_query_samples'])}"
    )
    g1_samples = payload["per_query_samples"]["g1"]
    assert isinstance(g1_samples, list)
    assert len(g1_samples) == 2, f"expected 2 samples for g1, got {len(g1_samples)}"
    for s in g1_samples:
        assert isinstance(s, dict)
        assert s.keys() == {"score", "reasoning"}, (
            f"each sample dict must have keys {{score, reasoning}}; got {sorted(s)}"
        )
        assert s["score"] == pytest.approx(0.77)
        assert s["reasoning"] == "ok-reasoning"


def test_cli_show_samples_flag_silently_ignored_in_text_format(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """--show-samples under --format text emits identical stdout to no flag."""
    from src.eval.cli import run_cli

    pack_dir = _make_synthetic_pack(tmp_path, samples_per_claim=2)
    _patch_full_chain(monkeypatch, judge_score=0.9)

    rc_no = run_cli(str(pack_dir), format="text", show_samples=False)
    out_no = capsys.readouterr().out

    rc_yes = run_cli(str(pack_dir), format="text", show_samples=True)
    out_yes = capsys.readouterr().out

    assert rc_no == 0 and rc_yes == 0
    assert out_no == out_yes, (
        "text format must ignore show_samples — outputs diverged"
    )


def test_cli_argv_parses_show_samples_flag(tmp_path: Path, monkeypatch, capsys) -> None:
    """main(['run', path, '--format', 'json', '--show-samples']) threads the
    flag through to run_cli → _emit_json (per_query_samples present)."""
    from src.eval.cli import main

    pack_dir = _make_synthetic_pack(tmp_path, samples_per_claim=2)
    _patch_full_chain(monkeypatch, judge_score=0.5)

    rc = main(["run", str(pack_dir), "--format", "json", "--show-samples"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert "per_query_samples" in payload, (
        "argparse must thread --show-samples through to _emit_json"
    )
    assert "g1" in payload["per_query_samples"]
