# @summary
# Tests for P6b eval CLI runner.
# 10 fast tests covering exit codes (0/1/2/3), --k propagation through
# retrieval AND the recall_at_<k> gate key, --format json/text output
# shape, argparse → run_cli dispatch, and chat_model_factory injection.
# Mutation A target: test_cli_gate_pass_returns_zero
# Mutation B target: test_cli_k_parameter_propagates
# Exports: test_cli_pack_not_found_exits_2,
#          test_cli_invalid_pack_exits_2,
#          test_cli_gate_pass_returns_zero,
#          test_cli_gate_fail_returns_one,
#          test_cli_runtime_error_exits_3,
#          test_cli_k_parameter_propagates,
#          test_cli_json_format_valid,
#          test_cli_text_format_human_readable,
#          test_cli_main_argv_parsing,
#          test_cli_chat_model_factory_injected
# Deps: src.eval.cli, src.eval.runner, src.eval.pack
# @end-summary
"""P6b — CLI runner tests (10 fast)."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Synthetic pack builder + fakes
# ---------------------------------------------------------------------------


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _make_synthetic_pack(
    tmp_path: Path,
    *,
    recall_floor: float = 0.5,
    faithfulness_floor: float | None = 0.5,
    recall_metric_key: str = "recall_at_5",
) -> Path:
    """Create a minimal valid eval_pack dir on disk and return its path.

    Single doc, single factoid golden, thresholds with declared floors on
    ``factoid`` qtype keyed by ``recall_metric_key`` (e.g. recall_at_5 or
    recall_at_3) and optional ``faithfulness``.
    """
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

    # corpus_pin = sha256 of sorted "<path>:<sha256>" joined by \n
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

    defaults: dict[str, dict[str, float]] = {
        "factoid": {recall_metric_key: recall_floor}
    }
    if faithfulness_floor is not None:
        defaults["factoid"]["faithfulness"] = faithfulness_floor

    thresholds = {
        "profile": "generic",
        "defaults": defaults,
        "min_goldens_per_qtype": {"factoid": 1},
    }
    (pack_dir / "thresholds.yaml").write_text(yaml.safe_dump(thresholds))
    return pack_dir


class _FakeJudgeClient:
    def __init__(self, score_value: float = 0.9) -> None:
        self.score_value = score_value
        self.calls: list = []

    def score(self, question):
        from src.eval.runner.judge import JudgmentScore

        self.calls.append(question)
        return JudgmentScore(score=self.score_value, reasoning="ok")


class _FakeEmbedder:
    def embed_documents(self, texts):
        return [[0.0] * 8 for _ in texts]


def _patch_full_chain(
    monkeypatch,
    *,
    retrieved_source: str = "doc1.md",
    chunks: tuple[str, ...] = ("alpha is the first",),
    judge_score: float = 0.9,
    recorded_k: list[int] | None = None,
):
    """Patch execute_plan + retrieve infra + build_judge_client to deterministic fakes.

    Returns the FakeJudgeClient so the test can inspect calls if needed.
    """
    from src.eval.runner import execute as execute_mod
    from src.eval.runner import retrieve as retrieve_mod
    from src.eval.runner import report as report_mod
    from src.vector_db.common.schemas import SearchResult

    fake_judge = _FakeJudgeClient(score_value=judge_score)

    def fake_execute_plan(plan, *, fresh: bool = True):
        return report_mod.IngestReport(
            collection_name=plan.collection_name,
            processed=1,
            skipped=0,
            failed=0,
            stored_chunks=1,
            document_count=1,
            plan=plan,
            errors=(),
        )

    def fake_search(client, query, query_embedding, alpha, limit, **kwargs):
        if recorded_k is not None:
            recorded_k.append(limit)
        return [
            SearchResult(
                text=chunks[0] if chunks else "",
                score=1.0,
                metadata={"source": retrieved_source},
            )
        ]

    monkeypatch.setattr(execute_mod, "execute_plan", fake_execute_plan)
    # cli imports execute_plan from src.eval.runner — patch the re-export too.
    import src.eval.runner as runner_pkg

    monkeypatch.setattr(runner_pkg, "execute_plan", fake_execute_plan)

    monkeypatch.setattr(retrieve_mod, "search", fake_search)
    monkeypatch.setattr(retrieve_mod, "get_embedding_provider", lambda: _FakeEmbedder())
    monkeypatch.setattr(retrieve_mod, "create_persistent_client", lambda: object())
    monkeypatch.setattr(retrieve_mod, "close_client", lambda _c: None)

    # Patch build_judge_client at the runner package level — run_cli lazy-imports it.
    monkeypatch.setattr(
        runner_pkg, "build_judge_client", lambda pack, **kw: fake_judge
    )
    return fake_judge


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_cli_pack_not_found_exits_2(tmp_path: Path) -> None:
    """A nonexistent pack path returns exit code 2 (PackValidationError)."""
    from src.eval.cli import run_cli

    bogus = tmp_path / "no_such_pack"
    rc = run_cli(str(bogus))
    assert rc == 2


def test_cli_invalid_pack_exits_2(tmp_path: Path, capsys) -> None:
    """A malformed pack (missing pack.yaml keys) returns 2 with stderr message."""
    from src.eval.cli import run_cli

    bad_pack = tmp_path / "bad_pack"
    bad_pack.mkdir()
    (bad_pack / "pack.yaml").write_text("name: only-name\n")  # missing required keys
    rc = run_cli(str(bad_pack))
    assert rc == 2
    err = capsys.readouterr().err
    assert "pack.yaml" in err or "pack" in err.lower()


def test_cli_gate_pass_returns_zero(tmp_path: Path, monkeypatch) -> None:
    """All chain mocked, recall=1.0, faithfulness=0.9 above floors → exit 0.

    MUTATION A TARGET: flipping the gate→exit mapping makes this red.
    """
    from src.eval.cli import run_cli

    pack_dir = _make_synthetic_pack(
        tmp_path, recall_floor=0.5, faithfulness_floor=0.5
    )
    _patch_full_chain(monkeypatch, judge_score=0.9)

    rc = run_cli(str(pack_dir))
    assert rc == 0


def test_cli_gate_fail_returns_one(tmp_path: Path, monkeypatch, capsys) -> None:
    """Recall below floor → exit 1, with GateFailure listed on stderr."""
    from src.eval.cli import run_cli

    pack_dir = _make_synthetic_pack(
        tmp_path, recall_floor=0.5, faithfulness_floor=0.5
    )
    # Make retrieval miss → recall=0.0 (returns wrong source)
    _patch_full_chain(monkeypatch, retrieved_source="other.md", judge_score=0.9)

    rc = run_cli(str(pack_dir))
    assert rc == 1
    err = capsys.readouterr().err
    assert "factoid" in err
    assert "recall_at_5" in err


def test_cli_missing_pack_judge_prompt_exits_2(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """A Phase-2 FileNotFoundError (missing pack-referenced judge prompt) maps to
    exit 2 (pack error), not exit 3 — pins the P8a.0 refactor's exit-code contract.

    The pack authors a ``prompts/`` directory but omits ``judge_v1.md`` for its
    declared ``tier1_prompt_version: v1``. Phase 1 (load_pack) succeeds — the
    missing prompt is NOT a load-time concern — so we exercise the REAL Phase-2
    judge path: ``build_judge_client`` → ``load_judge_prompt`` raises a genuine
    ``FileNotFoundError`` (see src/eval/runner/judge.py: when ``prompts/`` exists
    we do not fall back to the packaged default). The single
    ``except (PackValidationError, FileNotFoundError)`` branch must catch it →
    exit 2, NOT the ``except Exception`` → exit 3 branch.
    """
    from src.eval.cli import run_cli

    pack_dir = _make_synthetic_pack(tmp_path)
    # Author a prompts/ dir WITHOUT judge_v1.md → load_judge_prompt raises a
    # real FileNotFoundError (no silent fallback to the packaged default).
    (pack_dir / "prompts").mkdir()

    # Patch execute_plan + retrieval infra so Phase 2 reaches the judge step
    # WITHOUT touching Weaviate, but leave build_judge_client REAL so the
    # FileNotFoundError originates specifically from the judge-prompt path.
    from src.eval.runner import retrieve as retrieve_mod
    from src.eval.runner import report as report_mod
    from src.vector_db.common.schemas import SearchResult
    import src.eval.runner as runner_pkg

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

    # Real build_judge_client constructs a chat model BEFORE loading the prompt;
    # inject a no-op factory so no live LLM is touched and the FileNotFoundError
    # comes solely from load_judge_prompt.
    def noop_factory(*, model_alias, temperature):
        return object()

    rc = run_cli(str(pack_dir), chat_model_factory=noop_factory)
    assert rc == 2
    err = capsys.readouterr().err
    assert "pack load error" in err
    assert "judge_v1.md" in err


def test_cli_runtime_error_exits_3(tmp_path: Path, monkeypatch, capsys) -> None:
    """execute_plan raises → exit 3 (infra error)."""
    from src.eval.cli import run_cli

    pack_dir = _make_synthetic_pack(tmp_path)

    import src.eval.runner as runner_pkg

    def boom(*_a, **_k):
        raise RuntimeError("weaviate exploded")

    monkeypatch.setattr(runner_pkg, "execute_plan", boom)

    rc = run_cli(str(pack_dir))
    assert rc == 3
    err = capsys.readouterr().err
    assert "weaviate exploded" in err or "RuntimeError" in err or "error" in err.lower()


def test_cli_k_parameter_propagates(tmp_path: Path, monkeypatch) -> None:
    """--k=3 propagates to retrieve.search(limit=3) AND the gate compares
    recall_at_3 (not recall_at_5).

    MUTATION B TARGET: hardcoding k=5 in run_cli makes this red because
    the pack only declares recall_at_3 and the search would receive 5.
    """
    from src.eval.cli import run_cli

    # Pack thresholds key is recall_at_3 ONLY — if cli passes k=5, the gate
    # check key (recall_at_5) won't match defaults['factoid'], so the gate
    # would pass trivially even though we *want* it to apply the floor.
    # The discriminating signal is: did search receive limit=3?
    pack_dir = _make_synthetic_pack(
        tmp_path,
        recall_floor=0.5,
        faithfulness_floor=0.5,
        recall_metric_key="recall_at_3",
    )
    recorded_k: list[int] = []
    _patch_full_chain(
        monkeypatch, judge_score=0.9, recorded_k=recorded_k
    )

    rc = run_cli(str(pack_dir), k=3)
    # Gate should evaluate recall_at_3 against the floor; recall=1.0 → pass.
    assert rc == 0
    assert recorded_k == [3], (
        f"expected search to receive limit=3, got {recorded_k!r}"
    )


def test_cli_json_format_valid(tmp_path: Path, monkeypatch, capsys) -> None:
    """--format json emits a single parseable JSON object with required keys."""
    from src.eval.cli import run_cli

    pack_dir = _make_synthetic_pack(tmp_path)
    _patch_full_chain(monkeypatch, judge_score=0.9)

    rc = run_cli(str(pack_dir), format="json")
    assert rc == 0
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    for key in (
        "collection_name",
        "k",
        "passed",
        "failures",
        "recall_by_qtype",
        "faithfulness_by_qtype",
        "mrr_by_qtype",
        "per_query_recall",
        "per_query_mrr",
        "per_query_faithfulness",
        "total_queries_scored",
        "total_queries_skipped",
        "total_queries_judged",
    ):
        assert key in payload, f"json output missing key {key!r}: {payload}"
    assert payload["passed"] is True
    assert payload["k"] == 5
    assert payload["failures"] == []


def test_cli_text_format_human_readable(tmp_path: Path, monkeypatch, capsys) -> None:
    """--format text prints per-qtype rows and a PASS/FAIL summary line."""
    from src.eval.cli import run_cli

    pack_dir = _make_synthetic_pack(tmp_path)
    _patch_full_chain(monkeypatch, judge_score=0.9)

    rc = run_cli(str(pack_dir), format="text")
    assert rc == 0
    out = capsys.readouterr().out
    # Per-qtype row (factoid) AND a pass/fail summary line.
    assert "factoid" in out
    assert "PASS" in out.upper()


def test_cli_main_argv_parsing(tmp_path: Path, monkeypatch, capsys) -> None:
    """main(['run', path, '--k', '3', '--format', 'json']) dispatches correctly.

    Discriminating partner test for the argparse → run_cli wiring.
    """
    from src.eval.cli import main

    pack_dir = _make_synthetic_pack(
        tmp_path, recall_metric_key="recall_at_3"
    )
    recorded_k: list[int] = []
    _patch_full_chain(monkeypatch, judge_score=0.9, recorded_k=recorded_k)

    rc = main(
        ["run", str(pack_dir), "--k", "3", "--format", "json"]
    )
    assert rc == 0
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert payload["k"] == 3
    assert recorded_k == [3]


def test_cli_chat_model_factory_injected(tmp_path: Path, monkeypatch) -> None:
    """A chat_model_factory kwarg is plumbed through to build_judge_client."""
    from src.eval.cli import run_cli

    pack_dir = _make_synthetic_pack(tmp_path)

    # Don't use _patch_full_chain — we need to spy on build_judge_client's kwargs.
    from src.eval.runner import execute as execute_mod
    from src.eval.runner import retrieve as retrieve_mod
    from src.eval.runner import report as report_mod
    from src.vector_db.common.schemas import SearchResult
    import src.eval.runner as runner_pkg

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

    captured: dict[str, Any] = {}
    fake_judge = _FakeJudgeClient(score_value=0.9)

    def spy_build(pack, *, chat_model_factory=None, pack_dir=None):
        captured["chat_model_factory"] = chat_model_factory
        return fake_judge

    monkeypatch.setattr(runner_pkg, "build_judge_client", spy_build)

    sentinel_factory_calls: list = []

    def my_factory(*, model_alias, temperature):
        sentinel_factory_calls.append((model_alias, temperature))
        return object()

    rc = run_cli(str(pack_dir), chat_model_factory=my_factory)
    assert rc == 0
    assert captured["chat_model_factory"] is my_factory


# ---------------------------------------------------------------------------
# P7g — _emit_json metric exposure (direct unit tests on _emit_json).
# ---------------------------------------------------------------------------


def _make_eval_report(
    *,
    recall_by_qtype: dict[str, float] | None = None,
    faithfulness_by_qtype: dict[str, float] | None = None,
    mrr_by_qtype: dict[str, float] | None = None,
    per_query_recall: dict[str, float] | None = None,
    per_query_mrr: dict[str, float] | None = None,
    per_query_faithfulness: dict[str, float] | None = None,
):
    """Construct a minimal valid EvalReport for _emit_json unit tests."""
    from src.eval.runner.report import EvalReport

    return EvalReport(
        collection_name="c",
        k=5,
        per_query_recall=per_query_recall or {},
        recall_by_qtype=recall_by_qtype or {},
        total_queries_scored=1,
        total_queries_skipped=0,
        faithfulness_by_qtype=faithfulness_by_qtype or {},
        total_queries_judged=1,
        mrr_by_qtype=mrr_by_qtype or {},
        per_query_mrr=per_query_mrr or {},
        per_query_faithfulness=per_query_faithfulness or {},
    )


def test_emit_json_failure_carries_populated_qid() -> None:
    """A per-qid GateFailure serializes its qid verbatim in the JSON payload.

    Kills a hardcoded-empty mutation on the failures comprehension.
    """
    from src.eval.cli import _emit_json
    from src.eval.runner.gate import GateFailure, GateResult

    report = _make_eval_report()
    gate = GateResult(
        passed=False,
        failures=(
            GateFailure(
                qtype="factoid",
                metric="recall_at_5",
                expected=0.8,
                actual=0.3,
                qid="factoid_017",
            ),
        ),
    )
    buf = io.StringIO()
    _emit_json(report, gate, stdout=buf)
    payload = json.loads(buf.getvalue())
    assert payload["failures"][0]["qid"] == "factoid_017"


def test_emit_json_qtype_failure_has_empty_qid() -> None:
    """A qtype-level GateFailure (no qid) serializes qid as empty string.

    One-step-below partner: present+empty vs present+populated gives the
    qid serialization teeth.
    """
    from src.eval.cli import _emit_json
    from src.eval.runner.gate import GateFailure, GateResult

    report = _make_eval_report()
    gate = GateResult(
        passed=False,
        failures=(
            GateFailure(
                qtype="factoid",
                metric="recall_at_5",
                expected=0.8,
                actual=0.3,
            ),
        ),
    )
    buf = io.StringIO()
    _emit_json(report, gate, stdout=buf)
    payload = json.loads(buf.getvalue())
    assert payload["failures"][0]["qid"] == ""


def test_emit_json_per_query_maps_carry_distinct_sources() -> None:
    """The three per-query maps each serialize from their OWN source.

    Distinct values per map mean a copy-paste-wrong-source mutation reds
    exactly one assertion.
    """
    from src.eval.cli import _emit_json
    from src.eval.runner.gate import GateResult

    report = _make_eval_report(
        per_query_recall={"g1": 0.5},
        per_query_mrr={"g1": 0.25},
        per_query_faithfulness={"g1": 0.9},
    )
    gate = GateResult(passed=True, failures=())
    buf = io.StringIO()
    _emit_json(report, gate, stdout=buf)
    payload = json.loads(buf.getvalue())
    assert payload["per_query_recall"]["g1"] == 0.5
    assert payload["per_query_mrr"]["g1"] == 0.25
    assert payload["per_query_faithfulness"]["g1"] == 0.9


def test_emit_json_includes_mrr_by_qtype() -> None:
    """mrr_by_qtype is exposed as a top-level key with the carried value."""
    from src.eval.cli import _emit_json
    from src.eval.runner.gate import GateResult

    report = _make_eval_report(mrr_by_qtype={"factoid": 0.42})
    gate = GateResult(passed=True, failures=())
    buf = io.StringIO()
    _emit_json(report, gate, stdout=buf)
    payload = json.loads(buf.getvalue())
    assert payload["mrr_by_qtype"]["factoid"] == 0.42
