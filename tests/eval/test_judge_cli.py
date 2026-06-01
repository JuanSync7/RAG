# @summary
# Tests for the headless `claude` CLI LLM-as-judge backend (judge_cli.py).
# Headline E2E proves run_eval(... judge_client_factory=ClaudeCliJudgeClient
# with a fake run_fn ...) carries faithfulness ~0.8; a dispatch test pins the
# argv→backend wiring through main(["run", ..., "--judge-backend", "claude-cli"]);
# unit tests cover envelope parsing, fence stripping, clamping, and every
# JudgeBackendError raise path; an argv-contract test pins the cost/safety
# constraints; a model-gated real test scores one trivial question live.
# Exports: test_run_eval_scores_via_claude_cli_backend,
#          test_main_dispatch_routes_judge_backend_claude_cli,
#          test_score_parses_envelope_success, test_score_strips_json_fence,
#          test_score_clamps_out_of_range, test_score_is_error_true_raises,
#          test_score_bad_subtype_raises, test_score_unparseable_result_raises,
#          test_score_nonzero_returncode_raises,
#          test_build_claude_argv_contains_constraints,
#          test_claude_cli_judge_smoke_real
# Deps: src.eval.runner.judge_cli, src.eval.cli, src.eval.runner.judge
# @end-summary
"""Tests for the headless `claude` CLI judge backend.

All fast tests inject a fake ``run_fn`` so NO ``claude`` process is spawned and
NO Weaviate/embeddings/LLM are touched. The one live test is double-gated on a
``claude`` binary on PATH AND an explicit opt-in env var.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# Fake subprocess result + run_fn factory
# ---------------------------------------------------------------------------


def _fake_completed(stdout: str, *, returncode: int = 0, stderr: str = ""):
    """Mimic the slice of subprocess.CompletedProcess that score() reads."""
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def _success_envelope(result_payload: str) -> str:
    """Serialize a VERIFIED-shape success envelope with a stringified result."""
    return json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": result_payload,
        }
    )


def _make_run_fn(envelope_stdout: str, *, returncode: int = 0):
    """Return a (argv, *, timeout) -> result callable serving a canned stdout."""

    def _run_fn(argv, *, timeout):
        return _fake_completed(envelope_stdout, returncode=returncode)

    return _run_fn


def _question():
    from src.eval.runner.judge import JudgeQuestion

    return JudgeQuestion(
        qid="g1",
        qtype="factoid",
        query="what is alpha?",
        expected_answer_span="alpha",
        chunk_texts=("alpha is the first letter",),
    )


def _client(run_fn, *, template: str = "RUBRIC {query} {chunk_block}"):
    from src.eval.runner.judge_cli import ClaudeCliJudgeClient

    return ClaudeCliJudgeClient(
        template=template,
        model="claude-haiku-4-5-20251001",
        run_fn=run_fn,
    )


# ---------------------------------------------------------------------------
# Synthetic pack (mirrors tests/eval/test_cli.py + src/eval/smoke.py)
# ---------------------------------------------------------------------------


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _make_synthetic_pack(tmp_path: Path) -> Path:
    """Minimal valid eval_pack on disk; single factoid golden."""
    import yaml

    pack_dir = tmp_path / "synthetic_pack"
    pack_dir.mkdir()
    (pack_dir / "corpus").mkdir()
    (pack_dir / "goldens").mkdir()

    doc_text = "alpha beta gamma\n"
    (pack_dir / "corpus" / "doc1.md").write_text(doc_text)
    doc_sha = _sha256_text(doc_text)
    (pack_dir / "corpus" / "manifest.json").write_text(
        json.dumps({"entries": [{"path": "doc1.md", "sha256": doc_sha}]})
    )
    corpus_pin = hashlib.sha256(f"doc1.md:{doc_sha}".encode("utf-8")).hexdigest()

    pack_yaml = {
        "name": "synpack",
        "version": 1,
        "profile": "generic",
        "corpus_pin": corpus_pin,
        "description": "judge-cli test pack",
        "judge": {
            "tier1_model": "claude-haiku-4-5-20251001",
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

    thresholds = {
        "profile": "generic",
        "defaults": {"factoid": {"recall_at_5": 0.5, "faithfulness": 0.5}},
        "min_goldens_per_qtype": {"factoid": 1},
    }
    (pack_dir / "thresholds.yaml").write_text(yaml.safe_dump(thresholds))
    return pack_dir


class _FakeEmbedder:
    def embed_documents(self, texts):
        return [[0.0] * 8 for _ in texts]


def _patch_infra(monkeypatch, *, retrieved_source: str = "doc1.md"):
    """Patch execute_plan + retrieval infra to deterministic fakes (no Weaviate)."""
    from src.eval.runner import report as report_mod
    from src.eval.runner import retrieve as retrieve_mod
    from src.vector_db.common.schemas import SearchResult
    import src.eval.runner as runner_pkg

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
        return [
            SearchResult(
                text="alpha is the first",
                score=1.0,
                metadata={"source": retrieved_source},
            )
        ]

    monkeypatch.setattr(runner_pkg, "execute_plan", fake_execute_plan)
    monkeypatch.setattr(retrieve_mod, "search", fake_search)
    monkeypatch.setattr(
        retrieve_mod, "get_embedding_provider", lambda: _FakeEmbedder()
    )
    monkeypatch.setattr(retrieve_mod, "create_persistent_client", lambda: object())
    monkeypatch.setattr(retrieve_mod, "close_client", lambda _c: None)


# ---------------------------------------------------------------------------
# HEADLINE E2E — the integration-seam teeth
# ---------------------------------------------------------------------------


def test_run_eval_scores_via_claude_cli_backend(tmp_path: Path, monkeypatch) -> None:
    """run_eval with a ClaudeCliJudgeClient (fake run_fn) carries faithfulness 0.8.

    Proves the headless-CLI judge plugs into the real run_eval seam: the canned
    success envelope's result ({"score":0.8}) becomes the qtype faithfulness.
    """
    from src.eval.cli import run_eval
    from src.eval.runner.judge_cli import ClaudeCliJudgeClient

    pack_dir = _make_synthetic_pack(tmp_path)
    _patch_infra(monkeypatch)

    run_fn = _make_run_fn(_success_envelope(json.dumps({"score": 0.8, "reasoning": "ok"})))

    def factory(pack, *, chat_model_factory=None, pack_dir=None):
        return ClaudeCliJudgeClient(
            template="RUBRIC {query}\n{chunk_block}",
            model="claude-haiku-4-5-20251001",
            run_fn=run_fn,
        )

    result = run_eval(str(pack_dir), judge_client_factory=factory)
    assert result.report.faithfulness_by_qtype["factoid"] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# CLI dispatch wiring — argv→backend selection
# ---------------------------------------------------------------------------


def test_main_dispatch_routes_judge_backend_claude_cli(
    tmp_path: Path, monkeypatch
) -> None:
    """main(["run", ..., "--judge-backend", "claude-cli"]) selects the cli factory.

    Pins the argv→backend wiring: the claude-cli factory (spied) must be the one
    actually invoked, and main() returns an int exit code.
    """
    from src.eval import cli as cli_mod
    import src.eval.runner as runner_pkg
    from src.eval.runner.judge_cli import ClaudeCliJudgeClient

    pack_dir = _make_synthetic_pack(tmp_path)
    _patch_infra(monkeypatch)

    run_fn = _make_run_fn(_success_envelope(json.dumps({"score": 0.8, "reasoning": "ok"})))
    spy_calls: list = []

    def spy_factory(pack, *, chat_model_factory=None, pack_dir=None, run_fn=None):
        # Bind our OWN fake run_fn so no real `claude` process is ever spawned,
        # regardless of whether the CLI threads run_fn through.
        spy_calls.append(pack)
        return ClaudeCliJudgeClient(
            template="RUBRIC {query}\n{chunk_block}",
            model="claude-haiku-4-5-20251001",
            run_fn=run_fn or _make_run_fn(
                _success_envelope(json.dumps({"score": 0.8, "reasoning": "ok"}))
            ),
        )

    # Patch the symbol the CLI lazy-imports from the runner package.
    monkeypatch.setattr(
        runner_pkg, "build_claude_cli_judge_client", spy_factory, raising=False
    )

    rc = cli_mod.main(["run", str(pack_dir), "--judge-backend", "claude-cli"])
    assert isinstance(rc, int)
    assert spy_calls, "claude-cli judge factory was never invoked"


# ---------------------------------------------------------------------------
# score() parsing pipeline + error raises
# ---------------------------------------------------------------------------


def test_score_parses_envelope_success() -> None:
    """A clean success envelope yields the embedded score + reasoning."""
    run_fn = _make_run_fn(
        _success_envelope(json.dumps({"score": 0.7, "reasoning": "grounded"}))
    )
    out = _client(run_fn).score(_question())
    assert out.score == pytest.approx(0.7)
    assert out.reasoning == "grounded"


def test_score_strips_json_fence() -> None:
    """A ```json ... ``` fenced result string is stripped before json.loads."""
    fenced = "```json\n" + json.dumps({"score": 0.6, "reasoning": "ok"}) + "\n```"
    run_fn = _make_run_fn(_success_envelope(fenced))
    out = _client(run_fn).score(_question())
    assert out.score == pytest.approx(0.6)


def test_score_clamps_out_of_range() -> None:
    """A model score above 1.0 is clamped into [0, 1]."""
    run_fn = _make_run_fn(
        _success_envelope(json.dumps({"score": 1.7, "reasoning": "hot"}))
    )
    out = _client(run_fn).score(_question())
    assert out.score == pytest.approx(1.0)


def test_score_is_error_true_raises() -> None:
    """is_error: true in the envelope raises JudgeBackendError."""
    from src.eval.runner.judge_cli import JudgeBackendError

    # A VALID score payload so the is_error guard is the ONLY thing that can
    # raise — otherwise an empty "{}" result would raise KeyError on score
    # extraction and the test would pass even with the is_error guard removed.
    envelope = json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "is_error": True,
            "result": json.dumps({"score": 0.5, "reasoning": "x"}),
        }
    )
    run_fn = _make_run_fn(envelope)
    with pytest.raises(JudgeBackendError):
        _client(run_fn).score(_question())


def test_score_bad_subtype_raises() -> None:
    """A non-success subtype raises JudgeBackendError."""
    from src.eval.runner.judge_cli import JudgeBackendError

    envelope = json.dumps(
        {
            "type": "result",
            "subtype": "error_max_turns",
            "is_error": False,
            "result": json.dumps({"score": 0.5, "reasoning": "x"}),
        }
    )
    run_fn = _make_run_fn(envelope)
    with pytest.raises(JudgeBackendError):
        _client(run_fn).score(_question())


def test_score_unparseable_result_raises() -> None:
    """A non-JSON result payload raises JudgeBackendError."""
    from src.eval.runner.judge_cli import JudgeBackendError

    run_fn = _make_run_fn(_success_envelope("not json at all"))
    with pytest.raises(JudgeBackendError):
        _client(run_fn).score(_question())


def test_score_nonzero_returncode_raises() -> None:
    """A non-zero CLI returncode raises JudgeBackendError before parsing."""
    from src.eval.runner.judge_cli import JudgeBackendError

    run_fn = _make_run_fn(
        _success_envelope(json.dumps({"score": 0.9, "reasoning": "ok"})),
        returncode=1,
    )
    with pytest.raises(JudgeBackendError):
        _client(run_fn).score(_question())


# ---------------------------------------------------------------------------
# argv constraint contract — cost/safety guard
# ---------------------------------------------------------------------------


def test_build_claude_argv_contains_constraints() -> None:
    """build_claude_argv pins the constrained, non-coding-agent invocation.

    Asserts --model, --system-prompt, --allowed-tools, --output-format json,
    and --setting-sources are present so a future edit can't silently turn the
    judge into the full (expensive, tool-enabled, persona-contaminated) agent.
    """
    from src.eval.runner.judge_cli import build_claude_argv

    argv = build_claude_argv(
        "JUDGE THIS",
        model="claude-haiku-4-5-20251001",
        system_prompt="RUBRIC",
    )
    assert argv[0] == "claude"
    # -p / --print carries the prompt.
    assert "JUDGE THIS" in argv

    def _val(flag):
        return argv[argv.index(flag) + 1]

    assert "--model" in argv and _val("--model") == "claude-haiku-4-5-20251001"
    assert "--system-prompt" in argv and _val("--system-prompt") == "RUBRIC"
    assert "--output-format" in argv and _val("--output-format") == "json"
    # No tools, clean room: contamination guard.
    assert "--allowed-tools" in argv
    assert "--setting-sources" in argv


# ---------------------------------------------------------------------------
# MODEL-GATED REAL TEST — live claude CLI (double opt-in)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.integration
def test_claude_cli_judge_smoke_real() -> None:
    """Score one trivial question with a real `claude` CLI (default run_fn).

    Double-gated: skipped unless a `claude` binary is on PATH AND the explicit
    RAGWEAVE_EVAL_CLI_LIVE=1 opt-in is set, so it never runs by accident.
    """
    if not shutil.which("claude"):
        pytest.skip("no `claude` binary on PATH")
    if os.environ.get("RAGWEAVE_EVAL_CLI_LIVE") != "1":
        pytest.skip("RAGWEAVE_EVAL_CLI_LIVE != 1 (live judge opt-in not set)")

    from src.eval.runner.judge_cli import ClaudeCliJudgeClient

    template = (
        "You are a strict faithfulness judge. Score how well the chunks support "
        "answering the query.\nQuery: {query}\nChunks:\n{chunk_block}\n"
    )
    client = ClaudeCliJudgeClient(
        template=template, model="claude-haiku-4-5-20251001"
    )
    out = client.score(_question())
    assert 0.0 <= out.score <= 1.0
