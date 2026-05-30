# @summary
# P5.0 — Judge client + prompt loader contract tests (eight FAST tests).
# Exports: test_load_judge_prompt_reads_pack_prompts_dir,
#          test_load_judge_prompt_falls_back_to_packaged_default,
#          test_load_judge_prompt_missing_version_raises,
#          test_render_judge_prompt_substitutes_all_markers,
#          test_judge_client_score_calls_llm_with_structured_output,
#          test_judgment_score_clamped,
#          test_judge_question_is_frozen,
#          test_build_judge_client_uses_pack_judge_config
# Deps: src.eval.runner (JudgeClient, JudgeQuestion, JudgmentScore,
#       build_judge_client, load_judge_prompt, render_judge_prompt),
#       src.eval.pack.schema (EvalPack, PackMeta, JudgeConfig, Thresholds).
# @end-summary
"""P5.0 — judge client + prompt loader contract tests."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.eval.pack.schema import EvalPack, JudgeConfig, PackMeta, Thresholds
from src.eval.runner import (
    JudgeClient,
    JudgeQuestion,
    JudgmentScore,
    build_judge_client,
    load_judge_prompt,
    render_judge_prompt,
)

# ---------------------------------------------------------------------------
# Fake ChatModel (canonical structured-output stub for fast tests)
# ---------------------------------------------------------------------------


class _FakeStructured:
    def __init__(self, response):
        self.response = response
        self.invoked_with = None

    def invoke(self, prompt):
        self.invoked_with = prompt
        return self.response


class _FakeChatModel:
    def __init__(self, response):
        self.response = response
        self.structured_arg = None
        self._structured: _FakeStructured | None = None

    def with_structured_output(self, schema):
        self.structured_arg = schema
        self._structured = _FakeStructured(self.response)
        return self._structured


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_TEMPLATE_WITH_MARKERS = (
    "Q={query}\n"
    "SPAN={expected_answer_span}\n"
    "QTYPE={qtype}\n"
    "CHUNKS=\n{chunk_block}\n"
)


def _write_pack_prompt(pack_dir: Path, body: str, *, version: str = "v1") -> Path:
    prompts_dir = pack_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    p = prompts_dir / f"judge_{version}.md"
    p.write_text(body, encoding="utf-8")
    return p


def _make_pack(*, tier1_model: str = "fake-judge-model-xyz", temperature: float = 0.3) -> EvalPack:
    return EvalPack(
        meta=PackMeta(
            name="testpack",
            version=1,
            profile="generic",
            corpus_pin="deadbeef" * 8,
            description=None,
            judge=JudgeConfig(
                tier1_model=tier1_model,
                tier1_prompt_version="v1",
                temperature=temperature,
                samples_per_claim=1,
            ),
            collection_name_template="ragweave_{name}_{corpus_pin_short}",
        ),
        manifest=[],
        goldens={},
        thresholds=Thresholds(profile="generic"),
    )


# ---------------------------------------------------------------------------
# Tests 1-3 — load_judge_prompt
# ---------------------------------------------------------------------------


def test_load_judge_prompt_reads_pack_prompts_dir(tmp_path: Path):
    body = (
        "PACK-SPECIFIC HEADER\n"
        "Query: {query}\n"
        "Span: {expected_answer_span}\n"
        "Type: {qtype}\n"
        "Chunks: {chunk_block}\n"
    )
    _write_pack_prompt(tmp_path, body, version="v1")
    result = load_judge_prompt(tmp_path, version="v1")
    assert result == body
    for marker in ("{query}", "{expected_answer_span}", "{chunk_block}", "{qtype}"):
        assert marker in result


def test_load_judge_prompt_falls_back_to_packaged_default(tmp_path: Path):
    # No prompts/ directory in pack — must fall back to packaged default.
    assert not (tmp_path / "prompts").exists()
    result = load_judge_prompt(tmp_path, version="v1")
    assert isinstance(result, str)
    assert result.strip() != ""
    for marker in ("{query}", "{expected_answer_span}", "{chunk_block}", "{qtype}"):
        assert marker in result, f"packaged default missing marker {marker}"


def test_load_judge_prompt_missing_version_raises(tmp_path: Path):
    # prompts/ exists but the requested version file is absent — do NOT
    # silently fall back to the packaged default. (Discriminating gate.)
    (tmp_path / "prompts").mkdir()
    _write_pack_prompt(tmp_path, "irrelevant", version="v2")
    with pytest.raises(FileNotFoundError) as excinfo:
        load_judge_prompt(tmp_path, version="v1")
    msg = str(excinfo.value)
    assert "v1" in msg
    assert str(tmp_path) in msg or tmp_path.name in msg


# ---------------------------------------------------------------------------
# Test 4 — render_judge_prompt
# ---------------------------------------------------------------------------


def test_render_judge_prompt_substitutes_all_markers():
    q = JudgeQuestion(
        qid="q1",
        qtype="factoid",
        query="What is X?",
        expected_answer_span="42",
        chunk_texts=("chunk A", "chunk B"),
    )
    rendered = render_judge_prompt(_TEMPLATE_WITH_MARKERS, q)
    assert "What is X?" in rendered
    assert "42" in rendered
    assert "chunk A" in rendered
    assert "chunk B" in rendered
    assert "factoid" in rendered
    # Chunks separated by a blank-line block.
    idx_a = rendered.index("chunk A")
    idx_b = rendered.index("chunk B")
    assert idx_a < idx_b
    between = rendered[idx_a:idx_b]
    assert "\n\n" in between
    # Markers must be gone.
    for marker in ("{query}", "{expected_answer_span}", "{chunk_block}", "{qtype}"):
        assert marker not in rendered


# ---------------------------------------------------------------------------
# Test 5 — JudgeClient.score calls LLM with structured output
# ---------------------------------------------------------------------------


def test_judge_client_score_calls_llm_with_structured_output():
    expected = JudgmentScore(score=0.8, reasoning="ok")
    fake = _FakeChatModel(expected)
    client = JudgeClient(template=_TEMPLATE_WITH_MARKERS, chat_model=fake)
    q = JudgeQuestion(
        qid="q1",
        qtype="factoid",
        query="What is X?",
        expected_answer_span="42",
        chunk_texts=("chunk A", "chunk B"),
    )
    result = client.score(q)
    assert isinstance(result, JudgmentScore)
    assert result.score == 0.8
    assert result.reasoning == "ok"
    assert fake.structured_arg is JudgmentScore
    invoked = fake._structured.invoked_with
    assert isinstance(invoked, str)
    assert "What is X?" in invoked
    assert "42" in invoked
    assert "chunk A" in invoked
    assert "chunk B" in invoked


# ---------------------------------------------------------------------------
# Test 6 — JudgmentScore validation clamps to [0.0, 1.0]
# ---------------------------------------------------------------------------


def test_judgment_score_clamped():
    # Boundaries accepted.
    assert JudgmentScore(score=0.0, reasoning="x").score == 0.0
    assert JudgmentScore(score=1.0, reasoning="x").score == 1.0
    # Out of range — both ends rejected.
    with pytest.raises(ValidationError):
        JudgmentScore(score=-0.1, reasoning="x")
    with pytest.raises(ValidationError):
        JudgmentScore(score=1.1, reasoning="x")


# ---------------------------------------------------------------------------
# Test 7 — JudgeQuestion is a frozen dataclass with tuple chunk_texts
# ---------------------------------------------------------------------------


def test_judge_question_is_frozen():
    q = JudgeQuestion(
        qid="q1",
        qtype="factoid",
        query="What?",
        expected_answer_span=None,
        chunk_texts=("a", "b"),
    )
    assert isinstance(q.chunk_texts, tuple)
    assert dataclasses.is_dataclass(q)
    with pytest.raises(dataclasses.FrozenInstanceError):
        q.query = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Test 8 — build_judge_client reads pack.meta.judge and uses the factory seam
# ---------------------------------------------------------------------------


def test_build_judge_client_uses_pack_judge_config(tmp_path: Path, monkeypatch):
    pack = _make_pack(tier1_model="fake-judge-model-xyz", temperature=0.3)

    captured: dict = {}

    def fake_factory(*, model_alias: str, temperature: float):
        captured["model_alias"] = model_alias
        captured["temperature"] = temperature
        return _FakeChatModel(JudgmentScore(score=0.5, reasoning="ok"))

    # Force packaged-default prompt path (no pack_dir authored).
    client = build_judge_client(pack, chat_model_factory=fake_factory)
    assert isinstance(client, JudgeClient)
    assert captured["model_alias"] == "fake-judge-model-xyz"
    assert captured["temperature"] == 0.3
