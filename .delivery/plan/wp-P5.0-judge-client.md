---
slice_id: P5.0-judge-client
validable_outcome: |
  `uv run pytest tests/eval/test_judge_client.py -q` reports all FAST tests green. Specifically:
    1. `test_load_judge_prompt_reads_pack_prompts_dir` (FAST): writes a tmp pack dir with
       `prompts/judge_v1.md` containing the markers `{query}`, `{expected_answer_span}`,
       `{chunk_block}`, and `{qtype}`. `load_judge_prompt(pack_dir, version="v1")` returns
       the file contents verbatim as a string.
    2. `test_load_judge_prompt_falls_back_to_packaged_default` (FAST): when the pack has no
       `prompts/` directory, `load_judge_prompt(pack_dir, version="v1")` returns the
       packaged default template (read from `src/eval/runner/prompts/judge_v1.md`).
       Resulting string contains all four template markers above.
    3. `test_load_judge_prompt_missing_version_raises` (FAST): if a `prompts/` directory
       exists but the requested version file is missing, raises `FileNotFoundError` with
       a message naming the pack and version. (Don't silently fall back when the author
       explicitly authored prompts but mis-typed the version — discriminating gate per
       memory `feedback_discriminating_partner_test`.)
    4. `test_render_judge_prompt_substitutes_all_markers` (FAST): `render_judge_prompt(
       template, question)` for a `JudgeQuestion(qid="q1", qtype="factoid",
       query="What is X?", expected_answer_span="42", chunk_texts=("chunk A", "chunk B"))`
       returns a string where (a) `"What is X?"` appears, (b) `"42"` appears, (c) both
       chunks appear separated by a blank-line block, (d) `"factoid"` appears, and
       (e) the input markers (`{query}`, etc.) are NO LONGER present.
    5. `test_judge_client_score_calls_llm_with_structured_output` (FAST): a `JudgeClient`
       built with a fake ChatModel returns a `JudgmentScore(score=0.8, reasoning="...")`.
       Asserts:
       - `client.score(question)` returns the expected `JudgmentScore`.
       - The fake ChatModel's `with_structured_output` was called with `JudgmentScore`.
       - The rendered prompt (containing query + span + chunks) was passed to `.invoke`.
       Pattern: subclass / fake `ChatLLMAdapter`-shaped object via monkeypatch — see
       `tests/llm/test_provider_instrumentation.py:_FakeRouter` for the canonical
       stub-router pattern. P5.0 uses a `_FakeChatModel` exposing `with_structured_output`
       and `invoke`. Do NOT call litellm.
    6. `test_judgment_score_clamped` (FAST): pydantic validation of `JudgmentScore`
       rejects `score=-0.1` and `score=1.1`. Accepts `0.0` and `1.0` boundaries.
    7. `test_judge_question_is_frozen` (FAST): `JudgeQuestion` is a frozen dataclass with
       `chunk_texts: tuple[str, ...]`. Mutation raises `FrozenInstanceError`.
    8. `test_build_judge_client_uses_pack_judge_config` (FAST): `build_judge_client(
       pack)` reads `pack.meta.judge.tier1_model` and `pack.meta.judge.temperature`,
       constructs the underlying ChatModel via a `chat_model_factory` arg
       (monkeypatchable), and returns a `JudgeClient`. Default factory is
       `src.common.llm.provider.get_llm`. Test passes a fake factory and verifies
       it received the model + temperature from the pack.
  PLUS: `uv run pytest tests/eval/ tests/vector_db/ tests/retrieval/ -q -m "not slow and not integration"` stays green (regression).
touches:
  - src/eval/__init__.py
  - src/eval/runner/__init__.py
  - src/eval/runner/judge.py
  - src/eval/runner/prompts/__init__.py
  - src/eval/runner/prompts/judge_v1.md
  - evals/packs/opentitan_riscv/prompts/judge_v1.md
  - tests/eval/test_judge_client.py
depends_on:
  - P4-retrieval-metrics
  - P0.5-eval-pack-format
---

# P5.0 — Judge client + prompt loader (precondition for P5)

## End-state (validable, machine-checkable)

A pluggable judge contract lives at `src/eval/runner/judge.py`:

- `JudgeQuestion` — frozen dataclass: `qid`, `qtype`, `query`, `expected_answer_span: str | None`, `chunk_texts: tuple[str, ...]`.
- `JudgmentScore` — pydantic `BaseModel` with `score: float` (constrained 0.0–1.0) and `reasoning: str`. Used as the structured-output schema for the judge LLM call.
- `JudgeClient` — wraps a LangChain ChatModel; exposes `score(question: JudgeQuestion) -> JudgmentScore`. Renders the prompt internally and invokes the model via `.with_structured_output(JudgmentScore).invoke(rendered)`.
- `load_judge_prompt(pack_dir: Path, version: str) -> str` — reads `<pack_dir>/prompts/judge_<version>.md`; if `prompts/` does not exist, falls back to the packaged default `src/eval/runner/prompts/judge_<version>.md`. If `prompts/` exists but the version file does not, raises.
- `render_judge_prompt(template: str, question: JudgeQuestion) -> str` — substitutes `{query}`, `{expected_answer_span}`, `{qtype}`, and `{chunk_block}` (chunks joined with a blank line and a numeric prefix per chunk).
- `build_judge_client(pack: EvalPack, *, chat_model_factory: Callable | None = None) -> JudgeClient` — convenience builder. Resolves the model name and temperature from `pack.meta.judge` and uses `chat_model_factory or get_llm` to build the underlying ChatModel. The `chat_model_factory` parameter is the SOLE seam for fast tests.

The OpenTitan pack gains `evals/packs/opentitan_riscv/prompts/judge_v1.md` (real prompt body).
The packaged default at `src/eval/runner/prompts/judge_v1.md` is a generic chunk-level faithfulness rubric used when a pack does not author its own.

No scoring of goldens. No aggregation. No `EvalReport` extension. P5.0 is the contract surface; P5 wires it into the executor.

## Current state (surveyed by SA1)

- `src/common/llm/provider.py:215-258` exposes `get_llm(model_alias, *, temperature, ...) -> ChatLLMAdapter`. Returns a LangChain `BaseChatModel` supporting `.with_structured_output(PydanticModel)` and `.invoke(prompt)`.
- The underlying `ChatLLMAdapter` is a `BaseChatModel` — `with_structured_output` is inherited from `BaseChatModel` and parses structured pydantic output via tool-calling/JSON-mode.
- `src.eval.pack.schema.JudgeConfig` is already authored (`tier1_model`, `tier1_prompt_version`, `temperature`, `samples_per_claim`).
- Opentitan pack.yaml uses `tier1_model: claude-haiku-4-5-20251001`, `tier1_prompt_version: v1`, `temperature: 0.0`, `samples_per_claim: 3`.
- NO existing `prompts/` directory under `evals/packs/opentitan_riscv/`. P0.5 deferred this.
- NO existing judge/faithfulness eval scaffolding (`src/guardrails/` has runtime faithfulness but is monolithic and not reusable here).
- Canonical fake-ChatModel pattern: subclass `BaseChatModel` and stub `_generate`. SA1 cited `_FakeRouter` in `tests/llm/test_provider_instrumentation.py` but that's for the platform Router layer. For P5.0 fast tests we'll build a `_FakeChatModel` exposing `.with_structured_output()` and `.invoke()` to avoid taking on a litellm dep.

## Plan-spec alignment

P5.0 is the precondition micro-slice for P5 (per the P3.0/P3 precedent). It introduces the pluggable judge surface so P5 can focus on the executor logic (iterate retrieved chunks for each golden, call `JudgeClient.score`, aggregate per-qtype). Without P5.0, P5 would need to also author the prompt-loader, structured-output schema, and fake-ChatModel test pattern — too large for one slice.

Design decision (recorded for memory after slice closes): **chunk-level direct faithfulness**, not answer-synthesis-then-judge. The judge LLM receives the retrieved chunks and the `expected_answer_span` and scores whether the chunks support the expected fact. Cheaper, faster, and aligned with P4 (retrieval-quality measurement). Production-grade end-to-end RAG faithfulness lives in `src/guardrails/`, not here.

## Required deliverables

### Code

1. **`src/eval/runner/judge.py`** (new) — implements `JudgeQuestion`, `JudgmentScore`, `JudgeClient`, `load_judge_prompt`, `render_judge_prompt`, `build_judge_client`. `@summary` block at top.

2. **`src/eval/runner/prompts/__init__.py`** (new, empty) — makes prompts directory a package so the packaged default is locatable via `importlib.resources` OR via `Path(__file__).parent / "judge_v1.md"`. Pick the simpler `Path(__file__).parent` approach.

3. **`src/eval/runner/prompts/judge_v1.md`** (new) — generic chunk-level faithfulness rubric. Must contain markers `{query}`, `{expected_answer_span}`, `{qtype}`, `{chunk_block}`. Body should instruct the judge to score 0.0 (unsupported) to 1.0 (clearly supported) based ONLY on the retrieved chunks. Keep it short (~30 lines). Document the qtype branches:
   - factoid → look for `expected_answer_span` verbatim or paraphrased.
   - qfs / multi_topic / multi_aspect → score whether chunks would support a faithful summary (anchor-terms heuristic; phrased as a rubric, not a hard-coded check).
   - adversarial / out_of_corpus → score 1.0 ONLY if chunks DO NOT contain a confident answer (refusal is correct).
   - messy → treat as the underlying qtype's rules (typically factoid).

4. **`evals/packs/opentitan_riscv/prompts/judge_v1.md`** (new) — copy of the packaged default for now (pack-specific overrides happen in P5 or later). The DIFFERENCE that proves the loader respects the pack override: add ONE extra sentence in the header so test 1 can distinguish the pack version from the packaged default if needed. Keep the four markers verbatim.

5. **`src/eval/runner/__init__.py`** — re-export `JudgeClient`, `JudgeQuestion`, `JudgmentScore`, `build_judge_client`, `load_judge_prompt`, `render_judge_prompt`. Add to `__all__`. Keep alphabetical.

6. **`src/eval/__init__.py`** — surface the new exports alongside existing P0-P4 exports.

### Tests (`tests/eval/test_judge_client.py`)

Eight fast tests enumerated in `validable_outcome`. Key patterns:

- For tests 1-3 (prompt loader): use `tmp_path` to build a fake pack dir. Test 2 (packaged default) does NOT use a tmp pack — pass a tmp pack dir with no `prompts/`, and rely on the packaged default being readable.

- For test 5 (JudgeClient.score wiring), define a `_FakeChatModel` at module top with:
  ```python
  class _FakeStructured:
      def __init__(self, response): self.response = response; self.invoked_with = None
      def invoke(self, prompt): self.invoked_with = prompt; return self.response
  class _FakeChatModel:
      def __init__(self, response): self.response = response; self.structured_arg = None; self._structured = None
      def with_structured_output(self, schema): self.structured_arg = schema; self._structured = _FakeStructured(self.response); return self._structured
  ```
  Construct `JudgeClient(template="... {query} {expected_answer_span} {chunk_block} {qtype} ...", chat_model=_FakeChatModel(JudgmentScore(score=0.8, reasoning="ok")))`. Assert `client.score(question).score == 0.8`, `client.chat_model.structured_arg is JudgmentScore`, and the rendered prompt contains the query, span, and both chunks.

- For test 8 (`build_judge_client`), pass a fake factory:
  ```python
  captured = {}
  def fake_factory(*, model, temperature):
      captured["model"] = model; captured["temperature"] = temperature
      return _FakeChatModel(...)
  client = build_judge_client(pack, chat_model_factory=fake_factory)
  assert captured["model"] == pack.meta.judge.tier1_model
  assert captured["temperature"] == pack.meta.judge.temperature
  ```
  IMPORTANT: `get_llm` takes `model_alias`, not `model`. The signature of `chat_model_factory` should match the kwargs that `build_judge_client` passes — DOCUMENT this in the test by importing `get_llm` and inspecting. The simplest call shape: `chat_model_factory(model_alias=..., temperature=...)`. The DEFAULT factory passed into `build_judge_client` should wrap `get_llm` with the matching kwargs.

## Constraints

- Boundary discipline: ONLY the 7 files in `touches`.
- No new runtime deps. (`langchain_core` is already a runtime dep via `src/common/llm/provider.py`.)
- No live LLM calls in fast tests. All ChatModel interactions go through the fake.
- No actual scoring of OpenTitan goldens. No `EvalReport` extension. No aggregation. P5's job.
- No marker softening, no xfail, no skip.
- Do NOT import `src.platform.llm` directly from `src/eval/runner/judge.py`. Go through `src.common.llm.provider.get_llm` only.
- Module-top imports for everything `monkeypatch` may need to swap (per the consistent P3/P4 pattern).
- Pydantic `JudgmentScore` uses `Field(ge=0.0, le=1.0)` for score validation.

## Anti-gaming guards

- Test 3 must reject the silent-fallback case. A no-op loader that always returns the packaged default would pass tests 1, 2, 4–8 but FAIL test 3. This is the discriminating gate per `feedback_discriminating_partner_test`.

- Test 6 (`JudgmentScore` clamping) must include the `score=1.1` rejection. A schema that constrains to `>= 0.0` only would pass tests 1–5 + 7–8 but fail this case.

- Mutation probe (SA2 runs AFTER green): change the packaged default to be empty string (delete its body). Test 4 (`render_judge_prompt_substitutes_all_markers`) MUST still pass because it uses an inline template, NOT the loaded default — that's CORRECT independence. Test 2 (`load_judge_prompt_falls_back_to_packaged_default`) MUST RED-FAIL because the assertion is that the four markers are present in the returned string. Restore the default file.

- Mutation probe 2: in `build_judge_client`, swap `pack.meta.judge.tier1_model` for the hardcoded string `"claude-haiku-4-5-20251001"`. Test 8 MUST RED-FAIL ONLY IF the test pack's tier1_model differs from that string. To make this probe red, the test should construct a pack-like object with `tier1_model="some-other-model"` rather than using the real opentitan pack (which uses haiku and would mask the mutation). Per `feedback_mutation_probe_one_invariant`, design the test to discriminate.

- Red-reason proof: before any implementation in `src/eval/runner/judge.py`, run `uv run pytest tests/eval/test_judge_client.py -q`. Expected: `ImportError: cannot import name 'JudgeClient' from 'src.eval.runner'` (or similar import-level failure). NOT a `FileNotFoundError` on the prompt template. NOT a litellm connection error. Capture verbatim into `.delivery/plan/wp-P5.0-judge-client-red-proof.txt`.

## Out of scope (defer)

- Faithfulness scoring of OpenTitan goldens — P5.
- Per-qtype faithfulness aggregation — P5.
- `EvalReport` extension with judge fields — P5.
- Live judge LLM call against a real model — P5 (will run as dual-marker slow+integration).
- `samples_per_claim > 1` multi-sample averaging — P5 or later.
- Per-pack prompt versioning beyond `v1` — out of scope.
- Differentiating "good summary" anchor-terms scoring from chunk-presence — P5 (the prompt template DOCUMENTS the rubric; P5 verifies it).

## Agent green-gate vs. human-review checkpoint

- **Agent green-gate (this slice ends here):** 8 fast tests green, regression sweep green, commit on `feat/p5.0-judge-client`.
- **Human-review checkpoint (separate, unblocks P5):** reviewer reads `src/eval/runner/prompts/judge_v1.md` and confirms the rubric is sensible. Iterate on prompt body during P5 if reviewer flags issues. Out of scope for slice DoD.
