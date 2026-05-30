# @summary
# Pluggable judge contract for chunk-level faithfulness scoring (P5.0)
# + multi-sample helper (P6c).
# Exports: JudgeQuestion, JudgmentScore, JudgeClient, load_judge_prompt,
#          render_judge_prompt, build_judge_client, read_samples_per_claim,
#          read_max_parallel_judges.
# Deps: pydantic v2, src.common.llm.provider.get_llm,
#       src.eval.pack.schema.EvalPack.
# @end-summary
"""Judge client + prompt loader for the eval runner.

This module supplies the *contract surface* that P5's executor will wire
into per-golden faithfulness scoring. The runtime path is:

    template = load_judge_prompt(pack_dir, version)
    client   = build_judge_client(pack)
    score    = client.score(JudgeQuestion(...))

Tests inject a fake ChatModel through ``JudgeClient(chat_model=...)`` and a
fake factory through ``build_judge_client(pack, chat_model_factory=...)``.
No live LLM calls happen in fast tests.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

from src.eval.pack.schema import EvalPack

# ---------------------------------------------------------------------------
# Public dataclass / schema contracts
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class JudgeQuestion:
    """A single faithfulness-judgment input.

    Frozen because tests rely on identity stability and because hashability
    is convenient for future caching by ``qid``. ``chunk_texts`` is a tuple
    (not list) so the dataclass remains hashable.
    """

    qid: str
    qtype: str
    query: str
    expected_answer_span: str | None
    chunk_texts: tuple[str, ...]


class JudgmentScore(BaseModel):
    """Pydantic structured-output schema for the judge LLM call."""

    score: float = Field(ge=0.0, le=1.0)
    reasoning: str


# ---------------------------------------------------------------------------
# Prompt loader + renderer
# ---------------------------------------------------------------------------


_PACKAGED_DEFAULT_DIR = Path(__file__).parent / "prompts"


def load_judge_prompt(pack_dir: Path, version: str) -> str:
    """Load the judge prompt template for *pack_dir* at *version*.

    Resolution order:
      1. ``<pack_dir>/prompts/judge_<version>.md`` if ``prompts/`` exists.
         Missing the file when ``prompts/`` exists is an error — we do NOT
         silently fall back to the packaged default.
      2. Packaged default at ``src/eval/runner/prompts/judge_<version>.md``
         used only when the pack has no ``prompts/`` directory at all.
    """
    pack_prompts = Path(pack_dir) / "prompts"
    if pack_prompts.exists():
        candidate = pack_prompts / f"judge_{version}.md"
        if not candidate.exists():
            raise FileNotFoundError(
                f"pack {pack_dir!s} authored a prompts/ directory but is missing "
                f"judge_{version}.md (requested version={version!r})"
            )
        return candidate.read_text(encoding="utf-8")
    default = _PACKAGED_DEFAULT_DIR / f"judge_{version}.md"
    if not default.exists():
        raise FileNotFoundError(
            f"packaged default judge prompt missing for version={version!r} "
            f"(expected at {default!s})"
        )
    return default.read_text(encoding="utf-8")


def _format_chunk_block(chunks: tuple[str, ...]) -> str:
    """Numeric-prefixed chunks separated by a blank line."""
    return "\n\n".join(f"[{i + 1}] {text}" for i, text in enumerate(chunks))


def render_judge_prompt(template: str, question: JudgeQuestion) -> str:
    """Substitute ``{query}``, ``{expected_answer_span}``, ``{qtype}``, and
    ``{chunk_block}`` in *template* using values from *question*."""
    return template.format(
        query=question.query,
        expected_answer_span=question.expected_answer_span or "",
        qtype=question.qtype,
        chunk_block=_format_chunk_block(question.chunk_texts),
    )


# ---------------------------------------------------------------------------
# JudgeClient
# ---------------------------------------------------------------------------


class _SupportsStructuredOutput(Protocol):
    def with_structured_output(self, schema: type[BaseModel]) -> Any: ...


class JudgeClient:
    """Wraps a ChatModel and scores ``JudgeQuestion`` instances.

    Calls ``chat_model.with_structured_output(JudgmentScore).invoke(prompt)``.
    """

    def __init__(self, *, template: str, chat_model: _SupportsStructuredOutput):
        self.template = template
        self.chat_model = chat_model

    def score(self, question: JudgeQuestion) -> JudgmentScore:
        rendered = render_judge_prompt(self.template, question)
        structured = self.chat_model.with_structured_output(JudgmentScore)
        return structured.invoke(rendered)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def _default_factory(*, model_alias: str, temperature: float) -> Any:
    # Lazy import: the runtime LLM provider pulls in langchain_core which fast
    # tests stub out via conftest. Importing at call time keeps the contract
    # surface importable in stubbed environments.
    from src.common.llm.provider import get_llm

    return get_llm(model_alias=model_alias, temperature=temperature)


def build_judge_client(
    pack: EvalPack,
    *,
    chat_model_factory: Callable[..., Any] | None = None,
    pack_dir: Path | None = None,
) -> JudgeClient:
    """Build a JudgeClient from an EvalPack.

    Reads ``pack.meta.judge.tier1_model``, ``pack.meta.judge.temperature``,
    and ``pack.meta.judge.tier1_prompt_version``. Uses *chat_model_factory*
    if provided (the sole seam used by fast tests); otherwise wraps
    ``src.common.llm.provider.get_llm``.

    If *pack_dir* is supplied, the loader will check ``<pack_dir>/prompts/``
    first; otherwise it always falls back to the packaged default template.
    """
    factory = chat_model_factory or _default_factory
    judge_cfg = pack.meta.judge
    chat_model = factory(
        model_alias=judge_cfg.tier1_model,
        temperature=judge_cfg.temperature,
    )
    if pack_dir is not None:
        template = load_judge_prompt(pack_dir, version=judge_cfg.tier1_prompt_version)
    else:
        # No pack_dir provided — use the packaged default by passing a sentinel
        # directory that has no prompts/ subdir. We synthesize that by using a
        # non-existent path, which load_judge_prompt handles via the default.
        # Implementation detail: pass a path whose prompts/ does not exist.
        template = _load_default_template(version=judge_cfg.tier1_prompt_version)
    return JudgeClient(template=template, chat_model=chat_model)


def read_samples_per_claim(pack: EvalPack) -> int:
    """Read ``pack.meta.judge.samples_per_claim`` with a defensive default.

    Returns 1 when the field is missing (legacy packs). Raises ``ValueError``
    when the value is < 1, since callers will use it as a loop count.
    """
    n = getattr(pack.meta.judge, "samples_per_claim", 1)
    if n is None:
        n = 1
    try:
        n_int = int(n)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"samples_per_claim must be an integer >= 1, got {n!r}"
        ) from exc
    if n_int < 1:
        raise ValueError(f"samples_per_claim must be >= 1, got {n_int}")
    return n_int


def read_max_parallel_judges(pack: EvalPack) -> int:
    """Read ``pack.meta.judge.max_parallel_judges`` with a defensive default.

    Mirrors :func:`read_samples_per_claim`. Returns 1 when the field is
    missing (legacy packs) or ``None``. Raises ``ValueError`` when the value
    is < 1, since callers use it as a thread-pool width.
    """
    n = getattr(pack.meta.judge, "max_parallel_judges", 1)
    if n is None:
        n = 1
    try:
        n_int = int(n)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"max_parallel_judges must be an integer >= 1, got {n!r}"
        ) from exc
    if n_int < 1:
        raise ValueError(f"max_parallel_judges must be >= 1, got {n_int}")
    return n_int


def _load_default_template(*, version: str) -> str:
    default = _PACKAGED_DEFAULT_DIR / f"judge_{version}.md"
    if not default.exists():
        raise FileNotFoundError(
            f"packaged default judge prompt missing for version={version!r} "
            f"(expected at {default!s})"
        )
    return default.read_text(encoding="utf-8")
