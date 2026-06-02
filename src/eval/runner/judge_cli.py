# @summary
# Headless `claude` CLI LLM-as-judge backend (alternative to the litellm
# default). Wraps a constrained, non-coding-agent `claude -p ... --output-format
# json` invocation: no tools, clean-room settings, judge rubric as the system
# prompt. Parses the VERIFIED result-envelope shape into a JudgmentScore.
# Exports: JudgeBackendError, ClaudeCliJudgeClient, build_claude_argv,
#          build_claude_cli_judge_client.
# Deps: subprocess, json (stdlib); src.eval.runner.judge (JudgeQuestion,
#       JudgmentScore, render_judge_prompt, load_judge_prompt); src.eval.pack
#       .schema.EvalPack.
# @end-summary
"""Headless ``claude`` CLI judge backend.

This is an alternative LLM-as-judge backend to the default litellm-based
:class:`~src.eval.runner.judge.JudgeClient`. It shells out to the local
``claude`` CLI in a *constrained* mode — never the bare coding agent — so the
judge call is cheap (Haiku by default), tool-free, and uncontaminated by the
project/user agent persona or settings.

The runtime contract mirrors :func:`~src.eval.runner.judge.build_judge_client`:

    client = build_claude_cli_judge_client(pack, pack_dir=pack_dir)
    score  = client.score(JudgeQuestion(...))

The CLI is invoked as::

    claude -p "<rendered judge prompt + strict JSON-only instruction>"
      --output-format json
      --model <model>
      --system-prompt "<judge rubric template, full replace>"
      --allowed-tools ""        # no tools
      --setting-sources ""      # clean room: ignore project/user settings
      --permission-mode bypassPermissions   # non-prompting

The ``--output-format json`` envelope wraps the model's answer in a stringified
``.result`` field (optionally fenced); :meth:`ClaudeCliJudgeClient.score`
unwraps and parses it. ``run_fn`` is the testability seam: fast tests inject a
fake so no subprocess is ever spawned.
"""
from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.eval.pack.schema import EvalPack
from src.eval.runner.judge import (
    JudgeQuestion,
    JudgmentScore,
    load_judge_prompt,
    render_judge_prompt,
)

__all__ = [
    "JudgeBackendError",
    "ClaudeCliJudgeClient",
    "build_claude_argv",
    "build_claude_cli_judge_client",
]

# Default judge model: Haiku is cheap and adequate for chunk-level faithfulness.
_DEFAULT_MODEL = "claude-haiku-4-5-20251001"

# Non-prompting permission mode (confirmed from `claude --help` choices). With
# ``--allowed-tools ""`` no tools are reachable, so this only ensures the CLI
# never blocks on an interactive permission prompt in a headless run.
_PERMISSION_MODE = "bypassPermissions"

# Appended to the rendered judge prompt to force a parseable, prose-free answer.
_JSON_ONLY_INSTRUCTION = (
    "\n\nRespond with ONLY a JSON object of the form "
    '{"score": <float between 0 and 1>, "reasoning": <string>} '
    "— no prose, no markdown, no code fences."
)


class JudgeBackendError(RuntimeError):
    """Raised for any ``claude`` CLI invocation or response-parse failure.

    Covers a non-zero returncode, a timeout from the default runner, an
    ``is_error: true`` / non-success envelope, a missing/empty ``.result``, or
    an unparseable result payload. The eval CLI buckets this (like any other
    runtime error) into exit code 3.
    """


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp *value* into ``[low, high]``."""
    return max(low, min(high, value))


def _strip_json_fence(text: str) -> str:
    """Strip surrounding whitespace and an optional ```json ... ``` fence."""
    stripped = text.strip()
    if stripped.startswith("```"):
        # Drop the opening fence line (``` or ```json) and the closing fence.
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def build_claude_argv(
    prompt: str,
    *,
    model: str,
    system_prompt: str,
    permission_mode: str = _PERMISSION_MODE,
) -> list[str]:
    """Build the constrained ``claude`` CLI argv for a single judge call.

    Pure (no I/O) so the cost/safety constraint contract is unit-testable in
    isolation. The argv deliberately:

    * uses ``-p`` (print/non-interactive) with the rendered prompt,
    * forces ``--output-format json`` for a machine-parseable envelope,
    * pins ``--model`` (Haiku by default — cheap),
    * passes the rubric as ``--system-prompt`` (full replace),
    * sets ``--allowed-tools ""`` (no tools — not the coding agent),
    * sets ``--setting-sources ""`` (clean room — ignore project/user settings),
    * sets a non-prompting ``--permission-mode`` so a headless run never blocks.
    """
    return [
        "claude",
        "-p",
        prompt,
        "--output-format",
        "json",
        "--model",
        model,
        "--system-prompt",
        system_prompt,
        "--allowed-tools",
        "",
        "--setting-sources",
        "",
        "--permission-mode",
        permission_mode,
    ]


def _default_run_fn(argv: list[str], *, timeout: float):
    """Default runner: spawn the real ``claude`` CLI via subprocess.run."""
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


class ClaudeCliJudgeClient:
    """Scores :class:`JudgeQuestion` instances via the headless ``claude`` CLI.

    Satisfies the same single-method ``score(question) -> JudgmentScore``
    contract as :class:`~src.eval.runner.judge.JudgeClient`, so it is a
    drop-in ``judge_client`` for ``score_goldens``.

    :param template: the judge rubric template (passed as the CLI system
        prompt and also rendered into the ``-p`` prompt via
        :func:`~src.eval.runner.judge.render_judge_prompt`).
    :param model: the judge model alias/id (Haiku by default upstream).
    :param temperature: accepted for parity with the litellm backend; the CLI
        does not expose a temperature flag, so it is currently informational.
    :param run_fn: testability seam — a callable
        ``(argv: list[str], *, timeout: float) -> obj`` exposing
        ``.returncode`` / ``.stdout`` / ``.stderr``. When ``None`` (default) a
        real :func:`subprocess.run` is used; tests inject a fake so no process
        is spawned.
    :param timeout: per-call subprocess timeout in seconds.
    """

    def __init__(
        self,
        *,
        template: str,
        model: str,
        temperature: float = 0.0,
        run_fn: Callable[..., Any] | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.template = template
        self.model = model
        self.temperature = temperature
        self._run_fn = run_fn if run_fn is not None else _default_run_fn
        self.timeout = timeout

    def score(self, question: JudgeQuestion) -> JudgmentScore:
        """Render → invoke the CLI → parse the envelope → JudgmentScore.

        Raises :class:`JudgeBackendError` on any invocation/parse failure.
        """
        rendered = render_judge_prompt(self.template, question)
        prompt = rendered + _JSON_ONLY_INSTRUCTION
        argv = build_claude_argv(
            prompt, model=self.model, system_prompt=self.template
        )

        try:
            completed = self._run_fn(argv, timeout=self.timeout)
        except subprocess.TimeoutExpired as exc:
            raise JudgeBackendError(
                f"claude CLI judge timed out after {self.timeout}s"
            ) from exc

        if completed.returncode != 0:
            raise JudgeBackendError(
                f"claude CLI judge exited with code {completed.returncode}: "
                f"{(completed.stderr or '').strip()[:500]}"
            )

        return self._parse_envelope(completed.stdout)

    @staticmethod
    def _parse_envelope(stdout: str) -> JudgmentScore:
        """Parse the ``--output-format json`` envelope into a JudgmentScore."""
        try:
            envelope = json.loads(stdout)
        except (json.JSONDecodeError, TypeError) as exc:
            raise JudgeBackendError(
                f"claude CLI judge returned unparseable envelope JSON: {exc}"
            ) from exc

        if envelope.get("is_error") is not False:
            raise JudgeBackendError(
                f"claude CLI judge envelope reported an error: "
                f"is_error={envelope.get('is_error')!r}"
            )
        if envelope.get("subtype") != "success":
            raise JudgeBackendError(
                f"claude CLI judge envelope subtype was not 'success': "
                f"{envelope.get('subtype')!r}"
            )

        result = envelope.get("result")
        if not result or not isinstance(result, str):
            raise JudgeBackendError(
                f"claude CLI judge envelope missing/empty .result: {result!r}"
            )

        payload_text = _strip_json_fence(result)
        try:
            payload = json.loads(payload_text)
        except (json.JSONDecodeError, TypeError) as exc:
            raise JudgeBackendError(
                f"claude CLI judge .result was not parseable JSON: {exc}"
            ) from exc

        try:
            raw_score = float(payload["score"])
            reasoning = str(payload.get("reasoning", ""))
        except (KeyError, TypeError, ValueError) as exc:
            raise JudgeBackendError(
                f"claude CLI judge .result missing/invalid score field: {exc}"
            ) from exc

        return JudgmentScore(score=_clamp(raw_score), reasoning=reasoning)


def build_claude_cli_judge_client(
    pack: EvalPack,
    *,
    chat_model_factory: Callable[..., Any] | None = None,
    pack_dir: Path | None = None,
    run_fn: Callable[..., Any] | None = None,
) -> ClaudeCliJudgeClient:
    """Build a :class:`ClaudeCliJudgeClient` from an EvalPack.

    Mirrors :func:`~src.eval.runner.judge.build_judge_client`'s signature so it
    is a drop-in ``judge_client_factory``. ``chat_model_factory`` is accepted
    for signature parity and intentionally ignored (the CLI backend constructs
    no litellm chat model). The rubric template is loaded via
    :func:`~src.eval.runner.judge.load_judge_prompt`; model/temperature come
    from ``pack.meta.judge``.

    :param run_fn: passed through to the client as its subprocess seam (so the
        CLI dispatch path and offline tests can inject a fake runner).
    """
    judge_cfg = pack.meta.judge
    if pack_dir is not None:
        template = load_judge_prompt(
            pack_dir, version=judge_cfg.tier1_prompt_version
        )
    else:
        from src.eval.runner.judge import _load_default_template

        template = _load_default_template(version=judge_cfg.tier1_prompt_version)

    model = judge_cfg.tier1_model or _DEFAULT_MODEL
    return ClaudeCliJudgeClient(
        template=template,
        model=model,
        temperature=judge_cfg.temperature,
        run_fn=run_fn,
    )
