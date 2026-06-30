# @summary
# Model-driven chunk judge for the agentic loop: scores each retrieved candidate
# for relevance + faithfulness (generic [0,1] properties, no pattern matching =
# the KEEP gate), emits a listwise best->worst `ranking` (the ORDER signal that
# reproduces the measured near-ideal ranking), and judges the kept set for
# sufficiency. FAIL-OPEN: on empty/unparseable JSON or a count mismatch it returns
# keep-all verdicts with rank=-1 so a flaky model (e.g. qwopus emitting "{}")
# never drops the round AND the caller falls back to raw-hybrid order.
# Exports: judge_chunks
# Deps: src.retrieval.pipeline.agentic.state, src.retrieval.pipeline.deep_research (_parse_json_object)
# @end-summary
"""Generic, model-driven relevance/faithfulness/sufficiency judging.

The judge is the only model-driven component of the loop and the only place a
chunk can be dropped. It judges GENERIC properties of each ``(question, chunk)``
pair — never a vendor, document, heading, or phrasing match (CLAUDE.md §0).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from src.retrieval.common.schemas import RankedResult
from src.retrieval.pipeline.agentic.state import ChunkVerdict, PoolVerdict
from src.retrieval.pipeline.deep_research import _parse_json_object

logger = logging.getLogger(__name__)

# This module is at src/retrieval/pipeline/agentic/ — four parents up is the
# repo root (one deeper than deep_research.py, which uses parents[3]).
_PROMPT_DIR = Path(__file__).resolve().parents[4] / "prompts"
_JUDGE_PROMPT_FILE = _PROMPT_DIR / "agentic_chunk_judge.md"
_JUDGE_CONCISE_PROMPT_FILE = _PROMPT_DIR / "agentic_chunk_judge_concise.md"


def _load_prompt(path: Path = _JUDGE_PROMPT_FILE) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - packaging error
        raise RuntimeError(
            f"Agentic judge prompt not found at {path}: {exc}"
        ) from exc


def _render(template: str, **vars_: str) -> str:
    rendered = template
    for k, v in vars_.items():
        rendered = rendered.replace("{{ " + k + " }}", v)
    return rendered


def _clamp01(v: Any, default: float) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, f))


def _strip_reasoning(text: str) -> str:
    """Drop a leading ``<think>…</think>`` reasoning block (reasoning models emit
    one ahead of the answer). Keep only what follows the LAST ``</think>`` so a
    stray ``{``/``}`` inside the reasoning can't poison the first-{ to last-}
    salvage. No-op for plain output."""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1]
    return text


def _keep_all(candidates: list[RankedResult]) -> list[ChunkVerdict]:
    """Fail-open verdicts: keep every candidate (relevance=faithfulness=1.0).

    Used when the judge model returns nothing parseable. Dropping a whole round
    on a flaky JSON response would reintroduce the 'I cannot answer' refusals the
    project fought, so the loop degrades to 'keep what was retrieved'."""
    return [
        ChunkVerdict(index=i, relevance=1.0, faithfulness=1.0, keep=True,
                     reason="fail-open: judge output unusable")
        for i in range(len(candidates))
    ]


def _build_chunk_block(candidates: list[RankedResult], max_chars: int = 800) -> str:
    """Render the chunk block, capping each chunk's text so a deep pool of large
    chunks cannot blow the judge model's context window (relevance/ranking needs a
    representative excerpt, not the full chunk)."""
    lines: list[str] = []
    for i, c in enumerate(candidates):
        text = (c.text or "").replace("\n", " ").strip()
        if max_chars > 0 and len(text) > max_chars:
            text = text[:max_chars]
        lines.append(f"[{i}] {text}")
    return "\n".join(lines)


def _parse_concise(
    obj: Optional[dict[str, Any]], candidates: list[RankedResult]
) -> tuple[list[ChunkVerdict], Optional[PoolVerdict]]:
    """Build verdicts from the concise judge output (a ranked id-list).

    Listed indices are kept, in list order (``rank`` = position); every other
    candidate is dropped. A missing/non-list ``ranking`` is fail-open keep-all
    (rank -1, pool None) so a flaky judge never drops the round and ordering
    falls back to hybrid. An EMPTY list is a valid 'nothing relevant' verdict
    (drop all) — distinct from fail-open."""
    if not obj or not isinstance(obj.get("ranking"), list):
        logger.warning("agentic concise judge returned no ranking — keeping all")
        return _keep_all(candidates), None

    rank_of: dict[int, int] = {}
    pos = 0
    for ridx in obj["ranking"]:
        try:
            ri = int(ridx)
        except (TypeError, ValueError):
            continue
        if 0 <= ri < len(candidates) and ri not in rank_of:
            rank_of[ri] = pos
            pos += 1

    verdicts: list[ChunkVerdict] = []
    for i in range(len(candidates)):
        if i in rank_of:
            verdicts.append(ChunkVerdict(
                index=i, relevance=1.0, faithfulness=1.0, keep=True,
                reason="ranked", rank=rank_of[i],
            ))
        else:
            verdicts.append(ChunkVerdict(
                index=i, relevance=0.0, faithfulness=0.0, keep=False,
                reason="not ranked", rank=-1,
            ))

    pool = PoolVerdict(
        sufficient=bool(obj.get("sufficient", False)),
        confidence=_clamp01(obj.get("confidence"), 0.0),
        missing_information=str(obj.get("missing_information") or ""),
    )
    return verdicts, pool


async def judge_chunks(
    provider,
    *,
    model_alias: str,
    original_question: str,
    candidates: list[RankedResult],
    timeout_s: int,
    json_mode: bool = True,
    concise: bool = False,
    max_keep: Optional[int] = None,
    max_output_tokens: int = 1024,
    max_chars_per_chunk: int = 800,
) -> tuple[list[ChunkVerdict], Optional[PoolVerdict]]:
    """Judge ``candidates`` against ``original_question``.

    Returns ``(per_chunk_verdicts, pool_verdict)``. ``per_chunk_verdicts`` is
    always the same length as ``candidates`` (1:1 by index). On any failure the
    verdicts are keep-all (fail-open) and the pool verdict is ``None`` so the
    caller does not treat the round as 'sufficient' on a non-answer.

    ``concise`` selects the tiny-output judge: the model reasons then emits only a
    ranked id-list (+ sufficiency) instead of a scored object per chunk. Listed
    indices are kept (rank = position); unlisted are dropped. This collapses the
    gate + rank into one short output to cut decode latency (the dominant cost).
    ``max_keep`` caps the ranking length shown to the model (defaults to the
    candidate count).
    """
    if not candidates:
        return [], None

    chunk_block = _build_chunk_block(candidates, max_chars=max_chars_per_chunk)
    if concise:
        prompt = _render(
            _load_prompt(_JUDGE_CONCISE_PROMPT_FILE),
            original_question=original_question,
            chunks=chunk_block,
            max_keep=str(max_keep if max_keep is not None else len(candidates)),
        )
    else:
        prompt = _render(
            _load_prompt(),
            original_question=original_question,
            chunks=chunk_block,
        )
    kwargs: dict[str, Any] = dict(
        model_alias=model_alias,
        timeout=timeout_s,
        max_tokens=max(1, max_output_tokens),
        temperature=0.0,
    )
    if json_mode:
        # Only request guided JSON when configured (a JSON-reliable instruct
        # model). For a reasoning model this constraint causes empty/malformed
        # output, so it stays off by default and we salvage-parse free output.
        kwargs["response_format"] = {"type": "json_object"}
    try:
        resp = await provider.agenerate(
            [{"role": "user", "content": prompt}],
            **kwargs,
        )
    except Exception as exc:  # noqa: BLE001 — fail open
        logger.warning("agentic judge LLM call failed: %s — keeping all", exc)
        return _keep_all(candidates), None

    obj = _parse_json_object(_strip_reasoning(getattr(resp, "content", "") or ""))

    if concise:
        return _parse_concise(obj, candidates)

    if not obj or not isinstance(obj.get("chunks"), list):
        logger.warning("agentic judge returned unusable JSON — keeping all")
        return _keep_all(candidates), None

    # Map model verdicts back onto candidates by index. Any candidate the model
    # omitted is kept by default (fail-open per chunk, not drop-by-omission).
    by_index: dict[int, dict[str, Any]] = {}
    for entry in obj["chunks"]:
        if not isinstance(entry, dict):
            continue
        try:
            idx = int(entry.get("i", entry.get("index")))
        except (TypeError, ValueError):
            continue
        by_index[idx] = entry

    # Listwise ranking: an ordered list of chunk indices, best -> worst. This is
    # the ORDER signal (the artifact that reproduced the measured near-ideal
    # ranking); pointwise relevance/faithfulness below are only the keep gate.
    # rank[i] = position of candidate i in that order (0 = best); -1 if the judge
    # did not rank it (omitted, or no ranking emitted at all -> fail open to the
    # caller's hybrid-order fallback).
    rank_by_index: dict[int, int] = {}
    raw_ranking = obj.get("ranking")
    if isinstance(raw_ranking, list):
        pos = 0
        for ridx in raw_ranking:
            try:
                ri = int(ridx)
            except (TypeError, ValueError):
                continue
            if 0 <= ri < len(candidates) and ri not in rank_by_index:
                rank_by_index[ri] = pos
                pos += 1

    verdicts: list[ChunkVerdict] = []
    for i in range(len(candidates)):
        entry = by_index.get(i)
        if entry is None:
            verdicts.append(ChunkVerdict(
                index=i, relevance=1.0, faithfulness=1.0, keep=True,
                reason="fail-open: omitted by judge",
                rank=rank_by_index.get(i, -1),
            ))
            continue
        relevance = _clamp01(entry.get("relevance"), 1.0)
        faithfulness = _clamp01(entry.get("faithfulness"), 1.0)
        keep = bool(entry.get("keep", True))
        verdicts.append(ChunkVerdict(
            index=i, relevance=relevance, faithfulness=faithfulness, keep=keep,
            reason=str(entry.get("reason") or ""),
            rank=rank_by_index.get(i, -1),
        ))

    pool_obj = obj.get("pool")
    pool: Optional[PoolVerdict] = None
    if isinstance(pool_obj, dict):
        covered = pool_obj.get("covered_aspects") or []
        if isinstance(covered, str):
            covered = [covered]
        pool = PoolVerdict(
            sufficient=bool(pool_obj.get("sufficient", False)),
            confidence=_clamp01(pool_obj.get("confidence"), 0.0),
            missing_information=str(pool_obj.get("missing_information") or ""),
            covered_aspects=[str(a) for a in covered if str(a).strip()],
        )
    return verdicts, pool
