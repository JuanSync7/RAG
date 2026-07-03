# @summary
# DEEP_STUDY action for the turn loop: vet the controller-authored document
# reference against the legal target namespace (evidence pool, turn-context
# chunk refs, docs already studied — unvetted model-authored refs are skipped,
# never fetched), fetch the full source document through the injected
# fetch_document seam (MinIO-backed at the endpoint), anchor the starting
# window on the best pool chunk's refactored_char_start (guarding the -1
# no-anchor sentinel -> window 0), then read overlapping windows with the
# turn_deep_study_read.md prompt, accumulating notes into a StudiedDoc and
# feeding non-empty window findings into the evidence pool (provenance
# deep_study). Stops on answer_found, document end, window budget, or LLM
# ledger exhaustion; emits a deep_study event per window read.
# Exports: run_deep_study
# Deps: src.common.prompts, src.common.utils,
#       src.retrieval.pipeline.turn_loop.{schemas,events,controller}
# @end-summary
"""DEEP_STUDY stage: windowed full-document read (design §3.7/§5).

The document body is split into fixed-size overlapping character windows
(``TurnBudget.deep_study_window_chars`` / ``deep_study_window_overlap_chars``)
— a deterministic, content-agnostic segmentation (CLAUDE.md §0: no structural
pattern-matching on the markdown). The walk starts at the window containing
the highest-scoring pooled chunk's anchor offset when one exists (that chunk
is why the controller chose this document) and proceeds forward; with no
anchor it starts at window 0.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.common.prompts import load_prompt, render, strip_reasoning
from src.common.utils import parse_json_object
from src.retrieval.pipeline.turn_loop.controller import controller_model_alias
from src.retrieval.pipeline.turn_loop.events import TurnEventEmitter
from src.retrieval.pipeline.turn_loop.schemas import (
    DeepStudyArgs,
    EvidenceChunk,
    StudiedDoc,
    TurnBudget,
    TurnContext,
    TurnEventType,
    TurnLoopDeps,
    TurnState,
)

logger = logging.getLogger(__name__)

_PROMPT_FILE = "turn_deep_study_read.md"


def _notes_preview_chars() -> int:
    """Preview cap for deep_study event payloads (reuses the turn-context
    preview tunable — one knob for every preview surface)."""
    from config import settings

    return int(getattr(settings, "RAG_TURN_CONTEXT_PREVIEW_CHARS", 320))


def _window_stride(budget: TurnBudget) -> int:
    """Forward step between consecutive window starts (floor 1: the config
    validator enforces window > overlap, but a bad hand-built budget must
    degrade to progress, not an infinite loop)."""
    return max(
        1,
        budget.deep_study_window_chars
        - max(0, budget.deep_study_window_overlap_chars),
    )


def _window_count(content_len: int, budget: TurnBudget) -> int:
    """Total overlapping windows covering ``content_len`` characters."""
    if content_len <= budget.deep_study_window_chars:
        return 1
    stride = _window_stride(budget)
    remainder = content_len - budget.deep_study_window_chars
    return 1 + -(-remainder // stride)  # ceil division


def _anchor_offset(state: TurnState, document_id: str) -> int:
    """Best anchored offset into the document from the evidence pool.

    Scans pooled retrieve-provenance chunks of this document for the
    highest-scoring one with a real anchor, guarding the ``-1`` no-anchor
    sentinel (design risk #5: ``refactored_char_start == -1`` means the span
    was never located in the refactored markdown). Returns ``-1`` when no
    anchored chunk exists — the caller then starts at window 0.
    """
    best_offset = -1
    best_score = float("-inf")
    for chunk in state.pool:
        if chunk.document_id != document_id:
            continue
        if chunk.refactored_char_start < 0:
            continue  # -1 sentinel: no anchor recorded for this span
        if chunk.score > best_score:
            best_score = chunk.score
            best_offset = chunk.refactored_char_start
    return best_offset


def _allowed_targets(state: TurnState, context: TurnContext) -> set[str]:
    """Collect the legal DEEP_STUDY target namespace for this turn.

    The class enforced here: UNVALIDATED MODEL-AUTHORED RESOURCE REFERENCES.
    The controller prompt says study targets "MUST be copied verbatim from
    the evidence digest or the conversation-context refs", and the frozen
    ``TurnContext.chunk_refs`` contract names those refs "the only legal
    DEEP_STUDY target namespace" — this makes that a code invariant instead
    of prompt text, so a prompt-injected controller (document text renders
    un-fenced into its prompt) cannot fetch arbitrary store objects or
    escape a filter-scoped request's evidence.

    Sources (pure typed-state reads): every ``document_id``/``source_key``
    in the evidence pool, documents already studied this turn, and the
    turn-context ``chunk_refs`` / ``docs_studied`` from prior turns.
    """
    allowed: set[str] = set()
    for chunk in state.pool:
        for ref in (chunk.document_id, chunk.source_key):
            if ref:
                allowed.add(ref)
    allowed.update(key for key in state.studied if key)
    for ref in context.chunk_refs:
        if not isinstance(ref, dict):
            continue
        for value in (ref.get("document_id"), ref.get("source_key")):
            text = str(value or "").strip()
            if text:
                allowed.add(text)
    for doc in context.docs_studied:
        if not isinstance(doc, dict):
            continue
        text = str(doc.get("document_id") or "").strip()
        if text:
            allowed.add(text)
    return allowed


def _vetted_refs(
    args: DeepStudyArgs, state: TurnState, context: TurnContext
) -> tuple[Optional[str], Optional[str]]:
    """Return ``(document_id, source_key)`` with unvetted refs nulled out.

    Each ref is kept only if it appears verbatim in the allowed namespace —
    an unvetted ``document_id`` is dropped even when the ``source_key`` is
    valid (and vice versa), because the fetch seam resolves EITHER handle
    across the whole store.
    """
    allowed = _allowed_targets(state, context)
    document_id = args.document_id if args.document_id in allowed else None
    source_key = args.source_key if args.source_key in allowed else None
    return document_id, source_key


def _parse_window_read(raw: str) -> tuple[str, bool]:
    """Salvage-parse one window-read response into ``(notes, answer_found)``.

    Fail-open: unusable JSON reads as "no new notes, keep going" — the window
    walk continues (bounded by the window budget) instead of aborting the study.
    """
    payload = parse_json_object(strip_reasoning(raw or ""))
    notes = str(payload.get("notes") or "").strip()
    answer_found = bool(payload.get("answer_found", False))
    return notes, answer_found


async def run_deep_study(
    args: DeepStudyArgs,
    *,
    context: TurnContext,
    state: TurnState,
    budget: TurnBudget,
    deps: TurnLoopDeps,
    emitter: TurnEventEmitter,
) -> None:
    """Run one DEEP_STUDY action: windowed read of one source document.

    The controller-authored target is first vetted against the legal
    namespace (:func:`_allowed_targets`) — a ref that never appeared in the
    evidence pool, the turn-context chunk refs, or the studied-docs ledger is
    logged and skipped (degrade-not-crash), never fetched. Then resolves the
    document via ``deps.fetch_document`` (``None`` on a miss — the action
    degrades to a no-op instead of raising, per the DI contract), walks up to
    ``budget.deep_study_max_windows`` NEW windows forward from the anchor
    window, and accumulates notes into ``state.studied``. Non-empty window
    findings enter the evidence pool as ``deep_study``-provenance chunks so
    drafting can cite them (design §5: one pool shape, no origin-branching).
    Repeat studies of the same document resume — windows already read are
    skipped without charging the window budget.

    Args:
        args: The controller's study arguments (question + verbatim doc ref).
        context: The cross-turn context (legal-target namespace source).
        state: The turn's mutable state (pool, studied registry).
        budget: The frozen per-turn budget.
        deps: Injected capabilities (``fetch_document``, provider, emit).
        emitter: The turn's event emitter / charged-call wrapper.
    """
    round_index = max(0, state.iteration - 1)
    if not (args.document_id or args.source_key):
        logger.warning(
            "turn loop DEEP_STUDY decision carried no document ref — skipping"
        )
        return
    document_id, source_key = _vetted_refs(args, state, context)
    doc_key = document_id or source_key or ""
    if not doc_key:
        logger.warning(
            "turn loop DEEP_STUDY target document_id=%r source_key=%r is "
            "outside the evidence/context namespace — skipping the unvetted "
            "model-authored reference",
            args.document_id,
            args.source_key,
        )
        return
    if doc_key not in state.studied and len(state.studied) >= budget.deep_study_max_docs:
        logger.warning(
            "turn loop deep-study doc budget exhausted (%d) — skipping %s",
            budget.deep_study_max_docs,
            doc_key,
        )
        return

    document: Optional[object] = None
    try:
        document = await deps.fetch_document(document_id, source_key)
    except Exception as exc:  # noqa: BLE001 — a failed fetch must not kill the turn
        logger.warning("turn loop fetch_document failed for %s: %s", doc_key, exc)
    if document is None:
        logger.warning("turn loop could not resolve document %s — skipping", doc_key)
        return
    content = str(getattr(document, "content", "") or "")
    title = str(getattr(document, "title", "") or "")
    if not content:
        logger.warning("turn loop document %s has empty content — skipping", doc_key)
        return

    studied = state.studied.get(doc_key)
    if studied is None:
        studied = StudiedDoc(document_id=doc_key, title=title)
        state.studied[doc_key] = studied
    elif title and not studied.title:
        studied.title = title

    stride = _window_stride(budget)
    window_count = _window_count(len(content), budget)
    anchor = _anchor_offset(state, doc_key)
    start_window = 0 if anchor < 0 else min(window_count - 1, anchor // stride)

    preview_cap = _notes_preview_chars()
    template = load_prompt(_PROMPT_FILE)
    windows_read_this_call = 0
    window = start_window
    answer_found = False
    while window < window_count and windows_read_this_call < budget.deep_study_max_windows:
        if window in studied.windows_read:
            window += 1  # resume: already read in an earlier DEEP_STUDY action
            continue
        window_start = window * stride
        window_text = content[window_start : window_start + budget.deep_study_window_chars]
        prompt = render(
            template,
            question=args.question,
            document_title=title or doc_key,
            window_index=str(window + 1),
            window_count=str(window_count),
            window_text=window_text,
            notes_so_far=studied.notes or "(none yet)",
        )
        response = await emitter.charged_call(
            alias=controller_model_alias(),
            purpose="deep_study_read",
            prompt=prompt,
            temperature=0.0,
        )
        if response is None:
            break  # LLM ledger exhausted or provider down — stop the study
        notes, answer_found = _parse_window_read(
            getattr(response, "content", "") or ""
        )
        studied.windows_read.append(window)
        windows_read_this_call += 1
        if notes:
            studied.notes = (studied.notes + "\n" + notes).strip()
            finding_id = f"deep_study::{doc_key}::w{window}"
            if finding_id not in state.seen_chunk_ids:
                state.seen_chunk_ids.add(finding_id)
                state.pool.append(
                    EvidenceChunk(
                        chunk_id=finding_id,
                        document_id=doc_key,
                        source_key=source_key or "",
                        source=title or doc_key,
                        heading=title,
                        text=notes,
                        # Findings were extracted FOR the study question — enter
                        # the pool at full ranking weight.
                        score=1.0,
                        refactored_char_start=window_start,
                        refactored_char_end=min(
                            len(content),
                            window_start + budget.deep_study_window_chars,
                        ),
                        provenance=EvidenceChunk.PROVENANCE_DEEP_STUDY,
                        round_added=round_index,
                    )
                )
        await emitter.emit(
            TurnEventType.DEEP_STUDY,
            {
                "document_id": doc_key,
                "title": title,
                "window": window + 1,
                "of_windows": window_count,
                "notes_preview": notes[:preview_cap],
            },
        )
        if answer_found:
            break
        window += 1

    # The accumulated notes ARE the study's takeaway; the controller sees them
    # via the evidence digest and the next turn via docs_studied persistence.
    studied.conclusion = studied.notes


__all__ = ["run_deep_study"]
