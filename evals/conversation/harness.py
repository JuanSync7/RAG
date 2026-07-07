# @summary
# Multi-turn eval harness for the turn-level agentic conversation loop. Two
# drive modes: OFFLINE (default, zero infra — drives run_turn_loop directly
# through the TurnLoopDeps DI seam with a scripted fake LLM provider keyed by
# call purpose, a fixture-scripted fake retrieve_ranked, and a real TurnContext
# carried across turns) and LIVE (opt-in --api-base — POSTs /query/stream with
# one conversation_id per conversation and parses the typed SSE trace via
# urllib, never curl). Produces per-turn TraceResults and conversation-level
# metrics: action accuracy, terminal accuracy, anchor-doc retention/drift,
# expected-chunk recall, clarify-hint checks, and a clarify-quality placeholder
# hook (judge-scored later; N/A offline).
# Exports: FakeLLMResponse, ScriptedProvider, FakeRetrieveRanked,
#          FakeFetchDocument, EventCollector, TraceResult, TurnEvaluation,
#          ConversationReport, SuiteReport, classify_prompt_purpose,
#          build_turn_deps, run_conversation_offline, run_suite_offline,
#          run_suite_offline_sync, parse_sse, run_conversation_live,
#          run_suite_live, score_clarify_quality, main
# Deps: stdlib (argparse, asyncio, dataclasses, inspect, json, logging, time,
#       urllib.request, uuid, pathlib), src.retrieval.pipeline.turn_loop
#       (schemas; run_turn_loop injected by the caller),
#       src.common.prompts (template landmarks),
#       evals.retrieval.deep_research.harness (ChunkMatcher reuse),
#       config.settings lazily (RAG_TURN_LOOP_*/RAG_TURN_CONTEXT_* tunables)
# @end-summary
"""Multi-turn conversation eval harness for the turn loop (design section 10).

OFFLINE mode (the default; no Weaviate / Temporal / LLM endpoints) drives the
REAL ``run_turn_loop`` orchestrator through its :class:`TurnLoopDeps` seam:

* :class:`ScriptedProvider` — a fake LLM provider that classifies every call
  by *purpose* (controller / judge / hyde / deep_study_read / clarify /
  self_score / draft) and returns the fixture turn's canned response for that
  purpose. Classification is derived at runtime from the canonical prompt
  templates under ``PROMPTS_DIR`` (literal-segment overlap), so it keeps
  working when templates are reworded — no hardcoded phrase matching
  (CLAUDE.md section 0: the class solved is "which template produced this
  rendered prompt", not any specific wording instance).
* :class:`FakeRetrieveRanked` — returns fixture-defined chunk rounds mapped to
  :class:`EvidenceChunk`.
* A real :class:`TurnContext` built from the previous turns' results, so the
  cross-turn context transfer is genuinely exercised — anchor-doc retention is
  measured on what the loop actually carried, not on harness bookkeeping.

LIVE mode (opt-in via ``--api-base`` / ``RAG_EVAL_API_BASE``) threads one
``conversation_id`` through ``POST /query/stream`` and rebuilds the same
per-turn :class:`TraceResult` from the typed SSE event trace (python urllib
only — never curl/wget on this host).

CLI::

    python -m evals.conversation.harness \
        --fixtures evals/conversation/fixtures/golden_conversations.json \
        --output /tmp/conversation_eval_report.json [--api-base http://...]
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from evals.retrieval.deep_research.harness import ChunkMatcher
from src.retrieval.pipeline.turn_loop.schemas import (
    EvidenceChunk,
    TurnBudget,
    TurnContext,
    TurnEvent,
    TurnEventType,
    TurnLoopDeps,
    TurnLoopResult,
)

logger = logging.getLogger(__name__)

# Terminal vocabulary used by fixture expectations (design section 10).
TERMINAL_ANSWER = "answer"
TERMINAL_CLARIFY = "clarify"

# Offline clarify-quality is not judge-scored yet; this sentinel keeps the
# metric slot visible in reports until the live judge hook lands.
CLARIFY_QUALITY_NA = "N/A"


# ---------------------------------------------------------------------------
# Purpose classification (template-derived — no hardcoded phrase matching)
# ---------------------------------------------------------------------------

# Purpose -> the canonical prompt template file(s) that render that call. The
# landmarks below are DERIVED from these files at runtime, so a template
# rewording updates the classifier automatically.
_PURPOSE_TEMPLATES: dict[str, tuple[str, ...]] = {
    "controller": ("turn_controller_decide.md",),
    "deep_study_read": ("turn_deep_study_read.md",),
    "clarify": ("turn_clarify_generate.md",),
    "self_score": ("turn_answer_selfscore.md",),
    "judge": ("agentic_chunk_judge.md", "agentic_chunk_judge_concise.md"),
    "hyde": ("agentic_hyde_generate.md",),
}

# Minimum literal-line length considered a landmark (shorter lines are too
# generic to discriminate between templates).
_LANDMARK_MIN_CHARS = 24

# Minimum fraction of a template's landmarks that must appear in a prompt for
# the template to claim the call.
_CLASSIFY_THRESHOLD = 0.35

_LANDMARK_CACHE: dict[str, list[list[str]]] = {}


def _literal_segments(template: str) -> list[str]:
    """Extract the literal (non-placeholder) landmark lines of a template.

    Splits out ``{{ var }}`` placeholders by partitioning on the delimiter
    pairs (plain string ops — no regex, CLAUDE.md section 0) and keeps the
    stripped literal lines long enough to be discriminative.

    Args:
        template: Raw prompt template text.

    Returns:
        Landmark lines (order preserved, may be empty for tiny templates).
    """
    parts: list[str] = []
    rest = template
    while "{{" in rest:
        head, _, tail = rest.partition("{{")
        parts.append(head)
        _, _, rest = tail.partition("}}")
    parts.append(rest)
    lines: list[str] = []
    for part in parts:
        for raw_line in part.splitlines():
            line = raw_line.strip()
            if len(line) >= _LANDMARK_MIN_CHARS:
                lines.append(line)
    return lines


def _load_landmarks() -> dict[str, list[list[str]]]:
    """Load (and cache) per-purpose landmark line lists from ``PROMPTS_DIR``.

    A missing template file is tolerated (that purpose simply cannot be
    template-matched); the loader is shared with production code
    (``src.common.prompts.load_prompt``) so the classifier reads the same
    bytes the loop renders from.
    """
    if _LANDMARK_CACHE:
        return _LANDMARK_CACHE
    from src.common.prompts import load_prompt

    for purpose, filenames in _PURPOSE_TEMPLATES.items():
        landmark_lists: list[list[str]] = []
        for filename in filenames:
            try:
                landmark_lists.append(_literal_segments(load_prompt(filename)))
            except Exception as exc:  # noqa: BLE001 — absent template is not fatal
                logger.debug("landmark template %s unavailable: %s", filename, exc)
        _LANDMARK_CACHE[purpose] = landmark_lists
    return _LANDMARK_CACHE


def classify_prompt_purpose(prompt_text: str, model_alias: str = "") -> str:
    """Classify one rendered prompt to a loop call purpose.

    Scores every known template by the fraction of its landmark lines present
    in ``prompt_text`` and takes the argmax above :data:`_CLASSIFY_THRESHOLD`.
    Unmatched calls fall back to the alias prior (a call on the configured
    judge alias is a judge call) and finally to ``"draft"`` — the answer draft
    is the one loop call built from the generator's own message builder rather
    than a turn-loop template, so "matched nothing" is exactly the draft class.

    Args:
        prompt_text: Concatenated message contents of the call.
        model_alias: Router alias the call was made with (classification
            prior only, config-driven — never a hardcoded model name).

    Returns:
        One of ``controller`` / ``judge`` / ``hyde`` / ``deep_study_read`` /
        ``clarify`` / ``self_score`` / ``draft``.
    """
    best_purpose = "draft"
    best_score = 0.0
    for purpose, landmark_lists in _load_landmarks().items():
        for landmarks in landmark_lists:
            if not landmarks:
                continue
            hits = sum(1 for line in landmarks if line in prompt_text)
            score = hits / len(landmarks)
            if score > best_score:
                best_purpose, best_score = purpose, score
    if best_score >= _CLASSIFY_THRESHOLD:
        return best_purpose
    if model_alias:
        try:
            from config import settings

            if model_alias == getattr(
                settings, "RAG_TURN_LOOP_JUDGE_MODEL_ALIAS", "judge"
            ):
                return "judge"
        except Exception:  # noqa: BLE001 — settings unavailable → skip the prior
            pass
    return "draft"


# ---------------------------------------------------------------------------
# Fakes (the TurnLoopDeps DI-seam payloads — src/eval smoke style)
# ---------------------------------------------------------------------------


@dataclass
class FakeLLMResponse:
    """Duck-typed stand-in for the platform ``LLMResponse`` (getattr surface)."""

    content: str
    model: str = "scripted-fake"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


def _messages_text(messages: Any) -> str:
    """Flatten an OpenAI-style message list to one text blob for classification."""
    if isinstance(messages, str):
        return messages
    chunks: list[str] = []
    try:
        for message in messages:
            if isinstance(message, dict):
                chunks.append(str(message.get("content") or ""))
            else:
                chunks.append(str(getattr(message, "content", "") or ""))
    except TypeError:
        return str(messages)
    return "\n".join(chunks)


def _expand_judge_payload(verdict: dict) -> dict:
    """Expand a compact fixture judge verdict to satisfy BOTH judge parsers.

    Fixtures script the concise shape (``ranking`` / ``sufficient`` /
    ``confidence`` / ``missing_information``); the verbose judge parser wants
    ``chunks`` + ``pool`` as well. Adding the redundant keys here keeps the
    fixtures compact while staying agnostic to which judge prompt the
    retrieve stage composes.
    """
    payload = dict(verdict)
    ranking = payload.get("ranking") or []
    if "pool" not in payload:
        payload["pool"] = {
            "sufficient": bool(payload.get("sufficient", False)),
            "confidence": float(payload.get("confidence", 0.0) or 0.0),
            "missing_information": payload.get("missing_information", "") or "",
            "covered_aspects": list(payload.get("covered_aspects") or []),
        }
    if "chunks" not in payload:
        payload["chunks"] = [
            {
                "i": index,
                "relevance": float(payload.get("confidence", 0.0) or 0.0),
                "faithfulness": float(payload.get("confidence", 0.0) or 0.0),
                "keep": True,
                "reason": "scripted verdict",
            }
            for index in ranking
        ]
    return payload


class ScriptedProvider:
    """Scripted fake LLM provider: deterministic canned responses per purpose.

    One instance per fixture *turn*. Exposes the provider surface the loop may
    touch (``agenerate`` / ``generate_stream`` / ``agenerate_stream``); every
    call is classified by purpose and answered from the turn's script:

    * ``controller_decisions`` — consumed in order (controller calls).
    * ``judge_verdicts`` — consumed in order; the last one repeats if the loop
      judges more rounds than scripted.
    * ``hyde_responses`` — optional; default empty JSON (HyDE fail-open →
      plain-query search, the production fallback).
    * ``deep_study_reads`` — optional; default "no notes in this window".
    * ``clarify_response`` / ``self_score`` — single objects.
    * ``answer_draft`` — plain text returned for draft calls (the fallback
      class) and streamed by the stream surfaces.

    Unscripted calls (a purpose asked for more than the script provides) are
    recorded in :attr:`unscripted` and answered with a safe default so the
    loop always terminates; tests surface them for debugging.
    """

    def __init__(self, script: dict) -> None:
        """Bind the provider to one fixture turn's script dict."""
        script = script or {}
        self._controller: list[dict] = list(script.get("controller_decisions") or [])
        self._judges: list[dict] = list(script.get("judge_verdicts") or [])
        self._last_judge: Optional[dict] = None
        self._hyde: list[dict] = list(script.get("hyde_responses") or [])
        self._reads: list[dict] = list(script.get("deep_study_reads") or [])
        self._clarify: Optional[dict] = script.get("clarify_response")
        self._self_score: Optional[dict] = script.get("self_score")
        self._draft: str = str(script.get("answer_draft") or "")
        self.calls: list[dict] = []
        self.unscripted: list[str] = []

    # -- scripted response selection ------------------------------------

    def _respond(self, purpose: str) -> str:
        """Return the canned content string for one classified call."""
        if purpose == "controller":
            if self._controller:
                return json.dumps(self._controller.pop(0))
            self.unscripted.append("controller")
            # Fail-safe terminal so an over-asking loop still ends the turn.
            return json.dumps(
                {
                    "action": "ANSWER",
                    "reason": "scripted decisions exhausted — fail-safe terminal",
                    "confidence": 0.1,
                    "args": {},
                }
            )
        if purpose == "judge":
            if self._judges:
                self._last_judge = self._judges.pop(0)
            elif self._last_judge is None:
                self.unscripted.append("judge")
                self._last_judge = {
                    "ranking": [],
                    "sufficient": False,
                    "confidence": 0.0,
                    "missing_information": "no scripted judge verdict",
                }
            return json.dumps(_expand_judge_payload(self._last_judge))
        if purpose == "hyde":
            if self._hyde:
                return json.dumps(self._hyde.pop(0))
            # Unscripted HyDE is legitimate: empty output triggers the
            # production fail-open (plain search on the query text).
            return json.dumps({})
        if purpose == "deep_study_read":
            if self._reads:
                return json.dumps(self._reads.pop(0))
            self.unscripted.append("deep_study_read")
            return json.dumps(
                {"notes": "", "answer_found": False, "next_window_hint": ""}
            )
        if purpose == "clarify":
            if self._clarify is not None:
                return json.dumps(self._clarify)
            self.unscripted.append("clarify")
            return json.dumps(
                {"question": "", "hints": [], "scoping_questions": []}
            )
        if purpose == "self_score":
            if self._self_score is not None:
                return json.dumps(self._self_score)
            self.unscripted.append("self_score")
            return json.dumps({"self_score": 0.0, "unsupported_claims": []})
        # draft (and anything unmatched — see classify_prompt_purpose)
        if not self._draft:
            self.unscripted.append("draft")
        return self._draft

    def _record(self, surface: str, messages: Any, model_alias: str) -> str:
        """Classify, log and answer one provider call; returns the content."""
        purpose = classify_prompt_purpose(_messages_text(messages), model_alias)
        content = self._respond(purpose)
        self.calls.append(
            {"surface": surface, "purpose": purpose, "model_alias": model_alias}
        )
        return content

    # -- provider surface -------------------------------------------------

    async def agenerate(
        self, messages: Any, *, model_alias: str = "default", **_: Any
    ) -> FakeLLMResponse:
        """Async completion — the surface ``TurnEventEmitter.charged_call`` uses."""
        content = self._record("agenerate", messages, model_alias)
        return FakeLLMResponse(
            content=content,
            prompt_tokens=len(_messages_text(messages)) // 4,
            completion_tokens=len(content) // 4,
        )

    def generate_stream(
        self,
        messages: Any,
        *,
        model_alias: str = "default",
        include_reasoning: bool = False,
        **_: Any,
    ) -> Any:
        """Sync streaming surface (mirrors ``LLMProvider.generate_stream``)."""
        content = self._record("generate_stream", messages, model_alias)
        for piece in self._split_stream(content):
            yield ("content", piece) if include_reasoning else piece

    async def agenerate_stream(
        self, messages: Any, *, model_alias: str = "default", **_: Any
    ) -> Any:
        """Async streaming surface (mirrors ``LLMProvider.agenerate_stream``)."""
        content = self._record("agenerate_stream", messages, model_alias)
        for piece in self._split_stream(content):
            yield piece

    @staticmethod
    def _split_stream(content: str, pieces: int = 3) -> list[str]:
        """Split content into a few deterministic delta chunks for streaming."""
        if not content:
            return []
        step = max(1, len(content) // pieces)
        return [content[i : i + step] for i in range(0, len(content), step)]


class FakeRetrieveRanked:
    """Fixture-scripted ``retrieve_ranked`` seam: one chunk list per call.

    ``rounds`` is a list of chunk-dict lists — call N returns round N mapped
    to :class:`EvidenceChunk`. Exhausted rounds return ``[]`` ("no new
    evidence"), which is the honest degradation for an over-retrieving loop.
    """

    def __init__(self, rounds: list[list[dict]]) -> None:
        """Bind the fake to one fixture turn's ``retrieve_chunks`` rounds."""
        self._rounds: list[list[dict]] = [list(r) for r in (rounds or [])]
        self.calls: list[dict] = []

    async def __call__(
        self,
        query_text: str,
        hyde_text: Optional[str] = None,
        top_k: int = 0,
        *args: Any,
        **kwargs: Any,
    ) -> list[EvidenceChunk]:
        """Return the next scripted round (capped at ``top_k`` when > 0)."""
        round_index = len(self.calls)
        self.calls.append(
            {"query_text": query_text, "hyde_text": hyde_text, "top_k": top_k}
        )
        if not self._rounds:
            return []
        chunk_dicts = self._rounds.pop(0)
        chunks = [
            _chunk_from_fixture(entry, round_added=round_index)
            for entry in chunk_dicts
        ]
        if top_k and top_k > 0:
            return chunks[:top_k]
        return chunks


def _chunk_from_fixture(entry: dict, *, round_added: int = 0) -> EvidenceChunk:
    """Map one fixture chunk dict onto the :class:`EvidenceChunk` contract."""
    return EvidenceChunk(
        chunk_id=str(entry.get("chunk_id") or uuid.uuid4()),
        document_id=str(entry.get("document_id") or ""),
        source_key=str(entry.get("source_key") or ""),
        source=str(entry.get("source") or ""),
        heading=str(entry.get("heading") or ""),
        text=str(entry.get("text") or ""),
        score=float(entry.get("score", 0.5) or 0.0),
        refactored_char_start=int(entry.get("refactored_char_start", -1)),
        refactored_char_end=int(entry.get("refactored_char_end", -1)),
        provenance=str(
            entry.get("provenance") or EvidenceChunk.PROVENANCE_RETRIEVE
        ),
        round_added=round_added,
    )


class FakeFetchDocument:
    """Fixture-scripted ``fetch_document`` seam for DEEP_STUDY.

    ``documents`` maps a key (``document_id`` or ``source_key``) to
    ``{"title", "text", "document_id", "source_key"}``. Returns a duck-typed
    stored-document object exposing the commonly probed attributes, or
    ``None`` (the loop's documented fall-back path) when unresolvable.
    """

    def __init__(self, documents: Optional[dict[str, dict]] = None) -> None:
        """Bind the fake to one fixture turn's ``documents`` mapping."""
        self._documents = dict(documents or {})
        self.calls: list[dict] = []

    async def __call__(
        self,
        document_id: Optional[str] = None,
        source_key: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Optional[Any]:
        """Resolve by document_id first, then source_key; None when unknown."""
        self.calls.append({"document_id": document_id, "source_key": source_key})
        entry = None
        for key in (document_id, source_key):
            if key and key in self._documents:
                entry = self._documents[key]
                break
        if entry is None:
            return None
        text = str(entry.get("text") or "")
        return _StoredDocumentDouble(
            document_id=str(entry.get("document_id") or document_id or ""),
            source_key=str(entry.get("source_key") or source_key or ""),
            title=str(entry.get("title") or ""),
            text=text,
        )


@dataclass
class _StoredDocumentDouble:
    """Duck-typed stored document: text/markdown/content all carry the body."""

    document_id: str
    source_key: str
    title: str
    text: str

    @property
    def markdown(self) -> str:
        """Alias for consumers that read ``.markdown``."""
        return self.text

    @property
    def content(self) -> str:
        """Alias for consumers that read ``.content``."""
        return self.text


class EventCollector:
    """Async event sink capturing every emitted :class:`TurnEvent`."""

    def __init__(self) -> None:
        self.events: list[TurnEvent] = []

    async def __call__(self, event: TurnEvent) -> None:
        """Append one emitted event (the ``TurnLoopDeps.emit`` contract)."""
        self.events.append(event)


def build_turn_deps(turn_script: dict) -> tuple[TurnLoopDeps, ScriptedProvider, EventCollector]:
    """Assemble a fully-faked :class:`TurnLoopDeps` for one fixture turn.

    Returns the deps plus the provider and collector so the caller can read
    back the call log / unscripted markers / streamed events.
    """
    provider = ScriptedProvider(turn_script)
    collector = EventCollector()
    deps = TurnLoopDeps(
        retrieve_ranked=FakeRetrieveRanked(turn_script.get("retrieve_chunks") or []),
        fetch_document=FakeFetchDocument(turn_script.get("documents") or {}),
        llm_provider=provider,
        emit=collector,
    )
    return deps, provider, collector


# ---------------------------------------------------------------------------
# Adaptive run_turn_loop invocation
# ---------------------------------------------------------------------------

# Parameter-name aliases per harness role. Annotation matching (the contract
# type names) takes precedence; these cover unannotated/renamed parameters.
_ROLE_PARAM_NAMES: dict[str, frozenset[str]] = {
    "query": frozenset({"query", "user_query", "question", "query_text", "user_question"}),
    "context": frozenset({"context", "turn_context", "ctx", "conversation_context"}),
    "deps": frozenset({"deps", "dependencies", "turn_deps"}),
    "budget": frozenset({"budget", "turn_budget", "budgets"}),
}

_ROLE_ANNOTATION_TOKENS: dict[str, str] = {
    "context": "TurnContext",
    "deps": "TurnLoopDeps",
    "budget": "TurnBudget",
}


def bind_run_turn_loop_kwargs(fn: Callable[..., Any], available: dict[str, Any]) -> dict[str, Any]:
    """Map harness role values onto ``fn``'s actual parameter names.

    The orchestrator's exact signature is owned by another track; binding by
    contract annotation first (``TurnContext`` / ``TurnLoopDeps`` /
    ``TurnBudget``) and by conventional names second keeps the harness correct
    across naming choices without patching anything.

    Args:
        fn: The ``run_turn_loop`` callable.
        available: Role name -> value for ``query`` / ``context`` / ``deps`` /
            ``budget``.

    Returns:
        Keyword arguments ready for ``fn(**kwargs)``.

    Raises:
        RuntimeError: When a required parameter of ``fn`` cannot be filled
            from the available roles (a genuine contract mismatch to surface,
            not to paper over).
    """
    signature = inspect.signature(fn)
    kwargs: dict[str, Any] = {}
    unassigned = dict(available)
    for name, param in signature.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        role = None
        annotation = str(param.annotation) if param.annotation is not inspect.Parameter.empty else ""
        for candidate, token in _ROLE_ANNOTATION_TOKENS.items():
            if candidate in unassigned and token in annotation:
                role = candidate
                break
        if role is None:
            for candidate, names in _ROLE_PARAM_NAMES.items():
                if candidate in unassigned and name in names:
                    role = candidate
                    break
        if role is not None:
            kwargs[name] = unassigned.pop(role)
        elif param.default is inspect.Parameter.empty:
            raise RuntimeError(
                f"run_turn_loop parameter {name!r} cannot be filled by the "
                f"harness (available roles: {sorted(unassigned)}); update "
                "bind_run_turn_loop_kwargs role maps."
            )
    return kwargs


async def invoke_run_turn_loop(
    run_turn_loop: Callable[..., Any],
    *,
    query: str,
    context: TurnContext,
    deps: TurnLoopDeps,
    budget: TurnBudget,
) -> TurnLoopResult:
    """Invoke the real orchestrator with adaptively-bound keyword arguments."""
    kwargs = bind_run_turn_loop_kwargs(
        run_turn_loop,
        {"query": query, "context": context, "deps": deps, "budget": budget},
    )
    result = run_turn_loop(**kwargs)
    if inspect.isawaitable(result):
        result = await result
    return result


# ---------------------------------------------------------------------------
# Trace + report contracts
# ---------------------------------------------------------------------------


@dataclass
class TraceResult:
    """Everything observed for one conversation turn (both drive modes)."""

    turn_index: int
    query: str
    actions_taken: list[str] = field(default_factory=list)
    terminal: str = ""
    chunks_seen: list[dict] = field(default_factory=list)
    anchor_docs_retained: Optional[bool] = None
    clarification: Optional[dict] = None
    gate_scores: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    answer: str = ""
    confidence: float = 0.0
    stop_reason: str = ""
    llm_calls: int = 0
    unscripted_calls: list[str] = field(default_factory=list)
    # Snapshot of the TurnContext the turn STARTED with — proves the transfer.
    context_snapshot: dict = field(default_factory=dict)
    # Live-only richness (None offline; tests use skip-if-unpopulated).
    clarify_quality: Optional[float] = None
    error: Optional[str] = None

    def document_ids(self) -> set[str]:
        """Distinct non-empty document identities seen this turn."""
        ids = set()
        for chunk in self.chunks_seen:
            doc = chunk.get("document_id") or chunk.get("doc") or chunk.get("source")
            if doc:
                ids.add(str(doc))
        return ids


@dataclass
class TurnEvaluation:
    """Per-turn expectation checks (None = no expectation declared)."""

    turn_index: int
    action_ok: Optional[bool] = None
    terminal_ok: Optional[bool] = None
    anchor_ok: Optional[bool] = None
    chunk_recall: Optional[float] = None
    hints_ok: Optional[bool] = None
    details: dict = field(default_factory=dict)


@dataclass
class ConversationReport:
    """One conversation's turn traces, per-turn evaluations and metrics."""

    id: str
    turns: list[TraceResult] = field(default_factory=list)
    evaluations: list[TurnEvaluation] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


@dataclass
class SuiteReport:
    """Whole-fixture report: conversation reports plus aggregate metrics."""

    domain: str
    mode: str
    conversation_count: int
    conversations: list[ConversationReport] = field(default_factory=list)
    aggregate: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def score_clarify_quality(
    trace: TraceResult,
    judge: Optional[Callable[[TraceResult], Awaitable[float]]] = None,
) -> Optional[float]:
    """Clarify-quality hook — judge-scored later; ``None`` (N/A) offline.

    The eventual live implementation passes a judge callable that scores the
    clarification question/hints against the turn's ambiguity; offline runs
    keep the metric slot visible without fabricating a number.

    Args:
        trace: The turn to score (must carry a clarification).
        judge: Optional async scorer; when ``None`` the metric is N/A.

    Returns:
        ``None`` when unscored (offline), else the judge's [0, 1] score.
    """
    if judge is None or trace.clarification is None:
        return None
    return asyncio.get_event_loop().run_until_complete(judge(trace))  # pragma: no cover


def _evaluate_anchor(
    expect: dict, trace: TraceResult, previous_doc_ids: set[str]
) -> tuple[Optional[bool], bool]:
    """Evaluate the anchor-doc expectation for one turn.

    Returns ``(anchor_ok, retained)``. Anchors are the expectation's explicit
    ``anchor_docs`` list when given (ALL must reappear), else the previous
    turn's document set (ANY overlap counts as retention — drift is the total
    loss of the prior grounding, design section 10).
    """
    expected_flag = expect.get("anchor_docs_retained")
    if expected_flag is None:
        return None, False
    current = trace.document_ids()
    explicit = [str(d) for d in (expect.get("anchor_docs") or [])]
    if explicit:
        retained = all(doc in current for doc in explicit)
    else:
        retained = bool(previous_doc_ids & current)
    trace.anchor_docs_retained = retained
    return retained == bool(expected_flag), retained


def evaluate_turn(
    trace: TraceResult, expect: dict, previous_doc_ids: set[str]
) -> TurnEvaluation:
    """Check one turn's trace against its fixture expectations."""
    evaluation = TurnEvaluation(turn_index=trace.turn_index)

    allowed = expect.get("actions_allowed")
    if allowed is not None:
        allowed_set = {str(a).upper() for a in allowed}
        taken = {str(a).upper() for a in trace.actions_taken}
        evaluation.action_ok = bool(taken) and taken.issubset(allowed_set)
        evaluation.details["actions_taken"] = sorted(taken)
        evaluation.details["actions_allowed"] = sorted(allowed_set)

    expected_terminal = expect.get("terminal")
    if expected_terminal is not None:
        evaluation.terminal_ok = trace.terminal == str(expected_terminal)
        evaluation.details["terminal"] = trace.terminal

    anchor_ok, retained = _evaluate_anchor(expect, trace, previous_doc_ids)
    evaluation.anchor_ok = anchor_ok
    if anchor_ok is not None:
        evaluation.details["anchor_retained"] = retained

    expected_chunks = expect.get("chunks") or []
    if expected_chunks:
        matchers = [ChunkMatcher(**entry) for entry in expected_chunks]
        hits = sum(
            1
            for matcher in matchers
            if any(matcher.matches(chunk) for chunk in trace.chunks_seen)
        )
        evaluation.chunk_recall = hits / len(matchers)

    min_hints = expect.get("min_hints")
    if min_hints is not None:
        hints = (trace.clarification or {}).get("hints") or []
        evaluation.hints_ok = len(hints) >= int(min_hints)
        evaluation.details["hints"] = list(hints)

    return evaluation


def _conversation_metrics(report: ConversationReport) -> dict:
    """Roll per-turn evaluations up into one conversation's metric dict."""

    def _rate(values: list[bool]) -> Optional[float]:
        return (sum(1 for v in values if v) / len(values)) if values else None

    action_checks = [e.action_ok for e in report.evaluations if e.action_ok is not None]
    terminal_checks = [e.terminal_ok for e in report.evaluations if e.terminal_ok is not None]
    anchor_checks = [e.anchor_ok for e in report.evaluations if e.anchor_ok is not None]
    hint_checks = [e.hints_ok for e in report.evaluations if e.hints_ok is not None]
    recalls = [e.chunk_recall for e in report.evaluations if e.chunk_recall is not None]
    retention_rate = _rate(anchor_checks)
    return {
        "action_accuracy": _rate(action_checks),
        "terminal_accuracy": _rate(terminal_checks),
        "anchor_retention_rate": retention_rate,
        "anchor_drift_rate": (
            None if retention_rate is None else round(1.0 - retention_rate, 6)
        ),
        "clarify_hint_accuracy": _rate(hint_checks),
        "avg_chunk_recall": (sum(recalls) / len(recalls)) if recalls else None,
        "clarify_quality": CLARIFY_QUALITY_NA,
        "unscripted_call_count": sum(len(t.unscripted_calls) for t in report.turns),
    }


def _aggregate(reports: list[ConversationReport]) -> dict:
    """Aggregate conversation metrics across the suite (None-safe means)."""

    def _mean(key: str) -> Optional[float]:
        values = [
            r.metrics.get(key)
            for r in reports
            if isinstance(r.metrics.get(key), (int, float))
        ]
        return (sum(values) / len(values)) if values else None

    return {
        "action_accuracy": _mean("action_accuracy"),
        "terminal_accuracy": _mean("terminal_accuracy"),
        "anchor_retention_rate": _mean("anchor_retention_rate"),
        "anchor_drift_rate": _mean("anchor_drift_rate"),
        "clarify_hint_accuracy": _mean("clarify_hint_accuracy"),
        "avg_chunk_recall": _mean("avg_chunk_recall"),
        "clarify_quality": CLARIFY_QUALITY_NA,
        "conversation_count": len(reports),
        "turn_count": sum(len(r.turns) for r in reports),
        "error_count": sum(
            1 for r in reports for t in r.turns if t.error is not None
        ),
    }


# ---------------------------------------------------------------------------
# OFFLINE drive mode
# ---------------------------------------------------------------------------


def _budget_from_fixture(fixture: dict, conversation: dict) -> TurnBudget:
    """Build the turn budget: settings defaults + fixture overrides.

    Tunables stay the ``RAG_TURN_LOOP_*`` settings keys
    (:meth:`TurnBudget.from_settings`); a fixture may pin individual fields
    via a ``budget`` object (conversation-level overrides fixture-level) so
    goldens remain deterministic under env drift.
    """
    import dataclasses

    budget = TurnBudget.from_settings()
    overrides: dict[str, Any] = {}
    for scope in (fixture.get("budget") or {}, conversation.get("budget") or {}):
        overrides.update(scope)
    valid = {f.name for f in dataclasses.fields(TurnBudget)}
    overrides = {k: v for k, v in overrides.items() if k in valid}
    if overrides:
        budget = dataclasses.replace(budget, **overrides)
    return budget


def _events_to_dicts(events: list[TurnEvent]) -> list[dict]:
    """Serialize typed events for the report/trace."""
    return [
        {"type": e.type, "payload": e.payload, "ts_ms": e.ts_ms} for e in events
    ]


def _terminal_from_result(result: TurnLoopResult) -> str:
    """Map the loop's terminal action onto the fixture vocabulary."""
    action = getattr(result, "action", "") or ""
    if action == TurnLoopResult.ACTION_ANSWERED:
        return TERMINAL_ANSWER
    if action == TurnLoopResult.ACTION_ASK_USER:
        return TERMINAL_CLARIFY
    return action


def _chunk_to_dict(chunk: Any) -> dict:
    """Flatten an EvidenceChunk (or duck-typed equivalent) for the report."""
    return {
        "chunk_id": getattr(chunk, "chunk_id", ""),
        "document_id": getattr(chunk, "document_id", ""),
        "source_key": getattr(chunk, "source_key", ""),
        "source": getattr(chunk, "source", ""),
        "heading": getattr(chunk, "heading", ""),
        "text": getattr(chunk, "text", ""),
        "score": getattr(chunk, "score", 0.0),
        "provenance": getattr(chunk, "provenance", ""),
    }


def _context_snapshot(context: TurnContext) -> dict:
    """Record what the turn started with (context-transfer evidence)."""
    return {
        "recent_turns": len(context.recent_turns),
        "chunk_refs": len(context.chunk_refs),
        "chunk_ref_document_ids": sorted(
            {
                str(ref.get("document_id"))
                for ref in context.chunk_refs
                if isinstance(ref, dict) and ref.get("document_id")
            }
        ),
        "docs_studied": len(context.docs_studied),
        "pending_clarification": context.pending_clarification is not None,
    }


def _advance_context(
    context: TurnContext, query: str, result: TurnLoopResult, trace: TraceResult
) -> None:
    """Fold one turn's result into the TurnContext handed to the next turn.

    Mirrors the design section 7 memory contract: prose Q&A, preview-capped
    chunk refs (``RAG_TURN_CONTEXT_PREVIEW_CHARS`` /
    ``RAG_TURN_CONTEXT_MAX_CHUNK_REFS``), docs-studied records from the
    deep_study events, and the pending clarification for ask_user terminals.
    """
    from config import settings

    preview_chars = int(getattr(settings, "RAG_TURN_CONTEXT_PREVIEW_CHARS", 320))
    max_refs = int(getattr(settings, "RAG_TURN_CONTEXT_MAX_CHUNK_REFS", 24))

    answer_text = getattr(result, "answer", "") or ""
    clarification = getattr(result, "clarification", None)
    if clarification is not None and not answer_text:
        answer_text = getattr(clarification, "question", "") or ""
    context.recent_turns.append({"query": query, "answer": answer_text})

    known_ids = {
        str(ref.get("chunk_id"))
        for ref in context.chunk_refs
        if isinstance(ref, dict)
    }
    for chunk in getattr(result, "pool", None) or []:
        chunk_id = str(getattr(chunk, "chunk_id", ""))
        if not chunk_id or chunk_id in known_ids:
            continue
        known_ids.add(chunk_id)
        context.chunk_refs.append(
            {
                "chunk_id": chunk_id,
                "document_id": getattr(chunk, "document_id", ""),
                "source_key": getattr(chunk, "source_key", ""),
                "heading": getattr(chunk, "heading", ""),
                "score": getattr(chunk, "score", 0.0),
                "preview": (getattr(chunk, "text", "") or "")[:preview_chars],
            }
        )
    if len(context.chunk_refs) > max_refs:
        context.chunk_refs[:] = context.chunk_refs[-max_refs:]

    studied: dict[str, dict] = {
        str(doc.get("document_id")): dict(doc)
        for doc in context.docs_studied
        if isinstance(doc, dict) and doc.get("document_id")
    }
    for event in trace.events:
        if event.get("type") != TurnEventType.DEEP_STUDY:
            continue
        payload = event.get("payload") or {}
        document_id = str(payload.get("document_id") or "")
        if not document_id:
            continue
        record = studied.setdefault(
            document_id,
            {
                "document_id": document_id,
                "windows_read": 0,
                "sections": [],
                "conclusion": "",
                "ts": event.get("ts_ms"),
            },
        )
        record["windows_read"] = int(record.get("windows_read") or 0) + 1
        record["conclusion"] = str(payload.get("notes_preview") or record["conclusion"])
        record["ts"] = event.get("ts_ms")
    context.docs_studied[:] = list(studied.values())

    if trace.terminal == TERMINAL_CLARIFY and clarification is not None:
        context.pending_clarification = {
            "question": getattr(clarification, "question", "") or "",
            "hints": list(getattr(clarification, "hints", None) or []),
        }
    else:
        context.pending_clarification = None


async def run_conversation_offline(
    run_turn_loop: Callable[..., Any],
    conversation: dict,
    *,
    fixture: Optional[dict] = None,
) -> ConversationReport:
    """Drive one fixture conversation through the REAL orchestrator, offline.

    Per turn: fresh scripted deps, the carried :class:`TurnContext`, one
    ``run_turn_loop`` invocation, trace extraction, expectation evaluation and
    context advancement. Errors are captured per turn (the report shows them)
    rather than aborting the suite.
    """
    fixture = fixture or {}
    conversation_id = str(conversation.get("id") or f"conv-{uuid.uuid4().hex[:8]}")
    report = ConversationReport(id=conversation_id)
    context = TurnContext(conversation_id=conversation_id)
    budget = _budget_from_fixture(fixture, conversation)
    previous_doc_ids: set[str] = set()

    for turn_index, turn in enumerate(conversation.get("turns") or []):
        query = str(turn.get("query") or "")
        script = turn.get("script") or {}
        expect = turn.get("expect") or {}
        deps, provider, collector = build_turn_deps(script)
        trace = TraceResult(
            turn_index=turn_index,
            query=query,
            context_snapshot=_context_snapshot(context),
        )
        try:
            result = await invoke_run_turn_loop(
                run_turn_loop,
                query=query,
                context=context,
                deps=deps,
                budget=budget,
            )
        except Exception as exc:  # noqa: BLE001 — captured per turn, suite continues
            logger.exception(
                "turn %d of conversation %s failed", turn_index, conversation_id
            )
            trace.error = f"{type(exc).__name__}: {exc}"
            trace.unscripted_calls = list(provider.unscripted)
            report.turns.append(trace)
            report.evaluations.append(
                evaluate_turn(trace, expect, previous_doc_ids)
            )
            break

        loop_events = list(getattr(result, "trace", None) or []) or collector.events
        trace.events = _events_to_dicts(loop_events)
        trace.actions_taken = [
            str((e.get("payload") or {}).get("action") or "")
            for e in trace.events
            if e.get("type") == TurnEventType.TURN_ACTION
        ]
        trace.actions_taken = [a for a in trace.actions_taken if a]
        trace.gate_scores = [
            dict(e.get("payload") or {})
            for e in trace.events
            if e.get("type") == TurnEventType.GATE
        ]
        trace.terminal = _terminal_from_result(result)
        trace.chunks_seen = [
            _chunk_to_dict(chunk) for chunk in (getattr(result, "pool", None) or [])
        ]
        clarification = getattr(result, "clarification", None)
        if clarification is not None:
            trace.clarification = {
                "question": getattr(clarification, "question", "") or "",
                "hints": list(getattr(clarification, "hints", None) or []),
                "scoping_questions": list(
                    getattr(clarification, "scoping_questions", None) or []
                ),
            }
        trace.answer = getattr(result, "answer", "") or ""
        trace.confidence = float(getattr(result, "confidence", 0.0) or 0.0)
        trace.stop_reason = getattr(result, "stop_reason", "") or ""
        trace.llm_calls = int(getattr(result, "llm_calls", 0) or 0)
        trace.unscripted_calls = list(provider.unscripted)
        trace.clarify_quality = score_clarify_quality(trace)

        report.turns.append(trace)
        report.evaluations.append(evaluate_turn(trace, expect, previous_doc_ids))
        previous_doc_ids = trace.document_ids() or previous_doc_ids
        _advance_context(context, query, result, trace)

    report.metrics = _conversation_metrics(report)
    return report


async def run_suite_offline(
    run_turn_loop: Callable[..., Any], fixture: dict
) -> SuiteReport:
    """Run every fixture conversation offline; returns the suite report."""
    reports = [
        await run_conversation_offline(run_turn_loop, conversation, fixture=fixture)
        for conversation in fixture.get("conversations") or []
    ]
    return SuiteReport(
        domain=str(fixture.get("domain") or "unknown"),
        mode="offline",
        conversation_count=len(reports),
        conversations=reports,
        aggregate=_aggregate(reports),
    )


def run_suite_offline_sync(
    run_turn_loop: Callable[..., Any], fixture: dict
) -> SuiteReport:
    """Synchronous wrapper for pytest session fixtures / the CLI."""
    return asyncio.run(run_suite_offline(run_turn_loop, fixture))


# ---------------------------------------------------------------------------
# LIVE drive mode (urllib only — never curl/wget on this host)
# ---------------------------------------------------------------------------


def parse_sse(text: str) -> list[tuple[str, Optional[dict]]]:
    """Parse an SSE body into ``(event, data-dict)`` frames.

    Vendored from ``tests/server/test_query_endpoints.py::_parse_sse`` so the
    eval package has no import edge into ``tests/``.
    """
    frames: list[tuple[str, Optional[dict]]] = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        event = None
        data = None
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line[len("event: "):]
            elif line.startswith("data: "):
                try:
                    data = json.loads(line[len("data: "):])
                except json.JSONDecodeError:
                    data = {"raw": line[len("data: "):]}
        if event is not None:
            frames.append((event, data))
    return frames


def _post_stream(api_base: str, payload: dict, *, timeout_s: float) -> str:
    """POST ``payload`` to ``/query/stream`` and return the full SSE body.

    Uses ``urllib.request`` in-process (EDR on these hosts kills shell
    curl/wget). An optional bearer token is read from ``RAG_EVAL_API_TOKEN``.
    """
    import os
    import urllib.request

    url = api_base.rstrip("/") + "/query/stream"
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    token = os.environ.get("RAG_EVAL_API_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return response.read().decode("utf-8", errors="replace")


def _trace_from_frames(
    turn_index: int, query: str, frames: list[tuple[str, Optional[dict]]]
) -> TraceResult:
    """Rebuild a :class:`TraceResult` from live SSE frames (best effort).

    The typed loop events share names with :class:`TurnEventType`; frames
    outside that vocabulary (token / done / error envelopes) contribute the
    terminal signal only.
    """
    trace = TraceResult(turn_index=turn_index, query=query)
    saw_answer_output = False
    for event, data in frames:
        payload = data if isinstance(data, dict) else {}
        if event in TurnEventType.ALL:
            trace.events.append(
                {"type": event, "payload": payload, "ts_ms": int(time.time() * 1000)}
            )
        if event == TurnEventType.TURN_ACTION and payload.get("action"):
            trace.actions_taken.append(str(payload["action"]))
        elif event == TurnEventType.GATE:
            trace.gate_scores.append(dict(payload))
        elif event == TurnEventType.CLARIFY:
            trace.clarification = {
                "question": payload.get("question", ""),
                "hints": list(payload.get("hints") or []),
                "scoping_questions": list(payload.get("scoping_questions") or []),
            }
        elif event == TurnEventType.RETRIEVE_RESULT:
            for top in payload.get("top") or []:
                if isinstance(top, dict):
                    trace.chunks_seen.append(
                        {
                            "document_id": top.get("doc", ""),
                            "source": top.get("doc", ""),
                            "heading": top.get("heading", ""),
                            "text": "",
                            "score": top.get("score", 0.0),
                        }
                    )
        elif event == "token":
            saw_answer_output = True
            # The query routes emit token frames as {"token": <delta>}
            # (loop replay and single-shot streaming alike); "text"/"delta"
            # are tolerated for alternative emitters.
            trace.answer += str(
                payload.get("token") or payload.get("text") or payload.get("delta") or ""
            )
        elif event == "error":
            trace.error = json.dumps(payload)
    if trace.clarification is not None:
        trace.terminal = TERMINAL_CLARIFY
    elif saw_answer_output or trace.answer:
        trace.terminal = TERMINAL_ANSWER
    return trace


def run_conversation_live(
    api_base: str,
    conversation: dict,
    *,
    timeout_s: float = 180.0,
) -> ConversationReport:
    """Drive one conversation against a LIVE API via ``/query/stream``.

    Threads one ``conversation_id`` through every turn with ``turn_loop`` on;
    the server owns memory/context transfer, so anchor retention here
    measures the real end-to-end behavior.
    """
    conversation_id = (
        f"eval-{conversation.get('id', 'conv')}-{uuid.uuid4().hex[:8]}"
    )
    report = ConversationReport(id=str(conversation.get("id") or conversation_id))
    previous_doc_ids: set[str] = set()
    for turn_index, turn in enumerate(conversation.get("turns") or []):
        query = str(turn.get("query") or "")
        expect = turn.get("expect") or {}
        try:
            body = _post_stream(
                api_base,
                {
                    "query": query,
                    "conversation_id": conversation_id,
                    "turn_loop": True,
                },
                timeout_s=timeout_s,
            )
            trace = _trace_from_frames(turn_index, query, parse_sse(body))
        except Exception as exc:  # noqa: BLE001 — captured per turn
            logger.exception(
                "live turn %d of %s failed", turn_index, conversation_id
            )
            trace = TraceResult(
                turn_index=turn_index,
                query=query,
                error=f"{type(exc).__name__}: {exc}",
            )
        report.turns.append(trace)
        report.evaluations.append(evaluate_turn(trace, expect, previous_doc_ids))
        previous_doc_ids = trace.document_ids() or previous_doc_ids
    report.metrics = _conversation_metrics(report)
    return report


def run_suite_live(
    api_base: str, fixture: dict, *, timeout_s: float = 180.0
) -> SuiteReport:
    """Run every fixture conversation against a live API."""
    reports = [
        run_conversation_live(api_base, conversation, timeout_s=timeout_s)
        for conversation in fixture.get("conversations") or []
    ]
    return SuiteReport(
        domain=str(fixture.get("domain") or "unknown"),
        mode="live",
        conversation_count=len(reports),
        conversations=reports,
        aggregate=_aggregate(reports),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry: offline by default, live when ``--api-base`` is given."""
    import os

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--api-base",
        default=os.environ.get("RAG_EVAL_API_BASE") or None,
        help="Live mode: API base URL (or RAG_EVAL_API_BASE). Omit for offline.",
    )
    parser.add_argument("--live-timeout-s", type=float, default=180.0)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    fixture = json.loads(args.fixtures.read_text())

    if args.api_base:
        report = run_suite_live(
            args.api_base, fixture, timeout_s=args.live_timeout_s
        )
    else:
        from src.retrieval.pipeline.turn_loop import run_turn_loop

        report = run_suite_offline_sync(run_turn_loop, fixture)

    args.output.write_text(json.dumps(asdict(report), indent=2, default=str))
    print(f"Report written to {args.output}")
    print(json.dumps(report.aggregate, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
