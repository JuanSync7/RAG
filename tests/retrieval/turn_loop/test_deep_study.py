# @summary
# Tests for the turn-loop DEEP_STUDY stage: the refactored_char_start == -1
# no-anchor sentinel guard (start at window 0), anchored start-window
# selection from the best pool chunk, the overlapping window walk with the
# max_windows cap, answer_found early stop, resume without re-reading windows,
# notes feeding the evidence pool as deep_study-provenance chunks, document
# miss / empty-ref degradation, the deep-study doc budget, and the
# legal-target-namespace enforcement (unvetted model-authored refs are
# skipped, partially-vetted refs are nulled before the fetch).
# @end-summary
"""Tests for ``turn_loop/deep_study.py``."""

from __future__ import annotations

from src.retrieval.pipeline.turn_loop.deep_study import run_deep_study
from src.retrieval.pipeline.turn_loop.events import TurnEventEmitter
from src.retrieval.pipeline.turn_loop.schemas import (
    DeepStudyArgs,
    EvidenceChunk,
    StudiedDoc,
    TurnContext,
    TurnEventType,
    TurnState,
)

from tests.retrieval.turn_loop.conftest import (
    FakeDoc,
    FakeProvider,
    deep_read_json,
    events_of,
    make_budget,
    make_chunk,
    make_deps,
)

# Budget defaults: window_chars=100, overlap=20 -> stride 80.
# 250-char content -> windows starting at 0, 80, 160 -> 3 windows.
_CONTENT = "".join(chr(ord("a") + (i % 26)) for i in range(250))


def _ctx(*refs: dict) -> TurnContext:
    """TurnContext granting the given chunk_refs as legal DEEP_STUDY targets.

    Defaults to the ``doc-1`` / ``docs/d1.md`` pair every study in this file
    targets, so each test's namespace grant is explicit and local.
    """
    return TurnContext(
        conversation_id="conv-t",
        chunk_refs=list(refs)
        or [{"document_id": "doc-1", "source_key": "docs/d1.md"}],
    )


def _setup(provider, *, documents=None, budget=None):
    deps, emitted = make_deps(provider, documents=documents or {"doc-1": FakeDoc(_CONTENT)})
    state = TurnState()
    budget = budget or make_budget()
    emitter = TurnEventEmitter(deps=deps, state=state, budget=budget, stream_events=True)
    return deps, emitted, state, budget, emitter


async def test_no_anchor_sentinel_starts_at_window_zero_and_walks_all_windows():
    """Pool chunk has the -1 sentinel: the walk starts at window 0 and covers
    all 3 windows (max_windows=3)."""
    provider = FakeProvider(
        responses=[
            deep_read_json("fact one"),
            deep_read_json("fact two"),
            deep_read_json("fact three"),
        ]
    )
    deps, emitted, state, budget, emitter = _setup(provider)
    state.pool.append(make_chunk("c1", document_id="doc-1", refactored_char_start=-1))

    await run_deep_study(
        DeepStudyArgs(question="q?", document_id="doc-1", source_key="docs/d1.md"),
        context=_ctx(),
        state=state,
        budget=budget,
        deps=deps,
        emitter=emitter,
    )

    studied = state.studied["doc-1"]
    assert studied.windows_read == [0, 1, 2]
    windows = events_of(emitted, TurnEventType.DEEP_STUDY)
    assert [(w["window"], w["of_windows"]) for w in windows] == [(1, 3), (2, 3), (3, 3)]
    assert "fact one" in studied.notes and "fact three" in studied.notes
    assert studied.conclusion == studied.notes


async def test_anchor_offset_selects_the_containing_window():
    """Best anchored pool chunk at offset 170 -> stride 80 -> start window 2."""
    provider = FakeProvider(responses=[deep_read_json("anchored fact")])
    deps, emitted, state, budget, emitter = _setup(provider)
    state.pool.append(
        make_chunk("c1", document_id="doc-1", score=0.9, refactored_char_start=170)
    )
    state.pool.append(
        make_chunk("c2", document_id="doc-1", score=0.3, refactored_char_start=0)
    )

    await run_deep_study(
        DeepStudyArgs(question="q?", document_id="doc-1"),
        context=_ctx(),
        state=state,
        budget=budget,
        deps=deps,
        emitter=emitter,
    )

    assert state.studied["doc-1"].windows_read == [2]  # last window: walk ends


async def test_max_windows_caps_the_walk():
    provider = FakeProvider(
        responses=[deep_read_json("w0"), deep_read_json("w1"), deep_read_json("w2")]
    )
    budget = make_budget(deep_study_max_windows=2)
    deps, emitted, state, budget, emitter = _setup(provider, budget=budget)

    await run_deep_study(
        DeepStudyArgs(question="q?", document_id="doc-1"),
        context=_ctx(),
        state=state,
        budget=budget,
        deps=deps,
        emitter=emitter,
    )

    assert state.studied["doc-1"].windows_read == [0, 1]
    assert len(provider.calls) == 2


async def test_answer_found_stops_the_walk_early():
    provider = FakeProvider(
        responses=[deep_read_json("the answer", answer_found=True)]
    )
    deps, emitted, state, budget, emitter = _setup(provider)

    await run_deep_study(
        DeepStudyArgs(question="q?", document_id="doc-1"),
        context=_ctx(),
        state=state,
        budget=budget,
        deps=deps,
        emitter=emitter,
    )

    assert state.studied["doc-1"].windows_read == [0]
    assert len(provider.calls) == 1


async def test_repeat_study_resumes_past_windows_already_read():
    provider = FakeProvider(responses=[deep_read_json("new fact")])
    budget = make_budget(deep_study_max_windows=1)
    deps, emitted, state, budget, emitter = _setup(provider, budget=budget)
    state.studied["doc-1"] = StudiedDoc(
        document_id="doc-1", windows_read=[0], notes="old fact"
    )

    await run_deep_study(
        DeepStudyArgs(question="q?", document_id="doc-1"),
        context=_ctx(),
        state=state,
        budget=budget,
        deps=deps,
        emitter=emitter,
    )

    assert state.studied["doc-1"].windows_read == [0, 1]  # window 0 skipped free
    assert "old fact" in state.studied["doc-1"].notes
    assert "new fact" in state.studied["doc-1"].notes


async def test_window_findings_enter_the_pool_with_deep_study_provenance():
    provider = FakeProvider(responses=[deep_read_json("extracted value = 42")])
    deps, emitted, state, budget, emitter = _setup(provider)

    await run_deep_study(
        DeepStudyArgs(question="q?", document_id="doc-1", source_key="docs/d1.md"),
        context=_ctx(),
        state=state,
        budget=budget,
        deps=deps,
        emitter=emitter,
    )

    findings = [
        c for c in state.pool if c.provenance == EvidenceChunk.PROVENANCE_DEEP_STUDY
    ]
    assert len(findings) == 1
    assert findings[0].text == "extracted value = 42"
    assert findings[0].document_id == "doc-1"
    assert findings[0].source_key == "docs/d1.md"
    assert findings[0].refactored_char_start == 0


async def test_document_miss_degrades_to_noop():
    provider = FakeProvider()
    deps, emitted, state, budget, emitter = _setup(provider, documents={})

    await run_deep_study(
        DeepStudyArgs(question="q?", document_id="missing-doc"),
        context=_ctx({"document_id": "missing-doc"}),
        state=state,
        budget=budget,
        deps=deps,
        emitter=emitter,
    )

    assert state.studied == {}
    assert provider.calls == []
    assert events_of(emitted, TurnEventType.DEEP_STUDY) == []


async def test_missing_document_ref_degrades_to_noop():
    provider = FakeProvider()
    deps, emitted, state, budget, emitter = _setup(provider)

    await run_deep_study(
        DeepStudyArgs(question="q?"),  # controller gave no ref at all
        context=_ctx(),
        state=state,
        budget=budget,
        deps=deps,
        emitter=emitter,
    )

    assert state.studied == {}
    assert provider.calls == []


async def test_deep_study_doc_budget_blocks_new_documents():
    provider = FakeProvider()
    budget = make_budget(deep_study_max_docs=1)
    deps, emitted, state, budget, emitter = _setup(provider, budget=budget)
    state.studied["other-doc"] = StudiedDoc(document_id="other-doc")

    await run_deep_study(
        DeepStudyArgs(question="q?", document_id="doc-1"),
        context=_ctx(),
        state=state,
        budget=budget,
        deps=deps,
        emitter=emitter,
    )

    assert "doc-1" not in state.studied
    assert provider.calls == []


# ---------------------------------------------------------------------------
# Legal-target namespace (unvalidated model-authored references class)
# ---------------------------------------------------------------------------


async def test_unvetted_target_is_skipped_without_fetch():
    """A target never referenced by the pool/context/studied ledger is
    skipped — a prompt-injected controller cannot fetch arbitrary store
    objects (the fetch seam is never invoked)."""
    provider = FakeProvider()
    fetch_calls: list[tuple] = []
    deps, emitted, state, budget, emitter = _setup(
        provider, documents={"secret-doc": FakeDoc("classified body")}
    )
    real_fetch = deps.fetch_document

    async def recording_fetch(document_id, source_key):
        fetch_calls.append((document_id, source_key))
        return await real_fetch(document_id, source_key)

    deps.fetch_document = recording_fetch
    state.pool.append(make_chunk("c1", document_id="doc-1"))

    await run_deep_study(
        DeepStudyArgs(question="q?", document_id="secret-doc", source_key="secret/key.md"),
        context=_ctx(),  # grants doc-1 only — not the injected target
        state=state,
        budget=budget,
        deps=deps,
        emitter=emitter,
    )

    assert fetch_calls == []  # the unvetted ref never reached the store
    assert state.studied == {}
    assert provider.calls == []


async def test_pool_membership_grants_the_target():
    """A document that entered the evidence pool this turn is a legal target
    even with an empty cross-turn context (a different instance of the same
    namespace class than the context-granted cases above)."""
    provider = FakeProvider(responses=[deep_read_json("pool-granted fact")])
    deps, emitted, state, budget, emitter = _setup(provider)
    state.pool.append(make_chunk("c1", document_id="doc-1"))

    await run_deep_study(
        DeepStudyArgs(question="q?", document_id="doc-1"),
        context=TurnContext(conversation_id="conv-t"),  # nothing granted here
        state=state,
        budget=budget,
        deps=deps,
        emitter=emitter,
    )

    assert state.studied["doc-1"].windows_read  # the study ran


async def test_partially_vetted_ref_is_nulled_before_fetch():
    """When only the source_key is legal, the unvetted document_id must NOT
    reach the fetch seam (it would resolve verbatim across the whole store)."""
    provider = FakeProvider(responses=[deep_read_json("keyed fact")])
    fetch_calls: list[tuple] = []
    deps, emitted, state, budget, emitter = _setup(
        provider, documents={"docs/d1.md": FakeDoc(_CONTENT)}
    )
    real_fetch = deps.fetch_document

    async def recording_fetch(document_id, source_key):
        fetch_calls.append((document_id, source_key))
        return await real_fetch(document_id, source_key)

    deps.fetch_document = recording_fetch

    await run_deep_study(
        DeepStudyArgs(
            question="q?", document_id="forged-doc-id", source_key="docs/d1.md"
        ),
        context=_ctx(),  # grants doc-1 + docs/d1.md; forged-doc-id is not legal
        state=state,
        budget=budget,
        deps=deps,
        emitter=emitter,
    )

    assert fetch_calls == [(None, "docs/d1.md")]
    assert "docs/d1.md" in state.studied
