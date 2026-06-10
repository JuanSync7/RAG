# @summary
# Regression pin for a Temporal payload-decode contract bug discovered by the
# live ingest->serve e2e (tests/integration/test_ingest_serve_e2e.py).
#
# The embedding pipeline state types ``errors`` as ``list[str]`` (see
# src/ingest/embedding/state.py) and ``EmbeddingResult.errors`` is likewise
# ``list[str]``. BUT src/ingest/embedding/nodes/embedding_storage.py builds
# per-batch failures as DICTS (``_embed_batches`` -> ``errors.append({...})``,
# line ~144) and merges them into the shared ``errors`` field (line ~252).
# ``embedding_pipeline_activity`` then forwards ``result["errors"]`` straight
# into ``EmbeddingResult(errors=...)``.
#
# This serializes fine on the activity side, but when the workflow decodes the
# activity result Temporal coerces each element to ``str`` against the dataclass
# type hint and raises ``TypeError: Failed converting field errors on dataclass
# EmbeddingResult`` / ``Failed converting list[str] index 0`` — which fails the
# workflow task. Because workflow-task failures retry indefinitely, a single
# batch embedding failure wedges the whole ingest workflow forever.
#
# This test pins the behaviour at the Temporal payload-converter boundary with
# NO live infra, so a regression (or a future fix that makes dict errors decode)
# is caught in the offline gate.
# Exports: test_string_errors_round_trip, test_dict_errors_break_decode,
#          test_embedding_storage_emits_dict_errors
# Deps: temporalio.converter, src.ingest.temporal.activities,
#       src.ingest.embedding.nodes.embedding_storage
# @end-summary
"""Pin the EmbeddingResult Temporal payload-decode contract.

Discovered by the live ingest->serve e2e: ``embedding_storage`` appends
``dict`` error entries to a state field that ``EmbeddingResult`` declares as
``list[str]``. The mismatch is invisible to mocked tests (they never cross the
Temporal payload boundary) and only surfaces against a real Temporal server —
the exact gap this initiative exists to close. These tests reproduce it
offline and deterministically.

NOTE: this test currently asserts the BROKEN behaviour (dict errors fail to
decode). When the product bug is fixed — by coercing batch errors to ``str``
before they reach ``EmbeddingResult`` (e.g. ``str(err)`` / ``json.dumps(err)``
in embedding_storage or the activity) — ``test_dict_errors_break_decode`` will
start failing and should be flipped to assert a clean round-trip. That flip is
the intended signal that the fix landed.
"""
from __future__ import annotations

import pytest

from temporalio.converter import default as _default_converter

from src.ingest.temporal.activities import EmbeddingResult


def _round_trip(result: EmbeddingResult) -> EmbeddingResult:
    """Encode then decode an EmbeddingResult exactly as Temporal would.

    Mirrors the activity-result hand-off: the worker serializes the activity
    return value to payloads, and the workflow decodes them back using the
    dataclass type hint (``list[str]`` for ``errors``).
    """
    conv = _default_converter().payload_converter
    payloads = conv.to_payloads([result])
    return conv.from_payloads(payloads, [EmbeddingResult])[0]


def _make(errors: list) -> EmbeddingResult:
    return EmbeddingResult(
        errors=errors,
        stored_count=0,
        metadata_summary="",
        metadata_keywords=[],
        processing_log=[],
    )


def test_string_errors_round_trip() -> None:
    """Positive control: string errors (the declared contract) decode cleanly.

    Gives the dict-error assertion teeth — proving the failure is specifically
    the ``dict``-in-``list[str]`` violation, not a broken converter.
    """
    decoded = _round_trip(_make(["chunking:boom", "commit:kaboom"]))
    assert decoded.errors == ["chunking:boom", "commit:kaboom"]
    assert decoded.stored_count == 0


def test_dict_errors_break_decode() -> None:
    """The dict shape ``embedding_storage`` actually emits fails to decode.

    This is the exact failure the live ingest workflow hit
    (``Failed converting field errors on dataclass EmbeddingResult``). Pins the
    bug; flip to assert a clean round-trip once errors are coerced to ``str``.
    """
    batch_error = {
        "type": "batch_embedding_failure",
        "batch_index": 1,
        "chunk_range": "0-3",
        "error": "boom",
    }
    with pytest.raises(TypeError, match="Failed converting"):
        _round_trip(_make([batch_error]))


def test_sibling_result_dataclasses_honour_list_str_contract() -> None:
    """The OTHER ingest activity results round-trip with string errors/logs.

    A repo-wide audit (2026-06-10) found ``embedding_storage`` is the SOLE place
    that puts dicts into a ``list[str]`` error field — DocProcessingResult and
    DeleteSourceResult are fed only strings (f-strings) by their activities.
    This guard pins that: it round-trips both with realistic STRING payloads and
    asserts they decode. It turns RED if a future change starts feeding either a
    dict (the same regression class as the EmbeddingResult bug) BEFORE that
    change can wedge a live workflow.
    """
    from src.ingest.temporal.activities import (
        DeleteSourceResult,
        DocProcessingResult,
    )

    conv = _default_converter().payload_converter

    doc = DocProcessingResult(
        errors=["clean_store_write_failed: boom"],
        source_hash="abc",
        clean_hash="def",
        processing_log=["doc_processing:skipped:unchanged"],
    )
    decoded_doc = conv.from_payloads(conv.to_payloads([doc]), [DocProcessingResult])[0]
    assert decoded_doc.errors == ["clean_store_write_failed: boom"]
    assert decoded_doc.processing_log == ["doc_processing:skipped:unchanged"]

    dele = DeleteSourceResult(
        weaviate_deleted=0,
        minio_deleted=False,
        errors=["weaviate_delete_failed:boom", "minio_delete_failed:boom"],
    )
    decoded_del = conv.from_payloads(conv.to_payloads([dele]), [DeleteSourceResult])[0]
    assert decoded_del.errors == [
        "weaviate_delete_failed:boom",
        "minio_delete_failed:boom",
    ]


def test_embedding_storage_emits_dict_errors() -> None:
    """Source-level proof the producer emits dicts into a ``list[str]`` field.

    Drives the real ``_embed_batches`` with an embedder whose ``embed_documents``
    always raises, so the batch exhausts retries and the function records a
    failure. The recorded entry MUST be a ``dict`` (not ``str``) — which is what
    later breaks the Temporal decode pinned above. If a fix coerces these to
    strings, this assertion flips and flags that the contract was repaired.
    """
    from src.ingest.embedding.nodes.embedding_storage import _embed_batches

    class _AlwaysFailEmbedder:
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError("embed backend down")

    # One batch, retries exhaust immediately (max_retries=1) -> one failure entry.
    _vectors, errors, success_mask = _embed_batches(
        _AlwaysFailEmbedder(), [["chunk a", "chunk b"]], max_retries=1
    )

    assert success_mask == [False]
    assert len(errors) == 1
    # The load-bearing assertion: the entry is a dict, violating list[str].
    assert isinstance(errors[0], dict)
    assert errors[0]["type"] == "batch_embedding_failure"
