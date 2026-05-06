# @summary
# Tests the KGPhase2bRunner — Phase 2b sibling that reads clean documents
# from CleanDocumentStore and ingests them into the KG via KGIngestClient.
# Exports: (test module)
# Deps: pytest, pathlib, src.ingest.common.clean_store,
#       src.ingest.kg.phase2b, src.knowledge_graph.common.protocols
# @end-summary
"""Step 6a: KGPhase2bRunner contract tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from src.ingest.common.clean_store import CleanDocumentStore
from src.ingest.kg.phase2b import KGPhase2bRunner
from kgweave.knowledge_graph.common.protocols import KGIngestClient, KGIngestResult


class _RecorderClient:
    """KGIngestClient stub that records every extract_and_commit call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, Optional[Dict[str, Any]]]] = []

    def extract_and_commit(
        self,
        source_key: str,
        clean_text: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> KGIngestResult:
        self.calls.append((source_key, clean_text, meta))
        return KGIngestResult(
            source_key=source_key,
            entities_added=1,
            triples_added=0,
            chunks_processed=1,
            elapsed_ms=0.5,
        )


@pytest.fixture
def store(tmp_path: Path) -> CleanDocumentStore:
    s = CleanDocumentStore(tmp_path / "clean")
    s.write("doc1.md", "The CPU executes instructions.", {"source_name": "doc1"})
    s.write("doc2.md", "The GPU handles graphics.", {"source_name": "doc2"})
    return s


@pytest.fixture
def client() -> _RecorderClient:
    return _RecorderClient()


def test_recorder_satisfies_ingest_protocol(client: _RecorderClient) -> None:
    assert isinstance(client, KGIngestClient)


def test_run_one_calls_extract_and_commit(
    store: CleanDocumentStore, client: _RecorderClient
) -> None:
    runner = KGPhase2bRunner(clean_store=store, kg_client=client)
    result = runner.run_one("doc1.md")

    assert isinstance(result, KGIngestResult)
    assert result.source_key == "doc1.md"
    assert len(client.calls) == 1
    source_key, clean_text, meta = client.calls[0]
    assert source_key == "doc1.md"
    assert clean_text == "The CPU executes instructions."
    assert meta is not None
    assert meta.get("source_name") == "doc1"


def test_run_one_missing_source_raises(
    store: CleanDocumentStore, client: _RecorderClient
) -> None:
    runner = KGPhase2bRunner(clean_store=store, kg_client=client)
    with pytest.raises(FileNotFoundError):
        runner.run_one("missing.md")
    assert client.calls == []


def test_run_many_returns_results_in_order(
    store: CleanDocumentStore, client: _RecorderClient
) -> None:
    runner = KGPhase2bRunner(clean_store=store, kg_client=client)
    results = runner.run_many(["doc1.md", "doc2.md"])

    assert [r.source_key for r in results] == ["doc1.md", "doc2.md"]
    assert [c[0] for c in client.calls] == ["doc1.md", "doc2.md"]


def test_run_many_skip_missing_continues(
    store: CleanDocumentStore, client: _RecorderClient
) -> None:
    runner = KGPhase2bRunner(clean_store=store, kg_client=client)
    results = runner.run_many(
        ["doc1.md", "missing.md", "doc2.md"], skip_missing=True
    )
    assert [r.source_key for r in results] == ["doc1.md", "doc2.md"]


def test_run_many_default_propagates_missing(
    store: CleanDocumentStore, client: _RecorderClient
) -> None:
    runner = KGPhase2bRunner(clean_store=store, kg_client=client)
    with pytest.raises(FileNotFoundError):
        runner.run_many(["doc1.md", "missing.md"])


def test_run_all_processes_every_key(
    store: CleanDocumentStore, client: _RecorderClient
) -> None:
    runner = KGPhase2bRunner(clean_store=store, kg_client=client)
    results = runner.run_all()
    keys = sorted(r.source_key for r in results)
    assert keys == ["doc1.md", "doc2.md"]
