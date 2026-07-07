# @summary
# Tests for P4 retrieval + recall@k metrics over the ingested pack.
# Four FAST tests assert call wiring, pure-function behavior, aggregation
# rules, and EvalReport shape via monkeypatch + synthetic data; one
# SLOW+INTEGRATION test runs end-to-end against real Weaviate/MinIO.
# Exports: test_retrieve_for_goldens_wires_search_per_query,
#          test_recall_at_k_pure_function,
#          test_aggregate_recall_by_qtype,
#          test_eval_report_is_frozen,
#          test_opentitan_recall_at_5_meets_floor
# Deps: src.eval.runner, src.eval.pack
# @end-summary
"""P4 — retrieval + recall@k metrics tests."""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest

from src.eval.runner import (
    EvalReport,
    RetrievalResults,
    aggregate_recall_by_qtype,
    recall_at_k,
    retrieve_for_goldens,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PACK_DIR = REPO_ROOT / "evals" / "packs" / "opentitan_riscv"


class _FakeEmbedder:
    def embed_documents(self, texts):
        return [[0.0] * 1024 for _ in texts]


def _make_search_result(source: str):
    """Build a minimal SearchResult-like object exposing metadata['source']."""
    from src.vector_db.common.schemas import SearchResult

    return SearchResult(text="dummy", score=1.0, metadata={"source": source})


def test_retrieve_for_goldens_wires_search_per_query(monkeypatch) -> None:
    """retrieve_for_goldens calls search EXACTLY once per golden with the
    right collection, limit=k, and query=<golden.query> (LOCAL-name patch)."""
    from src.eval.pack.schema import Golden
    from src.eval.runner import retrieve as retrieve_mod

    calls: list[dict[str, Any]] = []

    def fake_search(client, query, query_embedding, alpha, limit, **kwargs):
        calls.append(
            {
                "client": client,
                "query": query,
                "alpha": alpha,
                "limit": limit,
                "collection": kwargs.get("collection"),
                "filters": kwargs.get("filters"),
            }
        )
        return [_make_search_result(f"docs/{query}.md")]

    monkeypatch.setattr(retrieve_mod, "search", fake_search)
    monkeypatch.setattr(retrieve_mod, "get_embedding_provider", lambda: _FakeEmbedder())
    monkeypatch.setattr(retrieve_mod, "create_persistent_client", lambda: object())
    monkeypatch.setattr(retrieve_mod, "close_client", lambda _client: None)

    goldens = {
        "factoid": [
            Golden(
                qid="f1",
                qtype="factoid",
                query="alpha",
                expected_answer_span="x",
                expected_source_docs=["docs/alpha.md"],
            ),
            Golden(
                qid="f2",
                qtype="factoid",
                query="beta",
                expected_answer_span="y",
                expected_source_docs=["docs/beta.md"],
            ),
        ],
        "qfs": [
            Golden(
                qid="q1",
                qtype="qfs",
                query="gamma",
                expected_source_docs=["docs/gamma.md"],
            ),
        ],
    }

    results = retrieve_for_goldens("EvalCollection", goldens, k=5)

    assert len(calls) == 3
    queries_in_order = [c["query"] for c in calls]
    assert queries_in_order == ["alpha", "beta", "gamma"]
    for c in calls:
        assert c["collection"] == "EvalCollection"
        assert c["limit"] == 5
    assert results.collection_name == "EvalCollection"
    assert results.k == 5
    assert set(results.per_query) == {"f1", "f2", "q1"}
    assert results.per_query["f1"].retrieved_sources == ("docs/alpha.md",)


@pytest.mark.parametrize(
    "retrieved, expected, score",
    [
        (["docs/a.md", "docs/b.md", "docs/c.md"], ["docs/a.md"], 1.0),
        (["docs/x.md", "docs/y.md"], ["docs/a.md"], 0.0),
        (["docs/a.md", "docs/y.md"], ["docs/a.md", "docs/b.md"], 0.5),
        (["docs/a.md", "docs/b.md"], ["docs/a.md", "docs/b.md"], 1.0),
        ([], ["docs/a.md"], 0.0),
    ],
)
def test_recall_at_k_pure_function(retrieved, expected, score) -> None:
    """recall_at_k is a pure set-intersection / len(expected) function."""
    assert recall_at_k(retrieved, expected) == pytest.approx(score)


def test_aggregate_recall_by_qtype() -> None:
    """Aggregation: mean per qtype over scoreable goldens; empty-gold qtypes omitted."""
    from src.eval.pack.schema import Golden
    from src.eval.runner.retrieve import QueryRetrievalResult

    goldens = {
        "factoid": [
            Golden(
                qid="f1",
                qtype="factoid",
                query="q1",
                expected_answer_span="x",
                expected_source_docs=["docs/a.md"],
            ),
            Golden(
                qid="f2",
                qtype="factoid",
                query="q2",
                expected_answer_span="y",
                expected_source_docs=["docs/b.md"],
            ),
        ],
        "qfs": [
            Golden(
                qid="q1",
                qtype="qfs",
                query="q3",
                expected_source_docs=["docs/c.md", "docs/d.md"],
            ),
        ],
        "adversarial": [
            Golden(
                qid="a1",
                qtype="adversarial",
                query="q4",
                expected_source_docs=[],
            ),
        ],
    }
    per_query = {
        "f1": QueryRetrievalResult(
            qid="f1",
            qtype="factoid",
            query="q1",
            retrieved_sources=("docs/a.md", "docs/z.md"),
            k=5,
        ),
        "f2": QueryRetrievalResult(
            qid="f2",
            qtype="factoid",
            query="q2",
            retrieved_sources=("docs/x.md", "docs/y.md"),
            k=5,
        ),
        "q1": QueryRetrievalResult(
            qid="q1",
            qtype="qfs",
            query="q3",
            retrieved_sources=("docs/c.md", "docs/m.md"),
            k=5,
        ),
        "a1": QueryRetrievalResult(
            qid="a1",
            qtype="adversarial",
            query="q4",
            retrieved_sources=("docs/whatever.md",),
            k=5,
        ),
    }
    results = RetrievalResults(collection_name="C", k=5, per_query=per_query)

    out = aggregate_recall_by_qtype(results, goldens)

    assert out["factoid"] == pytest.approx(0.5)
    assert out["qfs"] == pytest.approx(0.5)
    assert "adversarial" not in out


def test_eval_report_is_frozen() -> None:
    """EvalReport is a frozen dataclass with the spec'd field shape."""
    report = EvalReport(
        collection_name="C",
        k=5,
        per_query_recall={"f1": 1.0},
        recall_by_qtype={"factoid": 1.0},
        total_queries_scored=1,
        total_queries_skipped=0,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.k = 99  # type: ignore[misc]
    assert report.collection_name == "C"
    assert report.total_queries_scored == 1
    assert report.total_queries_skipped == 0


@pytest.mark.slow
@pytest.mark.integration
def test_opentitan_recall_at_5_meets_floor(tmp_path: Path) -> None:
    """Live: factoid recall@5 against the freshly-ingested OpenTitan pack
    must be >= 0.5. Skips on infra absence. Best-effort cleanup."""
    pytest.importorskip("weaviate")
    import socket

    from config.settings import MINIO_ENDPOINT
    from src.eval.pack import load_pack
    from src.eval.runner import execute_plan, plan_pack_ingest
    from src.vector_db import close_client, create_persistent_client

    try:
        client = create_persistent_client()
    except Exception as exc:  # pragma: no cover - infra-dependent
        pytest.skip(f"Live Weaviate not reachable: {exc}")

    try:
        host, _, port = MINIO_ENDPOINT.partition(":")
        with socket.create_connection(
            (host or "localhost", int(port or "9000")), timeout=2
        ):
            pass
    except Exception as exc:  # pragma: no cover - infra-dependent
        try:
            close_client(client)
        except Exception:
            pass
        pytest.skip(f"Live MinIO ({MINIO_ENDPOINT}) not reachable: {exc}")

    pack = load_pack(PACK_DIR)
    plan = plan_pack_ingest(pack, PACK_DIR)

    report = None
    try:
        report = execute_plan(plan)
        assert report.failed == 0, f"unexpected ingest failures: {report.errors}"

        results = retrieve_for_goldens(plan.collection_name, pack.goldens, k=5)
        recall_by_qtype = aggregate_recall_by_qtype(results, pack.goldens)
        print(f"recall_by_qtype={recall_by_qtype}")

        if recall_by_qtype.get("factoid", 0.0) < 0.5:
            pytest.fail(
                f"factoid recall@5 < 0.5 floor; full recall_by_qtype={recall_by_qtype}"
            )
    finally:
        try:
            target = report.collection_name if report else plan.collection_name
            if client.collections.exists(target):
                client.collections.delete(target)
        except Exception:
            pass
        try:
            close_client(client)
        except Exception:
            pass
